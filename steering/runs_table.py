"""One row per recorded run: settings, timing, clipping, gripper use, start pose and tip motion.

    uv run python steering/runs_table.py                       # scans steering/results/runs/
    uv run python steering/runs_table.py --runs DIR --out CSV

Writes steering/results/runs.csv (atomically: a temp file in the same directory, then os.replace) and
prints it. The existing CSV is read as ``utf-8-sig``, so a BOM added by a spreadsheet is fine.

The ``notes`` column (and any column added by hand) is carried over on every rewrite. An old row is
matched to a scanned run by ``run_dir`` first. Only an old row whose run directory no longer exists
(renamed or moved) may match by ``tag`` instead, each such row at most once, and it is then replaced by
the new row rather than kept as well. Old rows whose run directory has gone and that match nothing,
and hand-added rows with an empty ``run_dir``, are kept as they were (the latter at the end). Retired
columns (``gripper_close_events``) are dropped. Everything else is recomputed.

Columns (control-loop ticks only, phase 1; teardown/reset ticks are left out):
  run_dir, tag, start_time, task, inference_type, max_relative_target, display_data (on/off, from
  argv; off is lerobot-rollout's default), strategy, n_ticks,
  duration_s          sum over episodes of last - first tick time
  effective_hz        (ticks - episodes) / duration, as plot_run.py
  inference_median_ms, inference_max_ms, n_chunks
  pct_ticks_clipped   % of ticks with any joint clipped by max_relative_target
  gripper_min         measured gripper (0 closed .. 100 open)
  gripper_close_events_policy    closes in ``action_policy`` (the gripper the policy asked for, before
                                 the max_relative_target clamp)
  gripper_close_events_measured  closes in ``state`` (measured Present_Position)
                      A close is a drop below 5 after having been above 15 (hysteresis; NaN skipped),
                      counted per episode and summed. The measured gripper only counts a close when it
                      gets below 5, so a grasp that stalls on a thick object above 5 is not counted (on
                      the pens it closes to ~0.1-0.6). Replaces the former ``gripper_close_events``
                      (from ``action_sent``).
  start_pose          inside / OUTSIDE (joint +distance in deg) the checkpoint's trained state range
                      (q01..q99, arm frame; steering/check_state_range.py functions)
  end_pose            last measured state, pan/lift/elbow/wflex/wroll/grip
  tip_distance_m      path length of the gripper tip (steering/so101_fk.py), summed per episode
  tip_min_z_m         lowest tip height above base_link's origin (the table is ~-0.0024 m if the base
                      stands flat on it; the real table height is not recorded)
  notes               hand-typed, preserved
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from check_state_range import (
    JOINTS,
    POLICY_REPO,
    STATE_STATS_FILE,
    TOLERANCE,
    distance_outside,
    load_frame,
    load_quantiles,
    range_to_arm,
)
from safetensors.numpy import load_file
from so101_fk import SO101FK

STEERING = Path(__file__).parent
RUNS = STEERING / "results" / "runs"
OUT = STEERING / "results" / "runs.csv"
PHASE_POLICY = 1
GRIP_OPEN_ABOVE, GRIP_CLOSED_BELOW = 15.0, 5.0
SHORT = ("pan", "lift", "elbow", "wflex", "wroll", "grip")
COLUMNS = (
    "run_dir", "tag", "start_time", "task", "inference_type", "max_relative_target", "display_data",
    "strategy", "n_ticks", "duration_s", "effective_hz", "inference_median_ms", "inference_max_ms",
    "n_chunks", "pct_ticks_clipped", "gripper_min", "gripper_close_events_policy",
    "gripper_close_events_measured", "start_pose", "end_pose", "tip_distance_m", "tip_min_z_m", "notes",
)  # fmt: skip
RETIRED = frozenset({"gripper_close_events"})  # former columns: not carried over as hand-added ones

_RANGE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def argv_flag(meta: dict, name: str) -> str | None:
    """Value of the last ``--name=value`` (or ``--name value``) in the run's argv."""
    argv = meta.get("argv") or []
    val = None
    for i, a in enumerate(argv):
        if a.startswith(f"--{name}="):
            val = a.split("=", 1)[1]
        elif a == f"--{name}" and i + 1 < len(argv):
            val = argv[i + 1]
    return val


def trained_state_range(policy_path: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Arm-frame (low, high) of the checkpoint's trained state q01..q99 (local dir or Hub repo)."""
    key = policy_path or POLICY_REPO
    if key in _RANGE_CACHE:
        return _RANGE_CACHE[key]
    d = Path(key)
    if not d.is_absolute() and not d.exists():
        d = STEERING.parent / key
    if (d / "config.json").is_file():
        config = json.loads((d / "config.json").read_text())
        signs = np.asarray(config["joint_signs"], dtype=np.float64)
        offsets = np.asarray(config["joint_offsets"], dtype=np.float64)
        stats = load_file(str(d / STATE_STATS_FILE))
        q01 = stats["observation.state.q01"].astype(np.float64)
        q99 = stats["observation.state.q99"].astype(np.float64)
        mask = stats.get("observation.state.mask")
        if mask is not None and not mask.astype(bool).all():
            q01[~mask.astype(bool)], q99[~mask.astype(bool)] = -np.inf, np.inf
    else:
        repo = key if "/" in key else POLICY_REPO
        signs, offsets = load_frame(repo)
        q01, q99 = load_quantiles(repo, STATE_STATS_FILE, "observation.state")
    _RANGE_CACHE[key] = range_to_arm(q01, q99, signs, offsets)
    return _RANGE_CACHE[key]


def episode_segments(ticks: dict, rows: np.ndarray) -> list[np.ndarray]:
    """Split rows per episode label (one segment for single-episode runs)."""
    if "episode" not in ticks or not len(rows):
        return [rows] if len(rows) else []
    cuts = np.flatnonzero(np.diff(ticks["episode"][rows]) != 0) + 1
    return [s for s in np.split(rows, cuts) if len(s)]


def close_events(cmd: np.ndarray) -> int:
    """Drops of the command below GRIP_CLOSED_BELOW after having been above GRIP_OPEN_ABOVE."""
    armed, n = False, 0
    for v in cmd:
        if not np.isfinite(v):
            continue
        if v > GRIP_OPEN_ABOVE:
            armed = True
        elif v < GRIP_CLOSED_BELOW and armed:
            n, armed = n + 1, False
    return n


def _fmt(v: float, nd: int) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _num(v: object) -> str:
    """'6', 6 and 6.0 all as '6'; non-numbers as they are."""
    try:
        return f"{float(v):g}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "" if v is None else str(v)


def run_row(run_dir: Path, fk: SO101FK) -> dict[str, str]:
    """The table row of one run directory."""
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").is_file() else {}
    row = dict.fromkeys(COLUMNS, "")
    display = argv_flag(meta, "display_data")
    row.update(
        run_dir=run_dir.name,
        tag=str(meta.get("tag", run_dir.name.split("_", 1)[-1])),
        start_time=str(meta.get("started_at", "")),
        task=str(meta.get("task") or argv_flag(meta, "task") or ""),
        inference_type=str(meta.get("inference_type") or argv_flag(meta, "inference.type") or ""),
        max_relative_target=_num(
            (meta.get("robot") or {}).get("max_relative_target")
            or argv_flag(meta, "robot.max_relative_target")
        ),
        display_data="on" if (display or "").lower() in ("true", "1", "yes") else "off",
        strategy=str(meta.get("strategy_type") or argv_flag(meta, "strategy.type") or ""),
    )
    if not (run_dir / "ticks.npz").is_file():
        row["n_ticks"] = "0"
        return row
    ticks = dict(np.load(run_dir / "ticks.npz"))
    chunks = dict(np.load(run_dir / "chunks.npz")) if (run_dir / "chunks.npz").is_file() else {}
    phase = ticks.get("phase")
    rows = np.flatnonzero(phase == PHASE_POLICY) if phase is not None else np.arange(len(ticks["t"]))
    if phase is not None and len(rows) < 2:
        rows = np.arange(len(ticks["t"]))
    row["n_ticks"] = str(len(rows))
    dur = chunks.get("duration", np.zeros(0))
    row["n_chunks"] = str(len(dur))
    if len(dur):
        row["inference_median_ms"] = _fmt(float(np.median(dur)) * 1e3, 0)
        row["inference_max_ms"] = _fmt(float(dur.max()) * 1e3, 0)
    if len(rows) < 2:
        return row
    segs = episode_segments(ticks, rows)
    t = ticks["t"]
    duration = float(sum(t[s[-1]] - t[s[0]] for s in segs))
    state = ticks["state"][rows]
    grip = JOINTS.index("gripper")
    s_low, s_high = trained_state_range(meta.get("policy_path") or argv_flag(meta, "policy.path"))
    out = distance_outside(state[0], s_low, s_high)
    bad = [f"{SHORT[i]} {out[i]:+.1f}" for i in range(len(JOINTS)) if out[i] > TOLERANCE]
    tip_dist = 0.0
    tip_min = np.inf
    for s in segs:
        tip = fk.tip(ticks["state"][s])
        tip_dist += float(np.nansum(np.linalg.norm(np.diff(tip, axis=0), axis=1)))
        tip_min = min(tip_min, float(np.nanmin(tip[:, 2])))
    row.update(
        duration_s=_fmt(duration, 1),
        effective_hz=_fmt((len(rows) - len(segs)) / duration if duration else np.nan, 1),
        pct_ticks_clipped=_fmt(100.0 * float(ticks["clipped"][rows].any(axis=1).mean()), 1),
        gripper_min=_fmt(float(np.nanmin(state[:, grip])), 1),
        gripper_close_events_policy=str(sum(close_events(ticks["action_policy"][s, grip]) for s in segs)),
        gripper_close_events_measured=str(sum(close_events(ticks["state"][s, grip]) for s in segs)),
        start_pose="inside" if not bad else "OUTSIDE: " + ", ".join(bad),
        end_pose="/".join(f"{v:.0f}" for v in state[-1]),
        tip_distance_m=_fmt(tip_dist, 3),
        tip_min_z_m=_fmt(tip_min, 3),
    )
    return row


def read_existing(path: Path) -> list[dict[str, str]]:
    """Rows of an existing runs.csv (empty if none)."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def merge(new: list[dict[str, str]], old: list[dict[str, str]], runs_root: Path) -> list[dict[str, str]]:
    """New rows with the old notes and hand-added columns; see the module docstring for the matching."""
    present = {r["run_dir"] for r in new}
    by_dir = {r["run_dir"]: r for r in old if r.get("run_dir")}
    gone_by_tag: dict[str, list[dict[str, str]]] = {}  # old rows whose run directory no longer exists
    for r in old:
        d = r.get("run_dir") or ""
        if d and d not in present and not (runs_root / d).is_dir() and r.get("tag"):
            gone_by_tag.setdefault(r["tag"], []).append(r)
    used: set[int] = set()
    for r in new:
        prev = by_dir.get(r["run_dir"])
        if prev is None:
            prev = next((c for c in gone_by_tag.get(r["tag"], []) if id(c) not in used), None)
        if prev is None:
            continue
        used.add(id(prev))
        r["notes"] = prev.get("notes") or ""
        for k, v in prev.items():  # columns added to the CSV by hand survive too
            if k and k not in COLUMNS and k not in RETIRED:
                r[k] = v or ""
    kept = [
        {k: v or "" for k, v in r.items() if k and k not in RETIRED}
        for r in old
        if id(r) not in used and not (r.get("run_dir") and r["run_dir"] in present)
    ]
    rows = sorted(new + [r for r in kept if r.get("run_dir")], key=lambda r: r["run_dir"])
    return rows + [r for r in kept if not r.get("run_dir")]  # hand-added rows, in their old order


def write_atomic(path: Path, rows: list[dict[str, str]], cols: list[str]) -> None:
    """Write the CSV to a temp file next to ``path``, then os.replace it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def print_table(rows: list[dict[str, str]], width: int = 28) -> None:
    """Aligned print of every column, long cells cut to ``width``."""
    cols = list(COLUMNS) + sorted({k for r in rows for k in r if k and k not in COLUMNS})

    def cut(s: str) -> str:
        return s if len(s) <= width else s[: width - 1] + "…"

    w = {c: max(len(c), *(len(cut(r.get(c, "") or "")) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(cut(r.get(c, "") or "").ljust(w[c]) for c in cols))


def main() -> None:
    """Scan, merge, write, print."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    fk = SO101FK()
    run_dirs = sorted(d for d in args.runs.iterdir() if d.is_dir() and (d / "meta.json").is_file())
    rows = merge([run_row(d, fk) for d in run_dirs], read_existing(args.out), args.runs)
    cols = list(COLUMNS) + sorted({k for r in rows for k in r if k and k not in COLUMNS})
    write_atomic(args.out, rows, cols)
    print_table(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
