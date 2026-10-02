"""Plot and summarise a run recorded by steering/rollout.py.

    uv run python steering/plot_run.py steering/results/runs/<stamp>_<tag>
    uv run python steering/plot_run.py --self-test [DIR]  # fake run + fake episodic run (DIR_episodic), no hardware

Writes PNGs to RUN_DIR/plots/ and prints a one-screen summary (also saved as plots/summary.txt):
  joints.png    per joint over time: measured state, policy command, sent command, the checkpoint's
                trained state range (q01..q99, arm frame, shaded) and red marks on clipped ticks.
  chunks.png    every predicted chunk as a short segment at fps, for shoulder_lift and elbow_flex, over
                the sent command: sync chunks from the end of their inference call, RTC chunks
                (inference_delay >= 0) with step k at t_start + k/fps (an approximation of when the
                steps run).
  cadence.png   tick interval over time against the 33.3 ms target, inference calls shaded, and a
                histogram of the intervals.
  frames.png    contact sheet of the saved camera frames with their timestamps.

The trained range comes from the run's policy path (a local checkpoint directory, or a Hub repo via
steering/check_state_range.py), falling back to that script's default repo. Only control-loop ticks
(phase 1) are summarised; teardown ticks (return to the initial pose) are drawn greyed, with the
measured state dotted there (those rows repeat one stale reading).

Multi-episode runs (``--strategy.type=episodic``, or more than one attempt in meta.json "episodes")
also get, besides the whole-run plots above (resets greyed, like teardown):
  ep<k>/joints.png, chunks.png, frames.png   attempt k's policy phase, time from its start, with its
                reset greyed after it (reset ticks are only the 1 s return to the start pose).
  episodes.png  every attempt's policy phase overlaid per joint (measured state), time from the
                episode start, one colour per saved episode; discarded / unsaved attempts grey dashed.
  summary.txt   a session line and one row per attempt (duration, Hz, % clipped, distance per
                joint, gripper min/max, ended early, discarded) on top of the usual summary.
Single-episode runs are plotted exactly as before.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from check_state_range import (  # noqa: E402
    ACTION_STATS_FILE,
    JOINTS,
    POLICY_REPO,
    STATE_STATS_FILE,
    distance_outside,
    load_frame,
    load_quantiles,
    model_to_arm,
    range_to_arm,
)
from safetensors.numpy import load_file  # noqa: E402

STEERING = Path(__file__).parent
LOCAL_1CAM = STEERING / "checkpoints" / "MolmoAct2-SO100_101-LeRobot-1cam"
TARGET_MS = 1000.0 / 30.0
TOL = 1e-3

# Reference categorical palette (dataviz skill), fixed order; status red reserved for "clipped".
C_STATE, C_POLICY, C_SENT = "#2a78d6", "#eb6834", "#1baf7a"
C_CLIP, C_BAND, C_INK, C_MUTED, C_GRID = "#d03b3b", "#cde2fb", "#0b0b0b", "#898781", "#e8e7e3"
# Episode overlay: the reference categorical order (light mode), never cycled; a 9th+ saved
# episode falls back to muted grey (named in the legend), as do discarded attempts (dashed).
C_EPISODES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
PHASE_POLICY, PHASE_TEARDOWN, PHASE_RESET = 1, 2, 3  # rollout.py's codes (0 = pre)
GREYED = {PHASE_TEARDOWN: "teardown", PHASE_RESET: "reset"}
SPAN_ALPHA = {"teardown": 0.7, "reset": 0.4}  # reset windows a lighter grey than teardown
MAX_GAP_S = 1.0  # episodic runs: no line across a longer tick gap (the reset loop sends nothing)


# ---------------------------------------------------------------------------
# Checkpoint stats (arm frame)
# ---------------------------------------------------------------------------


def _local_quantiles(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Same as check_state_range.load_quantiles, for a local file."""
    stats = load_file(str(path))
    q01, q99 = stats[f"{key}.q01"].astype(np.float64), stats[f"{key}.q99"].astype(np.float64)
    mask = stats.get(f"{key}.mask")
    if mask is not None and not mask.astype(bool).all():
        q01, q99 = q01.copy(), q99.copy()
        q01[~mask.astype(bool)], q99[~mask.astype(bool)] = -np.inf, np.inf
    return q01, q99


def load_stats(policy_path: str | None) -> dict[str, np.ndarray]:
    """signs, offsets and model-frame state/action q01/q99 for a local dir or a Hub repo."""
    candidates = [p for p in (policy_path, str(LOCAL_1CAM)) if p]
    for cand in candidates:
        d = Path(cand)
        if not d.is_absolute() and not d.exists():
            d = STEERING.parent / cand
        if d.is_dir() and (d / "config.json").is_file():
            config = json.loads((d / "config.json").read_text())
            s01, s99 = _local_quantiles(d / STATE_STATS_FILE, "observation.state")
            a01, a99 = _local_quantiles(d / ACTION_STATS_FILE, "action")
            return {
                "signs": np.asarray(config["joint_signs"], dtype=np.float64),
                "offsets": np.asarray(config["joint_offsets"], dtype=np.float64),
                "s01": s01,
                "s99": s99,
                "a01": a01,
                "a99": a99,
                "source": str(d),
            }
    repo = (
        policy_path if policy_path and "/" in policy_path and not Path(policy_path).exists() else POLICY_REPO
    )
    signs, offsets = load_frame(repo)
    s01, s99 = load_quantiles(repo, STATE_STATS_FILE, "observation.state")
    a01, a99 = load_quantiles(repo, ACTION_STATS_FILE, "action")
    return {
        "signs": signs,
        "offsets": offsets,
        "s01": s01,
        "s99": s99,
        "a01": a01,
        "a99": a99,
        "source": repo,
    }


def chunk_norm_to_arm(chunk_norm: np.ndarray, policy_path: str | None) -> np.ndarray:
    """Postprocessor in numpy: clamp to [-1, 1], unnormalise (QUANTILES), model frame -> arm frame."""
    st = load_stats(policy_path)
    n = len(JOINTS)
    y = np.clip(np.asarray(chunk_norm, dtype=np.float64)[..., :n], -1.0, 1.0)
    q01, q99 = st["a01"], st["a99"]
    finite = np.isfinite(q01) & np.isfinite(q99)
    span, low = np.where(finite, q99 - q01, 2.0), np.where(finite, q01, -1.0)
    x = np.where(finite, (y + 1.0) / 2.0 * span + low, y)  # masked dims pass through
    return model_to_arm(x, st["signs"], st["offsets"]).astype(np.float32)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    """ticks, chunks and meta of a run dir (chunks/meta empty if missing)."""
    ticks = dict(np.load(run_dir / "ticks.npz"))
    chunks = dict(np.load(run_dir / "chunks.npz")) if (run_dir / "chunks.npz").is_file() else {}
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").is_file() else {}
    return ticks, chunks, meta


def _spans(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) times of each contiguous run of True in mask."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            stop = i if m else i - 1
            out.append((float(t[start]), float(t[stop])))
            start = None
    return out


def _gapped(t: np.ndarray, gaps: np.ndarray, *ys: np.ndarray) -> tuple[np.ndarray, ...]:
    """Insert a NaN row into t and each of ys at every index in gaps, so lines break there."""
    if not len(gaps):
        return (t, *ys)
    return (
        np.insert(t.astype(np.float64), gaps, np.nan),
        *(np.insert(y.astype(np.float64), gaps, np.nan) for y in ys),
    )


def _gap_index(ticks: dict) -> np.ndarray:
    """Insert positions for _gapped: tick gaps > MAX_GAP_S in runs with reset rows (else none)."""
    if not (ticks["phase"] == PHASE_RESET).any():
        return np.zeros(0, dtype=int)
    return np.flatnonzero(np.diff(ticks["t"]) > MAX_GAP_S) + 1


def episode_list(ticks: dict, meta: dict) -> list[dict]:
    """meta.json's per-attempt entries, or ones derived from the tick labels (older/partial runs)."""
    eps = [dict(e) for e in meta.get("episodes") or []]
    if eps or "episode" not in ticks:
        return eps
    pol = ticks["phase"] == PHASE_POLICY
    for k in np.unique(ticks["episode"][pol]):
        t = ticks["t"][pol & (ticks["episode"] == k)]
        eps.append({"index": int(k), "policy": {"start": float(t[0]), "end": float(t[-1])}, "reset": None})
    return eps


def is_multi_episode(ticks: dict, meta: dict) -> bool:
    """True for an episodic session (per-episode outputs); False keeps the single-run layout."""
    if "episode" not in ticks:
        return False
    return len(episode_list(ticks, meta)) > 1 or meta.get("strategy_type") == "episodic"


def episode_segments(ticks: dict, mask: np.ndarray) -> list[np.ndarray]:
    """Indices of mask's rows split per episode label (one segment if the run has no labels)."""
    idx = np.flatnonzero(mask)
    if "episode" not in ticks or not len(idx):
        return [idx]
    ep = ticks["episode"][idx]
    cuts = np.flatnonzero(np.diff(ep) != 0) + 1
    return np.split(idx, cuts)


_FRAME_RE = re.compile(
    r"^\d+_(?P<t>[\d.]+)s_(?:(?P<tag>pre|teardown)|ep(?P<ep>\d+)_(?P<ph>policy|reset))_(?P<cam>.+)$"
)


def parse_frame_name(path: Path) -> dict:
    """{file, t, camera, phase, episode} from <index>_<t>s[_<label>]_<camera>.jpg (old names too)."""
    m = _FRAME_RE.match(path.stem)
    if m:
        phase = m["tag"] or m["ph"]
        return {
            "file": f"frames/{path.name}",
            "t": float(m["t"]),
            "camera": m["cam"],
            "phase": phase,
            "episode": int(m["ep"]) if m["ep"] else -1,
        }
    return {
        "file": f"frames/{path.name}",
        "t": float(path.stem.split("_")[1].rstrip("s")),
        "camera": path.stem.split("_", 2)[2],
    }


def _style(ax: plt.Axes) -> None:
    ax.grid(True, color=C_GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_MUTED)
    ax.tick_params(colors=C_MUTED, labelsize=8)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_joints(
    ticks: dict,
    st: dict,
    out: Path,
    spans: list[tuple[float, float, str]] | None = None,
    title: str | None = None,
    xlabel: str = "time (s)",
) -> None:
    """joints.png: state, policy and sent command per joint, trained band, clipped ticks.

    Rows in teardown/reset (phases 2/3) are greyed; ``spans`` (start, end, label) overrides the
    greyed windows, which otherwise come from those rows.
    """
    t, phase = ticks["t"], ticks["phase"]
    greyed = np.isin(phase, list(GREYED))
    gaps = _gap_index(ticks)
    if spans is None:
        spans = [(a, b, GREYED[p]) for p in GREYED for a, b in _spans(t, phase == p)]
    s_low, s_high = range_to_arm(st["s01"], st["s99"], st["signs"], st["offsets"])
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for i, (ax, joint) in enumerate(zip(axes.flat, JOINTS, strict=True)):
        if np.isfinite(s_low[i]) and np.isfinite(s_high[i]):
            ax.axhspan(
                s_low[i], s_high[i], color=C_BAND, alpha=0.6, linewidth=0, label="trained state q01..q99"
            )
        # Teardown/reset rows (the return to the start pose) repeat one stale reading: dotted, thin.
        live = np.where(greyed, np.nan, ticks["state"][:, i])
        stale = np.where(greyed, ticks["state"][:, i], np.nan)
        tg, live, stale, pol, sent = _gapped(
            t, gaps, live, stale, ticks["action_policy"][:, i], ticks["action_sent"][:, i]
        )
        ax.plot(tg, live, color=C_STATE, lw=2, label="measured state")
        if greyed.any():
            names = "/".join(GREYED[p] for p in GREYED if (phase == p).any())
            ax.plot(tg, stale, color=C_STATE, lw=1, ls=":", alpha=0.6, label=f"state (stale, {names})")
        ax.plot(tg, pol, color=C_POLICY, lw=1.2, label="policy command")
        ax.plot(tg, sent, color=C_SENT, lw=1.2, ls="--", label="sent (clipped) command")
        clip = ticks["clipped"][:, i]
        if clip.any():
            ax.scatter(
                t[clip], ticks["action_sent"][clip, i], s=10, color=C_CLIP, zorder=5, label="clipped tick"
            )
        for a, b, lab in spans:
            ax.axvspan(a, b, color=C_GRID, alpha=SPAN_ALPHA.get(lab, 0.7), lw=0, label=lab)
        ax.set_title(joint, fontsize=10, color=C_INK, loc="left")
        ax.set_ylabel("deg" if joint != "gripper" else "0-100", fontsize=8, color=C_MUTED)
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=9, color=C_MUTED)
    handles, labels = {}, {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
            labels[lab] = lab
    if title:
        fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=11, color=C_INK)
        fig.legend(
            list(handles.values()),
            list(labels),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=6,
            frameon=False,
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.legend(
            list(handles.values()), list(labels), loc="upper center", ncol=6, frameon=False, fontsize=9
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "joints.png", dpi=110)
    plt.close(fig)


def plot_chunks(
    ticks: dict,
    chunks: dict,
    fps: float,
    out: Path,
    spans: list[tuple[float, float, str]] | None = None,
    title: str | None = None,
    xlabel: str = "time (s)",
) -> None:
    """chunks.png: predicted chunks over the sent command for shoulder_lift and elbow_flex.

    ``spans`` (start, end, label) are drawn greyed (reset windows); none by default.
    """
    sel = [JOINTS.index("shoulder_lift"), JOINTS.index("elbow_flex")]
    fig, axes = plt.subplots(len(sel), 1, figsize=(14, 7), sharex=True)
    arm = chunks.get("chunk_arm")
    for ax, j in zip(axes, sel, strict=True):
        for n, (a, b, lab) in enumerate(spans or []):
            ax.axvspan(
                a, b, color=C_GRID, alpha=SPAN_ALPHA.get(lab, 0.7), lw=0, label=lab if n == 0 else None
            )
        tg, state, sent = _gapped(
            ticks["t"], _gap_index(ticks), ticks["state"][:, j], ticks["action_sent"][:, j]
        )
        ax.plot(tg, state, color=C_STATE, lw=2, label="measured state")
        ax.plot(tg, sent, color=C_SENT, lw=1, ls="--", label="sent command")
        if arm is not None and len(arm):
            delays = chunks.get("inference_delay", np.full(len(arm), -1))
            for k in range(len(arm)):
                # RTC (inference_delay >= 0): step k of the chunk is aligned with t_start + k/fps;
                # sync: the chunk starts running when the inference call returns.
                rtc = k < len(delays) and delays[k] >= 0
                t_exec = chunks["t_start"][k] + (0.0 if rtc else chunks["duration"][k])
                steps = np.arange(arm.shape[1]) / fps
                ax.plot(
                    t_exec + steps,
                    arm[k, :, j],
                    color=C_POLICY,
                    lw=1,
                    alpha=0.8,
                    label="predicted chunk" if k == 0 else None,
                )
                ax.plot(t_exec, arm[k, 0, j], "o", ms=3, color=C_POLICY)
        ax.set_title(JOINTS[j], fontsize=10, color=C_INK, loc="left")
        ax.set_ylabel("deg", fontsize=8, color=C_MUTED)
        _style(ax)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[-1].set_xlabel(
        f"{xlabel}; chunks at fps, from the end of their inference call (sync) or its start (RTC)",
        fontsize=9,
        color=C_MUTED,
    )
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=C_INK)
    fig.tight_layout()
    fig.savefig(out / "chunks.png", dpi=110)
    plt.close(fig)


def plot_cadence(ticks: dict, chunks: dict, out: Path) -> None:
    """cadence.png: tick intervals over time and as a histogram, inference calls shaded."""
    run = ticks["phase"] == 1
    if run.sum() <= 1:
        run = np.ones_like(run, dtype=bool)
    # Intervals within each episode only: the gap across a reset is not a slow tick.
    segs = [s for s in episode_segments(ticks, run) if len(s) > 1]
    t = np.concatenate([ticks["t"][s[1:]] for s in segs]) if segs else np.zeros(0)
    dt_ms = np.concatenate([np.diff(ticks["t"][s]) * 1e3 for s in segs]) if segs else np.zeros(0)
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    for k in range(len(chunks.get("t_start", []))):
        s, d = chunks["t_start"][k], chunks["duration"][k]
        ax.axvspan(s, s + d, color=C_POLICY, alpha=0.15, lw=0, label="inference call" if k == 0 else None)
    if len(dt_ms):
        # A NaN between episodes so the line does not bridge a reset (one segment: no change).
        bounds = np.cumsum([len(seg) - 1 for seg in segs])[:-1]
        tg, dg = _gapped(t, bounds, dt_ms)
        ax.plot(tg, dg, color=C_STATE, lw=1, marker=".", ms=3, label="tick interval")
        axh.hist(dt_ms, bins=60, color=C_STATE, edgecolor="white", linewidth=0.5)
    ax.axhline(TARGET_MS, color=C_INK, lw=1, ls="--", label="target 33.3 ms")
    axh.axvline(TARGET_MS, color=C_INK, lw=1, ls="--")
    ax.set_xlabel("time (s)", fontsize=9, color=C_MUTED)
    ax.set_ylabel("ms between sent actions", fontsize=9, color=C_MUTED)
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    axh.set_xlabel("tick interval (ms)", fontsize=9, color=C_MUTED)
    axh.set_ylabel("ticks", fontsize=9, color=C_MUTED)
    for a in (ax, axh):
        _style(a)
    ax.set_title("loop cadence (control-loop ticks)", fontsize=10, color=C_INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "cadence.png", dpi=110)
    plt.close(fig)


def all_frames(run_dir: Path, meta: dict) -> list[dict]:
    """meta.json's frame list, or one parsed from the file names."""
    return meta.get("frames") or [parse_frame_name(p) for p in sorted((run_dir / "frames").glob("*.jpg"))]


def plot_frames(
    run_dir: Path,
    meta: dict,
    out: Path,
    max_frames: int = 48,
    frames: list[dict] | None = None,
    t0: float = 0.0,
    title: str | None = None,
) -> bool:
    """frames.png contact sheet (times relative to t0; reset frames marked); False if no frames."""
    frames = all_frames(run_dir, meta) if frames is None else frames
    if not frames:
        return False
    frames = sorted(frames, key=lambda f: f["t"])
    if len(frames) > max_frames:
        frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, max_frames)]
    cols = min(8, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.9 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, f in zip(axes.flat, frames, strict=False):
        ax.imshow(plt.imread(run_dir / f["file"]))
        reset = f.get("phase") == "reset" and title is not None  # per-episode sheets only
        ax.set_title(
            f"{f['t'] - t0:.1f} s {f.get('camera', '')}" + (" reset" if reset else ""),
            fontsize=7,
            color=C_MUTED if reset else C_INK,
        )
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=C_INK)
    fig.tight_layout()
    fig.savefig(out / "frames.png", dpi=110)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summary(ticks: dict, chunks: dict, meta: dict, st: dict) -> str:
    """One-screen text summary of the control-loop ticks."""
    run = ticks["phase"] == 1
    if run.sum() < 2:
        run = np.ones_like(ticks["phase"], dtype=bool)
    t, state, clipped = ticks["t"][run], ticks["state"][run], ticks["clipped"][run]
    # Per episode (one segment for a single run), so resets add no time, intervals or travel.
    segs = [s for s in episode_segments(ticks, run) if len(s)]
    duration = float(sum(ticks["t"][s[-1]] - ticks["t"][s[0]] for s in segs))
    dt = np.concatenate([np.diff(ticks["t"][s]) * 1e3 for s in segs]) if segs else np.zeros(0)
    s_low, s_high = range_to_arm(st["s01"], st["s99"], st["signs"], st["offsets"])
    a_low, a_high = range_to_arm(st["a01"], st["a99"], st["signs"], st["offsets"])
    start, end = state[0], state[-1]
    s_out = distance_outside(start, s_low, s_high)
    a_out = distance_outside(start, a_low, a_high)
    travel = sum(np.nansum(np.abs(np.diff(ticks["state"][s], axis=0)), axis=0) for s in segs)
    lines = [
        f"run: {meta.get('tag', '?')}  task: {meta.get('task', '?')!r}  inference: {meta.get('inference_type', '?')}"
        f"  max_relative_target: {(meta.get('robot') or {}).get('max_relative_target', '?')}",
        f"connect: {(meta.get('connect') or {}).get('path', '?')}   stats: {st['source']}",
        f"duration {duration:.1f} s, {len(t)} ticks, effective {(len(t) - len(segs)) / duration if duration else 0:.1f} Hz"
        + (
            f" (interval median {np.median(dt):.1f} ms, p95 {np.percentile(dt, 95):.1f}, max {dt.max():.0f})"
            if len(dt)
            else ""
        ),
    ]
    if len(chunks.get("duration", [])):
        d = chunks["duration"] * 1e3
        lines.append(f"inference: {len(d)} calls, median {np.median(d):.0f} ms, max {d.max():.0f} ms")
    lines += [
        "",
        f"{'joint':<14}{'clip %':>7}{'travel':>9}{'start':>9}{'end':>9}  start vs trained state range",
    ]
    for i, j in enumerate(JOINTS):
        flag = "ok"
        if s_out[i] > TOL:
            flag = f"OUTSIDE by {s_out[i]:.1f}" + (
                f" (action range by {a_out[i]:.1f})" if a_out[i] > TOL else ""
            )
        lines.append(
            f"{j:<14}{100 * clipped[:, i].mean():7.1f}{travel[i]:9.1f}{start[i]:9.1f}{end[i]:9.1f}  {flag}"
        )
    lines.append("")
    lines.append(
        "start: inside the trained state range"
        if not (s_out > TOL).any()
        else f"start: OUTSIDE the trained state range on {', '.join(j for j, o in zip(JOINTS, s_out, strict=True) if o > TOL)}"
    )
    lines.append(f"any joint clipped on {100 * clipped.any(axis=1).mean():.1f}% of ticks")
    errs = meta.get("recording_errors") or {}
    if errs:
        lines.append(f"recording errors (dropped records): {errs}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episodes (episodic sessions)
# ---------------------------------------------------------------------------

SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "wflex",
         "wrist_roll": "wroll", "gripper": "grip"}  # fmt: skip


def _rows(d: dict, mask: np.ndarray) -> dict:
    """Restrict per-row arrays of d to mask's rows (``joints`` is kept whole)."""
    n = len(mask)
    return {
        k: v[mask] if isinstance(v, np.ndarray) and k != "joints" and v.ndim and len(v) == n else v
        for k, v in d.items()
    }


def _t0(ep: dict, ticks: dict) -> float:
    """Start of attempt ep's policy phase (meta), else its first policy tick."""
    start = (ep.get("policy") or {}).get("start")
    if start is not None:
        return float(start)
    m = (ticks["episode"] == ep["index"]) & (ticks["phase"] == PHASE_POLICY)
    return float(ticks["t"][m][0]) if m.any() else 0.0


def _reset_window(ep: dict, ticks: dict) -> tuple[float, float] | None:
    """Attempt ep's reset window: meta's, else the span of its reset ticks."""
    reset = ep.get("reset") or {}
    if reset.get("start") is not None and reset.get("end") is not None:
        return float(reset["start"]), float(reset["end"])
    m = (ticks["episode"] == ep["index"]) & (ticks["phase"] == PHASE_RESET)
    return (float(ticks["t"][m].min()), float(ticks["t"][m].max())) if m.any() else None


def _ep_name(ep: dict) -> str:
    k, d = ep["index"], ep.get("dataset_episode")
    parts = [f"ep{k}"]
    if ep.get("discarded"):
        parts.append("DISCARDED (re-recorded)")
    elif d is not None:
        parts.append(f"dataset episode {d}")
    if ep.get("ended_early") and ep.get("end_reason") != "rerecord":
        parts.append(f"ended early ({ep.get('end_reason')})")
    return " · ".join(parts)


def session_spans(ticks: dict, eps: list[dict]) -> list[tuple[float, float, str]]:
    """Greyed windows of a whole session: every reset (meta windows) and the teardown ticks."""
    spans = [(*w, "reset") for ep in eps if (w := _reset_window(ep, ticks)) is not None]
    return spans + [(a, b, "teardown") for a, b in _spans(ticks["t"], ticks["phase"] == PHASE_TEARDOWN)]


def episode_stats(ticks: dict, ep: dict) -> dict | None:
    """Duration, Hz, % clipped, per-joint distance and gripper range of attempt ep's policy phase."""
    m = (ticks["episode"] == ep["index"]) & (ticks["phase"] == PHASE_POLICY)
    if not m.any():
        return None
    t, state = ticks["t"][m], ticks["state"][m]
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    grip = state[:, JOINTS.index("gripper")]
    return {
        "duration": duration,
        "n": int(m.sum()),
        "hz": (len(t) - 1) / duration if duration else 0.0,
        "clip_pct": 100.0 * float(ticks["clipped"][m].any(axis=1).mean()),
        "distance": np.nansum(np.abs(np.diff(state, axis=0)), axis=0),
        "grip_min": float(np.nanmin(grip)),
        "grip_max": float(np.nanmax(grip)),
    }


def episode_table(ticks: dict, meta: dict, eps: list[dict]) -> str:
    """Session line + one row per attempt."""
    stats = [episode_stats(ticks, ep) for ep in eps]
    saved = sum(1 for ep in eps if ep.get("dataset_episode") is not None)
    discarded = sum(1 for ep in eps if ep.get("discarded"))
    early = sum(1 for ep in eps if ep.get("ended_early"))
    policy_s = sum(x["duration"] for x in stats if x)
    total_s = float(ticks["t"][-1] - ticks["t"][0]) if len(ticks["t"]) > 1 else 0.0
    ds = meta.get("dataset") or {}
    limits = f", limits {ds.get('episode_time_s')} s + reset {ds.get('reset_time_s')} s" if ds else ""
    lines = [
        f"session: {len(eps)} attempts, {saved} saved, {discarded} discarded, {early} ended early; "
        f"policy {policy_s:.1f} s of {total_s:.1f} s recorded{limits}"
        + (f"; dataset {ds.get('repo_id')}" if ds.get("repo_id") else ""),
        "",
        "per attempt (policy phase; distance = measured travel, deg / gripper 0-100):",
        f"{'ep':>3}{'dataset':>8}{'dur s':>7}{'Hz':>6}{'clip%':>7}"
        + "".join(f"{SHORT[j]:>7}" for j in JOINTS)
        + f"{'grip min':>9}{'max':>6}  {'ended early':<19}discarded",
    ]
    for ep, x in zip(eps, stats, strict=True):
        d = ep.get("dataset_episode")
        dataset = "-" if d is None else str(d)
        early_s = f"yes ({ep.get('end_reason')})" if ep.get("ended_early") else "no"
        disc = "yes" if ep.get("discarded") else "no"
        if x is None:
            lines.append(f"{ep['index']:>3}{dataset:>8}  (no policy ticks)  {early_s:<19}{disc}")
            continue
        lines.append(
            f"{ep['index']:>3}{dataset:>8}{x['duration']:7.1f}{x['hz']:6.1f}{x['clip_pct']:7.1f}"
            + "".join(f"{v:7.1f}" for v in x["distance"])
            + f"{x['grip_min']:9.1f}{x['grip_max']:6.1f}  {early_s:<19}{disc}"
        )
    return "\n".join(lines)


def plot_episode(
    run_dir: Path, ticks: dict, chunks: dict, meta: dict, st: dict, fps: float, ep: dict, out: Path
) -> None:
    """plots/ep<k>/: joints, chunks and frames of attempt k, time from its start, reset greyed."""
    k = ep["index"]
    t0 = _t0(ep, ticks)
    out.mkdir(parents=True, exist_ok=True)
    m = (ticks["episode"] == k) & np.isin(ticks["phase"], [PHASE_POLICY, PHASE_RESET])
    sub = _rows(ticks, m)
    sub["t"] = sub["t"] - t0
    window = _reset_window(ep, ticks)
    spans = [] if window is None else [(window[0] - t0, window[1] - t0, "reset")]
    title = _ep_name(ep)
    xlabel = "time from episode start (s)"
    plot_joints(sub, st, out, spans=spans, title=title, xlabel=xlabel)
    if len(chunks.get("t_start", [])):
        if "episode" in chunks:
            cm = chunks["episode"] == k
        else:  # no labels: the chunks that start inside the attempt's window
            end = window[1] if window else (ep.get("policy") or {}).get("end", np.inf)
            cm = (chunks["t_start"] >= t0) & (chunks["t_start"] <= end)
        csub = _rows(chunks, cm)
        csub["t_start"] = csub["t_start"] - t0
    else:
        csub = chunks
    plot_chunks(sub, csub, fps, out, spans=spans, title=title, xlabel=xlabel)
    frames = all_frames(run_dir, meta)
    if frames and "episode" in frames[0]:
        frames = [f for f in frames if f.get("episode") == k]
    else:
        end = window[1] if window else (ep.get("policy") or {}).get("end", np.inf)
        frames = [f for f in frames if t0 <= f["t"] <= end]
    if frames:
        plot_frames(run_dir, meta, out, frames=frames, t0=t0, title=title)


def plot_episodes(ticks: dict, st: dict, eps: list[dict], out: Path) -> None:
    """episodes.png: every attempt's policy-phase state per joint, from its own start."""
    s_low, s_high = range_to_arm(st["s01"], st["s99"], st["signs"], st["offsets"])
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    slot = 0
    styles = []
    for ep in eps:
        if ep.get("dataset_episode") is None:  # discarded, or never saved (e.g. stopped / save failed)
            why = "discarded" if ep.get("discarded") else "not saved"
            styles.append({"color": C_MUTED, "lw": 1.2, "ls": "--", "label": f"ep{ep['index']} {why}"})
            continue
        color = C_EPISODES[slot] if slot < len(C_EPISODES) else C_MUTED
        slot += 1
        d = ep.get("dataset_episode")
        name = f"ep{ep['index']} (dataset {d})"
        if ep.get("ended_early"):
            name += " early"
        styles.append({"color": color, "lw": 2.0, "ls": "-", "label": name})
    for i, (ax, joint) in enumerate(zip(axes.flat, JOINTS, strict=True)):
        if np.isfinite(s_low[i]) and np.isfinite(s_high[i]):
            ax.axhspan(
                s_low[i], s_high[i], color=C_BAND, alpha=0.6, linewidth=0, label="trained state q01..q99"
            )
        for ep, style in zip(eps, styles, strict=True):
            m = (ticks["episode"] == ep["index"]) & (ticks["phase"] == PHASE_POLICY)
            if m.any():
                ax.plot(ticks["t"][m] - _t0(ep, ticks), ticks["state"][m, i], **style)
        ax.set_title(joint, fontsize=10, color=C_INK, loc="left")
        ax.set_ylabel("deg" if joint != "gripper" else "0-100", fontsize=8, color=C_MUTED)
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("time from episode start (s)", fontsize=9, color=C_MUTED)
    handles: dict = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    fig.suptitle(
        "measured state per episode (policy phase)",
        x=0.01,
        y=0.995,
        ha="left",
        va="top",
        fontsize=11,
        color=C_INK,
    )
    fig.legend(
        list(handles.values()),
        list(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=6,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "episodes.png", dpi=110)
    plt.close(fig)


def plot_run(run_dir: Path) -> None:
    """Render every plot into RUN_DIR/plots/ and print the summary."""
    ticks, chunks, meta = load_run(run_dir)
    if len(ticks["t"]) == 0:
        raise SystemExit(f"{run_dir}: no ticks recorded")
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    st = load_stats(meta.get("policy_path"))
    if "chunk_arm" not in chunks and len(chunks.get("chunk_norm", [])):
        chunks["chunk_arm"] = chunk_norm_to_arm(chunks["chunk_norm"], meta.get("policy_path"))
    fps = float(meta.get("fps") or 30.0)
    multi = is_multi_episode(ticks, meta)
    eps = episode_list(ticks, meta) if multi else []
    spans = session_spans(ticks, eps) if multi else None
    plot_joints(ticks, st, out, spans=spans)
    plot_chunks(ticks, chunks, fps, out, spans=[x for x in spans if x[2] == "reset"] if spans else None)
    plot_cadence(ticks, chunks, out)
    has_frames = plot_frames(run_dir, meta, out)
    text = summary(ticks, chunks, meta, st)
    if multi:
        for ep in eps:
            plot_episode(run_dir, ticks, chunks, meta, st, fps, ep, out / f"ep{ep['index']}")
        plot_episodes(ticks, st, eps, out)
        text = episode_table(ticks, meta, eps) + "\n\nwhole session (policy ticks of every attempt):\n" + text
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nplots: {', '.join(sorted(p.name for p in out.glob('*.png')))} in {out}")
    if multi:
        names = ", ".join(f"ep{ep['index']}/" for ep in eps)
        print(f"per episode: {names} (joints, chunks, frames), episodes.png")
    if not has_frames:
        print("(no frames saved)")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def make_fake_run(run_dir: Path, seconds: float = 12.0, fps: float = 30.0) -> None:
    """A plausible run: chunked policy, 5 deg clamp, an elbow start outside range, sync-style pauses."""
    from PIL import Image

    rng = np.random.default_rng(0)
    st = load_stats(str(LOCAL_1CAM))
    s_low, s_high = range_to_arm(st["s01"], st["s99"], st["signs"], st["offsets"])
    median = np.asarray([3.1, -33.2, 34.4, 57.9, -11.0, 9.2])
    start = median.copy()
    start[2] = s_high[2] + 8.0 if np.isfinite(s_high[2]) else 120.0  # elbow outside the trained range
    max_rel, steps = 5.0, 30
    pos, t, k = start.copy(), 0.0, 0
    ticks: dict[str, list] = {
        k_: [] for k_ in ("t", "state", "action_policy", "action_sent", "clipped", "phase")
    }
    chunk_t, chunk_d, chunk_arm = [], [], []
    queue: list[np.ndarray] = []
    while t < seconds:
        if not queue:
            d = 0.27 + 0.03 * rng.random()
            chunk_t.append(t)
            chunk_d.append(d)
            t += d
            goal = median + rng.normal(0, [8, 6, 6, 5, 3, 4])
            chunk = pos + (goal - pos) * np.linspace(0.1, 1.0, steps)[:, None]
            chunk_arm.append(chunk)
            queue = list(chunk)
        cmd = queue.pop(0)
        sent = pos + np.clip(cmd - pos, -max_rel, max_rel)
        ticks["t"].append(t)
        ticks["state"].append(pos.copy())
        ticks["action_policy"].append(cmd)
        ticks["action_sent"].append(sent)
        ticks["clipped"].append(np.abs(cmd - sent) > 1e-4)
        ticks["phase"].append(1)
        pos = pos + 0.6 * (sent - pos) + rng.normal(0, 0.05, 6)
        t += 1.0 / fps + rng.normal(0, 0.002)
        k += 1
    n = len(ticks["t"])
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "ticks.npz",
        t=np.asarray(ticks["t"]),
        t_wall=1.7e9 + np.asarray(ticks["t"]),
        state=np.asarray(ticks["state"]),
        action_policy=np.asarray(ticks["action_policy"]),
        action_sent=np.asarray(ticks["action_sent"]),
        clipped=np.asarray(ticks["clipped"]),
        phase=np.asarray(ticks["phase"], dtype=np.int8),
        joints=np.asarray(JOINTS),
    )
    arm = np.asarray(chunk_arm, dtype=np.float32)
    np.savez_compressed(
        run_dir / "chunks.npz",
        t_start=np.asarray(chunk_t),
        duration=np.asarray(chunk_d),
        chunk_norm=np.zeros_like(arm),
        chunk_arm=arm,
        inference_delay=np.full(len(chunk_t), -1),
    )
    (run_dir / "frames").mkdir(exist_ok=True)
    frames = []
    for i, ft in enumerate(np.arange(0, seconds, 0.5)):
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        img[..., 0] = int(255 * ft / seconds)
        img[40:80, 20 + i * 4 % 120 : 40 + i * 4 % 120, 1] = 255
        name = f"{i:05d}_{ft:08.3f}s_cam0.jpg"
        Image.fromarray(img).save(run_dir / "frames" / name, quality=85)
        frames.append({"file": f"frames/{name}", "t": float(ft), "camera": "cam0"})
    meta = {
        "tag": "selftest",
        "task": "pick up the red cube",
        "fps": fps,
        "policy_path": str(LOCAL_1CAM),
        "inference_type": "sync",
        "robot": {"max_relative_target": max_rel},
        "connect": {"path": "torque_safe"},
        "n_ticks": n,
        "frames": frames,
        "argv": ["steering/rollout.py", "--self-test"],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _fake_frame_label(phase: int, episode: int) -> str:
    """rollout.frame_label, duplicated so the self-test does not import rollout.py."""
    name = {0: "pre", PHASE_POLICY: "policy", PHASE_TEARDOWN: "teardown", PHASE_RESET: "reset"}[phase]
    return f"ep{episode:02d}_{name}" if phase in (PHASE_POLICY, PHASE_RESET) else name


def make_fake_episodic_run(run_dir: Path, fps: float = 30.0) -> None:
    """An episodic session: 3 saved episodes and one discarded attempt (left arrow at 3 s).

    Attempts: ep0 full 8 s, ep1 discarded, ep2 full 8 s, ep3 ended early (right arrow at 5 s). After
    every attempt but the last: a 1 s return to the start pose at 50 Hz (phase reset, stale state),
    a 3 s reset loop (no ticks), and after the discard a second return. Then a 3 s teardown return.
    RTC-style chunks (inference_delay 3, no pause), labelled like rollout.py's.
    """
    from PIL import Image

    rng = np.random.default_rng(1)
    median = np.asarray([3.1, -33.2, 34.4, 57.9, -11.0, 9.2])
    start_pose = median.copy()
    episode_time_s, reset_time_s, max_rel, steps = 8.0, 3.0, 6.0, 30
    plan = [(8.0, "time", False), (3.0, "rerecord", True), (8.0, "time", False), (5.0, "right_arrow", False)]
    keys = ("t", "state", "action_policy", "action_sent", "clipped", "phase", "episode")
    ticks: dict[str, list] = {k: [] for k in keys}
    chunk_keys = ("t_start", "duration", "chunk_arm", "inference_delay", "phase", "episode")
    chunks: dict[str, list] = {k: [] for k in chunk_keys}
    pos, t = start_pose.copy(), 0.4  # 0.4 s of "pre" (model warm, nothing sent)
    episodes: list[dict] = []
    windows: list[tuple[float, float, int, int]] = []  # (start, end, phase, episode) for frame labels

    def tick(cmd: np.ndarray, state: np.ndarray, phase: int, k: int) -> np.ndarray:
        sent = pos + np.clip(cmd - pos, -max_rel, max_rel)
        for key, v in zip(
            keys, (t, state.copy(), cmd, sent, np.abs(cmd - sent) > 1e-4, phase, k), strict=True
        ):
            ticks[key].append(v)
        return sent

    def return_to(target: np.ndarray, seconds: float, phase: int, k: int) -> None:
        nonlocal pos, t
        stale, origin = pos.copy(), pos.copy()  # no observation inside the move: state is stale
        n = int(seconds * 50)
        for i in range(1, n + 1):
            cmd = origin + (target - origin) * i / n
            tick(cmd, stale, phase, k)
            pos = cmd
            t += 1 / 50

    saved = 0
    for k, (seconds, reason, discarded) in enumerate(plan):
        p_start = t
        offset = rng.normal(0, [10, 6, 6, 5, 4, 0])  # each episode heads somewhere else
        queue: list[np.ndarray] = []
        while t < p_start + seconds:
            if len(queue) <= 6:  # RTC: the next chunk is requested before the queue runs dry
                goal = median + offset + rng.normal(0, [3, 2, 2, 2, 1, 4])
                base = queue[-1] if queue else pos
                chunk = base + (goal - base) * np.linspace(0.1, 1.0, steps)[:, None]
                for key, v in zip(
                    chunk_keys, (t, 0.27 + 0.03 * rng.random(), chunk, 3, PHASE_POLICY, k), strict=True
                ):
                    chunks[key].append(v)
                queue = queue[:3] + list(chunk[3:])
            cmd = queue.pop(0)
            sent = tick(cmd, pos, PHASE_POLICY, k)
            pos = pos + 0.6 * (sent - pos) + rng.normal(0, 0.05, 6)
            t += 1.0 / fps + rng.normal(0, 0.002)
        p_end = t
        windows.append((p_start, p_end, PHASE_POLICY, k))
        ep = {
            "index": k,
            "dataset_episode": None if discarded else saved,
            "discarded": discarded,
            "ended_early": reason != "time",
            "end_reason": reason,
            "policy": {"start": round(p_start, 4), "end": round(p_end, 4), "limit_s": episode_time_s},
            "reset": None,
        }
        saved += 0 if discarded else 1
        if k < len(plan) - 1 or discarded:
            r_start = t
            return_to(start_pose, 1.0, PHASE_RESET, k)
            t += reset_time_s  # the reset loop: observations only, no ticks
            returns = 1
            if discarded:
                return_to(start_pose, 1.0, PHASE_RESET, k)
                returns = 2
            ep["reset"] = {
                "start": round(r_start, 4),
                "end": round(t, 4),
                "limit_s": reset_time_s,
                "loop_s": reset_time_s,
                "ended_early": False,
                "end_reason": "time",
                "returns": returns,
            }
            windows.append((r_start, t, PHASE_RESET, k))
        episodes.append(ep)
    td_start = t
    return_to(start_pose, 3.0, PHASE_TEARDOWN, -1)
    windows.append((td_start, t, PHASE_TEARDOWN, -1))

    run_dir.mkdir(parents=True, exist_ok=True)
    n_j = len(JOINTS)
    np.savez_compressed(
        run_dir / "ticks.npz",
        t=np.asarray(ticks["t"]),
        t_wall=1.7e9 + np.asarray(ticks["t"]),
        state=np.asarray(ticks["state"]).reshape(-1, n_j),
        action_policy=np.asarray(ticks["action_policy"]).reshape(-1, n_j),
        action_sent=np.asarray(ticks["action_sent"]).reshape(-1, n_j),
        clipped=np.asarray(ticks["clipped"]).reshape(-1, n_j),
        phase=np.asarray(ticks["phase"], dtype=np.int8),
        episode=np.asarray(ticks["episode"], dtype=np.int16),
        joints=np.asarray(JOINTS),
    )
    arm = np.asarray(chunks["chunk_arm"], dtype=np.float32)
    np.savez_compressed(
        run_dir / "chunks.npz",
        t_start=np.asarray(chunks["t_start"]),
        duration=np.asarray(chunks["duration"]),
        chunk_norm=np.zeros_like(arm),
        chunk_arm=arm,
        inference_delay=np.asarray(chunks["inference_delay"]),
        phase=np.asarray(chunks["phase"], dtype=np.int8),
        episode=np.asarray(chunks["episode"], dtype=np.int16),
    )
    (run_dir / "frames").mkdir(exist_ok=True)
    frames = []
    for i, ft in enumerate(np.arange(0, t, 0.5)):
        phase, k = next(((ph, e) for a, b, ph, e in windows if a <= ft < b), (0, -1))
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        img[..., 0] = int(255 * ft / t)
        img[..., 2] = 200 if phase == PHASE_RESET else 0
        img[40:80, 20 + i * 4 % 120 : 40 + i * 4 % 120, 1] = 255
        name = f"{i:05d}_{ft:08.3f}s_{_fake_frame_label(phase, k)}_cam0.jpg"
        Image.fromarray(img).save(run_dir / "frames" / name, quality=85)
        frames.append(
            {
                "file": f"frames/{name}",
                "t": float(ft),
                "camera": "cam0",
                "phase": {0: "pre", 1: "policy", 2: "teardown", 3: "reset"}[phase],
                "episode": k,
            }
        )
    meta = {
        "tag": "selftest_episodic",
        "task": "pick up the red cube",
        "fps": fps,
        "policy_path": str(LOCAL_1CAM),
        "inference_type": "rtc",
        "strategy_type": "episodic",
        "robot": {"max_relative_target": max_rel},
        "dataset": {
            "repo_id": "local/rollout_selftest",
            "num_episodes": 3,
            "episode_time_s": episode_time_s,
            "reset_time_s": reset_time_s,
        },
        "connect": {"path": "torque_safe"},
        "n_ticks": len(ticks["t"]),
        "frames": frames,
        "episodes": episodes,
        "argv": ["steering/rollout.py", "--self-test"],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _self_test(run_dir: Path) -> None:
    """Fake single run (layout as before) + fake episodic session; exits non-zero on a failure."""
    single, episodic = run_dir, run_dir.with_name(run_dir.name + "_episodic")
    for d in (single, episodic):
        if d.exists():
            shutil.rmtree(d)
    make_fake_run(single)
    plot_run(single)
    expected = {"joints.png", "chunks.png", "cadence.png", "frames.png", "summary.txt"}
    names = {p.name for p in (single / "plots").iterdir()}
    problems = [f"single: missing {sorted(expected - names)}"] if expected - names else []
    if names - expected:
        problems.append(f"single: unexpected {sorted(names - expected)}")
    print("\n" + "=" * 78 + "\n")
    make_fake_episodic_run(episodic)
    plot_run(episodic)
    plots = episodic / "plots"
    want = [plots / n for n in sorted(expected | {"episodes.png"})]
    want += [plots / f"ep{k}" / n for k in range(4) for n in ("joints.png", "chunks.png", "frames.png")]
    problems += [f"episodic: missing {p.relative_to(episodic)}" for p in want if not p.is_file()]
    text = (plots / "summary.txt").read_text()
    if not text.startswith("session: 4 attempts, 3 saved, 1 discarded, 2 ended early"):
        problems.append(f"episodic: summary session line is {text.splitlines()[0]!r}")
    meta = json.loads((episodic / "meta.json").read_text())
    for f in meta["frames"]:  # file-name fallback agrees with meta.json
        parsed = parse_frame_name(episodic / f["file"])
        if (parsed["episode"], parsed["phase"], parsed["camera"]) != (f["episode"], f["phase"], f["camera"]):
            problems.append(f"episodic: frame name parse {parsed} != {f}")
            break
    if problems:
        sys.exit("self-test FAILED:\n  " + "\n  ".join(problems))
    print(f"\nself-test OK: {single} and {episodic}")


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", nargs="?", type=Path, help="steering/results/runs/<stamp>_<tag>")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="synthesise a fake run (RUN_DIR) and a fake 3-episode session (RUN_DIR_episodic), plot both",
    )
    args = parser.parse_args()
    if args.self_test:
        _self_test(args.run_dir or Path(tempfile.mkdtemp(prefix="plot_run_selftest_")) / "selftest")
        return
    if args.run_dir is None:
        parser.error("RUN_DIR is required (or --self-test)")
    plot_run(args.run_dir)


if __name__ == "__main__":
    main()
