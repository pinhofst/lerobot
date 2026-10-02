"""Plan consistency of RTC runs: how much each new chunk disagrees with the one it replaces.

    uv run python steering/plan_consistency.py              # every RTC run in steering/results/runs/
    uv run python steering/plan_consistency.py RUN_DIR ...  # some runs (no cross-run summary)

Writes RUN_DIR/plots/plan_consistency.png and plan_consistency.json per run, and with no arguments
steering/results/plan_consistency_summary.csv / .png (one row / line per run). CPU only, no model.

Alignment. In RTC each new chunk k replaces the queue of the chunk p it follows: its first d_k steps
are dropped (they were "executed" by p during the inference), step d_k runs on the merge tick, and step
j runs (j - d_k) ticks after it. The step of p planned for that same tick is q_k + j with
q_k = (the index of p that would have run on the merge tick) - d_k. So chunk k's step j and chunk p's
step q_k + j are two plans for the same tick, and the overlap is j in [0, H - q_k) (about 16 of 30
steps here). The script gets (merge tick, d_k, q_k) from, in this order:

  1. "merge data": rollout.py's RTC merge columns in chunks.npz (``merged``, ``merge_t``,
     ``merge_delay``, ``merge_prev_consumed``; patch c). Exact.
  2. "tick matching": every tick's ``action_policy`` is one row of one chunk's ``chunk_arm`` (it is
     the same postprocessed value, matched to < 1e-3 deg). Decoding which chunk and step each tick ran
     gives the merge tick, d_k (first step used) and the index of p just before it. Exact when the
     rows match, which they do in every run recorded so far (2 Oct).
  3. "inference_delay heuristic" (plot_run.py's): d_k = inference_delay, merge at t_start + duration,
     one step per tick in between. Its d_k is the latency tracker's running maximum, not the steps
     actually dropped (min(latency steps, actions consumed)); on 2 Oct it is 1-2 steps too large.
  The mode used is printed, written to the JSON and shown in the plot title.

Regions of the overlap (new-chunk step j):
  guided    j < execution_horizon (10 by default; --inference.rtc.execution_horizon): RTC's prefix
            guidance pulls these towards p's remaining plan. They are mostly dropped steps.
  unguided  j >= execution_horizon: the honest measure of how much the plan changed.
  seam      j = d_k: the command actually jumps by this much on the merge tick.
Disagreement = RMS over the region of (new - old) per joint in degrees (arm frame, after the
postprocessor's clamp), and of the tip-position difference in mm (steering/so101_fk.py).

Behaviour, per execution window (the ticks between merges): tracking error RMS(action_policy -
measured state) per joint, the fraction of ticks clipped by max_relative_target (any arm joint), and
the window's label from steering/label_provisional.py (hover/realign, grasp, park, other).

How to read it: hovering with HIGH unguided disagreement (the plans change a lot from one chunk to
the next, typically flipping direction) = indecision; LOW disagreement with high tracking error or
clipping = the arm cannot follow a consistent plan (tracking or the cap); low disagreement and low
tracking error while moving back and forth = the policy plans the oscillation itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from label_provisional import (  # noqa: E402
    HOVER_OPEN,
    HOVER_S,
    find_grasps,
    load as load_policy_phase,  # noqa: E402
    long_runs,
    parked_mask,
)
from so101_fk import SO101FK  # noqa: E402

STEERING = Path(__file__).parent
RUNS = STEERING / "results" / "runs"
SUMMARY_CSV = STEERING / "results" / "plan_consistency_summary.csv"
SUMMARY_PNG = STEERING / "results" / "plan_consistency_summary.png"
PHASE_POLICY = 1
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
SHORT = ("pan", "lift", "elbow", "wflex", "wroll")
MATCH_TOL = 1e-3  # deg
DEFAULT_HORIZON = 10
# Reference categorical palette (dataviz skill), fixed order: one slot per arm joint.
C_JOINT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
C_INK, C_MUTED, C_GRID, C_BAND = "#0b0b0b", "#898781", "#e8e7e3", "#cde2fb"
C_LABEL = {"hover": "#cde2fb", "grasp": "#f6d5d5", "park": "#e8e7e3"}


# ---------------------------------------------------------------------------
# Loading and alignment
# ---------------------------------------------------------------------------


@dataclass
class Merge:
    """One executed chunk and where it sits against the chunk it replaced."""

    k: int  # chunk row
    tick: int  # first policy tick that ran it
    drop: int  # d_k: first step used
    prev: int  # chunk row it replaced (-1: none, e.g. the first chunk)
    q: int  # step of prev planned for the same tick as step 0 of k


def argv_value(meta: dict, name: str) -> str | None:
    """Last ``--...name=value`` in argv (any prefix)."""
    argv = meta.get("argv") or []
    argv = argv if isinstance(argv, list) else str(argv).split()
    val = None
    for a in argv:
        m = re.match(rf"--[\w.]*{re.escape(name)}=(.*)", a)
        if m:
            val = m.group(1)
    return val


def has_merge_data(chunks: dict) -> bool:
    """RTC merge columns recorded by rollout.py (patch c)."""
    return "merged" in chunks and bool(chunks.get("merge_hooked", True)) and bool(np.any(chunks["merged"]))


def merges_from_merge_data(chunks: dict, t: np.ndarray) -> list[Merge]:
    """Mode 1: rollout.py's merge columns."""
    merged = np.asarray(chunks["merged"], dtype=bool)
    mt = np.asarray(chunks["merge_t"], dtype=np.float64)
    delay = np.asarray(chunks["merge_delay"], dtype=np.int64)
    consumed = np.asarray(chunks["merge_prev_consumed"], dtype=np.int64)
    prev_len = np.asarray(chunks["merge_prev_len"], dtype=np.int64)
    out: list[Merge] = []
    for k in sorted(np.flatnonzero(merged), key=lambda i: mt[i]):
        tick = int(np.searchsorted(t, mt[k]))
        if tick >= len(t):
            continue
        prev = out[-1].k if out and prev_len[k] > 0 else -1
        q = int(delay[prev] + consumed[k] - delay[k]) if prev >= 0 else 0
        out.append(Merge(int(k), tick, int(delay[k]), prev, q))
    return out


def decode_ticks(t: np.ndarray, act: np.ndarray, ts: np.ndarray, du: np.ndarray, arm: np.ndarray,
                 fps: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Mode 2: (chunk row, step) each tick ran, -1 where no chunk row matches."""
    n = len(t)
    k_of, i_of = np.full(n, -1), np.full(n, -1)
    cur_k, cur_i = -1, -1
    for j in range(n):
        done = np.flatnonzero(ts + du <= t[j] + 1e-3)  # inference finished: may have merged
        for k in done[::-1][:4]:
            if k < cur_k:
                break  # the queue never goes back to an older chunk
            err = np.abs(arm[k] - act[j][None, :]).max(axis=1)
            ok = np.flatnonzero(err < MATCH_TOL)
            if not len(ok):
                continue
            expect = cur_i + 1 if k == cur_k else round((t[j] - ts[k]) * fps)
            i = int(ok[np.argmin(np.abs(ok - expect))])
            cur_k, cur_i = int(k), i
            k_of[j], i_of[j] = cur_k, cur_i
            break
    return k_of, i_of


def merges_from_ticks(k_of: np.ndarray, i_of: np.ndarray, breaks: set[int]) -> list[Merge]:
    """Merges from the decoded ticks: where the chunk in use changes."""
    out = []
    for j in range(len(k_of)):
        if k_of[j] < 0 or (j > 0 and k_of[j] == k_of[j - 1] and j not in breaks):
            continue
        if any(m.k == k_of[j] for m in out):
            continue  # same chunk again after an unmatched tick
        prev, q = -1, 0
        if j > 0 and k_of[j - 1] >= 0 and j not in breaks:
            prev = int(k_of[j - 1])
            q = int(i_of[j - 1] + 1 - i_of[j])
        out.append(Merge(int(k_of[j]), j, int(i_of[j]), prev, q))
    return out


def merges_from_heuristic(chunks: dict, t: np.ndarray, keep: np.ndarray) -> list[Merge]:
    """Mode 3: inference_delay heuristic (plot_run.py)."""
    ts, du, dl = chunks["t_start"], chunks["duration"], chunks["inference_delay"]
    out: list[Merge] = []
    for k in keep:
        if dl[k] < 0:
            continue
        tick = int(np.searchsorted(t, ts[k] + du[k]))
        if tick >= len(t):
            continue
        prev, q = -1, 0
        if out:
            p = out[-1]
            prev = p.k
            q = int(p.drop + (tick - p.tick) - dl[k])
        out.append(Merge(int(k), tick, int(dl[k]), prev, q))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rms(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Root mean square (NaN for an empty slice)."""
    if x.shape[axis] == 0:
        shape = list(x.shape)
        del shape[axis]
        return np.full(shape, np.nan)
    return np.sqrt(np.mean(np.square(x), axis=axis))


def pair_metrics(arm: np.ndarray, m: Merge, horizon: int, fk: SO101FK) -> dict | None:
    """Disagreement of chunk m.k with m.prev over their overlap."""
    if m.prev < 0:
        return None
    h = arm.shape[1]
    j = np.arange(max(0, -m.q), min(h, h - m.q))
    if not len(j):
        return None
    new, old = arm[m.k, j, :5].astype(np.float64), arm[m.prev, j + m.q, :5].astype(np.float64)
    diff = new - old
    pad = np.zeros((len(j), 1))
    tip = np.linalg.norm(fk.tip(np.hstack([new, pad])) - fk.tip(np.hstack([old, pad])), axis=1) * 1e3
    g, u = j < horizon, j >= horizon
    seam = j == m.drop
    # alignment check: the guided steps should agree best at the chosen offset
    shifts = {}
    for s in (-2, -1, 0, 1, 2):
        jj = np.arange(max(0, -(m.q + s)), min(horizon, h, h - (m.q + s)))
        if len(jj):
            shifts[s] = float(rms(arm[m.k, jj, :5] - arm[m.prev, jj + m.q + s, :5]).mean())
    # direction of the unguided revision per joint (sign of the mean difference)
    return {
        "overlap": int(len(j)),
        "guided": rms(diff[g]),
        "unguided": rms(diff[u]),
        "all": rms(diff),
        "seam": np.abs(diff[seam][0]) if seam.any() else np.full(5, np.nan),
        "tip_guided": float(rms(tip[g])) if g.any() else np.nan,
        "tip_unguided": float(rms(tip[u])) if u.any() else np.nan,
        "tip_seam": float(tip[seam][0]) if seam.any() else np.nan,
        "sign_unguided": np.sign(diff[u].mean(axis=0)) if u.any() else np.zeros(5),
        "best_shift": min(shifts, key=lambda s: shifts[s]) if shifts else 0,
    }


def window_labels(run, n_ticks: int) -> np.ndarray:  # noqa: ANN001 (label_provisional.Run)
    """Per-tick behaviour label: hover / grasp / park / other."""
    lab = np.array(["other"] * n_ticks, dtype=object)
    held = np.zeros(n_ticks, dtype=bool)
    for gr in find_grasps(run):
        held[gr.i0 : gr.i1] = True
    park = parked_mask(run)
    hov = np.zeros(n_ticks, dtype=bool)
    for t0, t1 in long_runs(run, (run.policy[:, 5] > HOVER_OPEN) & ~park & ~held, HOVER_S):
        hov |= (run.t >= t0) & (run.t <= t1)
    lab[hov] = "hover"
    lab[held] = "grasp"
    lab[park] = "park"
    return lab


def analyse(run_dir: Path, fk: SO101FK) -> dict | None:
    """All per-pair and per-window numbers of one run, or None if it is not an RTC run."""
    ticks = dict(np.load(run_dir / "ticks.npz"))
    chunks = dict(np.load(run_dir / "chunks.npz")) if (run_dir / "chunks.npz").is_file() else {}
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").is_file() else {}
    if len(ticks["t"]) < 2 or not len(chunks.get("t_start", [])):
        return None
    if not np.any(np.asarray(chunks["inference_delay"]) >= 0):
        print(f"{run_dir.name}: not an RTC run (no inference_delay), skipped", file=sys.stderr)
        return None
    run = load_policy_phase(run_dir, fk)
    rows = np.flatnonzero(ticks["phase"] == PHASE_POLICY) if "phase" in ticks else np.arange(len(ticks["t"]))
    t, act, state, clipped = (
        ticks["t"][rows],
        ticks["action_policy"][rows],
        ticks["state"][rows],
        ticks["clipped"][rows],
    )
    fps = float(meta.get("fps") or 30.0)
    horizon = int(argv_value(meta, "execution_horizon") or DEFAULT_HORIZON)
    arm = chunks.get("chunk_arm")
    phase_c = chunks.get("phase")
    keep = np.arange(len(chunks["t_start"])) if phase_c is None else np.flatnonzero(phase_c == PHASE_POLICY)
    breaks: set[int] = set()
    if "episode" in ticks:
        breaks = set((np.flatnonzero(np.diff(ticks["episode"][rows]) != 0) + 1).tolist())
    note = ""
    if has_merge_data(chunks):
        mode, merges = "merge data", merges_from_merge_data(chunks, t)
    else:
        k_of, i_of = (np.full(len(t), -1), np.full(len(t), -1))
        if arm is not None and len(arm):
            k_of, i_of = decode_ticks(t, act, chunks["t_start"], chunks["duration"], arm, fps)
        matched = float(np.mean(k_of >= 0))
        if matched >= 0.95:
            mode, merges = "tick matching", merges_from_ticks(k_of, i_of, breaks)
            heur = {m.k: m for m in merges_from_heuristic(chunks, t, keep)}
            dd = [heur[m.k].drop - m.drop for m in merges if m.k in heur and m.prev >= 0]
            note = (f"{matched:.1%} of ticks matched a chunk row; inference_delay heuristic drop - actual "
                    f"drop: median {np.median(dd):+.0f} steps (range {min(dd):+d}..{max(dd):+d})") if dd else ""  # fmt: skip
        else:
            mode, merges = "inference_delay heuristic", merges_from_heuristic(chunks, t, keep)
            note = f"only {matched:.0%} of ticks matched a chunk row"
    if arm is None:
        return None
    pairs = []
    for m in merges:
        pm = pair_metrics(arm, m, horizon, fk)
        if pm is not None:
            pm.update(t=float(t[m.tick] - t[0]), k=m.k, prev=m.prev, q=m.q, drop=m.drop, tick=m.tick)
            pairs.append(pm)
    # per execution window: from this merge's tick to the next one
    labels = (
        window_labels(run, len(t))
        if run is not None and len(run.t) == len(t)
        else np.array(["other"] * len(t))
    )
    track = act[:, :5] - state[:, :5]
    for i, pm in enumerate(pairs):
        a = pm["tick"]
        b = pairs[i + 1]["tick"] if i + 1 < len(pairs) else len(t)
        w = slice(a, b)
        pm["track"] = rms(track[w])
        pm["clip_any"] = float(clipped[w, :5].any(axis=1).mean()) if b > a else np.nan
        pm["clip"] = clipped[w, :5].mean(axis=0) if b > a else np.full(5, np.nan)
        lab, cnt = np.unique(labels[w], return_counts=True)
        pm["label"] = str(lab[np.argmax(cnt)]) if len(lab) else "other"
        steps = np.abs(np.diff(act[a:b, :5], axis=0))
        pm["cmd_step"] = np.median(steps, axis=0) if len(steps) else np.full(5, np.nan)
    # sign flips of the unguided revision between consecutive pairs (dithering)
    signs = np.array([p["sign_unguided"] for p in pairs]) if pairs else np.zeros((0, 5))
    flips = (signs[1:] * signs[:-1] < 0).mean(axis=0) if len(signs) > 1 else np.full(5, np.nan)
    return {
        "run": run_dir.name,
        "tag": str(meta.get("tag", run_dir.name)),
        "task": str(meta.get("task", "")),
        "cap": (meta.get("robot") or {}).get("max_relative_target") if isinstance(meta.get("robot"), dict)
        else argv_value(meta, "max_relative_target"),
        "mode": mode,
        "note": note,
        "horizon": horizon,
        "fps": fps,
        "t": t - t[0],
        "act": act,
        "state": state,
        "clipped": clipped,
        "labels": labels,
        "pairs": pairs,
        "flips": flips,
        "arm": arm,
        "merges": merges,
        "chunk_t0": {m.k: float(t[m.tick] - t[0]) for m in merges},
    }  # fmt: skip


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def stack(pairs: list[dict], key: str, label: str | None = None) -> np.ndarray:
    """(n, ...) array of one metric over the pairs (optionally one window label only)."""
    sel = [p[key] for p in pairs if label is None or p["label"] == label]
    return np.array(sel, dtype=np.float64) if sel else np.zeros((0, 5))


def summarise(a: dict) -> dict[str, str]:
    """One summary row (medians over the run's chunk pairs; per label where it matters)."""
    p = a["pairs"]
    row = {
        "tag": a["tag"],
        "task": a["task"],
        "cap_deg": str(a["cap"]),
        "mode": a["mode"],
        "pairs": str(len(p)),
    }

    def med(x: np.ndarray) -> float:
        return float(np.nanmedian(x)) if np.size(x) and np.isfinite(x).any() else np.nan

    def fmt(v: float, nd: int = 1) -> str:
        return "" if not np.isfinite(v) else f"{v:.{nd}f}"

    ung, gui, seam = stack(p, "unguided"), stack(p, "guided"), stack(p, "seam")
    row["overlap_steps"] = fmt(med(np.array([x["overlap"] for x in p])), 0)
    row["q_median"] = fmt(med(np.array([x["q"] for x in p])), 0)
    row["drop_median"] = fmt(med(np.array([x["drop"] for x in p])), 0)
    row["aligned_pct"] = fmt(100 * np.mean([x["best_shift"] == 0 for x in p]) if p else np.nan, 0)
    row["guided_deg"] = fmt(med(rms(gui, axis=1)) if len(gui) else np.nan, 2)
    row["unguided_deg"] = fmt(med(rms(ung, axis=1)) if len(ung) else np.nan)
    for j, s in enumerate(SHORT):
        row[f"unguided_{s}_deg"] = fmt(med(ung[:, j]) if len(ung) else np.nan)
    row["seam_wroll_deg"] = fmt(med(seam[:, 4]) if len(seam) else np.nan)
    row["tip_guided_mm"] = fmt(med(np.array([x["tip_guided"] for x in p])))
    row["tip_unguided_mm"] = fmt(med(np.array([x["tip_unguided"] for x in p])))
    row["tip_seam_mm"] = fmt(med(np.array([x["tip_seam"] for x in p])))
    row["track_deg"] = fmt(med(rms(stack(p, "track"), axis=1)) if p else np.nan)
    row["clip_pct"] = fmt(100 * med(np.array([x["clip_any"] for x in p])), 0)
    row["flip_wroll_pct"] = fmt(100 * a["flips"][4], 0)
    for lab in ("hover", "other", "grasp"):
        u = stack(p, "unguided", lab)
        tr = stack(p, "track", lab)
        n = len(u)
        row[f"n_{lab}"] = str(n)
        row[f"unguided_deg_{lab}"] = fmt(med(rms(u, axis=1)) if n else np.nan)
        row[f"unguided_wroll_deg_{lab}"] = fmt(med(u[:, 4]) if n else np.nan)
        row[f"tip_unguided_mm_{lab}"] = fmt(
            med(np.array([x["tip_unguided"] for x in p if x["label"] == lab]))
        )
        row[f"track_deg_{lab}"] = fmt(med(rms(tr, axis=1)) if n else np.nan)
        row[f"track_wroll_deg_{lab}"] = fmt(med(tr[:, 4]) if n else np.nan)
        row[f"clip_pct_{lab}"] = fmt(100 * med(np.array([x["clip_any"] for x in p if x["label"] == lab])), 0)
    return row


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes) -> None:
    ax.grid(color=C_GRID, lw=0.6)
    ax.tick_params(labelsize=8, colors=C_INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)


def _shade(ax: plt.Axes, a: dict) -> None:
    """Behaviour labels as background bands."""
    t, lab = a["t"], a["labels"]
    for name, col in C_LABEL.items():
        m = np.concatenate([[False], lab == name, [False]])
        d = np.diff(m.astype(int))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
            ax.axvspan(t[s], t[e - 1], color=col, lw=0, zorder=0)


def write_run_plot(a: dict, out: Path) -> None:
    """Five stacked panels over time (see the module docstring)."""
    p = a["pairs"]
    tp = np.array([x["t"] for x in p])
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True, gridspec_kw={"hspace": 0.32})
    fig.suptitle(
        f"{a['tag']}  -  '{a['task']}', cap {a['cap']} deg   plan consistency (alignment: {a['mode']})\n"
        f"background: hover/realign (blue), grasp (red), park (grey), from label_provisional.py",
        fontsize=11, color=C_INK, x=0.06, ha="left", y=0.995,
    )  # fmt: skip
    ung = stack(p, "unguided")
    ax = axes[0]
    _shade(ax, a)
    for j, s in enumerate(SHORT):
        ax.plot(tp, ung[:, j], color=C_JOINT[j], lw=1.4, marker="o", ms=3, label=s)
    ax.set_ylabel("deg", fontsize=9)
    ax.set_title(f"new vs previous chunk, unguided overlap (steps >= {a['horizon']}): RMS per joint", fontsize=10,
                 loc="left")  # fmt: skip
    ax.legend(fontsize=8, ncol=5, loc="upper right", frameon=False)
    ax = axes[1]
    _shade(ax, a)
    ax.plot(tp, [x["tip_unguided"] for x in p], color=C_INK, lw=1.4, marker="o", ms=3, label="unguided steps")
    ax.plot(tp, [x["tip_guided"] for x in p], color=C_MUTED, lw=1.0, marker="o", ms=3,
            label=f"guided steps (< {a['horizon']}, mostly dropped)")  # fmt: skip
    ax.set_ylabel("mm", fontsize=9)
    ax.set_title("tip position (FK), new vs previous chunk: RMS over the region", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax = axes[2]
    _shade(ax, a)
    tr = stack(p, "track")
    for j, s in enumerate(SHORT):
        ax.plot(tp, tr[:, j], color=C_JOINT[j], lw=1.2, label=s)
    ax.set_ylabel("deg", fontsize=9)
    ax.set_title(
        "tracking error per execution window: RMS(policy command - measured)", fontsize=10, loc="left"
    )
    ax.legend(fontsize=8, ncol=5, loc="upper right", frameon=False)
    ax = axes[3]
    _shade(ax, a)
    ax.plot(tp, [100 * x["clip_any"] for x in p], color=C_INK, lw=1.2, drawstyle="steps-post")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of ticks", fontsize=9)
    ax.set_title(f"ticks clipped by the {a['cap']} deg cap (any arm joint), per execution window", fontsize=10,
                 loc="left")  # fmt: skip
    ax = axes[4]
    _shade(ax, a)
    jw = 4
    arm, fps = a["arm"], a["fps"]
    for m in a["merges"]:
        steps = np.arange(arm.shape[1])
        ax.plot(
            a["chunk_t0"][m.k] + (steps - m.drop) / fps, arm[m.k, :, jw], color=C_MUTED, lw=0.6, alpha=0.6
        )
    ax.plot(a["t"], a["act"][:, jw], color=C_JOINT[jw], lw=1.4, label="policy command (executed)")
    ax.plot(a["t"], a["state"][:, jw], color=C_INK, lw=1.0, label="measured")
    ax.plot([], [], color=C_MUTED, lw=0.6, label="each chunk's full 30-step plan")
    ax.set_ylabel("deg", fontsize=9)
    ax.set_title("wrist_roll: every chunk's plan at its execution time, the executed command and the measured joint",
                 fontsize=10, loc="left")  # fmt: skip
    ax.legend(fontsize=8, ncol=3, loc="upper right", frameon=False)
    ax.set_xlabel("time from policy start (s)", fontsize=9)
    ax.set_xlim(0, float(a["t"][-1]))
    for ax in axes:
        _style(ax)
    if a["note"]:
        fig.text(0.06, 0.005, a["note"], fontsize=8, color=C_MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_summary_plot(rows: list[dict[str, str]], out: Path) -> None:
    """Dot plot per run: unguided disagreement, tip disagreement, tracking, clipping; hover vs not."""

    def num(r: dict, k: str) -> float:
        try:
            return float(r[k])
        except (KeyError, ValueError):
            return np.nan

    rows = rows[::-1]
    y = np.arange(len(rows))
    panels = (
        ("unguided disagreement, arm RMS (deg)", "unguided_deg"),
        ("unguided disagreement, wrist_roll (deg)", "unguided_wroll_deg"),
        ("unguided disagreement, tip (mm)", "tip_unguided_mm"),
        ("tracking error, arm RMS (deg)", "track_deg"),
        ("ticks clipped (%)", "clip_pct"),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 0.36 * len(rows) + 1.8), sharey=True,
                             gridspec_kw={"wspace": 0.08})  # fmt: skip
    groups = (("hover", C_JOINT[0], "hover/realign windows"), ("other", C_JOINT[1], "other moving windows"),
              ("grasp", C_JOINT[2], "grasp windows"))  # fmt: skip
    for ax, (title, key) in zip(axes, panels, strict=True):
        for off, (lab, col, name) in zip((-0.22, 0.0, 0.22), groups, strict=True):
            vals = [num(r, f"{key}_{lab}") for r in rows]
            ax.scatter(vals, y + off, s=28, color=col, edgecolor="white", linewidth=1.2, zorder=3, label=name)
        ax.set_title(title, fontsize=9, loc="left")
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [f"{r['tag'].replace('median_rtc_', '')}  ({r['cap_deg']} deg)" for r in rows], fontsize=8
    )
    axes[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=3, frameon=False)
    fig.suptitle("Plan consistency across RTC runs, 2 Oct (medians over chunk pairs; windows labelled by "
                 "label_provisional.py)", fontsize=11, x=0.01, ha="left", y=1.0 + 0.6 / (0.36 * len(rows) + 1.8))  # fmt: skip
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def to_json(a: dict) -> dict:
    """The per-run numbers worth keeping (no arrays of ticks)."""

    def clean(v: object) -> object:
        if isinstance(v, np.ndarray):
            return [None if not np.isfinite(x) else round(float(x), 3) for x in v.ravel()]
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else round(float(v), 3)
        if isinstance(v, np.integer):
            return int(v)
        return v

    return {
        "run": a["run"], "tag": a["tag"], "mode": a["mode"], "note": a["note"], "horizon": a["horizon"],
        "joints": list(SHORT),
        "pairs": [{k: clean(v) for k, v in p.items()} for p in a["pairs"]],
        "sign_flip_fraction": clean(a["flips"]),
    }  # fmt: skip


def main() -> None:
    """Analyse, plot, summarise."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path)
    args = ap.parse_args()
    dirs = args.runs or sorted(d for d in RUNS.iterdir() if d.is_dir())
    fk = SO101FK()
    rows = []
    for d in dirs:
        if not (d / "ticks.npz").is_file():
            continue
        a = analyse(d, fk)
        if a is None or not a["pairs"]:
            continue
        write_run_plot(a, d / "plots" / "plan_consistency.png")
        (d / "plots" / "plan_consistency.json").write_text(json.dumps(to_json(a), indent=1))
        r = summarise(a)
        rows.append(r)
        print(f"{a['tag']:<45} {a['mode']:<14} pairs {r['pairs']:>3}  unguided {r['unguided_deg']:>4} deg "
              f"(wroll {r['unguided_wroll_deg']:>4}), tip {r['tip_unguided_mm']:>4} mm; guided {r['guided_deg']} deg;"
              f" hover/other unguided {r['unguided_deg_hover'] or '-'}/{r['unguided_deg_other'] or '-'} deg,"
              f" track {r['track_deg_hover'] or '-'}/{r['track_deg_other'] or '-'} deg,"
              f" clip {r['clip_pct_hover'] or '-'}/{r['clip_pct_other'] or '-'} %")  # fmt: skip
        if a["note"]:
            print(f"    {a['note']}")
    if not args.runs and rows:
        cols = list(rows[0])
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        write_summary_plot(rows, SUMMARY_PNG)
        print(f"\nwrote {SUMMARY_CSV} and {SUMMARY_PNG} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
