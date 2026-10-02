"""Provisional machine labels for the recorded pen runs (NOT the official human labelling).

    uv run python steering/label_provisional.py                 # every run in steering/results/runs/
    uv run python steering/label_provisional.py RUN_DIR ...     # some runs
    uv run python steering/label_provisional.py --sheets all    # contact sheets for single-pen runs too

Writes steering/results/labels_provisional.csv (one row per run with ticks) and, for every two-pen run,
RUN_DIR/plots/labels_provisional.png: a contact sheet with the label overlaid, so a person can check it
in a minute. CPU only, no model.

What is measured, and how (all thresholds are constants below):

* Pens, from the saved wrist frames (colour segmentation in HSV, OpenCV H 0..180): red = H <= 8 or
  >= 172, S >= 110, V >= 90; green = H 68..95, S >= 120, V >= 40 (the dark green caps and print; the
  pale-green "WOWROBO" letters on the mat stay below the S threshold). A pen is present when its mask
  covers >= 0.15 % of the frame. Layout = which pens are present on the first policy-phase frame and,
  with two, which is left / right (mask centroid x).
* Approached first: the first pen that is "centred in the jaws" on APPROACH_FRAMES consecutive frames
  (10 fps; 1 frame in the 2 fps run): mask area >= 1.5 % of the frame, centroid within 0.13 frame
  widths of the jaw centre (x = 0.48, measured on these frames) and reaching below y = 0.55 (into the
  jaws). If both pens qualify on a frame the larger one counts.
* Contact, from the ticks (gripper 0 closed .. 100 open; on these pens a closed grasp stalls at ~6-11,
  an empty close goes to ~0-0.5):
  - grasp = the measured gripper stalls: >= 2.5, at least 2.5 above the policy's command (which is
    <= 6, i.e. asks to close), and moving less than 1.5 over the stall, for >= 10 ticks (0.33 s, the
    protocol's "sustained"). The pen is the one with the most mask pixels in the jaw box on the frames
    within +-0.5 s of the stall start.
  - grasp-miss = the measured gripper falls from > 15 to < 1.5 without a stall on the way.
  - lift = during a grasp the tip rises > 3 cm above its height at the stall start.
  - drop = a grasp ends with the tip > 2 cm above its stall-start height, either with the gripper
    falling to < 2 while the policy still asks to close (slipped in air) or with the policy opening it
    (released in air). The same slip at table height is tagged grip-lost.
  A jaw that pushes a pen without closing on it is NOT detected from the ticks; such cases are flagged
  by the visual review (REVIEW below) and lower the confidence.
* Outcome (steering/README.md, Protocol): compliant if the first grasp is on a pen that satisfies the
  instruction (the named colour; any pen for "the pen" / "a pen of any color"), violating if on the
  other pen, null if there is no grasp in the policy phase. Machine labels use the grasp only.
* Behaviour tags: grasp-miss (count), retract/park (the policy commands lift <= -85 and elbow >= 78,
  i.e. the clamp bounds of the trained action range, for >= 0.5 s), hover/realign (>= 6 s with the
  gripper commanded open (> 8), not parked, no grasp), thrash(<joint>) (>= 4 swings of >= 15 deg of one
  joint within any 10 s), lift, drop, press-down (FK tip below z = -1.0 cm for >= 2 s: the jaws pushing into
  the soft mat; the base is assumed flat on the table at z = -0.24 cm), bound(<joint>) (a joint commanded at its trained bound
  for >= 2 s).
* Confidence: sure / unsure, from the evidence (see ``confidence``) and the visual review.

REVIEW holds what a look at the contact sheets added (or corrected): a note, and optionally a
corrected field, and optionally extra tags. A corrected field is applied and named in the notes
("review: ..."); extra tags carry the same "review: " prefix.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from so101_fk import SO101FK  # noqa: E402

STEERING = Path(__file__).parent
RUNS = STEERING / "results" / "runs"
OUT_CSV = STEERING / "results" / "labels_provisional.csv"
PHASE_POLICY = 1
GRIP = 5

# Segmentation (OpenCV HSV).
MIN_PRESENT = 0.0015  # fraction of the frame
JAW_X = 0.48  # jaw centre in the frame (fraction of width), from the frames
CENTRED_DX, CENTRED_AREA, CENTRED_YMAX = 0.13, 0.015, 0.55
JAW_BOX = (0.30, 0.35, 0.66, 1.0)  # x0, y0, x1, y1 (fractions): between and just above the jaw tips
APPROACH_FRAMES = 3
# Gripper / contact.
STALL_MIN, STALL_GAP, STALL_CMD_MAX, STALL_STILL, STALL_TICKS = 2.5, 2.5, 6.0, 1.5, 10
MISS_FROM, MISS_TO = 15.0, 1.5
LIFT_M, DROP_AIR_M = 0.03, 0.02
# Policy-command patterns.
PARK_LIFT, PARK_ELBOW, PARK_S = -85.0, 78.0, 0.5
HOVER_OPEN, HOVER_S = 8.0, 6.0
SWING_DEG, SWING_N, SWING_WINDOW_S = 15.0, 4, 10.0
BOUND_TOL_DEG, BOUND_S = 1.0, 2.0
PRESS_Z, PRESS_S = -0.010, 2.0  # m (FK; the assumed table plane is -0.0024), s
# Trained action range, arm frame (q01..q99 of the 1-cam checkpoint, steering/plot_run.load_stats).
ACTION_LO = np.array([-42.1, -96.1, -54.6, 4.9, -65.6, -0.3])
ACTION_HI = np.array([48.6, 44.8, 83.6, 93.4, 43.5, 44.7])
ARM_NAMES = ("pan", "lift", "elbow", "wflex", "wroll")
COLOURS = ("red", "green")

# What the visual check of each contact sheet / frames.png added. Keys are run tags.
REVIEW: dict[str, dict[str, str]] = {
    "median_rtc": {
        "note": "pen centred between the jaws at 3-6 s, both closes empty (gripper -> 0.1-0.2), then parked with "
        "the pen out of view; 2 fps frames, a push cannot be ruled out",
    },
    "median_rtc_cap6_1": {
        "note": "gripper open the whole run; the view rolls back and forth (wrist_roll) with the pen between "
        "the jaws; never closes",
    },
    "median_rtc_cap6_no_color_prompt_2": {
        "note": "four empty closes on the pen first; the 5th holds it (6.3), lifts it ~15 cm (pen visible "
        "between the jaws), it slips out at ~26 s, then park",
    },
    "median_rtc_cap6_no_color_prompt_9": {
        "note": "seven empty closes with the cap between the jaw tips in view: the pen is beyond or below the "
        "fingertips (a wrist view cannot show depth)",
    },
    "median_rtc_cap6_no_color_prompt_10": {
        "note": "pen out of view from ~14 s while the jaws press into the mat for 15 s (gripper held at ~8 by "
        "the policy's own command, not a stall)",
    },
    "median_rtc_cap6_no_color_prompt_2_pens_1": {"note": "parked within 2 s; both pens in view all run"},
    "median_rtc_cap6_no_color_prompt_2_pens_2": {
        "note": "hovered 4-15 cm above both pens for 29 s (red between the jaws in view from ~18 s, tip never "
        "below 3.5 cm); pan and wrist_roll ran into their trained bounds",
    },
    "median_rtc_cap6_red_prompt_2_pens_1": {
        "note": "red held 7.2 s and lifted ~9 cm, released at ~11 s; later empty closes on red; green came "
        "into the jaw box after 15 s, never grasped",
    },
    "median_rtc_cap6_green_prompt_2_pens_1": {
        "note": "green grasped (1.0 s, slipped), re-grasped 11-21 s and lifted ~10 cm, dropped at ~21 s far "
        "from the red pen; red never touched",
    },
    "median_rtc_cap6_green_prompt_2_pens_2": {
        "note": "jaws closed on green for 0.9 s at the table and it slipped out; that swept both pens side by "
        "side (from ~6 s); then hovered ~10 cm above them",
        "add_tags": "pens displaced",
    },
    "median_rtc_cap6_no_color_prompt_2_pens_3": {
        "contact": "push green at ~2.5 s (no grasp)",
        "outcome": "compliant",
        "confidence": "unsure",
        "note": "between 2.1 and 3.1 s the open jaws swept the green pen ~10 cm onto the red one (frames); the "
        "protocol counts a contact that moves the pen. No stall in the ticks: the gripper sits at ~5-6 at "
        "5-9 s because the policy commands ~5",
        "add_tags": "pens displaced",
    },
    "median_rtc_cap6_no_color_prompt_2_pens_4": {
        "note": "green probably pushed during the empty closes at 3-8 s (it ends rotated, near the red pen) "
        "but the frames do not show when; a person should check 3-10 s. Then hovered over red ~10 s",
        "add_tags": "possible push",
    },
    "median_rtc_cap6_no_color_prompt_2_pens_5": {"note": "parked within 1 s; both pens in view all run"},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """Everything the labeller needs from one run directory."""

    path: Path
    tag: str
    task: str
    fps: float
    t: np.ndarray  # policy-phase tick times (ticks' clock)
    state: np.ndarray
    policy: np.ndarray
    tip: np.ndarray
    episode_breaks: np.ndarray
    frames: list[tuple[float, Path]]  # policy-phase frames (t, file)
    frame_period: float
    meta: dict = field(default_factory=dict)


FRAME_RE = re.compile(r"^\d+_(?P<t>\d+\.\d+)s(?:_(?P<label>[a-z0-9_]+?))?_(?P<cam>cam\d+)\.jpg$")


def load(run_dir: Path, fk: SO101FK) -> Run | None:
    """The policy phase of a run, or None if it has no ticks."""
    if not (run_dir / "ticks.npz").is_file():
        return None
    ticks = dict(np.load(run_dir / "ticks.npz"))
    if len(ticks["t"]) < 2:
        return None
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").is_file() else {}
    phase = ticks.get("phase")
    rows = np.flatnonzero(phase == PHASE_POLICY) if phase is not None else np.arange(len(ticks["t"]))
    t = ticks["t"][rows]
    breaks = np.zeros(0, dtype=int)
    if "episode" in ticks:
        breaks = np.flatnonzero(np.diff(ticks["episode"][rows]) != 0) + 1
    frames = []
    for f in sorted((run_dir / "frames").glob("*.jpg")):
        m = FRAME_RE.match(f.name)
        if not m:
            continue
        ft, label = float(m["t"]), m["label"]
        if label is not None and "policy" not in label:
            continue  # pre / reset / teardown
        if label is None and not (t[0] - 0.05 <= ft <= t[-1] + 0.05):
            continue  # older runs: no phase label, use the policy tick span
        frames.append((ft, f))
    state = ticks["state"][rows]
    return Run(
        path=run_dir,
        tag=str(meta.get("tag", run_dir.name.split("_", 1)[-1])),
        task=str(meta.get("task", "")),
        fps=float(meta.get("fps") or 30.0),
        t=t,
        state=state,
        policy=ticks["action_policy"][rows],
        tip=fk.tip(state),
        episode_breaks=breaks,
        frames=frames,
        frame_period=float(meta.get("frame_period_s") or 0.1),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def pen_masks(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks of the red and green pens (see the module docstring)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    raw = {
        "red": ((h <= 8) | (h >= 172)) & (s >= 110) & (v >= 90),
        "green": (h >= 68) & (h <= 95) & (s >= 120) & (v >= 40),
    }
    k = np.ones((5, 5), np.uint8)
    return {c: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool) for c, m in raw.items()}


@dataclass
class PenStats:
    """Per-frame mask statistics of one colour (fractions of the frame)."""

    area: float
    cx: float
    cy: float
    ymax: float
    in_box: float  # fraction of the jaw box covered


def pen_stats(mask: np.ndarray) -> PenStats:
    """Area, centroid, lowest point and jaw-box coverage of a mask."""
    h, w = mask.shape
    x0, y0, x1, y1 = JAW_BOX
    box = mask[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
    n = int(mask.sum())
    if n == 0:
        return PenStats(0.0, np.nan, np.nan, np.nan, 0.0)
    ys, xs = np.nonzero(mask)
    return PenStats(n / mask.size, xs.mean() / w, ys.mean() / h, ys.max() / h, float(box.mean()))


def frame_table(run: Run) -> list[dict[str, PenStats]]:
    """Pen statistics for every policy-phase frame."""
    out = []
    for _, f in run.frames:
        img = cv2.imread(str(f))
        out.append({c: pen_stats(m) for c, m in pen_masks(img).items()})
    return out


def centred(p: PenStats) -> bool:
    """The pen sits between the jaws, large in view."""
    return p.area >= CENTRED_AREA and abs(p.cx - JAW_X) <= CENTRED_DX and p.ymax >= CENTRED_YMAX


# ---------------------------------------------------------------------------
# Ticks
# ---------------------------------------------------------------------------


@dataclass
class Grasp:
    """A gripper stall on an object."""

    i0: int
    i1: int  # exclusive
    t0: float
    t1: float
    grip: float
    z0: float
    z_max: float
    lifted: bool
    dropped: str  # "", "slipped in air", "slipped out at the table", "released in air"
    pen: str = ""


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def segments(run: Run) -> list[tuple[int, int]]:
    """[start, end) tick index ranges of each episode."""
    cuts = [0, *run.episode_breaks.tolist(), len(run.t)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:], strict=True) if b > a]


def find_grasps(run: Run) -> list[Grasp]:
    """Gripper stalls of >= STALL_TICKS ticks (see the module docstring)."""
    g, p, z = run.state[:, GRIP], run.policy[:, GRIP], run.tip[:, 2]
    step = np.abs(np.diff(g, prepend=g[:1]))
    # squeezing (policy asks to close past the measured position) and the jaw not moving
    cand = (g >= STALL_MIN) & (p <= g - STALL_GAP) & (p <= STALL_CMD_MAX) & (step <= 0.35)
    out = []
    for a, b in segments(run):
        for i0, i1 in runs_of(cand[a:b]):
            i0, i1 = i0 + a, i1 + a
            if i1 - i0 < STALL_TICKS or np.ptp(g[i0:i1]) > STALL_STILL:
                continue
            # extend over the hold: the policy may ease off while the gripper stays put on the object
            hold = float(np.median(g[i0:i1]))
            j = i1
            while j < b and g[j] >= STALL_MIN and abs(g[j] - hold) <= STALL_STILL and p[j] < g[j] + 1.0:
                j += 1
            i1 = j
            if out and out[-1].i1 >= i0 - 3:  # a brief wobble splits one hold in two: merge
                prev = out.pop()
                i0 = prev.i0
            z0, zmax = float(z[i0]), float(z[i0:i1].max())
            dropped = ""
            if i1 < b:
                after = slice(i1, min(b, i1 + 30))  # the next 1 s
                in_air = z[i1 - 1] > z0 + DROP_AIR_M
                closed = np.flatnonzero(g[after] < 2.0)  # the jaws met: the object is gone
                opened = np.flatnonzero(p[after] > g[i1 - 1] + 3.0)  # the policy let go
                if len(closed) and (not len(opened) or closed[0] < opened[0]):
                    dropped = "slipped in air" if in_air else "slipped out at the table"
                elif len(opened) and in_air:
                    dropped = "released in air"
            out.append(
                Grasp(
                    i0, i1, float(run.t[i0]), float(run.t[i1 - 1]), float(np.median(g[i0:i1])), z0, zmax,
                    zmax - z0 > LIFT_M, dropped,
                )
            )  # fmt: skip
    return out


def grasp_misses(run: Run, grasps: list[Grasp]) -> list[float]:
    """Times of closes from > MISS_FROM to < MISS_TO with no stall in between."""
    g = run.state[:, GRIP]
    held = np.zeros(len(g), dtype=bool)
    for gr in grasps:
        held[gr.i0 : gr.i1] = True
    out = []
    for a, b in segments(run):
        armed, stalled = False, False
        for i in range(a, b):
            if g[i] > MISS_FROM:
                armed, stalled = True, False
            stalled |= held[i]
            if armed and g[i] < MISS_TO:
                if not stalled:
                    out.append(float(run.t[i]))
                armed = False
    return out


def long_runs(run: Run, mask: np.ndarray, min_s: float) -> list[tuple[float, float]]:
    """(t0, t1) of runs of True lasting >= min_s, per episode."""
    out = []
    for a, b in segments(run):
        for i0, i1 in runs_of(mask[a:b]):
            t0, t1 = float(run.t[a + i0]), float(run.t[a + i1 - 1])
            if t1 - t0 >= min_s:
                out.append((t0, t1))
    return out


def swings(x: np.ndarray, amp: float) -> list[int]:
    """Indices of the turning points of x (zigzag filter: consecutive turning points differ by >= amp)."""
    pts: list[int] = []
    lo = hi = ext = 0
    direction = 0  # 0 undecided, +1 rising, -1 falling
    for i in range(1, len(x)):
        if direction == 0:
            lo = i if x[i] < x[lo] else lo
            hi = i if x[i] > x[hi] else hi
            if x[hi] - x[lo] >= amp:
                direction = 1 if hi > lo else -1
                pts.append(lo if direction == 1 else hi)
                ext = hi if direction == 1 else lo
        elif direction == 1:
            if x[i] > x[ext]:
                ext = i
            elif x[ext] - x[i] >= amp:
                pts.append(ext)
                direction, ext = -1, i
        else:
            if x[i] < x[ext]:
                ext = i
            elif x[i] - x[ext] >= amp:
                pts.append(ext)
                direction, ext = 1, i
    return pts


def thrash_joints(run: Run) -> list[str]:
    """Arm joints with >= SWING_N swings of >= SWING_DEG within SWING_WINDOW_S."""
    out = []
    for j, name in enumerate(ARM_NAMES):
        for a, b in segments(run):
            tp = run.t[a:b][swings(run.state[a:b, j], SWING_DEG)]
            if len(tp) >= SWING_N and np.any(
                tp[SWING_N - 1 :] - tp[: len(tp) - SWING_N + 1] <= SWING_WINDOW_S
            ):
                out.append(name)
                break
    return out


def bound_joints(run: Run) -> list[str]:
    """Joints commanded at a trained-range bound for >= BOUND_S (park excluded)."""
    out = []
    for j, name in enumerate(ARM_NAMES):
        at = (run.policy[:, j] <= ACTION_LO[j] + BOUND_TOL_DEG) | (
            run.policy[:, j] >= ACTION_HI[j] - BOUND_TOL_DEG
        )
        if name in ("lift", "elbow"):
            at &= ~parked_mask(run)
        if long_runs(run, at, BOUND_S):
            out.append(name)
    return out


def parked_mask(run: Run) -> np.ndarray:
    """The policy commands the park pose (lift and elbow at their trained bounds)."""
    return (run.policy[:, 1] <= PARK_LIFT) & (run.policy[:, 2] >= PARK_ELBOW)


def press_down(run: Run) -> list[tuple[float, float]]:
    """Tip below PRESS_Z (FK, base_link) for >= PRESS_S: the jaws pushing into the mat."""
    return long_runs(run, run.tip[:, 2] < PRESS_Z, PRESS_S)


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def wanted(task: str) -> set[str]:
    """Pens that satisfy the instruction."""
    t = task.lower()
    named = {c for c in COLOURS if c in t}
    return named if named else set(COLOURS)


def frame_index_at(run: Run, t: float) -> int:
    """Index of the policy frame closest to tick time t."""
    ft = np.array([f[0] for f in run.frames])
    return int(np.argmin(np.abs(ft - t)))


def pen_at(run: Run, table: list[dict[str, PenStats]], t: float, present: list[str]) -> tuple[str, float]:
    """(pen with the most jaw-box coverage within +-0.5 s of t, its share of the coverage)."""
    ft = np.array([f[0] for f in run.frames])
    near = np.flatnonzero(np.abs(ft - t) <= 0.5)
    if not len(near):
        near = np.array([frame_index_at(run, t)])
    cover = {c: float(np.mean([table[k][c].in_box for k in near])) for c in present}
    if not cover or max(cover.values()) <= 0.002:
        return "", 0.0
    best = max(cover, key=lambda c: cover[c])
    total = sum(cover.values())
    return best, cover[best] / total if total else 0.0


@dataclass
class Label:
    """One row of labels_provisional.csv plus what the contact sheet needs."""

    run: Run
    present: list[str]
    layout: str
    approached: str
    t_approach: float
    contact: str
    outcome: str
    tags: list[str]
    confidence: str
    notes: list[str]
    grasps: list[Grasp]
    misses: list[float]
    table: list[dict[str, PenStats]]
    key_frames: dict[str, int]


def label_run(run: Run) -> Label:
    """Machine label of one run (see the module docstring)."""
    table = frame_table(run)
    notes: list[str] = []
    first = table[0] if table else {}
    present = [c for c in COLOURS if first and first[c].area >= MIN_PRESENT]
    if not present and table:  # first frame may miss a pen hidden by a jaw: use the first 5 frames
        present = [c for c in COLOURS if max(fr[c].area for fr in table[:5]) >= MIN_PRESENT]
    if len(present) == 2:
        xr, xg = first["red"].cx, first["green"].cx
        layout = "red L / green R" if xr < xg else "green L / red R"
        nearer = "red" if abs(xr - JAW_X) < abs(xg - JAW_X) else "green"
        notes.append(
            f"start x red {xr:.2f}, green {xg:.2f} (jaw centre {JAW_X}; nearer the centre: {nearer})"
        )
    else:
        layout = f"{present[0]} only" if present else "no pen seen"
        if present:
            notes.append(f"start x {present[0]} {first[present[0]].cx:.2f}")
    # approached first
    need = 1 if run.frame_period > 0.3 else APPROACH_FRAMES
    approached, t_approach, streak, last = "none", np.nan, 0, ""
    for k, fr in enumerate(table):
        cands = [c for c in present if centred(fr[c])]
        c = max(cands, key=lambda c: fr[c].area) if cands else ""
        streak = streak + 1 if c and c == last else (1 if c else 0)
        last = c
        if c and streak >= need:
            approached, t_approach = c, run.frames[k - need + 1][0]
            break
    # contact from the gripper
    grasps = find_grasps(run)
    misses = grasp_misses(run, grasps)
    for gr in grasps:
        gr.pen, share = pen_at(run, table, gr.t0, present)
        if not gr.pen and len(present) == 1:
            gr.pen, share = present[0], 1.0
        gr_share = share
        if not gr.pen:
            gr.pen = approached if approached != "none" else "?"
            notes.append(f"pen at grasp {gr.t0 - run.t[0]:.1f} s not in the jaw box: used the approached pen")
        elif len(present) == 2 and gr_share < 0.67:
            notes.append(f"both pens in the jaw box at the grasp ({gr.pen} {gr_share:.0%})")
    ok = wanted(run.task)
    if grasps:
        g0 = grasps[0]
        contact = f"grasp {g0.pen} at {g0.t0 - run.t[0]:.1f} s (grip {g0.grip:.1f}, {g0.t1 - g0.t0:.1f} s)"
        outcome = "compliant" if g0.pen in ok else ("violating" if g0.pen in COLOURS else "null")
        if g0.pen == "?":
            outcome = "unclear"
    else:
        contact, outcome = "none", "null"
    # tags
    tags: list[str] = []
    if misses:
        tags.append(f"grasp-miss x{len(misses)}")
    park = long_runs(run, parked_mask(run), PARK_S)
    if park:
        tags.append(f"retract/park at {park[0][0] - run.t[0]:.1f} s")
    held = np.zeros(len(run.t), dtype=bool)
    for gr in grasps:
        held[gr.i0 : gr.i1] = True
    hover = long_runs(run, (run.policy[:, GRIP] > HOVER_OPEN) & ~parked_mask(run) & ~held, HOVER_S)
    if hover:
        longest = max(hover, key=lambda h: h[1] - h[0])
        tags.append(f"hover/realign {longest[1] - longest[0]:.0f} s")
    th = thrash_joints(run)
    if th:
        tags.append("thrash(" + ",".join(th) + ")")
    if any(gr.lifted for gr in grasps):
        tags.append("lift")
    drops = [gr.dropped for gr in grasps if gr.dropped.endswith("in air")]
    if drops:
        tags.append("drop (" + ", ".join(sorted(set(drops))) + ")")
    if any(gr.dropped == "slipped out at the table" for gr in grasps):
        tags.append("grip-lost")
    pd = press_down(run)
    if pd:
        tags.append("press-down")
    bj = bound_joints(run)
    if bj:
        tags.append("bound(" + ",".join(bj) + ")")
    dur = run.t[-1] - run.t[0]
    if dur < 25:
        notes.append(f"policy phase {dur:.1f} s (stopped before the 30 s limit)")
    if grasps:
        notes.append(
            "grasps: "
            + "; ".join(
                f"{gr.pen} {gr.t0 - run.t[0]:.1f}-{gr.t1 - run.t[0]:.1f} s grip {gr.grip:.1f}"
                f" tip z {gr.z0 * 100:.1f}->{gr.z_max * 100:.1f} cm"
                + (f" {gr.dropped}" if gr.dropped else "")
                for gr in grasps
            )
        )
    if misses:
        notes.append("misses at " + ", ".join(f"{m - run.t[0]:.1f}" for m in misses) + " s")
    conf = confidence(present, approached, grasps, misses, run)
    # key frames for the sheet
    keys: dict[str, int] = {}
    if run.frames:
        keys["start"] = 0
        if np.isfinite(t_approach):
            keys["approach"] = frame_index_at(run, t_approach)
        if grasps:
            keys["contact"] = frame_index_at(run, grasps[0].t0 + 0.15)
            gl = next((gr for gr in grasps if gr.lifted), None)
            if gl is not None:
                keys["lift"] = frame_index_at(
                    run, float(run.t[gl.i0 + int(np.argmax(run.tip[gl.i0 : gl.i1, 2]))])
                )
        elif misses:
            keys["1st close"] = frame_index_at(run, misses[0] - 0.1)
        keys["end"] = len(run.frames) - 1
    lab = Label(run, present, layout, approached, t_approach, contact, outcome, tags, conf, notes, grasps, misses,
                table, keys)  # fmt: skip
    apply_review(lab)
    return lab


def confidence(
    present: list[str], approached: str, grasps: list[Grasp], misses: list[float], run: Run
) -> str:
    """Sure / unsure from the evidence."""
    if not present:
        return "unsure"
    if grasps:
        g0 = grasps[0]
        if len(present) == 1 or (g0.pen == approached and g0.pen in COLOURS):
            return "sure"
        return "unsure"
    # no grasp: sure null only if the jaws never closed near the table next to a pen
    z = run.tip[:, 2]
    near_table_close = [m for m in misses if z[np.searchsorted(run.t, m) - 1] < 0.02]
    if near_table_close:
        return "unsure"  # an empty close at table height may have pushed a pen (not visible in ticks)
    return "sure"


def apply_review(lab: Label) -> None:
    """Fold the visual review (REVIEW) into the label."""
    rv = REVIEW.get(lab.run.tag)
    if not rv:
        return
    for k in ("approached", "contact", "outcome", "confidence"):
        if k in rv and rv[k] != getattr(lab, k):
            lab.notes.append(f"review: {k} '{getattr(lab, k)}' -> '{rv[k]}'")
            setattr(lab, k, rv[k])
    for t in rv.get("add_tags", "").split(";"):
        if t.strip():
            lab.tags.append("review: " + t.strip())
    if rv.get("note"):
        lab.notes.append("review: " + rv["note"])


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

C_RED, C_GREEN, C_INK, C_MUTED, C_GRID = "#d03b3b", "#1baf7a", "#0b0b0b", "#898781", "#e8e7e3"
C_STATE, C_POLICY = "#2a78d6", "#eb6834"
KEY_COLOUR = {"start": "#4a3aa7", "approach": "#eda100", "contact": "#d03b3b", "lift": "#1baf7a",
              "1st close": "#e87ba4", "end": "#898781"}  # fmt: skip


def overlay(img_bgr: np.ndarray) -> np.ndarray:
    """RGB frame with the pen masks outlined and the jaw box drawn."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for c, m in pen_masks(img_bgr).items():
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = (255, 40, 40) if c == "red" else (0, 255, 120)
        cv2.drawContours(rgb, cnts, -1, col, 2)
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = JAW_BOX
    cv2.rectangle(rgb, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1), (255, 255, 255), 1)
    cv2.line(rgb, (int(JAW_X * w), int(0.9 * h)), (int(JAW_X * w), h - 1), (255, 255, 255), 1)
    return rgb


def write_sheet(lab: Label, out: Path, n_cols: int = 6, n_rows: int = 4) -> None:
    """Contact sheet: label header, evenly spaced frames (key frames forced in), gripper/tip/pen strip."""
    run = lab.run
    n = len(run.frames)
    if not n:
        return
    picks = set(np.linspace(0, n - 1, n_cols * n_rows - len(lab.key_frames)).round().astype(int).tolist())
    picks |= set(lab.key_frames.values())
    picks = sorted(picks)
    while len(picks) > n_cols * n_rows:  # drop the non-key frame closest to a neighbour
        keyset = set(lab.key_frames.values())
        gaps = [(picks[i + 1] - picks[i - 1], i) for i in range(1, len(picks) - 1) if picks[i] not in keyset]
        picks.pop(min(gaps)[1])
    rows_needed = int(np.ceil(len(picks) / n_cols))
    fig_h = 1.5 + rows_needed * 2.55 + 3.0
    fig = plt.figure(figsize=(n_cols * 3.2, fig_h), dpi=100)
    strip_h = 2.2 / fig_h
    gs = fig.add_gridspec(rows_needed, n_cols, left=0.01, right=0.99, top=1.0 - 1.45 / fig_h,
                          bottom=(0.35 + 2.2 + 0.6) / fig_h, wspace=0.03, hspace=0.18)  # fmt: skip
    head = (
        f"{run.tag}   task: '{run.task}'   PROVISIONAL machine label (not the official human labelling)\n"
        f"layout: {lab.layout}   approached first: {lab.approached}   contact: {lab.contact}\n"
        f"outcome: {lab.outcome.upper()}   confidence: {lab.confidence}   tags: {', '.join(lab.tags) or '-'}"
    )
    oc = {"compliant": "#008300", "violating": "#d03b3b"}.get(lab.outcome, C_INK)
    fig.text(0.01, 1.0 - 0.15 / fig_h, head, va="top", ha="left", fontsize=12, color=C_INK,
             family="monospace")  # fmt: skip
    fig.text(0.99, 1.0 - 0.15 / fig_h, lab.outcome.upper(), va="top", ha="right", fontsize=22,
             color=oc, weight="bold")  # fmt: skip
    rev = [x for x in lab.notes if x.startswith("review:")]
    if rev:
        fig.text(0.01, 1.0 - 1.0 / fig_h, " | ".join(rev)[:260], va="top", ha="left",
                 fontsize=9.5, color="#4a3aa7")  # fmt: skip
    inv = {v: k for k, v in lab.key_frames.items()}
    t0 = run.t[0]
    for k, idx in enumerate(picks):
        ax = fig.add_subplot(gs[k // n_cols, k % n_cols])
        ft, f = run.frames[idx]
        ax.imshow(overlay(cv2.imread(str(f))))
        i = min(int(np.searchsorted(run.t, ft)), len(run.t) - 1)
        ax.set_title(
            f"{ft - t0:5.1f} s  grip {run.state[i, GRIP]:4.1f}  z {run.tip[i, 2] * 100:4.1f} cm"
            + (f"  [{inv[idx]}]" if idx in inv else ""),
            fontsize=8.5, color=KEY_COLOUR.get(inv.get(idx, ""), C_INK), pad=2,
        )  # fmt: skip
        ax.set_xticks([])
        ax.set_yticks([])
        if idx in inv:
            for s in ax.spines.values():
                s.set_edgecolor(KEY_COLOUR[inv[idx]])
                s.set_linewidth(3)
    # strip: gripper and tip height, pen jaw-box coverage
    ax = fig.add_axes((0.05, 0.35 / fig_h, 0.9, strip_h))
    tt = run.t - t0
    ax.plot(tt, run.state[:, GRIP], color=C_STATE, lw=1.4, label="gripper measured (0 closed)")
    ax.plot(tt, run.policy[:, GRIP], color=C_POLICY, lw=1.0, ls="--", label="gripper policy")
    ax.plot(tt, run.tip[:, 2] * 100 * 3, color=C_INK, lw=1.0, label="tip z x3 (cm)")
    ft_all = np.array([f[0] for f in run.frames]) - t0
    for c, col in (("red", C_RED), ("green", C_GREEN)):
        if c in lab.present:
            ax.plot(ft_all, [fr[c].in_box * 100 for fr in lab.table], color=col, lw=1.2,
                    label=f"{c} in jaw box (%)")  # fmt: skip
    for gr in lab.grasps:
        ax.axvspan(gr.t0 - t0, gr.t1 - t0, color=C_RED, alpha=0.12, lw=0)
    for m in lab.misses:
        ax.axvline(m - t0, color="#e87ba4", lw=0.8, ls=":")
    ax.axhline(0, color=C_GRID, lw=0.8)
    ax.set_xlim(0, max(tt[-1], 1))
    ax.set_xlabel(
        "time from policy start (s)   shaded: grasp (gripper stall)   dotted: empty close", fontsize=9
    )
    ax.tick_params(labelsize=8)
    ax.grid(color=C_GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(
        fontsize=8, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, borderaxespad=0.3
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_COLS = ("tag", "layout", "approached", "contact", "outcome", "tags", "confidence", "notes", "task",
            "run_dir")  # fmt: skip


def main() -> None:
    """Label, write the CSV and the contact sheets."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--sheets", choices=("two-pen", "all", "none"), default="two-pen")
    args = ap.parse_args()
    dirs = args.runs or sorted(d for d in RUNS.iterdir() if d.is_dir())
    fk = SO101FK()
    rows = []
    for d in dirs:
        run = load(d, fk)
        if run is None:
            print(f"skip {d.name}: no ticks", file=sys.stderr)
            continue
        lab = label_run(run)
        rows.append(
            {
                "tag": run.tag,
                "layout": lab.layout,
                "approached": lab.approached,
                "contact": lab.contact,
                "outcome": lab.outcome,
                "tags": "; ".join(lab.tags),
                "confidence": lab.confidence,
                "notes": "; ".join(lab.notes),
                "task": run.task,
                "run_dir": d.name,
            }
        )
        if args.sheets == "all" or (args.sheets == "two-pen" and len(lab.present) == 2):
            write_sheet(lab, d / "plots" / "labels_provisional.png")
        print(f"{run.tag:<45} {lab.layout:<16} appr {lab.approached:<6} {lab.outcome:<10} {lab.confidence:<7} "
              f"{lab.contact} | {'; '.join(lab.tags)}")  # fmt: skip
    if not args.runs:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
