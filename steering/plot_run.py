"""Plot and summarise a run recorded by steering/rollout.py.

    uv run python steering/plot_run.py steering/results/runs/<stamp>_<tag>
    uv run python steering/plot_run.py --self-test [DIR]  # fake run (default: a temp dir), plotted; no hardware

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
"""

from __future__ import annotations

import argparse
import json
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


def plot_joints(ticks: dict, st: dict, out: Path) -> None:
    """joints.png: state, policy and sent command per joint, trained band, clipped ticks."""
    t, phase = ticks["t"], ticks["phase"]
    s_low, s_high = range_to_arm(st["s01"], st["s99"], st["signs"], st["offsets"])
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for i, (ax, joint) in enumerate(zip(axes.flat, JOINTS, strict=True)):
        if np.isfinite(s_low[i]) and np.isfinite(s_high[i]):
            ax.axhspan(
                s_low[i], s_high[i], color=C_BAND, alpha=0.6, linewidth=0, label="trained state q01..q99"
            )
        # Teardown rows (phase 2) repeat one stale reading: draw the state there dashed and thin.
        live = np.where(phase == 2, np.nan, ticks["state"][:, i])
        stale = np.where(phase == 2, ticks["state"][:, i], np.nan)
        ax.plot(t, live, color=C_STATE, lw=2, label="measured state")
        if (phase == 2).any():
            ax.plot(t, stale, color=C_STATE, lw=1, ls=":", alpha=0.6, label="state (stale, teardown)")
        ax.plot(t, ticks["action_policy"][:, i], color=C_POLICY, lw=1.2, label="policy command")
        ax.plot(t, ticks["action_sent"][:, i], color=C_SENT, lw=1.2, ls="--", label="sent (clipped) command")
        clip = ticks["clipped"][:, i]
        if clip.any():
            ax.scatter(
                t[clip], ticks["action_sent"][clip, i], s=10, color=C_CLIP, zorder=5, label="clipped tick"
            )
        if (phase == 2).any():
            ax.axvspan(
                t[phase == 2].min(), t[phase == 2].max(), color=C_GRID, alpha=0.7, lw=0, label="teardown"
            )
        ax.set_title(joint, fontsize=10, color=C_INK, loc="left")
        ax.set_ylabel("deg" if joint != "gripper" else "0-100", fontsize=8, color=C_MUTED)
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=9, color=C_MUTED)
    handles, labels = {}, {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
            labels[lab] = lab
    fig.legend(list(handles.values()), list(labels), loc="upper center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "joints.png", dpi=110)
    plt.close(fig)


def plot_chunks(ticks: dict, chunks: dict, fps: float, out: Path) -> None:
    """chunks.png: predicted chunks over the sent command for shoulder_lift and elbow_flex."""
    sel = [JOINTS.index("shoulder_lift"), JOINTS.index("elbow_flex")]
    fig, axes = plt.subplots(len(sel), 1, figsize=(14, 7), sharex=True)
    arm = chunks.get("chunk_arm")
    for ax, j in zip(axes, sel, strict=True):
        ax.plot(ticks["t"], ticks["state"][:, j], color=C_STATE, lw=2, label="measured state")
        ax.plot(ticks["t"], ticks["action_sent"][:, j], color=C_SENT, lw=1, ls="--", label="sent command")
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
        "time (s); chunks at fps, from the end of their inference call (sync) or its start (RTC)",
        fontsize=9,
        color=C_MUTED,
    )
    fig.tight_layout()
    fig.savefig(out / "chunks.png", dpi=110)
    plt.close(fig)


def plot_cadence(ticks: dict, chunks: dict, out: Path) -> None:
    """cadence.png: tick intervals over time and as a histogram, inference calls shaded."""
    run = ticks["phase"] == 1
    t = ticks["t"][run] if run.sum() > 1 else ticks["t"]
    dt_ms = np.diff(t) * 1e3
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    for k in range(len(chunks.get("t_start", []))):
        s, d = chunks["t_start"][k], chunks["duration"][k]
        ax.axvspan(s, s + d, color=C_POLICY, alpha=0.15, lw=0, label="inference call" if k == 0 else None)
    if len(dt_ms):
        ax.plot(t[1:], dt_ms, color=C_STATE, lw=1, marker=".", ms=3, label="tick interval")
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


def plot_frames(run_dir: Path, meta: dict, out: Path, max_frames: int = 48) -> bool:
    """frames.png contact sheet; False if the run has no frames."""
    frames = meta.get("frames") or [
        {
            "file": f"frames/{p.name}",
            "t": float(p.stem.split("_")[1].rstrip("s")),
            "camera": p.stem.split("_", 2)[2],
        }
        for p in sorted((run_dir / "frames").glob("*.jpg"))
    ]
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
        ax.set_title(f"{f['t']:.1f} s {f.get('camera', '')}", fontsize=7, color=C_INK)
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
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    dt = np.diff(t) * 1e3
    s_low, s_high = range_to_arm(st["s01"], st["s99"], st["signs"], st["offsets"])
    a_low, a_high = range_to_arm(st["a01"], st["a99"], st["signs"], st["offsets"])
    start, end = state[0], state[-1]
    s_out = distance_outside(start, s_low, s_high)
    a_out = distance_outside(start, a_low, a_high)
    travel = np.nansum(np.abs(np.diff(state, axis=0)), axis=0)
    lines = [
        f"run: {meta.get('tag', '?')}  task: {meta.get('task', '?')!r}  inference: {meta.get('inference_type', '?')}"
        f"  max_relative_target: {(meta.get('robot') or {}).get('max_relative_target', '?')}",
        f"connect: {(meta.get('connect') or {}).get('path', '?')}   stats: {st['source']}",
        f"duration {duration:.1f} s, {len(t)} ticks, effective {(len(t) - 1) / duration if duration else 0:.1f} Hz"
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
    plot_joints(ticks, st, out)
    plot_chunks(ticks, chunks, fps, out)
    plot_cadence(ticks, chunks, out)
    has_frames = plot_frames(run_dir, meta, out)
    text = summary(ticks, chunks, meta, st)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nplots: {', '.join(sorted(p.name for p in out.glob('*.png')))} in {out}")
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


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", nargs="?", type=Path, help="steering/results/runs/<stamp>_<tag>")
    parser.add_argument("--self-test", action="store_true", help="synthesise a fake run and plot it")
    args = parser.parse_args()
    if args.self_test:
        run_dir = args.run_dir or Path(tempfile.mkdtemp(prefix="plot_run_selftest_")) / "selftest"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        make_fake_run(run_dir)
        plot_run(run_dir)
        expected = {"joints.png", "chunks.png", "cadence.png", "frames.png", "summary.txt"}
        missing = expected - {p.name for p in (run_dir / "plots").iterdir()}
        if missing:
            sys.exit(f"self-test FAILED: missing {sorted(missing)}")
        print("self-test OK")
        return
    if args.run_dir is None:
        parser.error("RUN_DIR is required (or --self-test)")
    plot_run(args.run_dir)


if __name__ == "__main__":
    main()
