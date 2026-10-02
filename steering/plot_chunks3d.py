"""3D view of a recorded run: measured gripper-tip path and every predicted chunk, via SO-101 FK.

    uv run --with plotly python steering/plot_chunks3d.py steering/results/runs/<stamp>_<tag>
    uv run python steering/plot_chunks3d.py RUN_DIR --png-only          # no plotly needed

Writes to RUN_DIR/plots/:
  chunks3d.html  one self-contained file (plotly.js inlined, works offline):
                 * measured tip path (control-loop ticks only), coloured by time;
                 * every predicted chunk as a thin polyline of its 30 steps (FK of chunk_arm);
                 * a slider / play button over chunks: the selected chunk drawn thick, the measured tip
                   over the same horizon (what the arm actually did while that chunk ran), the tip at
                   the chunk's start tick and the arm's link skeleton at that tick;
                 * buttons to show/hide the skeleton; the base and the table plane.
  chunks3d.png   matplotlib 3D: the tip path plus every 5th chunk, for a quick look.

Frames and conventions (see steering/so101_fk.py): base_link frame, metres, x forward, z up, z = 0 at
base_link's origin. The table height is not recorded: the plane is drawn at the base mesh's underside
(z = -0.0024 m), i.e. assuming the base stands flat on the table. Only policy-phase chunks
(``phase == PHASE_POLICY``) and ticks are drawn; teardown/reset ticks (the return to the start pose)
are left out.

Executed part of a chunk (drawn thick on the selected chunk; the whole prediction is drawn dotted):

* Runs recorded with RTC merge data (rollout.py patch c, ``merged`` in chunks.npz): steps
  ``[merge_delay_k, merge_delay_k + merge_prev_consumed_j)``, j the next chunk to merge, i.e. from the
  steps the merge dropped to the offset at which chunk j replaced the queue. When no later merge
  replaced it (last chunk, or the queue was cleared by an engine reset first) the end is estimated
  from the policy ticks between the merge and the next merge / reset, capped at the chunk length. A
  chunk that never merged (discarded) has no executed part. The chunk starts at its merge time.
* Older runs and sync runs (fallback, said so in the title and legend): plot_run.py's heuristic, the
  steps ``[inference_delay, end)``; RTC chunks start at t_start, sync chunks when the inference call
  returns. "The chunk's start tick" is the first tick at or after that time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402
from so101_fk import SKELETON, SO101FK, base_mesh_z_range  # noqa: E402

PHASE_POLICY = 1
TABLE_Z_DEFAULT = -0.0024  # base mesh underside, used when the meshes are missing
# Reference palette (dataviz skill), same roles as plot_run.py: blue = measured, orange = policy.
C_STATE, C_POLICY, C_INK, C_MUTED, C_GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e8e7e3"
C_TABLE = "#d9d7d0"
# Sequential blue ramp (steps 250..700) for time along the measured path.
TIME_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    """ticks, chunks, meta of a run dir (same files plot_run.py reads)."""
    ticks = dict(np.load(run_dir / "ticks.npz"))
    chunks = dict(np.load(run_dir / "chunks.npz")) if (run_dir / "chunks.npz").is_file() else {}
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").is_file() else {}
    return ticks, chunks, meta


def policy_rows(ticks: dict) -> np.ndarray:
    """Indices of control-loop ticks (all ticks for recordings without phase labels)."""
    phase = ticks.get("phase")
    if phase is None or (phase == PHASE_POLICY).sum() < 2:
        return np.arange(len(ticks["t"]))
    return np.flatnonzero(phase == PHASE_POLICY)


def segment_breaks(ticks: dict, rows: np.ndarray) -> np.ndarray:
    """Positions in rows where a new episode starts or the clock jumps by > 1 s (break the line)."""
    t = ticks["t"][rows]
    cut = np.diff(t) > 1.0
    if "episode" in ticks:
        cut |= np.diff(ticks["episode"][rows]) != 0
    return np.flatnonzero(cut) + 1


def with_gaps(xyz: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Insert NaN rows at breaks so polylines do not bridge episodes."""
    return np.insert(xyz.astype(np.float64), breaks, np.nan, axis=0) if len(breaks) else xyz


def has_merge_data(chunks: dict) -> bool:
    """True for RTC runs recorded with rollout.py's merge columns (patch c)."""
    delays = chunks.get("inference_delay")
    return (
        "merged" in chunks
        and bool(chunks.get("merge_hooked", True))
        and delays is not None
        and bool(np.any(np.asarray(delays) >= 0))
    )


def executed_spans(ticks: dict, chunks: dict, rows: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2) [lo, hi) executed steps of every chunk row from the merge data, and estimated-end flags.

    See the module docstring. Unmerged chunks get [0, 0).
    """
    n = len(chunks["t_start"])
    span = np.zeros((n, 2), dtype=np.int64)
    estimated = np.zeros(n, dtype=bool)
    merged = np.asarray(chunks["merged"], dtype=bool)
    merge_t = np.asarray(chunks["merge_t"], dtype=np.float64)
    drop = np.asarray(chunks["merge_delay"], dtype=np.int64)
    consumed = np.asarray(chunks["merge_prev_consumed"], dtype=np.int64)
    prev_len = np.asarray(chunks["merge_prev_len"], dtype=np.int64)
    resets = np.asarray(chunks.get("engine_reset_t", np.zeros(0)), dtype=np.float64)
    order = [int(i) for i in np.argsort(merge_t) if merged[i]]  # merge (completion) order
    t_tick = ticks["t"][rows]
    for pos, i in enumerate(order):
        lo = int(min(max(drop[i], 0), h))
        j = order[pos + 1] if pos + 1 < len(order) else None
        if j is not None and prev_len[j] > 0:
            span[i] = lo, min(h, lo + int(consumed[j]))
            continue
        bound = merge_t[j] if j is not None else np.inf
        later = resets[resets > merge_t[i]]
        if len(later):
            bound = min(bound, float(later[0]))
        n_ticks = int(np.count_nonzero((t_tick >= merge_t[i]) & (t_tick < bound)))
        span[i] = lo, min(h, lo + n_ticks)
        estimated[i] = True
    return span, estimated


def analyse(run_dir: Path, fk: SO101FK) -> dict:
    """Everything both views need, in base_link metres."""
    ticks, chunks, meta = load_run(run_dir)
    rows = policy_rows(ticks)
    if len(rows) < 2:
        raise SystemExit(f"{run_dir}: fewer than 2 control-loop ticks, nothing to plot")
    fps = float(meta.get("fps") or 30.0)
    t, state = ticks["t"][rows], ticks["state"][rows]
    tip = fk.tip(state)
    arm = chunks.get("chunk_arm")
    if arm is None or not len(arm):
        arm = np.zeros((0, 30, 6))
    h = arm.shape[1]
    n_all = len(arm)
    delays = np.asarray(chunks.get("inference_delay", np.full(n_all, -1)))[:n_all]
    t_exec = np.asarray(
        [chunks["t_start"][k] + (0.0 if delays[k] >= 0 else chunks["duration"][k]) for k in range(n_all)]
    )
    merge_mode = has_merge_data(chunks)
    if merge_mode:
        span, estimated = executed_spans(ticks, chunks, rows, h)
        span, estimated = span[:n_all], estimated[:n_all]
        merged = np.asarray(chunks["merged"], dtype=bool)[:n_all]
        t_exec = np.where(merged, np.asarray(chunks["merge_t"], dtype=np.float64)[:n_all], t_exec)
        discard = np.asarray(chunks.get("discard_reason", np.zeros(n_all)), dtype=np.int64)[:n_all]
    else:
        span = np.stack([np.clip(delays, 0, h), np.full(n_all, h)], axis=1).astype(np.int64)
        estimated = np.zeros(n_all, dtype=bool)
        merged = np.ones(n_all, dtype=bool)  # unknown: the heuristic treats every chunk as executed
        discard = np.zeros(n_all, dtype=np.int64)
    # Policy-phase chunks only (recordings without phase labels: all chunks).
    phase = chunks.get("phase")
    keep = np.arange(n_all) if phase is None else np.flatnonzero(np.asarray(phase)[:n_all] == PHASE_POLICY)
    arm, delays, t_exec = arm[keep], delays[keep], t_exec[keep]
    span, estimated, merged, discard = span[keep], estimated[keep], merged[keep], discard[keep]
    k_n = len(keep)
    chunk_tip = fk.tip(arm.reshape(-1, 6)).reshape(k_n, h, 3) if k_n else np.zeros((0, h, 3))
    tick_at = np.clip(np.searchsorted(t, t_exec), 0, len(t) - 1) if k_n else np.zeros(0, dtype=int)
    zr = base_mesh_z_range(fk.urdf_path)
    return {
        "run": run_dir.name,
        "meta": meta,
        "fps": fps,
        "t": t,
        "t0": float(t[0]),
        "tip": tip,
        "breaks": segment_breaks(ticks, rows),
        "skeleton": fk.skeleton(state[tick_at]) if k_n else np.zeros((0, len(SKELETON) + 1, 3)),
        "start_skeleton": fk.skeleton(state[:1])[0],
        "chunk_tip": chunk_tip,
        "t_exec": t_exec,
        "tick_at": tick_at,
        "delays": np.asarray(delays),
        "durations": np.asarray(chunks.get("duration", np.zeros(n_all)))[:n_all][keep],
        "rows": keep,
        "span": span,
        "estimated": estimated,
        "merged": merged,
        "discard": discard,
        "merge_mode": merge_mode,
        "table_z": zr[0] if zr else TABLE_Z_DEFAULT,
    }


def bounds(a: dict, pad: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis min/max over the path, the chunks and the base, padded."""
    pts = [a["tip"], a["chunk_tip"].reshape(-1, 3), a["start_skeleton"], np.zeros((1, 3))]
    p = np.concatenate([x for x in pts if len(x)])
    lo, hi = np.nanmin(p, axis=0) - pad, np.nanmax(p, axis=0) + pad
    lo[2] = min(lo[2], a["table_z"])
    return lo, hi


def span_source(a: dict) -> str:
    """How the executed part of a chunk was found (title and legend)."""
    if a["merge_mode"]:
        return "executed span from the recorded RTC merges"
    return "executed span = [inference_delay, end): heuristic fallback, no merge data"


def title_of(a: dict) -> str:
    """Run name, task, inference type and max_relative_target in one line."""
    m = a["meta"]
    return (
        f"{a['run']} · task {m.get('task', '?')!r} · {m.get('inference_type', '?')} · "
        f"max_relative_target {(m.get('robot') or {}).get('max_relative_target', '?')}"
    )


# ---------------------------------------------------------------------------
# PNG (matplotlib)
# ---------------------------------------------------------------------------


def write_png(a: dict, out: Path, every: int = 5) -> None:
    """chunks3d.png: tip path coloured by time, every ``every``-th chunk, base and table."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    lo, hi = bounds(a)
    xx, yy = np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]])
    ax.plot_surface(xx, yy, np.full_like(xx, a["table_z"]), color=C_TABLE, alpha=0.35, linewidth=0)
    cmap = LinearSegmentedColormap.from_list("time", TIME_RAMP)
    tip, t = a["tip"], a["t"] - a["t0"]
    seg = np.stack([tip[:-1], tip[1:]], axis=1)
    keep = np.ones(len(seg), dtype=bool)
    keep[a["breaks"] - 1] = False
    lc = Line3DCollection(seg[keep], cmap=cmap, linewidth=2.2)
    lc.set_array(t[:-1][keep])
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, shrink=0.55, pad=0.08)
    cb.set_label("time since loop start (s)", color=C_MUTED, fontsize=9)
    for n, k in enumerate(range(0, len(a["chunk_tip"]), every)):
        c = a["chunk_tip"][k]
        ax.plot(
            *c.T, color=C_POLICY, lw=0.9, alpha=0.75, label="predicted chunk (every 5th)" if n == 0 else None
        )
        ax.scatter(*c[0], color=C_POLICY, s=8)
    sk = a["start_skeleton"]
    ax.plot(*sk[:-1].T, color=C_INK, lw=1.5, marker="o", ms=3, alpha=0.6, label="arm at loop start")
    ax.plot(*np.stack([sk[-3], sk[-1]]).T, color=C_INK, lw=1.0, alpha=0.6)
    ax.scatter(0, 0, 0, color=C_INK, marker="s", s=30, label="base_link origin")
    ax.scatter(*tip[0], color=C_STATE, s=40, marker="o", label="tip at start")
    ax.scatter(*tip[-1], color=C_STATE, s=40, marker="X", label="tip at end")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    for lab, f in (
        ("x forward (m)", ax.set_xlabel),
        ("y left (m)", ax.set_ylabel),
        ("z up (m)", ax.set_zlabel),
    ):
        f(lab, color=C_MUTED, fontsize=9)
    ax.tick_params(colors=C_MUTED, labelsize=7)
    ax.view_init(elev=22, azim=-125)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title(
        title_of(a)
        + f"\nevery 5th predicted chunk, whole; executed parts are in chunks3d.html ({span_source(a)})"
        + f"\ntable plane at z = {a['table_z']:.4f} m (assumed: base underside; real height not recorded)",
        fontsize=9,
        color=C_INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out / "chunks3d.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML (plotly)
# ---------------------------------------------------------------------------


def _xyz(p: np.ndarray) -> dict:
    """Plotly x/y/z lists, NaN -> None (breaks the line)."""
    return {
        ax: [None if not np.isfinite(v) else round(float(v), 5) for v in p[:, i]]
        for i, ax in enumerate("xyz")
    }


def _chunk_frame(a: dict, k: int) -> list[dict]:
    """Data for the animated traces (executed part, whole prediction, measured horizon, tip, skeleton)."""
    c = a["chunk_tip"][k]
    lo, hi = (int(x) for x in a["span"][k])
    executed = c[lo:hi] if a["merged"][k] and hi > lo else np.full((1, 3), np.nan)
    t = a["t"]
    lo_t = a["t_exec"][k]
    hi_t = lo_t + max(hi - lo - 1, 0) / a["fps"]
    horizon = a["tip"][(t >= lo_t) & (t <= hi_t)] if a["merged"][k] and hi > lo else np.zeros((0, 3))
    sk = a["skeleton"][k]
    sk_pts = np.concatenate([sk[:-1], [[np.nan] * 3], sk[[-3, -1]]])  # arm, then wrist_roll -> moving jaw
    return [
        _xyz(executed),
        _xyz(c),
        _xyz(horizon if len(horizon) else np.full((1, 3), np.nan)),
        _xyz(a["tip"][a["tick_at"][k] : a["tick_at"][k] + 1]),
        _xyz(sk_pts),
    ]


def write_html(a: dict, out: Path) -> None:
    """chunks3d.html with plotly.js inlined."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise SystemExit(
            "plotly is not installed; run with `uv run --with plotly python steering/plot_chunks3d.py ...` "
            "or pass --png-only"
        ) from e
    lo, hi = bounds(a)
    t_rel = a["t"] - a["t0"]
    tip_g = with_gaps(a["tip"], a["breaks"])
    t_g = with_gaps(t_rel[:, None], a["breaks"])[:, 0]
    k_n, h = a["chunk_tip"].shape[:2]
    # All chunks as one trace, NaN-separated.
    all_c = (
        np.concatenate([np.vstack([c, np.full((1, 3), np.nan)]) for c in a["chunk_tip"]])
        if k_n
        else np.full((1, 3), np.nan)
    )
    all_txt = [
        f"chunk {k} · step {i} · t₀ {a['t_exec'][k] - a['t0']:.2f} s"
        for k in range(k_n)
        for i in [*range(h), -1]
    ]
    colorscale = [[i / (len(TIME_RAMP) - 1), c] for i, c in enumerate(TIME_RAMP)]
    traces = [
        go.Surface(
            x=[[lo[0], hi[0]], [lo[0], hi[0]]],
            y=[[lo[1], lo[1]], [hi[1], hi[1]]],
            z=[[a["table_z"]] * 2] * 2,
            colorscale=[[0, C_TABLE], [1, C_TABLE]],
            showscale=False,
            opacity=0.35,
            name=f"table (assumed z = {a['table_z']:.4f} m)",
            showlegend=True,
            hovertemplate="table plane (assumed: base underside)<extra></extra>",
        ),
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[a["table_z"], 0],
            mode="lines+markers",
            line={"color": C_INK, "width": 6},
            marker={"size": 4, "symbol": "square", "color": C_INK},
            name="base (base_link origin)",
            hovertemplate="base_link origin<extra></extra>",
        ),
        go.Scatter3d(
            **_xyz(tip_g),
            mode="lines",
            line={
                "color": [None if not np.isfinite(v) else float(v) for v in t_g],
                "colorscale": colorscale,
                "width": 5,
                "colorbar": {"title": {"text": "time (s)"}, "len": 0.5, "x": 1.0},
            },
            name="measured tip (coloured by time)",
            text=[None if not np.isfinite(v) else f"t {v:.2f} s" for v in t_g],
            hovertemplate="%{text}<br>x %{x:.3f} y %{y:.3f} z %{z:.3f} m<extra>measured</extra>",
        ),
        go.Scatter3d(
            **_xyz(all_c),
            mode="lines",
            line={"color": C_POLICY, "width": 1.5},
            opacity=0.35,
            name="every predicted chunk",
            text=all_txt,
            hovertemplate="%{text}<br>z %{z:.3f} m<extra>chunk</extra>",
        ),
    ]
    n_static = len(traces)
    names = [
        "selected chunk: executed part" + ("" if a["merge_mode"] else " (heuristic: inference_delay..end)"),
        "selected chunk: whole prediction",
        "measured while it executed",
        "tip at chunk start",
        "arm skeleton",
    ]
    styles = [
        {
            "mode": "lines+markers",
            "line": {"color": C_POLICY, "width": 7},
            "marker": {"size": 2.5, "color": C_POLICY},
        },
        {"mode": "lines", "line": {"color": C_POLICY, "width": 2, "dash": "dot"}},
        {"mode": "lines", "line": {"color": C_STATE, "width": 9}},
        {
            "mode": "markers",
            "marker": {"size": 6, "color": C_STATE, "line": {"color": "#ffffff", "width": 2}},
        },
        {
            "mode": "lines+markers",
            "line": {"color": C_INK, "width": 6},
            "marker": {"size": 4, "color": C_INK},
        },
    ]
    first = _chunk_frame(a, 0) if k_n else [_xyz(np.full((1, 3), np.nan))] * len(names)
    sk_labels = [lab for lab, _ in SKELETON] + ["", "wrist_roll", "moving jaw tip"]
    for name, style, data in zip(names, styles, first, strict=True):
        extra = (
            {"text": sk_labels, "hovertemplate": "%{text}<br>z %{z:.3f} m<extra>skeleton</extra>"}
            if name == "arm skeleton"
            else {"hoverinfo": "x+y+z+name"}
        )
        traces.append(go.Scatter3d(name=name, **data, **style, **extra))
    anim_idx = list(range(n_static, n_static + len(names)))
    sk_idx = anim_idx[-1]

    frames, steps = [], []
    for k in range(k_n):
        dur = a["durations"][k] * 1e3 if len(a["durations"]) else float("nan")
        label = f"{k}"
        info = (
            f"chunk {k}/{k_n - 1} (row {int(a['rows'][k])}) · starts t = {a['t_exec'][k] - a['t0']:.2f} s · "
            f"inference {dur:.0f} ms · inference_delay {int(a['delays'][k])} · {_exec_note(a, k)}"
            + _where(a, k)
        )
        frames.append(
            go.Frame(
                name=label,
                data=[go.Scatter3d(**d) for d in _chunk_frame(a, k)],
                traces=anim_idx,
                layout={"annotations": [_info(info)]},
            )
        )
        steps.append(
            {
                "method": "animate",
                "label": label,
                "args": [
                    [label],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )
    first_info = frames[0].layout.annotations[0] if frames else _info("no chunks recorded")
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title={"text": title_of(a), "font": {"size": 13, "color": C_INK}, "x": 0.01},
        paper_bgcolor="#fcfcfb",
        font={"family": "system-ui, sans-serif", "color": C_INK},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        legend={"x": 0.01, "y": 0.92, "bgcolor": "rgba(252,252,251,0.8)", "font": {"size": 11}},
        annotations=[
            first_info,
            _info(
                f"{span_source(a)} · x forward, y left, z up (m), base_link frame; table height not "
                "recorded: plane drawn at the base underside",
                y=1.02,
                size=10,
            ),
        ],
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x forward (m)", "range": [lo[0], hi[0]], "gridcolor": C_GRID},
            "yaxis": {"title": "y left (m)", "range": [lo[1], hi[1]], "gridcolor": C_GRID},
            "zaxis": {"title": "z up (m)", "range": [lo[2], hi[2]], "gridcolor": C_GRID},
            "camera": {"eye": {"x": -1.2, "y": -1.5, "z": 0.8}},
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 0.06,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "▶ play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 250, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "❚❚ pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            },
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.25,
                "y": 0.06,
                "xanchor": "left",
                "buttons": [
                    {"label": "skeleton on", "method": "restyle", "args": [{"visible": True}, [sk_idx]]},
                    {"label": "skeleton off", "method": "restyle", "args": [{"visible": False}, [sk_idx]]},
                ],
            },
        ],
        sliders=[
            {
                "active": 0,
                "steps": steps,
                "x": 0.45,
                "len": 0.53,
                "y": 0.1,
                "currentvalue": {"prefix": "chunk "},
            }
        ]
        if steps
        else [],
    )
    fig.write_html(
        out / "chunks3d.html",
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
        default_width="100%",
        default_height="94vh",
        config={"responsive": True, "displaylogo": False},
    )


def _exec_note(a: dict, k: int) -> str:
    """Executed steps of chunk k, or why it was not executed."""
    if not a["merged"][k]:
        why = {1: "discarded: engine reset", 2: "discarded / never merged"}.get(
            int(a["discard"][k]), "not merged"
        )
        return f"not executed ({why})"
    lo, hi = (int(x) for x in a["span"][k])
    est = " (end estimated from ticks)" if a["estimated"][k] else ""
    src = "merge" if a["merge_mode"] else "heuristic"
    return f"executed steps [{lo}, {hi}) ({src}){est}"


def _where(a: dict, k: int) -> str:
    """Note for chunks that start before the first or after the last control-loop tick."""
    if a["t_exec"][k] < a["t"][0]:
        return " · computed before the first tick"
    if a["t_exec"][k] > a["t"][-1]:
        return " · after the last control-loop tick (not executed)"
    return ""


def _info(text: str, y: float = 0.97, size: int = 11) -> dict:
    return {
        "text": text,
        "xref": "paper",
        "yref": "paper",
        "x": 0.01,
        "y": y,
        "showarrow": False,
        "xanchor": "left",
        "font": {"size": size, "color": C_MUTED},
    }


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--png-only", action="store_true", help="skip the HTML (no plotly needed)")
    args = parser.parse_args()
    fk = SO101FK()
    a = analyse(args.run_dir, fk)
    out = args.run_dir / "plots"
    out.mkdir(exist_ok=True)
    write_png(a, out)
    print(f"wrote {out / 'chunks3d.png'}")
    if not args.png_only:
        write_html(a, out)
        print(f"wrote {out / 'chunks3d.html'}")
    z = a["tip"][:, 2]
    print(
        f"{len(a['t'])} ticks, {len(a['chunk_tip'])} policy chunks ({int(a['merged'].sum())} executed); "
        f"tip z min {np.nanmin(z):.3f} m, max {np.nanmax(z):.3f} m; table assumed at z = {a['table_z']:.4f} m; "
        f"{span_source(a)}"
    )


if __name__ == "__main__":
    main()
