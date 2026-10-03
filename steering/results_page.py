"""Build the one-page results dashboard for the 2 Oct 2026 SO-101 runs.

Reads, all under ``steering/results/``:
  runs.csv, labels_provisional.csv, plan_consistency_summary.csv, ANALYSIS_2026-10-02.md,
  prompt_counterfactual/{results.json, SUMMARY.md, frames.json},
  runs/<run>/{meta.json, chunks.npz, ticks.npz} and the first policy frame of each two-pen run.
Writes ``steering/results/results_page.html``: an Artifact page body (no doctype/html/head/body), with
the data inlined as JSON and the charts drawn as inline SVG by inline JS. No number is typed into the
template; every figure comes from the files above.

Run: uv run python steering/results_page.py
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import stats

STEERING = Path(__file__).resolve().parent
RES = STEERING / "results"
CF = RES / "prompt_counterfactual"
RUNS_DIR = RES / "runs"
OUT = RES / "results_page.html"
PHASE_POLICY = 1
TILE_W, TILE_H = 240, 180
WARNINGS: list[str] = []


def warn(msg: str) -> None:
    """Record a data problem; it is printed at the end and listed on the page."""
    WARNINGS.append(msg)
    print(f"warning: {msg}", file=sys.stderr)


def fnum(v: object) -> float | None:
    """A float, or None for an empty or non-numeric cell."""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV file as dicts."""
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- text helpers


def md_inline(text: str) -> str:
    """Escape text and render **bold**, *italic* and `code`."""
    s = html.escape(text.strip(), quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def md_block(text: str, marker: str, *, what: str) -> str:
    """The line holding ``marker`` plus its continuation lines, as one string.

    A bullet continues on more-indented non-bullet lines; a plain paragraph continues until a blank line,
    a bullet, a heading or a table row.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if marker not in line:
            continue
        is_bullet = line.lstrip().startswith("- ")
        indent = len(line) - len(line.lstrip())
        out = [line.strip().removeprefix("- ")]
        for nxt in lines[i + 1 :]:
            st = nxt.strip()
            if not st or st.startswith(("- ", "#", "|")) or re.match(r"^\d+\.\s", st):
                break
            if is_bullet and len(nxt) - len(nxt.lstrip()) <= indent:
                break
            out.append(st)
        return " ".join(out)
    warn(f"could not find {what!r} ({marker!r}) in the markdown")
    return ""


def md_bullets_after(text: str, marker: str, *, what: str) -> list[str]:
    """The bullets (with continuations) of the list that follows the line holding ``marker``."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if marker in line), None)
    if start is None:
        warn(f"could not find {what!r} ({marker!r}) in the markdown")
        return []
    items: list[str] = []
    indent: int | None = None
    for line in lines[start + 1 :]:
        st = line.strip()
        ind = len(line) - len(line.lstrip())
        if not st:
            if items:
                break
            continue
        if indent is None and not st.startswith("- "):
            continue  # the rest of the paragraph that holds the marker
        if st.startswith("- ") and (indent is None or ind == indent):
            indent = ind
            items.append(st[2:])
        elif items and indent is not None and ind > indent and not st.startswith("- "):
            items[-1] += " " + st
        elif items and indent is not None and ind > indent:
            continue  # deeper sub-bullets are left out
        else:
            break
    return items


def first_para(text: str) -> str:
    """The first paragraph of a markdown body, joined into one line."""
    out = []
    for line in text.strip().splitlines():
        st = line.strip()
        if not st or st.startswith(("- ", "#", "|")):
            break
        out.append(st)
    return " ".join(out)


def md_section(text: str, heading_prefix: str) -> str:
    """Body of the markdown section whose heading starts with ``heading_prefix``."""
    m = re.search(rf"^(#+) {re.escape(heading_prefix)}.*$", text, flags=re.M)
    if not m:
        warn(f"section {heading_prefix!r} not found")
        return ""
    level = len(m.group(1))
    rest = text[m.end() :]
    nxt = re.search(rf"^#{{1,{level}}} ", rest, flags=re.M)
    return rest[: nxt.start()] if nxt else rest


def numbered_items(text: str) -> list[dict[str, object]]:
    """Top-level ``1. ...`` items with their sub-bullets."""
    items: list[dict[str, object]] = []
    for line in text.splitlines():
        st = line.strip()
        m = re.match(r"^\d+\.\s+(.*)$", line)
        if m:
            items.append({"head": m.group(1), "sub": []})
        elif items and st.startswith("- "):
            items[-1]["sub"].append(st[2:])  # type: ignore[union-attr]
        elif items and st and line.startswith(" "):
            sub = items[-1]["sub"]
            if sub:
                sub[-1] += " " + st  # type: ignore[index]
            else:
                items[-1]["head"] = f"{items[-1]['head']} {st}"
    return items


def cap_first(s: str) -> str:
    """Upper-case the first character."""
    s = s.strip()
    return s[:1].upper() + s[1:]


def minus(s: str) -> str:
    """Typographic minus for negative numbers."""
    return s.replace("-", "−")


def f1(v: float | None, nd: int = 1, signed: bool = False) -> str:
    """Format a number, with a true minus sign; '' for None."""
    if v is None:
        return ""
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return minus(s)


def rng(vals: list[float], nd: int = 1, unit: str = "") -> str:
    """'a–b unit' from a list of values."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    sep = " to " if lo < 0 else "–"
    body = f1(lo, nd) if f1(lo, nd) == f1(hi, nd) else f"{f1(lo, nd)}{sep}{f1(hi, nd)}"
    return f"{body} {unit}".strip()


# --------------------------------------------------------------------------- run naming


def short_tag(tag: str) -> str:
    """median_rtc_cap6_no_color_prompt_2_pens_1 -> no_color_2_pens_1 (ANALYSIS naming)."""
    s = tag.removeprefix("median_rtc_")
    if s == tag:
        return tag
    s = re.sub(r"^cap\d+_(?=\D)", "", s)
    return s.replace("_prompt", "")


def prompt_kind(task: str) -> str:
    """Prompt family from the task string."""
    t = task.lower()
    if "red" in t:
        return "red"
    if "green" in t:
        return "green"
    if "any colo" in t:
        return "any"
    if "pen" in t:
        return "pen"
    return "null"


def layout_key(layout: str) -> str | None:
    """'red L / green R' -> 'red|green' (the counterfactual's naming); None for one pen."""
    m = re.match(r"(red|green) L / (red|green) R", layout)
    return f"{m.group(1)}|{m.group(2)}" if m else None


def layout_words(key: str) -> str:
    """'red|green' -> 'red pen left, green pen right'."""
    a, b = key.split("|")
    return f"{a} pen left, {b} pen right"


# --------------------------------------------------------------------------- per-run data


GRASP_RE = re.compile(
    r"(red|green) ([\d.]+)-([\d.]+) s grip ([\d.]+) tip z ([-\d.]+)->([-\d.]+) cm( [a-z ]+?)?(?:;|$)"
)


def parse_label(lab: dict[str, str]) -> dict[str, object]:
    """Grasps, misses, hover and park times from the provisional label text."""
    notes, tags = lab.get("notes", ""), lab.get("tags", "")
    grasps = [
        {
            "pen": m.group(1),
            "t0": float(m.group(2)),
            "t1": float(m.group(3)),
            "grip": float(m.group(4)),
            "z0": float(m.group(5)),
            "z1": float(m.group(6)),
            "end": (m.group(7) or "").strip(),
        }
        for m in GRASP_RE.finditer(notes)
    ]
    mm = re.search(r"misses at ([\d., ]+?) s", notes)
    misses = [float(x) for x in mm.group(1).split(",")] if mm else []
    nmiss = sum(int(x) for x in re.findall(r"grasp-miss x(\d+)", tags))
    if nmiss != len(misses):
        warn(f"{lab['tag']}: tags say {nmiss} empty closes, notes list {len(misses)} miss times")
    hv = re.search(r"hover/realign (\d+(?:\.\d+)?) s", tags)
    pk = re.search(r"retract/park at ([\d.]+) s", tags)
    jc = re.search(r"jaw centre ([\d.]+)", notes)
    return {
        "grasps": grasps,
        "misses": misses,
        "n_empty": nmiss,
        "hover_s": float(hv.group(1)) if hv else 0.0,
        "park_s": float(pk.group(1)) if pk else None,
        "jaw_centre": float(jc.group(1)) if jc else None,
    }


def loop_stats(run_dir: Path) -> dict[str, object]:
    """Inference time per chunk and tick-interval percentiles of one run."""
    out: dict[str, object] = {}
    cpath, tpath = run_dir / "chunks.npz", run_dir / "ticks.npz"
    if cpath.is_file():
        dur = np.load(cpath)["duration"] * 1e3
        if len(dur):
            p = np.percentile(dur, [5, 25, 50, 75, 95])
            out["inf"] = {
                k: round(float(v), 1) for k, v in zip(("p5", "p25", "p50", "p75", "p95"), p, strict=True)
            }
            out["inf"]["max"] = round(float(dur.max()), 1)  # type: ignore[index]
            out["inf"]["n"] = len(dur)  # type: ignore[index]
    if tpath.is_file():
        z = np.load(tpath)
        rows = np.flatnonzero(z["phase"] == PHASE_POLICY)
        t = z["t"]
        if "episode" in z.files and len(rows):
            cuts = np.flatnonzero(np.diff(z["episode"][rows]) != 0) + 1
            segs = [s for s in np.split(rows, cuts) if len(s) > 1]
        else:
            segs = [rows] if len(rows) > 1 else []
        dts = np.concatenate([np.diff(t[s]) for s in segs]) * 1e3 if segs else np.zeros(0)
        if len(dts):
            out["tick"] = {
                "p50": round(float(np.median(dts)), 1),
                "p95": round(float(np.percentile(dts, 95)), 1),
                "max": round(float(dts.max()), 1),
            }
    return out


def load_runs() -> tuple[list[dict[str, object]], list[dict[str, str]], dict[str, object]]:
    """Analysed runs (ticks present) joined with labels, plan consistency and loop stats."""
    table = read_csv(RES / "runs.csv")
    labels = {r["run_dir"]: r for r in read_csv(RES / "labels_provisional.csv")}
    plan = {r["tag"]: r for r in read_csv(RES / "plan_consistency_summary.csv")}
    empty = [r for r in table if not fnum(r["n_ticks"])]
    runs: list[dict[str, object]] = []
    setup: dict[str, object] = {"cameras": set(), "policy": set(), "fps": set(), "duration_cfg": set()}
    for r in table:
        if not fnum(r["n_ticks"]):
            continue
        rd = RUNS_DIR / r["run_dir"]
        meta = json.loads((rd / "meta.json").read_text()) if (rd / "meta.json").is_file() else {}
        cams = (meta.get("robot") or {}).get("cameras") or {}
        setup["cameras"].add(len(cams))  # type: ignore[union-attr]
        setup["policy"].add(Path(str(meta.get("policy_path", ""))).name)  # type: ignore[union-attr]
        setup["fps"].add(meta.get("fps"))  # type: ignore[union-attr]
        setup["duration_cfg"].add(meta.get("duration_cfg"))  # type: ignore[union-attr]
        lab = labels.get(r["run_dir"])
        if lab is None:
            warn(f"{r['run_dir']}: no provisional label")
            lab = {"tag": r["tag"], "layout": "", "approached": "", "contact": "", "outcome": ""}
        pc = plan.get(r["tag"])
        if pc is None:
            warn(f"{r['tag']}: no plan-consistency row")
            pc = {}
        parsed = parse_label(lab)
        lay = layout_key(lab.get("layout", ""))
        run = {
            "run_dir": r["run_dir"],
            "tag": r["tag"],
            "short": short_tag(r["tag"]),
            "time": r["start_time"][11:16],
            "start": r["start_time"],
            "task": r["task"],
            "kind": prompt_kind(r["task"]),
            "cap": fnum(r["max_relative_target"]),
            "display": r["display_data"],
            "inference_type": r["inference_type"],
            "start_pose": r["start_pose"],
            "dur": fnum(r["duration_s"]),
            "hz": fnum(r["effective_hz"]),
            "inf_med": fnum(r["inference_median_ms"]),
            "inf_max": fnum(r["inference_max_ms"]),
            "n_chunks": fnum(r["n_chunks"]),
            "clip": fnum(r["pct_ticks_clipped"]),
            "closes_policy": fnum(r["gripper_close_events_policy"]),
            "closes_measured": fnum(r["gripper_close_events_measured"]),
            "tip_min_cm": None if fnum(r["tip_min_z_m"]) is None else fnum(r["tip_min_z_m"]) * 100,
            "tip_dist_m": fnum(r["tip_distance_m"]),
            "layout": lab.get("layout", ""),
            "layout_key": lay,
            "pens": 2 if lay else 1,
            "approached": lab.get("approached", ""),
            "contact": lab.get("contact", ""),
            "outcome": lab.get("outcome", ""),
            "tags": lab.get("tags", ""),
            "confidence": lab.get("confidence", ""),
            **parsed,
            "guided": fnum(pc.get("guided_deg")),
            "unguided": fnum(pc.get("unguided_deg")),
            "tip_unguided_mm": fnum(pc.get("tip_unguided_mm")),
            "seam_wroll": fnum(pc.get("seam_wroll_deg")),
            "n_hover": fnum(pc.get("n_hover")),
            "n_other": fnum(pc.get("n_other")),
            "ung_hover": fnum(pc.get("unguided_deg_hover")),
            "ung_other": fnum(pc.get("unguided_deg_other")),
            "tip_hover": fnum(pc.get("tip_unguided_mm_hover")),
            "tip_other": fnum(pc.get("tip_unguided_mm_other")),
            "clip_hover": fnum(pc.get("clip_pct_hover")),
            "flip_wroll": fnum(pc.get("flip_wroll_pct")),
            **loop_stats(rd),
        }
        if pc and pc.get("task") and pc["task"] != r["task"]:
            warn(f"{r['tag']}: plan-consistency task {pc['task']!r} differs from runs.csv {r['task']!r}")
        runs.append(run)
    runs.sort(key=lambda x: str(x["start"]))
    setup = {k: sorted(v, key=str) for k, v in setup.items()}  # type: ignore[union-attr]
    return runs, empty, setup


# --------------------------------------------------------------------------- offline counterfactual


def t_interval(x: list[float]) -> tuple[float, float, float] | None:
    """Mean and 95% t-interval over per-run values."""
    if len(x) < 2:
        return None
    a = np.asarray(x, float)
    m = float(a.mean())
    h = float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return m, m - h, m + h


def load_counterfactual(runs_by_tag: dict[str, dict[str, object]]) -> dict[str, object]:
    """Per-run and aggregate colour effects, neutral drift, gating and on-arm percentiles."""
    d = json.loads((CF / "results.json").read_text())
    frames = {f["id"]: f for f in d["frames"]}
    per_run: dict[str, dict[str, object]] = {}
    for pf in d["per_frame"]:
        fr = frames[pf["id"]]
        tag = fr["tag"]
        run = per_run.setdefault(
            tag,
            {
                "tag": tag,
                "short": short_tag(tag),
                "run_dir": fr["run"],
                "layout": pf["layout"],
                "task": fr["run_task"],
                "frames": [],
            },
        )
        ce = pf.get("colour_effect") or {}
        eff = ce.get("toward_red_pan_deg")
        ci = ce.get("toward_red_pan_ci95") or [None, None]
        run["frames"].append(  # type: ignore[union-attr]
            {
                "offset": fr["offset_s"],
                "eff": eff,
                "lo": ci[0],
                "hi": ci[1],
                "n_moving": ce.get("n_moving"),
            }
        )
    uncommitted = max(
        (f["offset_s"] for f in d["frames"] if f["offset_s"] <= 1.0 + 1e-9), default=1.0
    )  # frames <= +1 s, as SUMMARY.md
    for run in per_run.values():
        fr_ok = [f for f in run["frames"] if f["eff"] is not None]  # type: ignore[union-attr]
        early = [f["eff"] for f in fr_ok if f["offset"] <= uncommitted]
        run["mean_early"] = float(np.mean(early)) if early else None
        first = [f["eff"] for f in fr_ok if f["offset"] == 0]
        run["first"] = first[0] if first else None
        lab = runs_by_tag.get(str(run["tag"]))
        if lab is not None:
            lk = lab["layout_key"]
            if lk and lk != run["layout"]:
                warn(f"{run['short']}: label layout {lk} vs counterfactual frame layout {run['layout']}")
    order = sorted(per_run.values(), key=lambda r: (r["layout"] != "red|green", str(r["run_dir"])))

    cfol = d["colour_following"]

    def agg(key: str, sel: list[dict[str, object]], val: str) -> dict[str, object]:
        c = cfol[key]
        ti = t_interval([float(r[val]) for r in sel if r[val] is not None])  # type: ignore[arg-type]
        return {
            "mean": c["mean"],
            "lo": c["ci95"][0],
            "hi": c["ci95"][1],
            "n_runs": c["n_runs"],
            "n_frames": c["n_frames"],
            "pos": c["frames_positive"],
            "t_lo": ti[1] if ti else None,
            "t_hi": ti[2] if ti else None,
            "t_mean": ti[0] if ti else None,
        }

    red_left = [r for r in order if r["layout"] == "red|green"]
    red_right = [r for r in order if r["layout"] == "green|red"]
    aggs = {
        "early": agg("uncommitted_le_1s", order, "mean_early"),
        "first": agg("first_frame_only", order, "first"),
        "red_left": agg("red_left_layout", red_left, "mean_early"),
        "red_right": agg("red_right_layout", red_right, "mean_early"),
        "all_frames": {
            "mean": cfol["all_frames"]["mean"],
            "lo": cfol["all_frames"]["ci95"][0],
            "hi": cfol["all_frames"]["ci95"][1],
        },
    }
    for k in ("early", "first", "red_left", "red_right"):
        a = aggs[k]
        if a["t_mean"] is not None and abs(float(a["t_mean"]) - float(a["mean"])) > 0.05:
            warn(f"colour effect {k}: mean of per-run means {a['t_mean']:.2f} vs results.json {a['mean']}")
    pos = d["position_bias"]
    neutral = {
        p: {
            "right": pos[p]["image_right_pan_uncommitted"]["mean"],
            "right_lo": pos[p]["image_right_pan_uncommitted"]["ci95"][0],
            "right_hi": pos[p]["image_right_pan_uncommitted"]["ci95"][1],
            "red": pos[p]["toward_red_pan_uncommitted"]["mean"],
            "red_lo": pos[p]["toward_red_pan_uncommitted"]["ci95"][0],
            "red_hi": pos[p]["toward_red_pan_uncommitted"]["ci95"][1],
        }
        for p in ("any", "pen", "null")
        if p in pos
    }
    by_layout = {
        lay: {p: {"mean": v["mean"], "lo": v["ci95"][0], "hi": v["ci95"][1]} for p, v in vals.items()}
        for lay, vals in d["image_right_pan_by_layout"].items()
    }
    gating = {
        w: {p: d["gating"][w][p]["fraction_moving"] for p in d["prompts"]}
        for w in ("first_frame_only", "uncommitted_le_1s", "all_frames")
        if w in d["gating"]
    }
    on_arm = []
    for run_dir, v in d["on_arm_vs_offline_first_frame"].items():
        tag = per_run_by_dir(per_run, run_dir)
        pct = v["offline"][v["prompt"]]["on_arm_percentile"]
        chunks = d["on_arm_chunks"].get(run_dir, {}).get("chunks", [])
        on_arm.append(
            {
                "short": short_tag(tag) if tag else run_dir,
                "prompt": v["prompt"],
                "pct": pct,
                "toward_named": [c.get("toward_named_mm") for c in chunks],
                "toward_red": [c.get("toward_red_mm") for c in chunks],
            }
        )
    on_arm.sort(key=lambda r: r["short"])
    pcts = [r["pct"] for r in on_arm]
    cal = d["image_pan_calibration"]
    return {
        "question": d["question"],
        "prompts": d["prompts"],
        "seeds": d["seeds"],
        "move_threshold_deg": d["move_threshold_deg"],
        "n_frames": d["n_frames"],
        "offsets": sorted({f["offset_s"] for f in d["frames"]}),
        "uncommitted_s": uncommitted,
        "verdict": d["verdict"],
        "runs": order,
        "agg": aggs,
        "neutral": neutral,
        "by_layout": by_layout,
        "gating": gating,
        "on_arm": on_arm,
        "on_arm_median": statistics.median(pcts) if pcts else None,
        "px_per_deg": cal["px_per_deg"]["shoulder_pan"],
        "px_per_deg_ci": cal["pan_px_per_deg_ci95"],
        "toward_named_fraction": d.get("toward_named_pen_fraction"),
        "timing": d.get("timing"),
    }


def per_run_by_dir(per_run: dict[str, dict[str, object]], run_dir: str) -> str | None:
    """Tag of the counterfactual run recorded in ``run_dir``."""
    for tag, r in per_run.items():
        if r["run_dir"] == run_dir:
            return tag
    return None


def contact_sheet(cf_runs: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    """One downscaled JPEG (data URI) with the first policy frame of each two-pen run, 4 per row."""
    frames = json.loads((CF / "frames.json").read_text())
    first = {f["tag"]: f for f in frames if f["offset_s"] == 0}
    tiles = []
    for r in cf_runs:
        f = first.get(str(r["tag"]))
        if f is None:
            continue
        path = RUNS_DIR / Path(f["run_dir"]).name / f["file"]
        if path.is_file():
            tiles.append((r, f, path))
        else:
            warn(f"first frame missing: {path}")
    if not tiles:
        return "", []
    cols = 4
    nrows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TILE_W, nrows * TILE_H), (40, 44, 48))
    meta = []
    for i, (r, f, path) in enumerate(tiles):
        im = Image.open(path).convert("RGB")
        w0, h0 = im.size
        im = im.resize((TILE_W, TILE_H), Image.Resampling.LANCZOS)
        dr = ImageDraw.Draw(im)
        sx, sy = TILE_W / w0, TILE_H / h0
        jaw = None
        lay = f.get("layout") or {}
        for pen in ("red", "green"):
            p = lay.get(pen) or {}
            if not p.get("visible"):
                continue
            x, y = p["x"] * sx, p["y"] * sy
            rr = 6
            if pen == "red":
                dr.ellipse((x - rr, y - rr, x + rr, y + rr), fill="white", outline="black", width=2)
            else:
                dr.polygon(
                    [(x, y - rr - 1), (x - rr, y + rr - 1), (x + rr, y + rr - 1)],
                    fill="white",
                    outline="black",
                    width=2,
                )
        jaw_vals = [run_jaw for run_jaw in [r.get("jaw_centre")] if run_jaw]
        jaw = jaw_vals[0] if jaw_vals else None
        if jaw:
            xj = jaw * TILE_W
            for y in range(0, TILE_H, 8):
                dr.line((xj, y, xj, y + 4), fill="white", width=1)
        sheet.paste(im, ((i % cols) * TILE_W, (i // cols) * TILE_H))
        meta.append({"col": i % cols, "row": i // cols})
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=68, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), meta


# --------------------------------------------------------------------------- page pieces


SVG_SHAPES = {
    "red": '<circle cx="7" cy="7" r="4.6"/>',
    "green": '<path d="M7 2.2 L12 11.2 L2 11.2 Z"/>',
    "none": '<path d="M2.5 7 H11.5" class="ln"/>',
}


def pen_mark(pen: str, hollow: bool = False) -> str:
    """Inline marker: red pen = circle, green pen = triangle, none = dash."""
    pen = pen if pen in SVG_SHAPES else "none"
    cls = f"mk c-{pen}" + (" hol" if hollow else "")
    return (
        f'<svg class="pm" viewBox="0 0 14 14" width="14" height="14" aria-hidden="true">'
        f'<g class="{cls}">{SVG_SHAPES[pen]}</g></svg>'
    )


def outcome_badge(outcome: str) -> str:
    """Semantic outcome pill."""
    o = outcome or "unlabelled"
    cls = {"compliant": "ok", "violating": "bad"}.get(o, "null")
    return f'<span class="badge b-{cls}">{html.escape(o)}</span>'


def esc(v: object) -> str:
    """HTML-escape any value."""
    return html.escape("" if v is None else str(v))


def two_pen_matrix(runs: list[dict[str, object]]) -> tuple[str, list[str]]:
    """Prompt x layout matrix of the two-pen runs, plus a list of cells with fewer than 2 runs."""
    two = [r for r in runs if r["pens"] == 2]
    tasks: list[str] = []
    for kind in ("red", "green", "pen", "any", "null"):
        for r in two:
            if r["kind"] == kind and r["task"] not in tasks:
                tasks.append(str(r["task"]))
    layouts = sorted({str(r["layout"]) for r in two}, key=lambda s: (not s.startswith("red"), s))
    head = "".join(
        f'<th scope="col">{pen_mark(lay.split()[0])} {esc(lay.split()[0])} left<br>'
        f'<span class="sub">{esc(lay)}</span></th>'
        for lay in layouts
    )
    rows_html = []
    gaps = []
    for task in tasks:
        cells = []
        for lay in layouts:
            sel = [r for r in two if r["task"] == task and r["layout"] == lay]
            kind = prompt_kind(task)
            if kind in ("red", "green") and len(sel) < 2:
                gaps.append(f"“{task}” with {layout_words(layout_key(lay) or lay)}: {len(sel)} run(s)")
            if not sel:
                cells.append('<td class="empty"><span class="sub">not run</span></td>')
                continue
            chips = []
            for r in sel:
                appr = str(r["approached"])
                went = "no approach" if appr in ("", "none") else f"went to {appr}"
                unsure = ' <span class="tag-unsure">unsure</span>' if r["confidence"] == "unsure" else ""
                chips.append(
                    f'<div class="chip">'
                    f'<div class="chip-top">{pen_mark(appr)}<span class="mono">{esc(r["short"])}</span>'
                    f"{outcome_badge(str(r['outcome']))}{unsure}</div>"
                    f'<div class="chip-txt">{esc(went)}; contact: {esc(r["contact"])}</div>'
                    f"</div>"
                )
            cells.append(f"<td>{''.join(chips)}</td>")
        rows_html.append(f'<tr><th scope="row">“{esc(task)}”</th>{"".join(cells)}</tr>')
    table = (
        '<div class="scroll"><table class="matrix"><thead><tr><th scope="col">Prompt</th>'
        f"{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )
    return table, gaps


def run_table(runs: list[dict[str, object]]) -> str:
    """Full per-run table; sortable by the page script."""
    cols = [
        ("time", "Start", "s"),
        ("short", "Run", "s"),
        ("task", "Prompt", "s"),
        ("layout", "Pens (wrist view)", "s"),
        ("cap", "Cap °", "n"),
        ("outcome", "Outcome", "s"),
        ("approached", "Went to", "s"),
        ("n_grasps", "Grasps", "n"),
        ("n_empty", "Empty closes", "n"),
        ("hover_s", "Hover s", "n"),
        ("clip", "Clip %", "n"),
        ("hz", "Rate Hz", "n"),
        ("inf_med", "Inference median ms", "n"),
        ("unguided", "Unguided °", "n"),
        ("tip_min_cm", "Tip min z cm", "n"),
        ("dur", "Policy s", "n"),
        ("confidence", "Label confidence", "s"),
    ]
    head = "".join(
        f'<th scope="col" data-type="{t}"><button type="button" class="sort" data-col="{i}">'
        f"{esc(lbl)}</button></th>"
        for i, (_, lbl, t) in enumerate(cols)
    )
    body = []
    for r in runs:
        tds = []
        for key, _, typ in cols:
            v = len(r["grasps"]) if key == "n_grasps" else r.get(key)  # type: ignore[arg-type]
            if key == "outcome":
                shown = outcome_badge(str(v))
            elif key == "approached":
                shown = f"{pen_mark(str(v))} {esc(v)}"
            elif key == "short":
                shown = f'<span class="mono">{esc(v)}</span>'
            elif key == "task":
                shown = esc(v)
            elif typ == "n":
                nd = 0 if key in ("cap", "n_grasps", "n_empty", "hover_s", "inf_med") else 1
                shown = f1(v if v is None else float(v), nd)  # type: ignore[arg-type]
            else:
                shown = esc(v)
            dv = "" if v is None else v
            num = ' class="num"' if typ == "n" else ""
            tds.append(f'<td{num} data-v="{esc(dv)}">{shown}</td>')
        body.append(f"<tr>{''.join(tds)}</tr>")
    return (
        '<div class="scroll"><table class="runs" id="runs-table"><thead><tr>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def bullets(items: list[str]) -> str:
    """A <ul> of markdown-inline items."""
    return "<ul>" + "".join(f"<li>{md_inline(i)}</li>" for i in items if i) + "</ul>"


def para(text: str, cls: str = "") -> str:
    """A <p> of markdown-inline text, or nothing."""
    c = f' class="{cls}"' if cls else ""
    return f"<p{c}>{md_inline(text)}</p>" if text else ""


# --------------------------------------------------------------------------- build


def build() -> str:
    """Assemble the page."""
    analysis = (RES / "ANALYSIS_2026-10-02.md").read_text()
    summary_md = (CF / "SUMMARY.md").read_text()
    runs, empty_runs, setup = load_runs()
    by_tag = {str(r["tag"]): r for r in runs}
    cf = load_counterfactual(by_tag)
    for r in cf["runs"]:  # type: ignore[union-attr]
        lab = by_tag.get(str(r["tag"]))
        r["approached"] = lab["approached"] if lab else ""
        r["outcome"] = lab["outcome"] if lab else ""
        r["jaw_centre"] = lab["jaw_centre"] if lab else None
    sheet_uri, _ = contact_sheet(cf["runs"])  # type: ignore[arg-type]

    n = len(runs)
    dates = sorted({str(r["start"])[:10] for r in runs})
    date_txt = ", ".join(datetime.strptime(dt, "%Y-%m-%d").strftime("%-d %b %Y") for dt in dates)
    caps = defaultdict(int)
    for r in runs:
        caps[r["cap"]] += 1
    cap_txt = ", ".join(
        f"{f1(c, 0)}° in {k} run{'s' if k > 1 else ''}"
        for c, k in sorted(caps.items(), key=lambda kv: -kv[1])
    )
    inf_types = sorted({str(r["inference_type"]).upper() for r in runs})
    cams = setup["cameras"]
    cam_txt = f"{cams[0]} wrist camera" if len(cams) == 1 else f"{'/'.join(map(str, cams))} cameras"
    pose_m = re.search(r"started from the ([^.]+?)\.", analysis)
    pose_txt = pose_m.group(1) if pose_m else "start pose not stated"
    if not pose_m:
        warn("start pose phrase not found in ANALYSIS")
    starts_inside = sum(1 for r in runs if r["start_pose"] == "inside")
    fps = setup["fps"][0] if len(setup["fps"]) == 1 else None
    dur_cfg = setup["duration_cfg"][0] if len(setup["duration_cfg"]) == 1 else None
    policy = ", ".join(map(str, setup["policy"]))
    displays_on = [r for r in runs if r["display"] == "on"]

    # single pen
    single = [r for r in runs if r["pens"] == 1]
    s_grasped = [r for r in single if r["grasps"]]
    s_empty = sum(int(r["n_empty"]) for r in single)
    s_first_grasp = [r["grasps"][0]["t0"] for r in s_grasped]  # type: ignore[index]
    s_colours = defaultdict(int)
    for r in single:
        s_colours[str(r["layout"]).split()[0]] += 1
    all_grips = [g["grip"] for r in runs for g in r["grasps"]]  # type: ignore[union-attr]

    # two pens
    two = [r for r in runs if r["pens"] == 2]
    named = [r for r in two if r["kind"] in ("red", "green")]
    named_hit = [r for r in named if r["approached"] == r["kind"]]
    named_first = [r["grasps"][0]["t0"] for r in named_hit if r["grasps"]]  # type: ignore[index]
    neutral_two = [r for r in two if r["kind"] not in ("red", "green")]
    neutral_by_layout = []
    for lay in sorted({str(r["layout"]) for r in neutral_two}, key=lambda s: not s.startswith("red")):
        sel = [r for r in neutral_two if r["layout"] == lay]
        went = [str(r["approached"]) for r in sel if r["approached"] not in ("", "none")]
        pens = ", ".join(sorted(set(went)))
        neutral_by_layout.append(
            f"{layout_words(layout_key(lay) or lay)}: {len(went)} of {len(sel)} neutral-prompt runs "
            f"approached a pen" + (f" (all to {pens})" if went and len(set(went)) == 1 else "")
        )
    matrix_html, gaps = two_pen_matrix(runs)

    # plan consistency
    paired = [r for r in runs if r["ung_hover"] is not None and r["ung_other"] is not None]
    ratios = [float(r["ung_hover"]) / float(r["ung_other"]) for r in paired if r["ung_other"]]  # type: ignore[arg-type]
    hover_higher = sum(1 for r in paired if r["ung_hover"] > r["ung_other"])  # type: ignore[operator]
    guided = [r["guided"] for r in runs if r["guided"] is not None]
    unguided = [r["unguided"] for r in runs if r["unguided"] is not None]
    tipu = [r["tip_unguided_mm"] for r in runs if r["tip_unguided_mm"] is not None]

    # control loop
    hz = [r["hz"] for r in runs]
    inf_med = [r["inf_med"] for r in runs]
    inf_max = [r["inf_max"] for r in runs]
    tick_p95 = [r["tick"]["p95"] for r in runs if "tick" in r]  # type: ignore[index]
    disp_on_med = [r["inf_med"] for r in displays_on]
    disp_off_med = [r["inf_med"] for r in runs if r["display"] != "on"]
    short_runs = [r for r in runs if dur_cfg and r["dur"] is not None and r["dur"] < 0.9 * float(dur_cfg)]
    table_floor = [r["tip_min_cm"] for r in runs if r["tip_min_cm"] is not None and r["tip_min_cm"] < 0]

    agg = cf["agg"]  # type: ignore[index]
    ea = agg["early"]
    verdict = str(cf["verdict"])

    # consistency checks the page should not hide
    for r in runs:
        meas, pol = r["closes_measured"], r["closes_policy"]
        if meas is not None and pol is not None and meas > pol:
            warn(f"{r['short']}: more measured gripper closes ({meas:g}) than commanded ({pol:g})")
    for r in two:
        if r["kind"] in ("red", "green") and r["approached"] == r["kind"] and not r["grasps"]:
            warn(f"{r['short']}: named pen approached but no grasp parsed")

    # ---- text pulled from the markdown (definitions use ANALYSIS wording)
    labels_status = md_block(analysis, "**Status of the labels:**", what="label status")
    label_bullets = md_bullets_after(analysis, "**Status of the labels:**", what="label status bullets")
    layout_def = md_block(analysis, "Layout is from colour segmentation", what="layout definition")
    hover_def = md_block(analysis, "**Hover/realign:**", what="hover definition")
    park_def = md_block(analysis, "**Retract/park:**", what="park definition")
    method_def = md_block(analysis, "**Method.**", what="plan-consistency method")
    gripper_ev = md_bullets_after(analysis, "**Gripper evidence:**", what="gripper evidence")
    height_ev = md_bullets_after(analysis, "**Height evidence:**", what="height evidence")
    hover_res = md_block(analysis, "**Hovering is indecision", what="hover result")
    rtc_res = md_bullets_after(analysis, "**RTC guidance works only where", what="RTC result")
    display_note = md_block(analysis, "**Display:**", what="display note")
    episode_note = md_block(analysis, "**Episode length:**", what="episode length note")
    cap_note = first_para(md_section(analysis, "3. 4° vs 6° cap"))
    direction_def = md_block(summary_md, "**Direction.**", what="colour-effect definition")
    verdict_rule = md_block(summary_md, "**Pre-registered rule:", what="verdict rule")
    plain_reading = md_block(summary_md, "**In plain language:**", what="plain reading")
    not_robust = md_block(summary_md, "**It is not robust across scenes.**", what="robustness note")
    cf_limits = md_bullets_after(summary_md, "## Limits", what="counterfactual limits")
    preliminary = md_bullets_after(analysis, "- **Preliminary:**", what="preliminary list")
    next_items = numbered_items(md_section(analysis, "6. Next experiments"))
    gating_line = md_block(summary_md, "**(iii) Gating.**", what="gating line")

    # ---- data for the charts
    keep = (
        "short",
        "tag",
        "time",
        "task",
        "kind",
        "cap",
        "display",
        "dur",
        "hz",
        "inf_med",
        "inf_max",
        "clip",
        "layout",
        "layout_key",
        "pens",
        "approached",
        "outcome",
        "confidence",
        "grasps",
        "misses",
        "n_empty",
        "hover_s",
        "park_s",
        "guided",
        "unguided",
        "tip_unguided_mm",
        "seam_wroll",
        "ung_hover",
        "ung_other",
        "tip_hover",
        "tip_other",
        "n_hover",
        "n_other",
        "clip_hover",
        "inf",
        "tick",
    )
    data = {
        "runs": [{k: r.get(k) for k in keep} for r in runs],
        "fps": fps,
        "duration_cfg": dur_cfg,
        "cf": {
            "runs": [
                {k: r.get(k) for k in ("short", "layout", "task", "frames", "approached", "outcome", "first")}
                for r in cf["runs"]  # type: ignore[union-attr]
            ],
            "agg": agg,
            "neutral": cf["neutral"],
            "on_arm": cf["on_arm"],
            "on_arm_median": cf["on_arm_median"],
            "uncommitted_s": cf["uncommitted_s"],
            "seeds": cf["seeds"],
            "prompts": cf["prompts"],
        },
    }
    data_json = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    # ---- static sections
    def tile(value: str, label: str, detail: str, extra: str = "") -> str:
        return (
            f'<div class="kpi{extra}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-detail">{detail}</div></div>'
        )

    t_txt = (
        f"t-interval over per-run means {f1(ea['t_lo'], 1, True)} to {f1(ea['t_hi'], 1, True)}"
        if ea["t_lo"] is not None
        else ""
    )
    kpis = "".join(
        [
            tile(
                f"{rng(hz, 1)}<span class='u'>Hz</span>",
                f"Control rate, all {n} runs",
                f"inference median {rng(inf_med, 0, 'ms')} per chunk, max {rng(inf_max, 0, 'ms')}",
            ),
            tile(
                f"{len(s_grasped)}<span class='u'>/ {len(single)}</span>",
                "Single-pen runs with a grasp",
                f"{s_empty} empty closes across the {len(single)} single-pen runs; grasps began at "
                + ", ".join(f"{f1(t, 1)} s" for t in s_first_grasp),
            ),
            tile(
                f"{len(named_hit)}<span class='u'>/ {len(named)}</span>",
                "Named-colour two-pen runs that went to the named pen",
                f"first grasp at {rng(named_first, 1, 's')}; n = {len(named)}",
            ),
            tile(
                f"{f1(ea['mean'], 1, True)}<span class='u'>°</span>",
                "Offline colour effect, shoulder_pan towards the red pen",
                f"95% CI {f1(ea['lo'], 1, True)} to {f1(ea['hi'], 1, True)} (run bootstrap, approximate); "
                f"{t_txt}. Pre-registered verdict: <strong>{esc(verdict)}</strong>",
            ),
        ]
    )

    key_points = [
        f"The named colour won {len(named_hit)} of {len(named)} named-colour two-pen runs. "
        + "; ".join(neutral_by_layout)
        + ".",
        f"Grasping is the bottleneck. In single-pen runs {len(s_grasped)} of {len(single)} grasped, after "
        f"{s_empty} empty closes in total. A held pen stalls the measured gripper at {rng(all_grips, 1)}.",
        f"RTC guidance matches the old plan on the guided steps to {rng(guided, 2, '°')} RMS, "
        f"but the executed, unguided overlap disagrees by {rng(unguided, 1, '°')} "
        f"({rng(tipu, 0, 'mm')} at the tip). Hover windows disagree more than other windows in "
        f"{hover_higher} of {len(paired)} runs that have both kinds of window (median ratio {f1(statistics.median(ratios), 1)}×).",
        f"Offline, with frame and state fixed, the colour word shifts shoulder_pan by "
        f"{f1(ea['mean'], 1, True)}° towards the named pen on average, positive on {ea['pos']} of "
        f"{ea['n_frames']} early frames. The verdict by the pre-registered rule is "
        f"“{verdict}”.",
    ]

    gaps_txt = "; ".join(gaps) if gaps else "none"
    two_pen_note = (
        f"Each cell lists the runs for that prompt and layout. Marker: {pen_mark('red')} red pen, "
        f"{pen_mark('green')} green pen, {pen_mark('none')} no approach. Badges are provisional machine "
        f"labels. Named-colour cells with fewer than 2 runs: {esc(gaps_txt)}."
    )

    sheet_html = sheet_css = ""
    if sheet_uri:
        figs = []
        for i, r in enumerate(cf["runs"]):  # type: ignore[arg-type]
            col, row = i % 4, i // 4
            pos_x = f"{col * 100 / 3:.4f}%"
            pos_y = "0%" if row == 0 else f"{row * 100 / max(1, (len(cf['runs']) - 1) // 4):.4f}%"  # type: ignore[arg-type]
            eff = r["first"]
            figs.append(
                f'<figure class="frame"><div class="frame-img" role="img" '
                f'aria-label="first policy frame of {esc(r["short"])}" '
                f'style="background-position:{pos_x} {pos_y}"></div>'
                f'<figcaption><span class="mono">{esc(r["short"])}</span><br>'
                f"“{esc(r['task'])}”<br>{pen_mark(str(r['approached']))} "
                f"{esc(r['approached'] or 'none')} {outcome_badge(str(r['outcome']))}<br>"
                f'<span class="sub">offline first-frame effect {f1(eff, 1, True)}°</span>'
                f"</figcaption></figure>"
            )
        rows_n = (len(cf["runs"]) + 3) // 4  # type: ignore[arg-type]
        sheet_css = f".frame-img{{background-image:url({sheet_uri});background-size:400% {rows_n * 100}%}}"
        sheet_html = (
            '<div class="frames">' + "".join(figs) + "</div>"
            '<p class="note">First policy frame of each two-pen run (wrist view), top row red pen on the '
            "left, bottom row green pen on the left. Overlay: white circle = red pen cap centroid, white "
            "triangle = green pen cap centroid, dashed line = jaw centre. Downscaled from the run recordings."
            "</p>"
        )

    gate = cf["gating"]  # type: ignore[index]
    gate_rows = "".join(
        f"<tr><th scope='row'>“{esc(cf['prompts'][p])}”</th>"  # type: ignore[index]
        + "".join(f"<td class='num'>{f1(100 * gate[w][p], 0)}%</td>" for w in gate)
        + "</tr>"
        for p in cf["prompts"]  # type: ignore[union-attr]
    )
    gate_names = {
        "first_frame_only": "first frame",
        "uncommitted_le_1s": f"frames ≤ +{f1(cf['uncommitted_s'], 0)} s",  # type: ignore[arg-type]
        "all_frames": "all frames",
    }
    gate_table = (
        '<div class="scroll"><table class="small"><thead><tr><th scope="col">Prompt</th>'
        + "".join(f"<th scope='col' class='num'>{gate_names[w]}</th>" for w in gate)
        + f"</tr></thead><tbody>{gate_rows}</tbody></table></div>"
    )

    lay_rows = []
    for lay, vals in cf["by_layout"].items():  # type: ignore[union-attr]
        for p in ("red", "green"):
            v = vals.get(p)
            if v:
                lay_rows.append(
                    f"<tr><th scope='row'>{esc(layout_words(lay.removesuffix('_layout').replace('red_left', 'red|green').replace('red_right', 'green|red')))}</th><td>“{esc(cf['prompts'][p])}”"  # type: ignore[index]
                    f"</td><td class='num'>{f1(v['mean'], 1, True)}°</td><td class='num'>"
                    f"{f1(v['lo'], 1, True)} to {f1(v['hi'], 1, True)}</td></tr>"
                )
    lay_table = (
        '<div class="scroll"><table class="small"><thead><tr><th scope="col">Layout</th>'
        '<th scope="col">Prompt</th><th scope="col" class="num">pan to image right</th>'
        '<th scope="col" class="num">95% CI</th></tr></thead><tbody>'
        + "".join(lay_rows)
        + "</tbody></table></div>"
    )

    name_note = ""
    two_short = [str(r["short"]) for r in cf["runs"]]  # type: ignore[union-attr]
    if any("_2_pens_" in s for s in two_short):
        ex = next(s for s in two_short if "_2_pens_" in s)
        name_note = (
            f"Run names here follow ANALYSIS_2026-10-02.md. SUMMARY.md drops “2_pens_”, so its "
            f"{ex.replace('2_pens_', '')} is {ex} here."
        )

    next_html = (
        "<ol class='next'>"
        + "".join(
            f"<li>{md_inline(str(it['head']))}"
            + (bullets(list(it["sub"])) if it["sub"] else "")  # type: ignore[arg-type]
            + "</li>"
            for it in next_items
        )
        + "</ol>"
    )

    limits = [
        "All outcome, contact and hover labels are provisional machine labels. The protocol's blind human "
        "labels have not been made.",
        f"Two-pen colour following rests on n = {len(named)} named-colour runs. Gaps: {gaps_txt}.",
        cap_note,
        episode_note,
        *[f"Offline test: {x}" for x in cf_limits if not x.startswith("Next")],
    ]

    defs = [
        ("Provisional machine labels", labels_status + " " + " ".join(label_bullets)),
        ("Layout", layout_def),
        ("Colour effect", direction_def),
        ("Pre-registered verdict", verdict_rule),
        ("Hover/realign", cap_first(hover_def.removeprefix("**Hover/realign:**"))),
        ("Retract/park", cap_first(park_def.removeprefix("**Retract/park:**"))),
        ("Plan consistency", cap_first(method_def.removeprefix("**Method.**"))),
        ("Empty close and grasp", " ".join(gripper_ev)),
    ]
    defs_html = "".join(f"<div><dt>{esc(k)}</dt><dd>{md_inline(v)}</dd></div>" for k, v in defs if v.strip())

    excluded_txt = ", ".join(f"{r['start_time'][11:19]} ({short_tag(r['tag'])})" for r in empty_runs)
    warn_html = ""
    if WARNINGS:
        warn_html = (
            "<section id='data-notes'><h2>Data notes</h2><p>Checks the generator ran on the source files:</p>"
            + bullets(list(WARNINGS))
            + "</section>"
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    sources = [
        "runs.csv",
        "labels_provisional.csv",
        "plan_consistency_summary.csv",
        "ANALYSIS_2026-10-02.md",
        "prompt_counterfactual/results.json",
        "prompt_counterfactual/SUMMARY.md",
        "prompt_counterfactual/frames.json",
        "runs/*/chunks.npz, ticks.npz, meta.json",
    ]

    body = f"""
<div class="wrap">
<header class="top">
  <p class="eyebrow">MolmoAct2 zero-shot on the SO-101 · {esc(date_txt)}</p>
  <h1>SO-101 Run Results</h1>
  <p class="lede">{n} real-robot runs analysed from {esc(date_txt)}, plus offline analyses on the recorded
  frames. {len(empty_runs)} run folder without ticks is left out ({esc(excluded_txt)}).</p>
  <dl class="setup">
    <div><dt>Checkpoint</dt><dd class="mono">{esc(policy)}</dd></div>
    <div><dt>Camera</dt><dd>{esc(cam_txt)}</dd></div>
    <div><dt>Inference</dt><dd>{esc(", ".join(inf_types))}</dd></div>
    <div><dt>Start pose</dt><dd>{esc(pose_txt)} ({starts_inside} of {n} inside the trained state range)</dd></div>
    <div><dt>Joint cap per tick</dt><dd>{esc(cap_txt)}</dd></div>
    <div><dt>Loop</dt><dd>{f1(fps, 0) if fps else "?"} Hz target, {f1(dur_cfg, 0) if dur_cfg else "?"} s
    episode limit</dd></div>
  </dl>
  <nav class="toc" aria-label="Sections">
    <a href="#two-pen">Two-pen outcomes</a><a href="#grasps">Grasp attempts</a>
    <a href="#offline">Offline colour effect</a><a href="#plan">Plan consistency</a>
    <a href="#loop">Control loop</a><a href="#table">All runs</a><a href="#limits">Limits and next</a>
    <a href="#defs">Definitions</a>
  </nav>
</header>

<aside class="callout" role="note"><strong>Provisional machine labels.</strong> Every outcome, contact,
approach and hover label on this page is a provisional machine label from gripper stalls plus a look at
each contact sheet. These are not the protocol's blind human labels; a person labelling should overrule
them.</aside>

<section id="summary" aria-label="Summary">
  <div class="kpis">{kpis}</div>
  <ul class="points">{"".join(f"<li>{md_inline(p)}</li>" for p in key_points)}</ul>
</section>

<section id="two-pen">
  <h2>Two-pen outcomes</h2>
  <p class="dek">Prompt × layout for the {len(two)} two-pen runs. Layout is left/right in the wrist view
  at the first policy frame. Outcomes are provisional machine labels.</p>
  {matrix_html}
  <p class="note">{two_pen_note}</p>
  {sheet_html}
</section>

<section id="grasps">
  <h2>Grasp attempts over time</h2>
  <p class="dek">Every gripper close in every run, from the provisional machine labels. Single-pen runs:
  {", ".join(f"{k} {c} pen" for c, k in sorted(s_colours.items(), key=lambda kv: -kv[1]))}.</p>
  <div class="legend" id="lg-timeline"></div>
  <div class="chart" id="ch-timeline"></div>
  {bullets(gripper_ev + height_ev[:2])}
</section>

<section id="offline">
  <h2>Offline colour effect</h2>
  <p class="dek">{md_inline(str(cf["question"])[:1].upper() + str(cf["question"])[1:])} {len(cf["runs"])} two-pen scenes,
  {len(cf["prompts"])} prompts × the same {cf["seeds"]} seeds, frames at
  {", ".join(f"+{f1(o, 0)}" for o in cf["offsets"])} s after the first policy observation.</p>
  <div class="verdict"><span class="v-label">Pre-registered verdict</span>
  <span class="v-value">{esc(verdict)}</span><p>{md_inline(verdict_rule)}</p>
  <p>The bootstrap CIs resample only {ea["n_runs"]} runs (4 per layout), so they are approximate and too
  narrow. The t-interval over per-run means is shown beside them.</p></div>

  <h3>Per run</h3>
  <p class="dek">Colour effect = shoulder_pan of the “{esc(cf["prompts"]["red"])}” seeds minus the
  “{esc(cf["prompts"]["green"])}” seeds, movers only, signed so + is towards the red pen.
  Error bars are seed-resampled 95% CIs for one frame; most are narrower than the marker.</p>
  <div class="legend" id="lg-cf-runs"></div>
  <div class="chart" id="ch-cf-runs"></div>
  <p class="note">{md_inline(not_robust)} {esc(name_note)}</p>

  <h3>Aggregates and neutral drift</h3>
  <div class="legend" id="lg-cf-agg"></div>
  <div class="chart" id="ch-cf-agg"></div>
  <p class="note">{md_inline(plain_reading)}</p>
  <div class="two-col">
    <div><h4>Pan to the image right by layout, frames ≤ +{f1(cf["uncommitted_s"], 0)} s</h4>{lay_table}</div>
    <div><h4>Seeds moving (threshold {f1(cf["move_threshold_deg"], 0)}°)</h4>{gate_table}
    <p class="note">{md_inline(gating_line.removeprefix("**(iii) Gating.**"))}</p></div>
  </div>

  <h3>The arm's own first chunk against the offline seeds</h3>
  <div class="chart" id="ch-onarm"></div>
  <p class="note">Percentile of the chunk the arm actually ran, among the {cf["seeds"]} offline seeds for the
  same prompt and frame. Median {f1(cf["on_arm_median"], 0)}th. Near the middle means offline sampling on
  the saved frames reproduces the arm.</p>
</section>

<section id="plan">
  <h2>Plan consistency (RTC)</h2>
  <p class="dek">{md_inline(method_def.removeprefix("**Method.**"))}</p>
  <h3>Hover windows against other moving windows</h3>
  <div class="legend" id="lg-plan-hover"></div>
  <div class="chart" id="ch-plan-hover"></div>
  <p class="note">{md_inline(hover_res)} Higher in {hover_higher} of {len(paired)} runs that have both kinds
  of window; median ratio {f1(statistics.median(ratios), 2)}×. Hover windows are from the provisional
  machine labeller.</p>
  <h3>Guided against unguided steps, every run</h3>
  <div class="legend" id="lg-plan-gu"></div>
  <div class="chart" id="ch-plan-gu"></div>
  {bullets(rtc_res)}
</section>

<section id="loop">
  <h2>Control-loop health</h2>
  <div class="two-col wide-left">
    <div><h3>Inference time per chunk</h3><div class="legend" id="lg-lat"></div>
    <div class="chart" id="ch-lat"></div></div>
    <div><h3>Effective control rate</h3><div class="chart" id="ch-rate"></div>
    <p class="note">Tick interval p95 {rng(tick_p95, 1, "ms")} across runs.
    {md_inline(cap_first(display_note.removeprefix("**Display:**")))}</p>
    <p class="note">Display on: median {rng(disp_on_med, 0, "ms")} ({len(displays_on)} runs); off:
    {rng(disp_off_med, 0, "ms")}. {len(short_runs)} runs ended before the {f1(dur_cfg, 0)} s limit.
    Computed tip minimum below the base origin in {len(table_floor)} runs ({rng(table_floor, 1, "cm")}).</p>
    </div>
  </div>
</section>

<section id="table">
  <h2>All runs</h2>
  <p class="dek">Click a column to sort. Outcome, went to, grasps, empty closes, hover and confidence are
  provisional machine labels. Unguided ° is the run-median RMS disagreement of the unguided steps.</p>
  {run_table(runs)}
</section>

<section id="limits">
  <h2>Limits and next experiments</h2>
  <div class="two-col">
    <div><h3>Limits</h3>{bullets(limits)}<h4>Preliminary, from the analysis</h4>{bullets(preliminary)}</div>
    <div><h3>Next, in priority order</h3>{next_html}</div>
  </div>
</section>

<section id="defs">
  <h2>Definitions</h2>
  <p class="dek">Wording from ANALYSIS_2026-10-02.md and prompt_counterfactual/SUMMARY.md.</p>
  <dl class="defs">{defs_html}</dl>
</section>
{warn_html}
<footer class="foot">Generated {esc(generated)} by <span class="mono">steering/results_page.py</span> from
<span class="mono">steering/results/</span>: {", ".join(f'<span class="mono">{esc(s)}</span>' for s in sources)}.
</footer>
</div>
<div id="tip" class="tip" hidden></div>
<script type="application/json" id="data">{data_json}</script>
"""
    return HEAD.replace("/*SHEET*/", sheet_css) + body + "<script>" + SCRIPT + "</script>\n"


HEAD = """<title>SO-101 Run Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: one left-aligned reading column (max 1080px), summary first, then sections ruled off;
   charts measure their container and redraw, wide tables scroll inside their own box. */
:root{
  --bg:#eff2f3; --surface:#fafbfb; --surface-2:#e4e9eb; --rule:#cfd7db; --band:#e8edef;
  --fg:#121a1f; --fg-2:#3f4d56; --fg-3:#5a6871;
  --accent:#7a4f00; --accent-soft:#f1e3c3;
  --ok:#0a6aa1; --ok-soft:#d9e9f4; --bad:#a3407a; --bad-soft:#f3dfea; --null:#6f7a81; --null-soft:#e1e6e8;
  --pen-red:#c4501b; --pen-green:#007d63; --ink:#1d2a32; --muted:#9aa5ab;
  --f-display:"Archivo","Arial Narrow",Arial,sans-serif;
  --f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#11161a; --surface:#171e23; --surface-2:#1f282e; --rule:#2c3740; --band:#1a2227;
  --fg:#e3e9ec; --fg-2:#b0bcc3; --fg-3:#8e9ba4;
  --accent:#e2b155; --accent-soft:#3a2f17;
  --ok:#5cb3e6; --ok-soft:#163246; --bad:#df8dbd; --bad-soft:#3d2333; --null:#97a2a9; --null-soft:#252e34;
  --pen-red:#f07a43; --pen-green:#33c39a; --ink:#dfe6ea; --muted:#5d6a73;
  color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#11161a; --surface:#171e23; --surface-2:#1f282e; --rule:#2c3740; --band:#1a2227;
  --fg:#e3e9ec; --fg-2:#b0bcc3; --fg-3:#8e9ba4;
  --accent:#e2b155; --accent-soft:#3a2f17;
  --ok:#5cb3e6; --ok-soft:#163246; --bad:#df8dbd; --bad-soft:#3d2333; --null:#97a2a9; --null-soft:#252e34;
  --pen-red:#f07a43; --pen-green:#33c39a; --ink:#dfe6ea; --muted:#5d6a73;
  color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.55 var(--f-body);padding-inline:16px;padding-block:28px 64px;margin:0}
.wrap{max-width:1080px;margin-inline:auto;display:flex;flex-direction:column;gap:40px}
h1,h2,h3,h4{text-wrap:balance;margin:0;color:var(--fg)}
h1{font-family:var(--f-display);font-stretch:75%;font-weight:750;font-size:clamp(34px,6vw,52px);line-height:1;letter-spacing:-.01em}
h2{font-family:var(--f-display);font-stretch:85%;font-weight:700;font-size:26px;line-height:1.15;padding-top:14px;border-top:2px solid var(--fg)}
h3{font-size:16px;font-weight:600;margin-top:26px}
h4{font-size:13px;font-weight:600;color:var(--fg-2);margin-bottom:6px}
p{margin:0}
section{display:flex;flex-direction:column;gap:12px;min-width:0}
code,.mono{font-family:var(--f-mono);font-size:.88em}
a{color:var(--fg);text-decoration-color:var(--accent);text-underline-offset:3px}
a:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.eyebrow,.kpi-label,dt,.v-label{font-family:var(--f-mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--fg-3)}
.eyebrow{color:var(--accent)}
.top{display:flex;flex-direction:column;gap:14px}
.lede{max-width:68ch;color:var(--fg-2);font-size:16px}
.setup{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr));gap:10px 24px;margin:6px 0 0;padding:14px 0;border-block:1px solid var(--rule)}
.setup div{min-width:0}.setup dd{margin:2px 0 0;font-size:14px}
.toc{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13.5px}
.callout{background:var(--accent-soft);color:var(--fg);padding:12px 16px;border-left:3px solid var(--accent);max-width:80ch;font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:20px 28px}
.kpi{border-top:1px solid var(--rule);padding-top:10px;display:flex;flex-direction:column;gap:4px;min-width:0}
.kpi-value{font-family:var(--f-display);font-stretch:75%;font-weight:700;font-size:44px;line-height:1.05;font-variant-numeric:tabular-nums;color:var(--fg)}
.kpi-value .u{font-size:22px;font-weight:500;color:var(--fg-2);margin-left:4px}
.kpi-detail{font-size:13px;color:var(--fg-2)}
.points{margin:4px 0 0;padding-left:18px;display:flex;flex-direction:column;gap:6px;max-width:82ch}
.dek{color:var(--fg-2);max-width:78ch;font-size:14.5px}
.note{color:var(--fg-3);font-size:13px;max-width:82ch}
section ul{margin:0;padding-left:18px;display:flex;flex-direction:column;gap:4px;max-width:82ch;font-size:14px}
.scroll{overflow-x:auto;max-width:100%;border:1px solid var(--rule);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;vertical-align:top;padding:7px 10px;border-bottom:1px solid var(--rule)}
thead th{background:var(--surface-2);font-weight:600;font-size:12px;color:var(--fg-2);white-space:nowrap}
.num{text-align:right;font-family:var(--f-mono);font-size:12px}
table.small{font-size:12.5px}
.sub{color:var(--fg-3);font-weight:400;font-size:11.5px}
.matrix{min-width:640px}
.matrix tbody th{font-weight:500;width:22%}
.matrix td{width:39%}
.matrix td.empty{background:var(--band)}
.chip{display:flex;flex-direction:column;gap:3px;padding:6px 0}
.chip+.chip{border-top:1px dashed var(--rule)}
.chip-top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.chip-txt{color:var(--fg-2);font-size:12.5px}
.badge{display:inline-block;font-family:var(--f-mono);font-size:11px;padding:1px 7px;border-radius:2px;border:1px solid}
.b-ok{background:var(--ok);border-color:var(--ok);color:var(--surface)}
.b-bad{background:var(--bad);border-color:var(--bad);color:var(--surface)}
.b-null{background:transparent;border-color:var(--null);color:var(--null)}
.tag-unsure{font-family:var(--f-mono);font-size:11px;color:var(--fg-3);border-bottom:1px dotted var(--fg-3)}
.pm{vertical-align:-2px;flex:none}
.mk{fill:var(--c);stroke:var(--c);stroke-width:1.4}
.mk.hol{fill:var(--surface)}
.mk .ln,.mk.ln{fill:none;stroke-width:2}
.c-red{--c:var(--pen-red)}.c-green{--c:var(--pen-green)}.c-none{--c:var(--fg-3)}
.c-ink{--c:var(--ink)}.c-muted{--c:var(--muted)}.c-ok{--c:var(--ok)}.c-null{--c:var(--null)}.c-accent{--c:var(--accent)}
.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
@media (max-width:640px){.frames{grid-template-columns:repeat(2,minmax(0,1fr))}}
.frame{margin:0;min-width:0}
.frame-img{width:100%;max-width:100%;aspect-ratio:4/3;background-repeat:no-repeat;background-color:var(--surface-2)}
.frame figcaption{font-size:12px;color:var(--fg-2);margin-top:4px;line-height:1.45;overflow-wrap:anywhere}
.verdict{display:grid;grid-template-columns:auto 1fr;gap:4px 16px;align-items:baseline;padding:12px 16px;border:1px solid var(--accent);background:var(--surface);max-width:86ch}
.verdict p{grid-column:1/-1;font-size:13.5px;color:var(--fg-2)}
.v-value{font-family:var(--f-display);font-stretch:80%;font-weight:700;font-size:26px;color:var(--fg)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12.5px;color:var(--fg-2)}
.legend span{display:inline-flex;align-items:center;gap:5px}
.chart{width:100%;min-width:0}
.chart svg{display:block;max-width:100%}
.chart text{font-family:var(--f-mono);font-size:11px}
.ax{fill:var(--fg-3)}.lab{fill:var(--fg-2)}.grp{fill:var(--fg);font-weight:500}.axt{fill:var(--fg-2)}
.grid{stroke:var(--rule);stroke-width:1}.zero{stroke:var(--fg-2);stroke-width:1.2}
.ref{stroke:var(--fg-3);stroke-width:1;stroke-dasharray:4 3}
.band{fill:var(--band)}
.trk{fill:var(--surface-2);stroke:var(--rule)}
.lnk{stroke:var(--muted);stroke-width:2}
.rng{stroke:var(--c);fill:none}
.box{fill:var(--surface-2);stroke:var(--c);stroke-width:1.2}
.box.on{fill:var(--accent-soft)}
.bar{fill:var(--c);opacity:.55}
.two-col{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:20px 32px}
.two-col>div{min-width:0;display:flex;flex-direction:column;gap:10px}
.two-col.wide-left{grid-template-columns:minmax(0,3fr) minmax(0,2fr)}
@media (max-width:760px){.two-col.wide-left{grid-template-columns:minmax(0,1fr)}}
table.runs{min-width:1280px}
table.runs td{white-space:nowrap}
table.runs th{padding:0}
.sort{all:unset;cursor:pointer;display:block;padding:7px 10px;white-space:nowrap}
.sort:hover{color:var(--fg)}
th[aria-sort="ascending"] .sort::after{content:" \\2191";color:var(--accent)}
th[aria-sort="descending"] .sort::after{content:" \\2193";color:var(--accent)}
table.runs tbody tr:hover{background:var(--band)}
.next{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:8px;font-size:14px}
.next ul{margin-top:4px;font-size:13px;color:var(--fg-2)}
.defs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:14px 28px;margin:0}
.defs div{min-width:0}.defs dd{margin:3px 0 0;font-size:13.5px;color:var(--fg-2)}
.foot{font-size:12px;color:var(--fg-3);border-top:1px solid var(--rule);padding-top:12px}
.tip{position:fixed;z-index:10;pointer-events:none;max-width:280px;background:var(--fg);color:var(--bg);font:12px/1.4 var(--f-mono);padding:6px 8px}
@media (prefers-reduced-motion:no-preference){.sort,a{transition:color .15s}}
/*SHEET*/
</style>
"""

SCRIPT = r"""
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
const MINUS = '−';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (v, nd=1, signed=false) => v==null ? '' : ((signed && v>0 ? '+' : '') + v.toFixed(nd)).replace('-', MINUS);
function niceTicks(a, b, n){
  const raw = (b-a)/Math.max(1,n), mag = Math.pow(10, Math.floor(Math.log10(raw))), e = raw/mag;
  const step = (e>=7.5?10:e>=3.5?5:e>=1.5?2:1)*mag, out=[];
  for(let v=Math.ceil(a/step-1e-9)*step; v<=b+1e-9; v+=step) out.push(+v.toFixed(10));
  return out;
}
function axisLines(txt, maxW){
  const n = Math.floor(maxW/6.7); if(txt.length <= n) return [txt];
  let cut = txt.lastIndexOf(' ', n); if(cut < 1) cut = n;
  return [txt.slice(0, cut), clip(txt.slice(cut).trim(), n)];
}
const clip = (t, n) => t.length > n ? t.slice(0, Math.max(1, n-1)) + '\u2026' : t;
function tickTxt(v){ const s = Math.abs(v) < 1e-9 ? '0' : String(+v.toFixed(6)); return s.replace('-', MINUS); }
function shapeSvg(sh, x, y, r, cls, hollow, tip){
  const c = `mk ${cls}${hollow?' hol':''}`, t = tip ? ` data-tip="${esc(tip)}"` : '';
  if(sh==='tri') return `<path class="${c}"${t} d="M${x} ${y-r*1.15}L${x+r*1.05} ${y+r*0.8}L${x-r*1.05} ${y+r*0.8}Z"/>`;
  if(sh==='sq') return `<rect class="${c}"${t} x="${x-r*0.85}" y="${y-r*0.85}" width="${r*1.7}" height="${r*1.7}"/>`;
  if(sh==='dia') return `<path class="${c}"${t} d="M${x} ${y-r*1.15}L${x+r*1.15} ${y}L${x} ${y+r*1.15}L${x-r*1.15} ${y}Z"/>`;
  if(sh==='x') return `<path class="${c} ln"${t} d="M${x-r*0.8} ${y-r*0.8}L${x+r*0.8} ${y+r*0.8}M${x+r*0.8} ${y-r*0.8}L${x-r*0.8} ${y+r*0.8}"/>`+
    (t?`<rect x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}" fill="transparent"${t}/>`:'');
  return `<circle class="${c}"${t} cx="${x}" cy="${y}" r="${r}"/>`;
}
const penShape = p => p==='red' ? 'circle' : p==='green' ? 'tri' : 'dash';
const penCls = p => p==='red' ? 'c-red' : p==='green' ? 'c-green' : 'c-none';
function legend(id, items){
  const el = document.getElementById(id); if(!el) return;
  el.innerHTML = items.map(it => {
    let g;
    if(it.kind==='range') g = `<line x1="1" x2="17" y1="7" y2="7" class="rng ${it.cls}" style="stroke-width:${it.w||2}${it.dash?';stroke-dasharray:3 2':''}"/>`;
    else if(it.kind==='box') g = `<rect x="2" y="2" width="14" height="10" class="box ${it.cls} ${it.on?'on':''}"/>`;
    else if(it.kind==='bar') g = `<rect x="1" y="4" width="16" height="6" class="mk bar ${it.cls}"/>`;
    else if(it.kind==='track') g = `<rect x="1" y="5" width="16" height="4" class="trk"/>`;
    else if(it.kind==='dash') g = `<path d="M3 7H15" class="mk ln ${it.cls}"/>`;
    else g = shapeSvg(it.shape||'circle', 9, 7, it.r||4.2, it.cls||'c-ink', it.hollow);
    return `<span><svg width="18" height="14" viewBox="0 0 18 14" aria-hidden="true">${g}</svg>${esc(it.label)}</span>`;
  }).join('');
}
/* rows: [{label, group?:bool, marks:[...]}]; marks: pt{x,shape,cls,hollow,r,tip}, rng{lo,hi,cls,w,dash,dy,tip},
   box{p5,p25,p50,p75,p95,max,cls,on,tip}, bar{lo,hi,cls,h,tip}, track{lo,hi}, link{lo,hi} */
function rowChart(host, spec){
  const W = Math.max(300, Math.floor(host.clientWidth));
  const lw = Math.min(spec.labelW||150, Math.round(W*0.34));
  const x0 = lw + 12, x1 = W - 14, rowH = spec.rowH||22, gH = 26;
  let y = 6; const rows = [];
  spec.rows.forEach(r => { if(r.group){ rows.push(Object.assign({}, r, {y: y+gH-9})); y += gH; } else { rows.push(Object.assign({}, r, {y: y+rowH/2})); y += rowH; } });
  const yb = y + 4, axl = axisLines(spec.axis, W - 8), H = yb + 40 + (axl.length-1)*14;
  const [a, b] = spec.domain, sx = v => x0 + (v-a)/(b-a)*(x1-x0);
  const ticks = spec.ticks || niceTicks(a, b, Math.max(3, Math.floor((x1-x0)/70)));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(spec.aria||'')}">`;
  let band = 0;
  rows.forEach(r => { if(r.group){ band = 0; return; } if(band++ % 2 === 1) s += `<rect class="band" x="${x0}" y="${r.y-rowH/2}" width="${x1-x0}" height="${rowH}"/>`; });
  ticks.forEach(t => { const X = sx(t); s += `<line class="grid" x1="${X}" x2="${X}" y1="4" y2="${yb}"/><text class="ax" x="${X}" y="${yb+15}" text-anchor="middle">${tickTxt(t)}</text>`; });
  if(spec.zero && a < 0 && b > 0) s += `<line class="zero" x1="${sx(0)}" x2="${sx(0)}" y1="2" y2="${yb}"/>`;
  (spec.refs||[]).forEach(rf => { s += `<line class="ref" x1="${sx(rf.x)}" x2="${sx(rf.x)}" y1="2" y2="${yb}"/>`; });
  s += `<line class="grid" x1="${x0}" x2="${x1}" y1="${yb}" y2="${yb}"/>`;
  axl.forEach((t, i) => { s += `<text class="axt" x="${W/2}" y="${yb+33+i*14}" text-anchor="middle">${esc(t)}</text>`; });
  rows.forEach(r => {
    if(r.group){ s += `<text class="grp" x="0" y="${r.y}"><title>${esc(r.label)}</title>${esc(clip(r.label, Math.floor(W/6.8)))}</text>`; return; }
    s += `<text class="lab" x="${lw}" y="${r.y+4}" text-anchor="end"><title>${esc(r.label)}</title>${esc(clip(r.label, Math.floor(lw/6.8)))}</text>`;
    const back = [], front = [];
    (r.marks||[]).forEach(m => {
      const yy = r.y + (m.dy||0), tip = m.tip ? ` data-tip="${esc(m.tip)}"` : '';
      if(m.t==='track') back.push(`<rect class="trk" x="${sx(m.lo)}" y="${yy-2.5}" width="${Math.max(0,sx(m.hi)-sx(m.lo))}" height="5"${tip}/>`);
      else if(m.t==='link') back.push(`<line class="lnk" x1="${sx(m.lo)}" x2="${sx(m.hi)}" y1="${yy}" y2="${yy}"/>`);
      else if(m.t==='rng') back.push(`<line class="rng ${m.cls||'c-ink'}" x1="${sx(m.lo)}" x2="${sx(m.hi)}" y1="${yy}" y2="${yy}" style="stroke-width:${m.w||1.6}${m.dash?';stroke-dasharray:3 2':''}"${tip}/>`);
      else if(m.t==='bar'){ const h = m.h||8; back.push(`<rect class="mk bar ${m.cls}" x="${sx(m.lo)}" y="${yy-h/2}" width="${Math.max(1.5,sx(m.hi)-sx(m.lo))}" height="${h}"${tip}/>`); }
      else if(m.t==='box'){ const h = 10, c = m.cls||'c-ink';
        back.push(`<line class="rng ${c}" x1="${sx(m.p5)}" x2="${sx(m.p95)}" y1="${yy}" y2="${yy}" style="stroke-width:1.2"/>`);
        back.push(`<rect class="box ${c} ${m.on?'on':''}" x="${sx(m.p25)}" y="${yy-h/2}" width="${Math.max(1,sx(m.p75)-sx(m.p25))}" height="${h}"${tip}/>`);
        back.push(`<line class="rng ${c}" x1="${sx(m.p50)}" x2="${sx(m.p50)}" y1="${yy-h/2}" y2="${yy+h/2}" style="stroke-width:2.2"/>`);
        if(m.max!=null) front.push(shapeSvg('x', sx(m.max), yy, 3.6, c, false, m.tip));
      }
      else if(m.t==='pt'){ if(m.shape==='dash') front.push(`<path class="mk ln ${m.cls}" d="M${sx(m.x)-4} ${yy}H${sx(m.x)+4}"${tip}/>`);
        else front.push(shapeSvg(m.shape||'circle', sx(m.x), yy, m.r||4.4, m.cls||'c-ink', m.hollow, m.tip)); }
    });
    s += back.join('') + front.join('');
  });
  host.innerHTML = s + '</svg>';
}
function dotStack(host, spec){
  const W = Math.max(260, Math.floor(host.clientWidth)), x0 = 16, x1 = W - 16, r = 5;
  const groups = {}; spec.values.forEach(v => { const k = v.x.toFixed(spec.nd||1); (groups[k] = groups[k]||[]).push(v); });
  const maxN = Math.max(...Object.values(groups).map(g => g.length));
  const yb = 14 + maxN*(2*r+2), axl = axisLines(spec.axis, W - 8), H = yb + 42 + (axl.length-1)*14;
  const [a, b] = spec.domain, sx = v => x0 + (v-a)/(b-a)*(x1-x0);
  const ticks = niceTicks(a, b, Math.max(3, Math.floor((x1-x0)/60)));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(spec.aria||'')}">`;
  ticks.forEach(t => { const X = sx(t); s += `<line class="grid" x1="${X}" x2="${X}" y1="4" y2="${yb}"/><text class="ax" x="${X}" y="${yb+15}" text-anchor="middle">${tickTxt(t)}</text>`; });
  (spec.refs||[]).forEach(rf => { const X = sx(rf.x); s += `<line class="ref" x1="${X}" x2="${X}" y1="2" y2="${yb}"/><text class="ax" x="${X-4}" y="10" text-anchor="end">${esc(rf.label)}</text>`; });
  s += `<line class="grid" x1="${x0}" x2="${x1}" y1="${yb}" y2="${yb}"/>`;
  Object.entries(groups).forEach(([k, g]) => g.forEach((v, i) => { s += shapeSvg('circle', sx(+k), yb - r - 2 - i*(2*r+2), r, v.cls||'c-ink', v.hollow, v.tip); }));
  axl.forEach((t, i) => { s += `<text class="axt" x="${W/2}" y="${yb+33+i*14}" text-anchor="middle">${esc(t)}</text>`; });
  host.innerHTML = s + '</svg>';
}
const pad = (lo, hi, f=0.06) => { const d = (hi-lo)||1; return [lo - d*f, hi + d*f]; };
const runs = D.runs;
const groupsByPens = () => {
  const out = [];
  [[1,'Single pen'],[2,'Two pens']].forEach(([n, name]) => { const sel = runs.filter(r => r.pens===n); if(sel.length){ out.push({group:true, label:`${name} (${sel.length} runs)`}); sel.forEach(r => out.push(r)); } });
  return out;
};
const charts = [];
/* 1. timeline of grasp attempts */
charts.push(['ch-timeline', host => {
  const maxT = Math.max(D.duration_cfg||0, ...runs.map(r => r.dur||0));
  const rows = groupsByPens().map(r => r.group ? r : {label: r.short, marks: [
    {t:'track', lo:0, hi:r.dur, tip:`${r.short}: policy phase ${fmt(r.dur)} s, "${r.task}"`},
    ...r.grasps.map(g => ({t:'bar', lo:g.t0, hi:g.t1, cls:penCls(g.pen), h:9, tip:`${r.short}: ${g.pen} pen held ${fmt(g.t0)}–${fmt(g.t1)} s, grip ${fmt(g.grip)}, tip z ${fmt(g.z0)} to ${fmt(g.z1)} cm${g.end?', '+g.end:''}`})),
    ...r.grasps.map(g => ({t:'pt', x:g.t0, shape:penShape(g.pen), cls:penCls(g.pen), r:4.6})),
    ...r.misses.map(t => ({t:'pt', x:t, shape:'x', cls:'c-ink', r:4, tip:`${r.short}: empty close at ${fmt(t)} s`})),
    ...(r.park_s!=null ? [{t:'pt', x:r.park_s, shape:'dia', cls:'c-null', hollow:true, r:4.4, tip:`${r.short}: retract/park at ${fmt(r.park_s)} s`}] : []),
  ]});
  rowChart(host, {rows, domain:[0, Math.ceil(maxT)], axis:'time since the policy started (s)', labelW:150, rowH:20,
    aria:'Timeline of gripper closes per run'});
}]);
legend('lg-timeline', [
  {kind:'track', label:'policy phase'}, {kind:'bar', cls:'c-red', label:'red pen held'}, {kind:'bar', cls:'c-green', label:'green pen held'},
  {shape:'circle', cls:'c-red', label:'grasp start, red'}, {shape:'tri', cls:'c-green', label:'grasp start, green'},
  {shape:'x', cls:'c-ink', label:'empty close'}, {shape:'dia', cls:'c-null', hollow:true, label:'retract/park'}]);
/* 2. offline per-run colour effect */
const cfr = D.cf.runs, early = D.cf.uncommitted_s;
charts.push(['ch-cf-runs', host => {
  const vals = []; cfr.forEach(r => r.frames.forEach(f => { if(f.eff!=null){ vals.push(f.eff, f.lo, f.hi); } }));
  const lim = Math.ceil(Math.max(...vals.map(Math.abs)) / 5) * 5;
  const rows = [];
  ['red|green','green|red'].forEach(lay => {
    const sel = cfr.filter(r => r.layout===lay); if(!sel.length) return;
    const [l, rr] = lay.split('|');
    rows.push({group:true, label:`${l} pen left, ${rr} pen right (${sel.length} runs)`});
    sel.forEach(r => rows.push({label: r.short, marks: r.frames.filter(f => f.eff!=null).flatMap(f => {
      const tip = `${r.short} +${fmt(f.offset,0)} s: ${fmt(f.eff,1,true)}° (CI ${fmt(f.lo,1,true)} to ${fmt(f.hi,1,true)}); arm prompt "${r.task}"`;
      if(f.offset > early) return [{t:'pt', x:f.eff, shape:'circle', cls:'c-muted', r:3, tip}];
      return [{t:'rng', lo:f.lo, hi:f.hi, cls:'c-ink', w:2}, {t:'pt', x:f.eff, shape:'circle', cls:'c-ink', hollow:f.offset>0, r:4.6, tip}];
    })}));
  });
  rowChart(host, {rows, domain:[-lim, lim], zero:true, labelW:150,
    axis:'colour effect (degrees of shoulder_pan, + towards the red pen)', aria:'Per-run colour effect'});
}]);
legend('lg-cf-runs', [{shape:'circle', cls:'c-ink', label:'+0 s frame'}, {shape:'circle', cls:'c-ink', hollow:true, label:`+${fmt(early,0)} s frame`},
  {shape:'circle', cls:'c-muted', r:3, label:'later frames (arm committing)'}, {kind:'range', cls:'c-ink', label:'95% CI over seeds'}]);
/* 3. aggregates */
charts.push(['ch-cf-agg', host => {
  const A = D.cf.agg, N = D.cf.neutral, P = D.cf.prompts;
  const aggRow = (label, a) => ({label, marks:[
    {t:'rng', lo:a.lo, hi:a.hi, cls:'c-ink', w:3, dy:-3, tip:`${label}: ${fmt(a.mean,1,true)}°, bootstrap CI ${fmt(a.lo,1,true)} to ${fmt(a.hi,1,true)}; ${a.pos} of ${a.n_frames} frames positive`},
    ...(a.t_lo!=null ? [{t:'rng', lo:a.t_lo, hi:a.t_hi, cls:'c-ink', w:1.4, dash:true, dy:4, tip:`${label}: t-interval ${fmt(a.t_lo,1,true)} to ${fmt(a.t_hi,1,true)}`}] : []),
    {t:'pt', x:a.mean, shape:'circle', cls:'c-ink', r:4.6, dy:-3}]});
  const rows = [
    {group:true, label:'Colour effect (red minus green prompt)'},
    aggRow(`all runs, ≤ +${fmt(early,0)} s`, A.early), aggRow('first frame only', A.first),
    aggRow('red pen left', A.red_left), aggRow('red pen right', A.red_right),
    {group:true, label:'Neutral prompts, pan to the image right'},
    ...Object.entries(N).map(([p, v]) => ({label: P[p] ? `"${P[p]}"` : '""', marks:[
      {t:'rng', lo:v.right_lo, hi:v.right_hi, cls:'c-ink', w:3, tip:`"${P[p]}": ${fmt(v.right,1,true)}° to the image right (CI ${fmt(v.right_lo,1,true)} to ${fmt(v.right_hi,1,true)}); towards red ${fmt(v.red,1,true)}°`},
      {t:'pt', x:v.right, shape:'sq', cls:'c-ink', r:4.4}]})),
  ];
  const vals = []; [A.early, A.first, A.red_left, A.red_right].forEach(a => vals.push(a.lo, a.hi, a.t_lo ?? 0, a.t_hi ?? 0));
  Object.values(N).forEach(v => vals.push(v.right_lo, v.right_hi));
  const lim = Math.ceil(Math.max(...vals.map(Math.abs)) / 5) * 5;
  rowChart(host, {rows, domain:[-lim, lim], zero:true, labelW:215, rowH:24,
    axis:'degrees of shoulder_pan', aria:'Aggregate colour effect and neutral drift'});
}]);
legend('lg-cf-agg', [{shape:'circle', cls:'c-ink', label:'mean'}, {kind:'range', cls:'c-ink', w:3, label:'95% CI, runs then seeds resampled'},
  {kind:'range', cls:'c-ink', w:1.4, dash:true, label:'95% t-interval over per-run means'}, {shape:'sq', cls:'c-ink', label:'neutral prompt mean'}]);
/* 4. on-arm percentile */
charts.push(['ch-onarm', host => {
  const rows = D.cf.on_arm.map(r => ({label: r.short, marks:[{t:'pt', x:r.pct, shape:'circle', cls:'c-ink', r:4.6,
    tip:`${r.short}: ${fmt(r.pct,0)}th percentile, prompt "${D.cf.prompts[r.prompt]}"`}]}));
  rowChart(host, {rows, domain:[0,100], ticks:[0,25,50,75,100], refs: D.cf.on_arm_median!=null ? [{x:D.cf.on_arm_median}] : [], labelW:150, rowH:20,
    axis:`percentile among ${D.cf.seeds} offline seeds (%), dashed line = median`, aria:'On-arm chunk percentile'});
}]);
/* 5. plan consistency, hover vs other */
charts.push(['ch-plan-hover', host => {
  const sel = runs.filter(r => r.ung_hover!=null && r.ung_other!=null);
  const hi = Math.max(...sel.map(r => Math.max(r.ung_hover, r.ung_other)));
  const rows = sel.map(r => ({label: r.short, marks:[
    {t:'link', lo:Math.min(r.ung_hover, r.ung_other), hi:Math.max(r.ung_hover, r.ung_other)},
    {t:'pt', x:r.ung_other, shape:'circle', cls:'c-ink', hollow:true, r:4.6, tip:`${r.short} other windows (n=${r.n_other}): ${fmt(r.ung_other)}°, tip ${fmt(r.tip_other)} mm`},
    {t:'pt', x:r.ung_hover, shape:'circle', cls:'c-ink', r:4.6, tip:`${r.short} hover windows (n=${r.n_hover}): ${fmt(r.ung_hover)}°, tip ${fmt(r.tip_hover)} mm, ${fmt(r.clip_hover,0)}% ticks clipped; ratio ${fmt(r.ung_hover/r.ung_other,2)}×`}]}));
  rowChart(host, {rows, domain:[0, Math.ceil(hi+0.5)], labelW:150, rowH:22,
    axis:'unguided disagreement, run median RMS (degrees)', aria:'Hover vs other disagreement'});
}]);
legend('lg-plan-hover', [{shape:'circle', cls:'c-ink', hollow:true, label:'other moving windows'}, {shape:'circle', cls:'c-ink', label:'hover windows'}]);
/* 6. guided vs unguided */
charts.push(['ch-plan-gu', host => {
  const hi = Math.max(...runs.map(r => r.unguided||0));
  const rows = groupsByPens().map(r => r.group ? r : {label: r.short, marks: r.unguided==null ? [] : [
    {t:'link', lo:r.guided, hi:r.unguided},
    {t:'pt', x:r.guided, shape:'sq', cls:'c-ink', hollow:true, r:4, tip:`${r.short} guided: ${fmt(r.guided,2)}°`},
    {t:'pt', x:r.unguided, shape:'circle', cls:'c-ink', r:4.4, tip:`${r.short} unguided: ${fmt(r.unguided)}°, tip ${fmt(r.tip_unguided_mm)} mm; wrist_roll seam ${fmt(r.seam_wroll)}°; ${fmt(r.clip)}% ticks clipped`}]});
  rowChart(host, {rows, domain:[0, Math.ceil(hi+0.5)], labelW:150, rowH:20,
    axis:'disagreement with the previous chunk, run median RMS (degrees)', aria:'Guided vs unguided disagreement'});
}]);
legend('lg-plan-gu', [{shape:'sq', cls:'c-ink', hollow:true, label:'guided steps'}, {shape:'circle', cls:'c-ink', label:'unguided, executed steps'}]);
/* 7. latency */
charts.push(['ch-lat', host => {
  const hi = Math.max(...runs.map(r => r.inf ? r.inf.max : (r.inf_max||0)));
  const rows = runs.map(r => ({label: r.short, marks: r.inf ? [{t:'box', ...r.inf, cls:'c-ink', on:r.display==='on',
    tip:`${r.short}: median ${fmt(r.inf.p50,0)} ms, IQR ${fmt(r.inf.p25,0)}–${fmt(r.inf.p75,0)}, p95 ${fmt(r.inf.p95,0)}, max ${fmt(r.inf.max,0)} ms, ${r.inf.n} chunks; display ${r.display}`}] : []}));
  rowChart(host, {rows, domain:[0, Math.ceil(hi/100)*100], labelW:150, rowH:20,
    axis:'inference time per chunk (ms)', aria:'Inference latency per run'});
}]);
legend('lg-lat', [{kind:'box', cls:'c-ink', label:'p25–p75, median line, p5–p95 whisker'}, {shape:'x', cls:'c-ink', label:'max'},
  {kind:'box', cls:'c-ink', on:true, label:'live display on'}]);
/* 8. rate */
charts.push(['ch-rate', host => {
  const xs = runs.map(r => r.hz).filter(v => v!=null), lo = Math.min(...xs), hi = Math.max(...xs, D.fps||0);
  dotStack(host, {values: runs.filter(r => r.hz!=null).map(r => ({x:r.hz, cls:'c-ink', hollow:r.display==='on', tip:`${r.short}: ${fmt(r.hz)} Hz, tick p95 ${r.tick?fmt(r.tick.p95):''} ms`})),
    domain:[Math.floor((lo-0.3)*10)/10, Math.ceil((hi+0.1)*10)/10], refs: D.fps ? [{x:D.fps, label:`target ${fmt(D.fps,0)} Hz`}] : [],
    axis:'effective control rate per run (Hz); hollow = display on', aria:'Control rate per run'});
}]);
function drawAll(){ charts.forEach(([id, fn]) => { const el = document.getElementById(id); if(el) fn(el); }); }
drawAll();
let rt; const ro = new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(drawAll, 80); });
document.querySelectorAll('.chart').forEach(el => ro.observe(el));
/* tooltip */
const tip = document.getElementById('tip');
function showTip(e){ const t = e.target.closest('[data-tip]'); if(!t){ tip.hidden = true; return; }
  tip.textContent = t.getAttribute('data-tip'); tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 12, y = e.clientY + 14;
  if(x + w > window.innerWidth - 8) x = e.clientX - w - 12;
  if(y + h > window.innerHeight - 8) y = e.clientY - h - 12;
  tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, y) + 'px'; }
document.addEventListener('pointermove', showTip); document.addEventListener('pointerdown', showTip);
document.addEventListener('scroll', () => { tip.hidden = true; }, {passive:true});
/* sortable table */
const table = document.getElementById('runs-table');
if(table){
  const ths = table.querySelectorAll('thead th');
  table.querySelectorAll('button.sort').forEach(btn => btn.addEventListener('click', () => {
    const i = +btn.dataset.col, th = ths[i], num = th.dataset.type === 'n';
    const dir = th.getAttribute('aria-sort') === 'ascending' ? -1 : 1;
    ths.forEach(t => t.removeAttribute('aria-sort')); th.setAttribute('aria-sort', dir > 0 ? 'ascending' : 'descending');
    const tb = table.tBodies[0], rows = Array.from(tb.rows);
    rows.sort((ra, rb) => { const a = ra.cells[i].dataset.v, b = rb.cells[i].dataset.v;
      if(num){ const fa = a === '' ? Infinity : +a, fb = b === '' ? Infinity : +b; return (fa - fb) * dir; }
      return a.localeCompare(b) * dir; });
    rows.forEach(r => tb.appendChild(r));
  }));
}
})();
"""


def main() -> None:
    """Write the page and report its size."""
    page = build()
    OUT.write_text(page)
    kb = len(page.encode()) / 1024
    print(f"wrote {OUT} ({kb:.0f} KB)")
    if kb > 200:
        print("warning: page is over 200 KB", file=sys.stderr)
    for w in WARNINGS:
        print(f"  note: {w}")


if __name__ == "__main__":
    main()
