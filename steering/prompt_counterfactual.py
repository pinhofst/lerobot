"""Offline counterfactual-prompt test on real two-pen recordings: does the colour word move the target?

    flock /tmp/claude-gpu.lock uv run python steering/prompt_counterfactual.py      # every two-pen run
    uv run python steering/prompt_counterfactual.py --frames-only                    # CPU: frames, layout, contact sheet
    uv run python steering/prompt_counterfactual.py --analyse-only                   # CPU: readouts from saved chunks
    flock /tmp/claude-gpu.lock uv run python steering/prompt_counterfactual.py RUN_DIR [RUN_DIR ...] \
        --prompt red="pick up the red pen" --prompt null="" --seeds 64 --offsets 0 1 2

On the arm, "pick up the red pen" went for the red pen and "pick up the green pen" for the green one,
but the named pen also sat on the left of the wrist view in those runs. Here the observation (wrist
frame + joint state) is held fixed and only the words change, with the same noise seeds for every
prompt, so differences between prompts are paired.

Steps:
1. Frames. For each run (``ticks.npz``/``frames/``/``chunks.npz``/``meta.json`` from rollout.py,
   runs with 0 ticks skipped), take the recorded wrist frame nearest to each ``--offsets`` time after
   the first policy chunk's observation (``chunks.t_start[0]``), keeping it only if both pens are
   segmented (else the nearest frame within 0.25 s that has both). The state is the first
   policy-phase tick at or after the frame (rollout.py records the Present_Position of the latest
   observation on every tick; before the first tick nothing has been sent, so the first tick's state
   is the state at the first frame). Layout: red / green hue masks (HSV), the centroid of the
   largest connected component of each (in practice the cap), compared left/right.
   ``contact_sheet.jpg`` shows every chosen frame with its masks and centroids for a human check.
2. Image <-> pan sign. From consecutive recorded frames (0.1 s apart) with both pens visible, the
   pens' mean horizontal image shift is regressed on the joint changes. Objects move left in the
   image when the camera turns right, so ``u = -sign(px per deg of pan)`` is +1 when +pan turns the
   wrist view to the right. Every pan readout is then expressed as "image-right pan" r = u * dpan.
3. Model. The 1-camera checkpoint is loaded once (``fast_load``), the RTC config is set as
   ``lerobot-rollout --inference.type=rtc`` does, and chunks are sampled through the same code as
   ``MolmoAct2Policy._generate_actions_from_inputs_with_rtc`` without a previous chunk (the first
   chunk of every run; later chunks on the arm also had RTC guidance, which offline is absent). The
   VLM runs once per (frame, prompt); its KV cache is expanded over a batch of seeds for the 10 flow
   steps, and the noise for seed s is ``torch.randn(1, H, D, generator=Generator('cuda').manual_seed(s))``,
   exactly what a stock batch-1 call with that generator draws. ``--validate`` compares against the
   stock ``predict_action_chunk`` at batch 1 (bit-exact expected) and with the seed batch.
   Chunks go through the stock postprocessor (clamp, unnormalise, arm frame).
4. Readouts per frame, on the postprocessed (arm-frame) chunks, d = chunk[-1] - chunk[0]:
   movers (frame_diagnostic.mover_stats: |d| > --move-threshold on any of joints 0-3), mover mean
   displacement, red - green pan difference among movers with a bootstrap CI
   (frame_diagnostic.mover_direction), and the direction-to-target readout: the colour effect
   ``(r_red - r_green) * side_red`` (deg of pan towards the red pen's side, red prompt relative to
   the green prompt; side_red = +1 when the red pen is right of the green one). Per prompt: mean r
   among movers, the fraction of movers whose r points towards the named pen, and the tip motion by
   forward kinematics (so101_fk): lateral (towards image right, mm), height and reach.
5. Aggregates: (i) colour-following, the mean colour effect over frames; (ii) position bias, r under
   the neutral and empty prompts by layout; (iii) gating, movers per prompt. CIs: hierarchical
   bootstrap (runs with replacement, then seeds within each frame), because frames of one run share
   the scene.

Outputs in --out (default steering/results/prompt_counterfactual/): frames.json, contact_sheet.jpg,
chunks.npz (raw normalised and arm-frame chunks, (frames, prompts, seeds, T, 6), and states),
results.json, direction.png, gating.png. ``--analyse-only`` redoes 4-5 and the plots from chunks.npz.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import zlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from frame_diagnostic import JOINTS, is_moving, mover_direction, mover_stats  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
from so101_fk import SO101FK  # noqa: E402

STEERING = Path(__file__).parent
RUNS = STEERING / "results" / "runs"
OUT = STEERING / "results" / "prompt_counterfactual"
CHECKPOINT = STEERING / "checkpoints" / "MolmoAct2-SO100_101-LeRobot-1cam"
IMAGE_KEY = "observation.images.cam0"
# Exact task strings of the recorded runs (meta.json "task"), plus the empty prompt.
DEFAULT_PROMPTS = {
    "red": "pick up the red pen",
    "green": "pick up the green pen",
    "any": "pick up a pen of any color",
    "pen": "pick up the pen",
    "null": "",
}
COLOURS = ("red", "green")
NEUTRAL = ("any", "pen", "null")
PHASE_POLICY = 1
SEED = 0  # base seed of every bootstrap generator
MIN_MOVERS = 5  # same floor as frame_diagnostic.mover_direction

# HSV thresholds (hue in degrees, s/v in 0..1), tuned on these recordings: the red cap/body and the
# green cap/body; the green "WOWROBO" letters on the mat are yellow-green (hue < 125) and excluded.
RED_HUE, GREEN_HUE = ((335.0, 15.0), (125.0, 185.0))
RED_SV, GREEN_SV = ((0.45, 0.30), (0.40, 0.20))
MIN_AREA = 400  # px at 640x480: smaller largest components count as "not visible"

# dataviz reference palette; red and green are only 7.2 CVD dE apart, so every series also has its
# own marker and a direct label.
STYLE = {
    "red": {"color": "#e34948", "marker": "o", "label": "red pen"},
    "green": {"color": "#008300", "marker": "^", "label": "green pen"},
    "pen": {"color": "#2a78d6", "marker": "s", "label": "the pen"},
    "any": {"color": "#4a3aa7", "marker": "D", "label": "any colour"},
    "null": {"color": "#898781", "marker": "X", "label": 'empty ""'},
}
C_INK, C_MUTED, C_GRID, C_ACCENT = "#0b0b0b", "#898781", "#e8e7e3", "#2a78d6"


def rng_for(key: str) -> np.random.Generator:
    """Bootstrap generator for one named readout, independent of what else is computed."""
    return np.random.default_rng([SEED, zlib.crc32(key.encode())])


def r2(x: Any) -> Any:
    """Round floats (and lists of floats) for JSON; None passes through."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r2(v) for v in x]
    return round(float(x), 3)


# ---------------------------------------------------------------------------
# Runs and frames
# ---------------------------------------------------------------------------


def find_runs(pattern: str = "2_pens") -> list[Path]:
    """Run dirs whose tag contains ``pattern`` and that recorded at least one tick."""
    out = []
    for d in sorted(RUNS.glob(f"*{pattern}*")):
        if (d / "ticks.npz").is_file() and len(np.load(d / "ticks.npz")["t"]):
            out.append(d)
    return out


def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    """ticks, chunks and meta of a run (the keys plot_run.load_run reads)."""
    ticks = dict(np.load(run_dir / "ticks.npz"))
    chunks = dict(np.load(run_dir / "chunks.npz")) if (run_dir / "chunks.npz").is_file() else {}
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").is_file() else {}
    return ticks, chunks, meta


_T_RE = re.compile(r"^\d+_(?P<t>[\d.]+)s_")


def frame_list(run_dir: Path, meta: dict) -> list[dict]:
    """meta.json's frame list (file, t), else parsed from the file names; sorted by time."""
    frames = meta.get("frames")
    if not frames:
        frames = []
        for p in sorted((run_dir / "frames").glob("*.jpg")):
            m = _T_RE.match(p.name)
            if m:
                frames.append({"file": f"frames/{p.name}", "t": float(m["t"])})
    return sorted(frames, key=lambda f: f["t"])


def policy_start(ticks: dict, chunks: dict) -> float:
    """Time of the first policy chunk's observation (its inference start), else the first policy tick."""
    if len(chunks.get("t_start", [])):
        starts = chunks["t_start"]
        if "phase" in chunks:
            starts = (
                starts[chunks["phase"] == PHASE_POLICY] if (chunks["phase"] == PHASE_POLICY).any() else starts
            )
        return float(starts[0])
    return float(ticks["t"][ticks["phase"] == PHASE_POLICY][0])


def state_at(ticks: dict, t: float) -> tuple[np.ndarray, float]:
    """State of the first policy tick at or after t (else the last one), and that tick's time."""
    pol = np.flatnonzero(ticks["phase"] == PHASE_POLICY)
    times = ticks["t"][pol]
    k = int(np.searchsorted(times, t - 1e-6))
    k = min(k, len(pol) - 1)
    return ticks["state"][pol[k]].astype(np.float32), float(times[k])


def load_image(run_dir: Path, frame: dict) -> np.ndarray:
    """RGB uint8 (H, W, 3) of a recorded frame."""
    return np.asarray(Image.open(run_dir / frame["file"]).convert("RGB"))


def pen_masks(img: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean red / green hue masks (after a small morphological opening)."""
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.float32)
    h, s, v = hsv[..., 0] * 360.0 / 255.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    red = ((h >= RED_HUE[0]) | (h < RED_HUE[1])) & (s > RED_SV[0]) & (v > RED_SV[1])
    green = (h >= GREEN_HUE[0]) & (h < GREEN_HUE[1]) & (s > GREEN_SV[0]) & (v > GREEN_SV[1])
    return {k: ndimage.binary_opening(m, iterations=2) for k, m in (("red", red), ("green", green))}


def pen_layout(img: np.ndarray) -> dict:
    """Largest component per colour: centroid (px), area, visible; and which pen is on the right."""
    out: dict[str, Any] = {"width": int(img.shape[1]), "height": int(img.shape[0])}
    for name, mask in pen_masks(img).items():
        lab, n = ndimage.label(mask)
        if n == 0:
            out[name] = {"x": None, "y": None, "area": 0, "visible": False}
            continue
        areas = ndimage.sum_labels(mask, lab, index=np.arange(1, n + 1))
        k = int(np.argmax(areas)) + 1
        ys, xs = np.nonzero(lab == k)
        out[name] = {
            "x": float(xs.mean()),
            "y": float(ys.mean()),
            "area": int(areas[k - 1]),
            "visible": bool(areas[k - 1] >= MIN_AREA),
        }
    both = out["red"]["visible"] and out["green"]["visible"]
    out["both_visible"] = bool(both)
    # +1: red pen right of the green pen in the wrist image; -1: left.
    out["red_side"] = (1 if out["red"]["x"] > out["green"]["x"] else -1) if both else 0
    out["layout"] = ("green|red" if out["red_side"] > 0 else "red|green") if both else "?"
    return out


def select_frames(run_dirs: list[Path], offsets: list[float], window: float = 0.25) -> list[dict]:
    """Frames at the requested offsets after the first policy observation, both pens visible."""
    selected = []
    for run_dir in run_dirs:
        ticks, chunks, meta = load_run(run_dir)
        t0 = policy_start(ticks, chunks)
        frames = frame_list(run_dir, meta)
        times = np.asarray([f["t"] for f in frames])
        used: set[str] = set()
        for off in offsets:
            target = t0 + off
            order = np.argsort(np.abs(times - target))
            pick = None
            for i in order:
                if abs(times[i] - target) > window:
                    break
                layout = pen_layout(load_image(run_dir, frames[i]))
                if layout["both_visible"]:
                    pick = (i, layout)
                    break
            if pick is None:
                print(f"  {run_dir.name}: no frame with both pens within {window} s of +{off} s; skipped")
                continue
            i, layout = pick
            if frames[i]["file"] in used:
                continue
            used.add(frames[i]["file"])
            state, t_tick = state_at(ticks, float(times[i]))
            start_state, _ = state_at(ticks, t0)
            selected.append(
                {
                    "id": f"{run_dir.name.split('_', 1)[0]}+{off:g}s",
                    "run": run_dir.name,
                    "run_dir": str(run_dir),
                    "tag": meta.get("tag", run_dir.name),
                    "run_task": meta.get("task"),
                    "offset_s": off,
                    "t": float(times[i]),
                    "t_from_start": round(float(times[i]) - t0, 3),
                    "file": frames[i]["file"],
                    "tick_t": t_tick,
                    "state": [float(v) for v in state],
                    "pan_moved_since_start_deg": round(float(state[0] - start_state[0]), 2),
                    "layout": layout,
                }
            )
    return selected


def contact_sheet(frames: list[dict], out: Path, cols: int = 4) -> None:
    """Every chosen frame with its red/green masks tinted and the centroids marked."""
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.05 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, f in zip(axes.flat, frames, strict=False):
        img = load_image(Path(f["run_dir"]), f)
        masks = pen_masks(img)
        vis = img.astype(np.float32)
        for name, rgb in (("red", (255, 40, 40)), ("green", (40, 255, 40))):
            vis[masks[name]] = 0.45 * vis[masks[name]] + 0.55 * np.asarray(rgb)
        ax.imshow(vis.astype(np.uint8))
        lay = f["layout"]
        for name, mk in (("red", "o"), ("green", "^")):
            p = lay[name]
            if p["x"] is not None:
                ax.plot(p["x"], p["y"], mk, ms=11, mfc="white", mec=C_INK, mew=1.5)
                ax.text(p["x"] + 14, p["y"], name[0].upper(), fontsize=11, color="white",
                        fontweight="bold", va="center")  # fmt: skip
        ax.axvline(lay["width"] / 2, color="white", lw=0.6, ls=":")
        short = f["tag"].replace("median_rtc_cap6_", "").replace("_prompt_2_pens", "")
        ax.set_title(f"{short} +{f['offset_s']:g} s (frame at {f['t_from_start']:+.2f} s)\nlayout {lay['layout']} (left|right)", fontsize=8,
                     color=C_INK)  # fmt: skip
    fig.suptitle(
        "Chosen frames: red/green masks tinted, largest-component centroid marked (R, G); "
        "dotted line = image centre",
        x=0.01, ha="left", fontsize=10, color=C_INK,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image <-> pan sign, from the recordings
# ---------------------------------------------------------------------------


def image_pan_calibration(run_dirs: list[Path], seconds: float = 6.0, n_boot: int = 2000) -> dict:
    """Regress the pens' mean horizontal image shift between consecutive frames on the joint changes."""
    rows, dxs = [], []
    for run_dir in run_dirs:
        ticks, chunks, meta = load_run(run_dir)
        t0 = policy_start(ticks, chunks)
        frames = [f for f in frame_list(run_dir, meta) if t0 - 0.05 <= f["t"] <= t0 + seconds]
        prev = None
        for f in frames:
            lay = pen_layout(load_image(run_dir, f))
            state, _ = state_at(ticks, f["t"])
            cur = (f["t"], lay, state)
            if (
                prev is not None
                and lay["both_visible"]
                and prev[1]["both_visible"]
                and f["t"] - prev[0] < 0.15
            ):
                ratios = [lay[c]["area"] / max(prev[1][c]["area"], 1) for c in COLOURS]
                if all(0.6 < q < 1.6 for q in ratios):  # same components, not a segmentation jump
                    dxs.append(np.mean([lay[c]["x"] - prev[1][c]["x"] for c in COLOURS]))
                    rows.append(state[:5] - prev[2][:5])
            prev = cur
    a, y = np.asarray(rows, np.float64), np.asarray(dxs, np.float64)
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    rng = rng_for("calibration")
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(len(y), size=len(y))
        boot.append(np.linalg.lstsq(a[idx], y[idx], rcond=None)[0][0])
    moving = np.abs(a[:, 0]) > 0.5
    simple = float(np.polyfit(a[moving, 0], y[moving], 1)[0]) if moving.sum() > 3 else None
    u = -int(np.sign(coef[0]))
    return {
        "n_pairs": len(y),
        "px_per_deg": dict(zip(JOINTS[:5], r2(coef), strict=True)),
        "pan_px_per_deg_ci95": r2(np.percentile(boot, [2.5, 97.5])),
        "pan_only_slope_px_per_deg": r2(simple),
        "u_image_right_per_plus_pan": u,
        "note": "dx (px, +right) between consecutive frames ~ joint changes (deg); objects shift left when "
        "the view turns right, so u = -sign(pan coefficient) is +1 if +pan turns the wrist view right",
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ChunkSampler:
    """The 1-camera MolmoAct2 checkpoint, sampled like the first RTC chunk of a rollout."""

    def __init__(self, checkpoint: Path, device: str = "cuda"):
        """Load once with fast_load (local re-save), RTC config as lerobot-rollout sets it."""
        import fast_load
        import torch

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        self.torch = torch
        self.device = device
        fast_load.install()
        t0 = time.perf_counter()
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.pretrained_path = checkpoint
        config.device = device
        self.policy = MolmoAct2Policy.from_pretrained(checkpoint, config=config)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        # lerobot-rollout --inference.type=rtc (rollout/context.py): rtc_config + rtc_processor.
        self.policy.config.rtc_config = RTCConfig()
        self.policy.init_rtc_processor()
        info = fast_load.LAST_LOAD
        self.load_info = {
            "seconds": round(time.perf_counter() - t0, 1),
            "fast_path": bool(info.fast) if info else None,
            "reason": info.reason if info else "",
            "checkpoint": str(checkpoint),
        }
        print(f"model loaded in {self.load_info['seconds']} s (fast path: {self.load_info['fast_path']})")

    def batch(self, image: np.ndarray, state: np.ndarray, task: str) -> dict:
        """Preprocessed policy batch for one observation and task (as the RTC engine builds it)."""
        from molmo_common import preprocess

        obs = {IMAGE_KEY: image, "observation.state": np.asarray(state, np.float32)}
        with self.torch.inference_mode():
            return preprocess(self.pre, obs, task, self.device)

    def stock(self, batch: dict, seed: int):
        """Stock predict_action_chunk (RTC path, no previous chunk) at batch 1: (1, T, D) raw."""
        torch = self.torch
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.inference_mode():
            return self.policy.predict_action_chunk(batch, generator=g)

    def sample(self, batch: dict, seeds: list[int], seed_batch: int):
        """Raw normalised chunks (len(seeds), T, D): one VLM pass, flow steps over a batch of seeds.

        Mirrors MolmoAct2Policy.predict_action_chunk -> _generate_actions_from_inputs_with_rtc with
        prev_chunk_left_over=None (RTCProcessor.denoise_step then returns the plain velocity).
        """
        from lerobot.policies.molmoact2.modeling_molmoact2 import _mask_action_dim_tensor

        torch = self.torch
        p = self.policy
        model_inputs = p._model_inputs(batch)
        action_dim = p._output_action_dim(batch)
        pad = batch.get("action_dim_is_pad")
        mask_pad = p.config.mask_action_dim_padding
        outs = []
        with torch.inference_mode(), p._autocast_context():
            backbone, expert = p._backbone(), p._action_expert()
            outputs = backbone(
                **model_inputs, use_cache=True, output_attentions=False, output_hidden_states=False
            )
            kv = backbone._extract_kv_states(outputs.past_key_values)
            enc_mask = p._encoder_attention_mask_for_action_expert(
                input_ids=model_inputs.get("input_ids"), attention_mask=model_inputs.get("attention_mask")
            )
            gate, depth_mask = backbone._depth_gate_from_condition(
                input_ids=model_inputs.get("input_ids"), encoder_attention_mask=enc_mask, layer_kv_states=kv
            )
            kv = backbone._apply_depth_gate_to_layer_kv_states(kv, depth_mask, gate)
            steps = int(
                getattr(p.config, "num_inference_steps", None) or backbone.config.flow_matching_num_steps
            )
            horizon, max_dim = p._generation_action_horizon(), int(backbone.config.max_action_dim)
            device = kv[0][0].device
            for i in range(0, len(seeds), seed_batch):
                group = seeds[i : i + seed_batch]
                b = len(group)
                kv_b = [(k.expand(b, *k.shape[1:]), v.expand(b, *v.shape[1:])) for k, v in kv]
                mask_b = None if enc_mask is None else enc_mask.expand(b, *enc_mask.shape[1:])
                traj = torch.cat(
                    [
                        torch.randn(
                            1,
                            horizon,
                            max_dim,
                            device=device,
                            dtype=torch.float32,
                            generator=torch.Generator(device=device).manual_seed(int(s)),
                        )  # fmt: skip
                        for s in group
                    ]
                )
                if mask_pad:
                    traj = _mask_action_dim_tensor(traj, pad)
                ctx = expert.prepare_context(
                    encoder_kv_states=kv_b,
                    encoder_attention_mask=mask_b,
                    state_embeddings=None,
                    batch_size=b,
                    seq_len=traj.shape[1],
                    device=device,
                    dtype=traj.dtype,
                )
                ts = [torch.full((b,), k / steps, device=device, dtype=traj.dtype) for k in range(steps)]
                mods = expert.get_or_prepare_modulation_cache(ts, cache_key=(steps, b, device, traj.dtype))
                for k in range(steps):
                    vel = expert.forward_with_context(
                        traj, mods[k].conditioning, context=ctx, modulation=mods[k]
                    )
                    if mask_pad:
                        vel = _mask_action_dim_tensor(vel, pad)
                    traj = traj + (1.0 / steps) * vel
                    if mask_pad:
                        traj = _mask_action_dim_tensor(traj, pad)
                outs.append(traj[:, : p.config.n_action_steps, :action_dim].to(dtype=torch.float32))
        return torch.cat(outs)

    def postprocess(self, raw) -> np.ndarray:
        """Stock postprocessor (clamp, unnormalise, model -> arm frame): (N, T, D) degrees."""
        torch = self.torch
        with torch.inference_mode():
            out = self.post(raw.clone())
        return torch.as_tensor(out).float().cpu().numpy().reshape(raw.shape)


def validate_sampler(sampler: ChunkSampler, image: np.ndarray, state: np.ndarray, prompts: dict[str, str],
                     seeds: list[int], seed_batch: int) -> dict:  # fmt: skip
    """Batched sampler against the stock predict_action_chunk (batch 1) on one frame."""
    out = {}
    for name, text in prompts.items():
        batch = sampler.batch(image, state, text)
        stock = sampler.torch.cat([sampler.stock(batch, s) for s in seeds])
        one = sampler.sample(batch, seeds, 1)
        many = sampler.sample(
            batch, seeds + list(range(10_000, 10_000 + seed_batch - len(seeds))), seed_batch
        )
        many = many[: len(seeds)]
        rows = np.stack([sampler.postprocess(stock[i : i + 1])[0] for i in range(len(seeds))])
        out[name] = {
            "seeds": seeds,
            "batch1_vs_stock_max_abs_raw": float((one - stock).abs().max()),
            "batched_vs_stock_max_abs_raw": float((many - stock).abs().max()),
            "batched_vs_stock_max_abs_deg": float(np.abs(sampler.postprocess(many) - rows).max()),
            "postprocess_batch_vs_rows_max_abs_deg": float(np.abs(sampler.postprocess(stock) - rows).max()),
        }
        print(f"validate {name}: {out[name]}")
    return out


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def boot_mean_ci(x: np.ndarray, key: str, n_boot: int) -> list[float] | None:
    """Bootstrap 95% CI of a mean."""
    if len(x) < 2:
        return None
    rng = rng_for(key)
    means = x[rng.integers(len(x), size=(n_boot, len(x)))].mean(1)
    return r2(np.percentile(means, [2.5, 97.5]))


def tip_motion(fk: SO101FK, chunks: np.ndarray, u: int) -> dict[str, np.ndarray]:
    """Per seed, tip motion over the chunk (mm): lateral towards image right, up, and radial reach."""
    first, last = chunks[:, 0].astype(np.float64), chunks[:, -1].astype(np.float64)
    p0, p1 = fk.tip(first), fk.tip(last)
    bumped = first.copy()
    bumped[:, 0] += 1.0  # direction the tip moves for +1 deg of pan, horizontal part
    e = fk.tip(bumped) - p0
    e[:, 2] = 0.0
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    d = (p1 - p0) * 1000.0
    return {
        "lateral_right_mm": u * np.einsum("ij,ij->i", d, e),
        "up_mm": d[:, 2],
        "reach_mm": (np.hypot(p1[:, 0], p1[:, 1]) - np.hypot(p0[:, 0], p0[:, 1])) * 1000.0,
    }


def frame_readouts(frame: dict, chunks: dict[str, np.ndarray], u: int, fk: SO101FK, threshold: float,
                   n_boot: int) -> tuple[dict, dict]:  # fmt: skip
    """Readouts on one frame; also the per-prompt mover arrays the aggregates resample."""
    side_red = frame["layout"]["red_side"]
    side = {"red": side_red, "green": -side_red}
    movers = mover_stats(chunks, threshold)
    direction = mover_direction({c: chunks[c] for c in COLOURS if c in chunks}, threshold, n_boot, SEED)
    per_prompt, arrays = {}, {}
    for name, c in chunks.items():
        d = c[:, -1] - c[:, 0]
        moving = is_moving(d, threshold)
        r = u * d[:, 0]
        tip = tip_motion(fk, c, u)
        arrays[name] = {"r": r[moving], "moving": moving, "r_all": r}
        entry = {
            "n_moving": int(moving.sum()),
            "mover_image_right_pan_deg": r2(r[moving].mean()) if moving.any() else None,
            "mover_image_right_pan_ci95": boot_mean_ci(r[moving], f"{frame['id']}_{name}_r", n_boot),
            "mover_tip_lateral_right_mm": r2(tip["lateral_right_mm"][moving].mean())
            if moving.any()
            else None,
            "mover_tip_up_mm": r2(tip["up_mm"][moving].mean()) if moving.any() else None,
            "mover_tip_reach_mm": r2(tip["reach_mm"][moving].mean()) if moving.any() else None,
            "mover_toward_red_pan_deg": r2(side_red * r[moving].mean()) if moving.any() else None,
        }
        if name in side and moving.any():
            entry["mover_fraction_toward_named_pen"] = r2(np.mean(side[name] * r[moving] > 0))
            entry["mover_tip_lateral_toward_named_mm"] = r2(
                side[name] * tip["lateral_right_mm"][moving].mean()
            )
        per_prompt[name] = entry
    colour = None
    pair = direction.get("red_vs_green")
    if pair is not None:
        # mover_pan_diff is red - green in raw pan degrees; u maps it to image right, side_red to "towards red".
        ci = sorted(u * side_red * v for v in pair["mover_pan_diff_ci95"])
        both = arrays["red"]["moving"] & arrays["green"]["moving"]
        paired = side_red * (arrays["red"]["r_all"] - arrays["green"]["r_all"])
        colour = {
            "toward_red_pan_deg": r2(u * side_red * pair["mover_pan_diff_deg"]),
            "toward_red_pan_ci95": r2(ci),
            "n_moving": pair["n_moving"],
            "paired_all_seeds_toward_red_pan_deg": r2(paired.mean()),
            "paired_all_seeds_ci95": boot_mean_ci(paired, f"{frame['id']}_paired", n_boot),
            "n_moving_both": int(both.sum()),
        }
    return {
        "movers": movers,
        "mover_direction": direction,
        "per_prompt": per_prompt,
        "colour_effect": colour,
    }, arrays


def tip_entry(fk: SO101FK, chunk: np.ndarray, u: int, side_red: int, side_named: int | None,
              threshold: float) -> dict:  # fmt: skip
    """Tip displacement of one chunk (arm frame, (T, 6)) relative to the pens' layout."""
    tip = {k: float(v[0]) for k, v in tip_motion(fk, chunk[None], u).items()}
    d = chunk[-1] - chunk[0]
    lateral = tip["lateral_right_mm"]
    out = {
        "moving": bool(is_moving(d[None], threshold)[0]),
        "pan_image_right_deg": r2(u * d[0]),
        "tip_lateral_right_mm": r2(lateral),
        "tip_reach_mm": r2(tip["reach_mm"]),
        "tip_up_mm": r2(tip["up_mm"]),
        "toward_red_mm": r2(side_red * lateral) if side_red else None,
        "heads": ("right" if lateral > 0 else "left") + (" (toward red)" if side_red * lateral > 0 else
                                                          " (toward green)") if side_red else None,
    }  # fmt: skip
    if side_named:
        out["toward_named_mm"] = r2(side_named * lateral)
    return out


def on_arm_readouts(frames: list[dict], prompts: dict[str, str], u: int, threshold: float,
                    n_chunks: int = 5) -> dict:  # fmt: skip
    """Per run: the chunks the arm actually predicted (first ``n_chunks`` policy chunks), tip direction.

    Each chunk is read against the pen layout segmented in the recorded frame nearest its
    observation time (chunks.t_start). Its prompt is the run's own task.
    """
    fk = SO101FK()
    by_text = {v: k for k, v in prompts.items()}
    out = {}
    for run in sorted({f["run"] for f in frames}):
        run_dir = Path(next(f["run_dir"] for f in frames if f["run"] == run))
        ticks, chunks, meta = load_run(run_dir)
        if "chunk_arm" not in chunks:
            continue
        idx = np.arange(len(chunks["t_start"]))
        if "phase" in chunks:
            idx = idx[chunks["phase"] == PHASE_POLICY]
        rec = frame_list(run_dir, meta)
        times = np.asarray([f["t"] for f in rec])
        prompt = by_text.get(meta.get("task"))
        rows = []
        for k in idx[:n_chunks]:
            ts = float(chunks["t_start"][k])
            fr = rec[int(np.argmin(np.abs(times - ts)))]
            lay = pen_layout(load_image(run_dir, fr))
            side_red = lay["red_side"]
            named = {"red": side_red, "green": -side_red}.get(prompt or "")
            e = tip_entry(fk, chunks["chunk_arm"][k], u, side_red, named, threshold)
            rows.append({"chunk": int(k), "t_from_start": r2(ts - policy_start(ticks, chunks)),
                         "frame": fr["file"], "layout": lay["layout"], **e})  # fmt: skip
        out[run] = {"task": meta.get("task"), "prompt": prompt, "chunks": rows}
    return out


def place_on_arm(on_arm: dict, frames: list[dict], arm: np.ndarray, names: list[str], u: int,
                 threshold: float) -> dict:  # fmt: skip
    """Offline seeds on each run's first frame, same tip readout; where the on-arm first chunk falls."""
    fk = SO101FK()
    out = {}
    for run, entry in on_arm.items():
        i = next((k for k, f in enumerate(frames) if f["run"] == run and f["offset_s"] == 0), None)
        if i is None or not entry["chunks"]:
            continue
        side_red = frames[i]["layout"]["red_side"]
        sample = entry["chunks"][0]
        per_prompt = {}
        for j, p in enumerate(names):
            c = arm[i, j]
            tip = tip_motion(fk, c, u)
            moving = is_moving(c[:, -1] - c[:, 0], threshold)
            toward_red = side_red * tip["lateral_right_mm"]
            per_prompt[p] = {
                "toward_red_mm_mean_all": r2(toward_red.mean()),
                "toward_red_mm_quartiles_all": r2(np.percentile(toward_red, [25, 50, 75])),
                "toward_red_mm_mean_movers": r2(toward_red[moving].mean()) if moving.any() else None,
                "fraction_movers_toward_red": r2(np.mean(toward_red[moving] > 0)) if moving.any() else None,
                "n_moving": int(moving.sum()),
                "reach_mm_mean_movers": r2(tip["reach_mm"][moving].mean()) if moving.any() else None,
                "up_mm_mean_movers": r2(tip["up_mm"][moving].mean()) if moving.any() else None,
                # Percentile of the on-arm first chunk within this prompt's 64 offline seeds.
                "on_arm_percentile": r2(100.0 * np.mean(toward_red <= sample["toward_red_mm"]))
                if sample["toward_red_mm"] is not None
                else None,
            }
        out[run] = {
            "prompt": entry["prompt"],
            "frame_layout": frames[i]["layout"]["layout"],
            "on_arm_first_chunk_toward_red_mm": sample["toward_red_mm"],
            "on_arm_first_chunk_moving": sample["moving"],
            "offline": per_prompt,
        }
    return out


def frame_stat(arrays: dict, kind: str, side_red: int, rng: np.random.Generator | None) -> float | None:
    """One frame's statistic, optionally with movers resampled: colour effect or a prompt's r."""

    def mean(x: np.ndarray) -> float:
        if rng is not None:
            x = x[rng.integers(len(x), size=len(x))]
        return float(x.mean())

    if kind == "colour":
        a, b = arrays.get("red"), arrays.get("green")
        if a is None or b is None or len(a["r"]) < MIN_MOVERS or len(b["r"]) < MIN_MOVERS:
            return None
        return side_red * (mean(a["r"]) - mean(b["r"]))
    prompt, _, frame_ref = kind.partition(":")
    a = arrays.get(prompt)
    if a is None or len(a["r"]) < MIN_MOVERS:
        return None
    sign = side_red if frame_ref == "red" else 1  # "towards red" or "towards image right"
    return sign * mean(a["r"])


def hierarchical(frames: list[dict], arrays: list[dict], kind: str, n_boot: int,
                 keep=lambda f: True) -> dict:  # fmt: skip
    """Mean over frames of a frame statistic; 95% CI resampling runs, then movers within frames."""
    idx = [i for i, f in enumerate(frames) if keep(f)]
    vals = {i: frame_stat(arrays[i], kind, frames[i]["layout"]["red_side"], None) for i in idx}
    idx = [i for i in idx if vals[i] is not None]
    if not idx:
        return {"mean": None, "ci95": None, "n_frames": 0, "n_runs": 0}
    by_run: dict[str, list[int]] = {}
    for i in idx:
        by_run.setdefault(frames[i]["run"], []).append(i)
    runs = sorted(by_run)
    rng = rng_for(f"hier_{kind}_{len(idx)}_{runs[0]}")
    boot = []
    for _ in range(n_boot):
        picked = [runs[j] for j in rng.integers(len(runs), size=len(runs))]
        xs = [
            frame_stat(arrays[i], kind, frames[i]["layout"]["red_side"], rng)
            for r in picked
            for i in by_run[r]
        ]
        boot.append(np.mean(xs))
    pos = [vals[i] > 0 for i in idx]
    return {
        "mean": r2(np.mean([vals[i] for i in idx])),
        "ci95": r2(np.percentile(boot, [2.5, 97.5])),
        "n_frames": len(idx),
        "n_runs": len(runs),
        "frames_positive": int(sum(pos)),
    }


def gating(frames: list[dict], chunks: np.ndarray, prompts: list[str], threshold: float, n_boot: int,
           keep=lambda f: True) -> dict:  # fmt: skip
    """Movers fraction per prompt over frames; CI resampling runs, then seeds."""
    idx = [i for i, f in enumerate(frames) if keep(f)]
    moving = {p: [is_moving(chunks[i, j, :, -1] - chunks[i, j, :, 0], threshold) for i in idx]
              for j, p in enumerate(prompts)}  # fmt: skip
    runs = sorted({frames[i]["run"] for i in idx})
    by_run = {r: [k for k, i in enumerate(idx) if frames[i]["run"] == r] for r in runs}
    out = {}
    for p in prompts:
        rng = rng_for(f"gating_{p}_{len(idx)}")
        boot = []
        for _ in range(n_boot):
            picked = [runs[j] for j in rng.integers(len(runs), size=len(runs))]
            fr = [moving[p][k][rng.integers(len(moving[p][k]), size=len(moving[p][k]))].mean()
                  for r in picked for k in by_run[r]]  # fmt: skip
            boot.append(np.mean(fr))
        out[p] = {
            "fraction_moving": r2(np.mean([m.mean() for m in moving[p]])),
            "ci95": r2(np.percentile(boot, [2.5, 97.5])),
            "per_frame": [int(m.sum()) for m in moving[p]],
        }
    return out


def analyse(frames: list[dict], arm: np.ndarray, prompts: dict[str, str], calibration: dict, threshold: float,
            n_boot: int) -> dict:  # fmt: skip
    """Per-frame readouts and the three aggregates."""
    u = int(calibration["u_image_right_per_plus_pan"])
    names = list(prompts)
    fk = SO101FK()
    per_frame, arrays = [], []
    for i, f in enumerate(frames):
        chunks = {p: arm[i, j] for j, p in enumerate(names)}
        res, arr = frame_readouts(f, chunks, u, fk, threshold, n_boot)
        per_frame.append(
            {"id": f["id"], "layout": f["layout"]["layout"], "t_from_start": f["t_from_start"], **res}
        )
        arrays.append(arr)

    # "uncommitted": frames up to +1 s, before the arm has turned towards a pen (it first moves ~0.8 s
    # in); by +2-3 s it is at or over a pen and the prompts converge.
    early = lambda f: f["offset_s"] <= 1.0  # noqa: E731
    subsets = {
        "all_frames": lambda f: True,
        "uncommitted_le_1s": early,
        "first_frame_only": lambda f: f["offset_s"] == 0,
        "red_left_layout": lambda f: early(f) and f["layout"]["red_side"] < 0,
        "red_right_layout": lambda f: early(f) and f["layout"]["red_side"] > 0,
        "red_left_layout_all_frames": lambda f: f["layout"]["red_side"] < 0,
        "red_right_layout_all_frames": lambda f: f["layout"]["red_side"] > 0,
    }
    colour = {k: hierarchical(frames, arrays, "colour", n_boot, keep) for k, keep in subsets.items()}
    # 2x2 table: image-right pan of each prompt's movers, by layout (uncommitted frames).
    table = {k: {p: hierarchical(frames, arrays, p, n_boot, subsets[k]) for p in names}
             for k in ("red_left_layout", "red_right_layout")}  # fmt: skip
    position = {
        p: {
            "image_right_pan_uncommitted": hierarchical(frames, arrays, p, n_boot, early),
            "toward_red_pan_uncommitted": hierarchical(frames, arrays, f"{p}:red", n_boot, early),
        }
        for p in names
    }
    toward_named = {}
    for c in COLOURS:
        fr = [pf["per_prompt"][c].get("mover_fraction_toward_named_pen") for pf in per_frame]
        n = [pf["per_prompt"][c]["n_moving"] for pf in per_frame]
        ok = [(a, b) for a, b in zip(fr, n, strict=True) if a is not None and b >= MIN_MOVERS]
        toward_named[c] = {
            "mover_weighted_fraction": r2(sum(a * b for a, b in ok) / sum(b for _, b in ok)) if ok else None,
            "n_frames": len(ok),
        }
    on_arm = on_arm_readouts(frames, prompts, u, threshold)
    return {
        "on_arm_chunks": on_arm,
        "on_arm_vs_offline_first_frame": place_on_arm(on_arm, frames, arm, names, u, threshold),
        "per_frame": per_frame,
        "colour_following": colour,
        "image_right_pan_by_layout": table,
        "position_bias": position,
        "toward_named_pen_fraction": toward_named,
        "gating": {
            k: gating(frames, arm, names, threshold, n_boot, subsets[k])
            for k in ("all_frames", "uncommitted_le_1s", "first_frame_only")
        },
    }


def verdict(res: dict) -> str:
    """Plain-language reading of the aggregates (the rule is written in SUMMARY.md)."""
    # Rule, on the uncommitted frames: "steers" if the run-resampled CI of the colour effect excludes 0
    # and the effect is positive in both layouts (so it follows the pen, not a side); "position" if the
    # CI includes 0 and |effect| < 1 deg (the words barely change the direction); else "unclear".
    c = res["colour_following"]
    agg, left, right = c["uncommitted_le_1s"], c["red_left_layout"], c["red_right_layout"]
    if agg["ci95"] is None:
        return "unclear"
    if agg["ci95"][0] > 0 and (left["mean"] or 0) > 0 and (right["mean"] or 0) > 0:
        return "the colour word steers the target"
    if agg["ci95"][0] <= 0 <= agg["ci95"][1] and abs(agg["mean"]) < 1.0:
        return "position/learned direction dominates"
    return "unclear"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", color=C_GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def plot_direction(frames: list[dict], res: dict, names: list[str], out: Path) -> None:
    """Left: colour effect per frame with CIs. Right: image-right pan per prompt by layout."""
    order = sorted(range(len(frames)), key=lambda i: (frames[i]["layout"]["red_side"], frames[i]["run"],
                                                       frames[i]["offset_s"]))  # fmt: skip
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 0.32 * len(frames) + 2.6),
                                 gridspec_kw={"width_ratios": [1.15, 1]})  # fmt: skip
    ys, labels = [], []
    for y, i in enumerate(order):
        ce = res["per_frame"][i]["colour_effect"]
        f = frames[i]
        short = f["tag"].replace("median_rtc_cap6_", "").replace("_prompt_2_pens", "")
        labels.append(f"{short} +{f['offset_s']:g}s  [{f['layout']['layout']}]")
        ys.append(y)
        if ce is None:
            ax.text(0, y, "too few movers", fontsize=7, color=C_MUTED, va="center", ha="center")
            continue
        lo, hi = ce["toward_red_pan_ci95"]
        ax.plot([lo, hi], [y, y], color=C_ACCENT, lw=2, solid_capstyle="round")
        ax.plot(ce["toward_red_pan_deg"], y, "o", ms=6, color=C_ACCENT, mec="white", mew=1)
    agg = res["colour_following"]["uncommitted_le_1s"]
    y_agg = len(order) + 0.8
    if agg["mean"] is not None:
        ax.plot(agg["ci95"], [y_agg, y_agg], color=C_INK, lw=2.5, solid_capstyle="round")
        ax.plot(agg["mean"], y_agg, "D", ms=8, color=C_INK, mec="white", mew=1)
    ax.set_yticks(
        [*ys, y_agg], [*labels, f"frames <= +1 s (runs resampled), n={agg['n_frames']}"], fontsize=7
    )
    ax.axvline(0, color=C_MUTED, lw=1)
    ax.invert_yaxis()
    ax.set_xlabel('pan towards the red pen: "red pen" minus "green pen" movers (deg over the 1 s chunk)',
                  fontsize=8, color=C_MUTED)  # fmt: skip
    ax.set_title("(i) colour-following on the same frame (95% CI)", fontsize=10, color=C_INK, loc="left")
    _style(ax)

    table = res["image_right_pan_by_layout"]
    xs = {"red_left_layout": 0, "red_right_layout": 1}
    offsets = np.linspace(-0.12, 0.12, len(names))
    for k, name in enumerate(names):
        st = STYLE.get(name, {"color": C_MUTED, "marker": "o", "label": name})
        pts = []
        for lay, x in xs.items():
            e = table[lay][name]
            if e["mean"] is None:
                continue
            xx = x + offsets[k]
            bx.plot([xx, xx], e["ci95"], color=st["color"], lw=2, solid_capstyle="round")
            pts.append((xx, e["mean"]))
        if pts:
            bx.plot(*zip(*pts, strict=True), color=st["color"], lw=1.2, alpha=0.6)
            bx.plot(*zip(*pts, strict=True), st["marker"], ms=8, color=st["color"], mec="white", mew=1.2,
                    label=f'"{st["label"]}"')  # fmt: skip
            bx.text(pts[-1][0] + 0.06, pts[-1][1], st["label"], fontsize=8, color=C_INK, va="center")
    bx.axhline(0, color=C_MUTED, lw=1)
    bx.set_xticks([0, 1], ["red pen LEFT\n(red | green)", "red pen RIGHT\n(green | red)"], fontsize=8)
    bx.set_xlim(-0.4, 1.55)
    bx.set_ylabel("image-right pan of moving seeds (deg; + = towards the right of the wrist view)", fontsize=8,
                  color=C_MUTED)  # fmt: skip
    bx.set_title("(ii) heading by layout, frames <= +1 s (95% CI)", fontsize=10, color=C_INK, loc="left")
    bx.legend(frameon=False, fontsize=8, loc="best")
    _style(bx)
    bx.grid(True, axis="y", color=C_GRID, lw=0.6)
    bx.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_on_arm(frames: list[dict], arm: np.ndarray, res: dict, names: list[str], u: int, out: Path) -> None:
    """Per run, first frame: offline tip motion towards the red pen per prompt vs the arm's own chunk.

    Offline seeds are shown as median, IQR and 10-90%; the chunk the arm actually predicted from that
    observation is a black star (its own prompt).
    """
    fk = SO101FK()
    runs = list(res["on_arm_vs_offline_first_frame"])
    fig, ax = plt.subplots(figsize=(10, 0.95 * len(runs) + 1.6))
    offs = np.linspace(-0.3, 0.3, len(names))
    labels = []
    for y, run in enumerate(runs):
        i = next(k for k, f in enumerate(frames) if f["run"] == run and f["offset_s"] == 0)
        side_red = frames[i]["layout"]["red_side"]
        entry = res["on_arm_vs_offline_first_frame"][run]
        for k, p in enumerate(names):
            st = STYLE.get(p, {"color": C_MUTED, "marker": "o", "label": p})
            x = side_red * tip_motion(fk, arm[i, k], u)["lateral_right_mm"]
            q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90])
            yy = y + offs[k]
            ax.plot([q10, q90], [yy, yy], color=st["color"], lw=0.8, alpha=0.6)
            ax.plot([q25, q75], [yy, yy], color=st["color"], lw=3, solid_capstyle="round")
            ax.plot(q50, yy, st["marker"], ms=6, color=st["color"], mec="white", mew=0.8,
                    label=f'"{st["label"]}" offline (median, IQR, 10-90%)' if y == 0 else None)  # fmt: skip
        if entry["on_arm_first_chunk_toward_red_mm"] is not None and entry["prompt"] in names:
            yy = y + offs[names.index(entry["prompt"])]
            ax.plot(entry["on_arm_first_chunk_toward_red_mm"], yy, "*", ms=15, color=C_INK, mec="white", mew=1,
                    label="chunk predicted on the arm (its own prompt)" if y == 0 else None)  # fmt: skip
        short = frames[i]["tag"].replace("median_rtc_cap6_", "").replace("_prompt_2_pens", "")
        labels.append(f'{short}\nran "{STYLE.get(entry["prompt"] or "", {"label": "?"})["label"]}", '
                      f'layout {entry["frame_layout"]}')  # fmt: skip
    ax.set_yticks(range(len(runs)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=C_MUTED, lw=1)
    ax.set_xlabel("tip lateral motion over the first chunk, mm (+ = towards the red pen's side, - = towards green)",
                  fontsize=8, color=C_MUTED)  # fmt: skip
    ax.set_title("First policy frame: on-arm chunk vs 64 offline seeds per prompt", fontsize=10, color=C_INK,
                 loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_gating(res: dict, names: list[str], out: Path) -> None:
    """Movers per prompt: one dot per frame, mean with run-resampled 95% CI."""
    g = res["gating"]["uncommitted_le_1s"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    rng = np.random.default_rng(0)
    for k, name in enumerate(names):
        st = STYLE.get(name, {"color": C_MUTED, "marker": "o", "label": name})
        per = np.asarray(g[name]["per_frame"], float)
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(per)), per, s=14, color=C_MUTED, alpha=0.55, lw=0)
        n_seeds = res["n_seeds"]
        mean, ci = g[name]["fraction_moving"] * n_seeds, [v * n_seeds for v in g[name]["ci95"]]
        ax.plot([k + 0.28, k + 0.28], ci, color=st["color"], lw=2.5, solid_capstyle="round")
        ax.plot(k + 0.28, mean, st["marker"], ms=9, color=st["color"], mec="white", mew=1.2)
        ax.text(k + 0.36, mean, f"{mean:.0f}", fontsize=8, color=C_INK, va="center")
    ax.set_xticks(range(len(names)), [f'"{STYLE.get(n, {"label": n})["label"]}"' for n in names], fontsize=8)
    ax.set_ylabel(f"moving seeds (of {res['n_seeds']})", fontsize=8, color=C_MUTED)
    ax.set_ylim(0, res["n_seeds"] * 1.05)
    ax.set_title("(iii) gating, frames <= +1 s: seeds that move per prompt (grey: each frame; mean, 95% CI)", fontsize=10,
                 color=C_INK, loc="left")  # fmt: skip
    _style(ax)
    ax.grid(True, axis="y", color=C_GRID, lw=0.6)
    ax.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_prompts(items: list[str] | None) -> dict[str, str]:
    """Repeated name=text, else the defaults."""
    if not items:
        return dict(DEFAULT_PROMPTS)
    out = {}
    for item in items:
        name, sep, text = item.partition("=")
        if not sep:
            raise SystemExit(f"--prompt needs name=text, got {item!r}")
        out[name] = text
    return out


def main() -> None:
    """Select frames, sample every prompt with shared seeds, analyse, plot."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "run_dirs", nargs="*", type=Path, help="run dirs (default: every *2_pens* run with ticks)"
    )
    ap.add_argument("--prompt", action="append", help='name="text" (repeatable); default: the five prompts')
    ap.add_argument("--seeds", type=int, default=64, help="seeds 0..N-1, the same for every prompt and frame")
    ap.add_argument("--seed-batch", type=int, default=16, help="seeds per flow-matching batch")
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0],
                    help="s after the first policy observation (the arm first moves ~0.8 s in)")  # fmt: skip
    ap.add_argument("--move-threshold", type=float, default=5.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--frames-only", action="store_true", help="frames, layout, calibration, contact sheet")
    ap.add_argument("--analyse-only", action="store_true", help="readouts and plots from OUT/chunks.npz")
    ap.add_argument("--validate", type=int, default=4, help="seeds checked against stock inference (0: skip)")
    args = ap.parse_args()
    prompts = parse_prompts(args.prompt)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.analyse_only:
        meta = json.loads((out / "results.json").read_text())
        frames = json.loads((out / "frames.json").read_text())
        data = np.load(out / "chunks.npz")
        prompts, calibration = meta["prompts"], meta["image_pan_calibration"]
        arm = data["arm"]
        extra = {k: meta[k] for k in ("validation", "model", "timing") if k in meta}
    else:
        run_dirs = args.run_dirs or find_runs()
        print(f"{len(run_dirs)} runs")
        frames = select_frames(run_dirs, args.offsets)
        print(f"{len(frames)} frames: " + ", ".join(f"{f['id']}[{f['layout']['layout']}]" for f in frames))
        calibration = image_pan_calibration(run_dirs)
        print("image <-> pan:", calibration)
        (out / "frames.json").write_text(json.dumps(frames, indent=1) + "\n")
        contact_sheet(frames, out / "contact_sheet.jpg")
        if args.frames_only:
            (out / "calibration.json").write_text(json.dumps(calibration, indent=1) + "\n")
            return
        seeds = list(range(args.seeds))
        sampler = ChunkSampler(args.checkpoint)
        first_img = load_image(Path(frames[0]["run_dir"]), frames[0])
        first_state = np.asarray(frames[0]["state"], np.float32)
        t_warm = time.perf_counter()
        for s in range(2):  # warm-up
            sampler.sample(sampler.batch(first_img, first_state, prompts[next(iter(prompts))]), [100 + s], 1)
        b = sampler.batch(first_img, first_state, prompts[next(iter(prompts))])
        deterministic = bool((sampler.sample(b, [0, 1], 2) == sampler.sample(b, [0, 1], 2)).all())
        validation: dict[str, Any] = {"same_seed_deterministic": deterministic}
        if args.validate:
            vp = {k: prompts[k] for k in list(prompts)[:1] + [k for k in prompts if prompts[k] == ""][:1]}
            validation.update(validate_sampler(sampler, first_img, first_state, vp, seeds[: args.validate],
                                               args.seed_batch))  # fmt: skip
        print(f"warm-up + validation {time.perf_counter() - t_warm:.1f} s")
        raw = np.zeros((len(frames), len(prompts), len(seeds), 30, 6), np.float32)
        arm = np.zeros_like(raw)
        t_run = time.perf_counter()
        for i, f in enumerate(frames):
            img = load_image(Path(f["run_dir"]), f)
            state = np.asarray(f["state"], np.float32)
            for j, text in enumerate(prompts.values()):
                r = sampler.sample(sampler.batch(img, state, text), seeds, args.seed_batch)
                if i == 0 and j == 0 and r.shape[1:] != raw.shape[3:]:
                    raw = np.zeros((len(frames), len(prompts), len(seeds), *r.shape[1:]), np.float32)
                    arm = np.zeros_like(raw)
                raw[i, j] = r.cpu().numpy()
                arm[i, j] = sampler.postprocess(r)
            print(f"  frame {i + 1}/{len(frames)} {f['id']} done ({time.perf_counter() - t_run:.0f} s)")
        np.savez_compressed(
            out / "chunks.npz",
            raw=raw,
            arm=arm,
            state=np.asarray([f["state"] for f in frames], np.float32),
            frame_ids=np.asarray([f["id"] for f in frames]),
            prompts=np.asarray(list(prompts)),
            seeds=np.asarray(seeds),
        )
        extra = {
            "validation": validation,
            "model": sampler.load_info,
            "timing": {"sampling_s": round(time.perf_counter() - t_run, 1), "per_frame_prompt_s": round(
                (time.perf_counter() - t_run) / (len(frames) * len(prompts)), 2)},
        }  # fmt: skip

    res = analyse(frames, arm, prompts, calibration, args.move_threshold, args.n_boot)
    res["n_seeds"] = int(arm.shape[2])
    result = {
        "question": "does the colour word change where MolmoAct2 heads, observation held fixed?",
        "prompts": prompts,
        "seeds": int(arm.shape[2]),
        "move_threshold_deg": args.move_threshold,
        "n_frames": len(frames),
        "frames": [{k: f[k] for k in ("id", "run", "tag", "run_task", "offset_s", "t_from_start", "file", "state",
                                      "pan_moved_since_start_deg")} | {"layout": f["layout"]} for f in frames],
        "image_pan_calibration": calibration,
        **extra,
        "verdict": verdict(res),
        **res,
    }  # fmt: skip
    (out / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    plot_direction(frames, res, list(prompts), out / "direction.png")
    plot_gating(res, list(prompts), out / "gating.png")
    plot_on_arm(
        frames, arm, res, list(prompts), int(calibration["u_image_right_per_plus_pan"]), out / "on_arm.png"
    )
    c = res["colour_following"]
    print("colour-following (deg of pan towards the red pen, red - green):")
    for k, v in c.items():
        print(f"  {k}: {v}")
    print("image-right pan by layout:")
    for lay, d in res["image_right_pan_by_layout"].items():
        print(f"  {lay}: " + ", ".join(f"{p} {v['mean']} {v['ci95']}" for p, v in d.items()))
    for k, g in res["gating"].items():
        print(f"gating {k}:", {p: (v["fraction_moving"], v["ci95"]) for p, v in g.items()})
    print("position (neutral prompts, uncommitted):", {p: (v["image_right_pan_uncommitted"]["mean"],
          v["image_right_pan_uncommitted"]["ci95"], v["toward_red_pan_uncommitted"]["mean"],
          v["toward_red_pan_uncommitted"]["ci95"]) for p, v in res["position_bias"].items()})  # fmt: skip
    print("on-arm chunks (tip lateral mm towards the red pen, first 5 policy chunks):")
    for run, e in res["on_arm_chunks"].items():
        print(f"  {run[16:]} ({e['prompt']}): " + ", ".join(f"{c['toward_red_mm']}{'' if c['moving'] else '(still)'}"
                                                         for c in e["chunks"]))  # fmt: skip
    for run, e in res["on_arm_vs_offline_first_frame"].items():
        own = e["offline"].get(e["prompt"] or "", {})
        print(f"  {run[16:]}: on-arm {e['on_arm_first_chunk_toward_red_mm']} mm, percentile in own prompt "
              f"{own.get('on_arm_percentile')}; offline medians " + ", ".join(
                  f"{p} {v['toward_red_mm_quartiles_all'][1]}" for p, v in e["offline"].items()))  # fmt: skip
    print("verdict:", result["verdict"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
