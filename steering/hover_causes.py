"""Why do consecutive RTC chunks disagree while MolmoAct2 hovers? Sampling noise vs observation sensitivity.

    uv run python steering/hover_causes.py --plan-only                              # CPU: windows, frames
    flock /tmp/claude-gpu.lock uv run python steering/hover_causes.py              # sample + analyse
    uv run python steering/hover_causes.py --analyse-only                           # CPU: from samples.npz

Windows. For each run with a hover window (plan_consistency.analyse, label_provisional labels), take 8
consecutive executed chunks from the middle of its longest hover segment and, where the run has one, 8
from the middle of its longest "other" (moving, not hover/grasp/park) segment. That gives 7
consecutive pairs (p -> k) per window, with the on-arm alignment (q, drop) and RTC inference_delay.
Each chunk's observation is the recorded wrist frame nearest its ``t_start``, with the state of the first
policy tick at or after that frame (prompt_counterfactual.state_at); the run's own task string.

Metric. D(a, b) = RMS over steps j in J = [10, 30 - q) and arm joints 0-4 (deg, arm frame, after the
stock postprocessor) of a[j] - b[j + q]; for two chunks from the same observation q = 0 with the
pair's J. This is plan_consistency's "unguided" arm RMS: the measured on-arm value is D(chunk_k,
chunk_p) on the recorded chunks.

Samples per window frame (1-cam checkpoint, prompt_counterfactual.ChunkSampler loading, one VLM pass
per frame, flow steps over seed batches; noise for seed s = randn(1, H, D, Generator(cuda).seed(s))):
  A   64 seeds, unguided (no previous chunk), temperature 1
  T7  seeds 0-31, x0 scaled by 0.7;  T5  x0 scaled by 0.5
For frames that are the k of a pair, also:
  S   seeds 0-31, tick-aligned noise: x0_k[j] = x0_p(s)[j + q] for j < H - q, fresh tail (seed 10^5 + s)
  R10, R20  seeds 0-15, RTC guidance through the policy's own RTCProcessor.denoise_step, with the
      recorded previous chunk's normalised leftover chunk_norm[p][q:] (padded/truncated to the horizon
      as lerobot's RTC engine does) and the recorded inference_delay; execution_horizon 10 / 20
And for hover pair frames, the next recorded frame (~0.1 s later, its own state):
  B, Bs  seeds 0-31, index-aligned / tick-aligned (shift = round(dt * fps)) noise
Validation: batch-1 flows against the stock predict_action_chunk (no prefix, and RTC with h 10 and 20),
plus the floor from seed batching (same seeds, different batch size).

Readouts per pair (medians over seeds / seed pairs):
  measured        on-arm D(chunk_k, chunk_p)
  rtc10_vs_rec    offline RTC h10 on frame k vs the recorded chunk p (reproduces "measured")
  free_vs_rec     unguided on frame k vs the recorded chunk p
  seed_spread     D between two seeds on frame k (H1: noise alone); guided_spread: same for R10
  fixed_index     same seed on frames p and k, index-aligned (remedy a: one noise sample per episode)
  fixed_tick      tick-aligned noise (S_k vs A_p): H2, observation change with the noise held fixed
  fixed_tick_0.1s same on frame k vs the frame 0.1 s later (hover only)
  fresh           different seeds on p and k, unguided (H1 + H2 together)
  temp0.7/0.5     fresh, at temperature 0.7 / 0.5;  rtc20_vs_rec
  best-of-N       chains over each window: chunk k picked among N of frame k's 64 seeds, closest
                  (arm RMS over the full overlap j < 30 - q) to the previous pick; N = 1 is fresh noise.
Plan shift (the cost of a remedy): best-of-N, RMS of (pick - mean of the N - 1 unpicked) over the steps
the previous chunk does not cover (j >= 30 - q), against a random pick from the same N; temperature,
RMS over all steps of mean(T) - mean(A) against the split-half noise of mean(A).
Multimodality: per frame and joint, 1- vs 2-component Gaussian fit (EM) of the 64 seeds' endpoint
(step 29); bimodal = delta BIC > 10, both weights > 0.1, separation > 2 pooled SD.

Outputs in --out (default steering/results/hover_causes/): windows.json, samples.npz, results.json,
causes.png, remedies.png, joints.png, wroll_hist.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plan_consistency import RUNS, analyse  # noqa: E402
from prompt_counterfactual import CHECKPOINT, ChunkSampler, frame_list, load_image, state_at  # noqa: E402
from so101_fk import SO101FK  # noqa: E402

STEERING = Path(__file__).parent
OUT = STEERING / "results" / "hover_causes"
TAGS = (
    "median_rtc_cap6_1",
    "median_rtc_cap6_no_color_prompt_6",
    "median_rtc_cap6_no_color_prompt_7",
    "median_rtc_cap6_no_color_prompt_8",
    "median_rtc_cap6_red_prompt_2_pens_1",
    "median_rtc_cap6_green_prompt_2_pens_2",
    "median_rtc_cap6_no_color_prompt_2_pens_2",
    "median_rtc_cap6_no_color_prompt_2_pens_3",
    "median_rtc_cap6_no_color_prompt_2_pens_4",
)
SHORT = ("pan", "lift", "elbow", "wflex", "wroll")
H = 30
G0 = 10  # first unguided step (execution_horizon on the arm)
N_A, N_T, N_R = 64, 32, 16
TEMPS = {"T7": 0.7, "T5": 0.5}
WINDOW = 8  # chunks per window
FRAME_TOL = 0.06  # s between t_start and the nearest recorded frame
TAIL_SEED = 100_000
BEST_N = (1, 4, 8, 16)
N_TRIALS = 300
C_COND = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948")
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#898781", "#e8e7e3"
C_HOVER, C_OTHER = "#2a78d6", "#eb6834"


def r3(x: Any) -> Any:
    """Round floats (recursively) for JSON; NaN -> None."""
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r3(v) for v in x]
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 3)
    return x


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def segments(labels: list[str], pairs: list[dict], label: str) -> list[list[int]]:
    """Runs of consecutive pairs with this label whose chunks chain (pair i+1's prev is pair i's k)."""
    out: list[list[int]] = []
    cur: list[int] = []
    for i, lab in enumerate(labels):
        chained = cur and pairs[i]["prev"] == pairs[cur[-1]]["k"]
        if lab == label and (not cur or chained):
            cur.append(i)
        else:
            if cur:
                out.append(cur)
            cur = [i] if lab == label else []
    if cur:
        out.append(cur)
    return out


def frame_entry(
    run_dir: Path, tag: str, meta: dict, chunks: dict, ticks: dict, rec: list[dict], k: int
) -> dict:
    """Chunk k's observation: the recorded frame nearest t_start, the state of the first tick after it."""
    times = np.asarray([f["t"] for f in rec])
    ts = float(chunks["t_start"][k])
    i = int(np.argmin(np.abs(times - ts)))
    if abs(times[i] - ts) > FRAME_TOL:
        raise RuntimeError(f"{tag} chunk {k}: nearest frame {times[i] - ts:+.3f} s away")
    state, t_tick = state_at(ticks, float(times[i]))
    nxt = i + 1 if i + 1 < len(rec) else None
    return {
        "run": run_dir.name, "run_dir": str(run_dir), "tag": tag, "task": meta["task"], "chunk": k,
        "t_start": ts, "file": rec[i]["file"], "t": float(times[i]), "state": [float(v) for v in state],
        "tick_t": t_tick,
        "next_file": rec[nxt]["file"] if nxt is not None else None,
        "next_t": float(times[nxt]) if nxt is not None else None,
        "next_state": [float(v) for v in state_at(ticks, float(times[nxt]))[0]] if nxt is not None else None,
    }  # fmt: skip


def plan_windows() -> tuple[list[dict], list[dict]]:
    """Frames (one per chunk observation) and windows (pairs between consecutive frames)."""
    fk = SO101FK()
    frames: list[dict] = []
    windows: list[dict] = []
    fkey: dict[tuple[str, int], int] = {}
    for tag in TAGS:
        run_dir = next(d for d in sorted(RUNS.iterdir()) if d.name.endswith("_" + tag))
        a = analyse(run_dir, fk)
        pairs = a["pairs"]
        labels = [p["label"] for p in pairs]
        ticks = dict(np.load(run_dir / "ticks.npz"))
        chunks = dict(np.load(run_dir / "chunks.npz"))
        meta = json.loads((run_dir / "meta.json").read_text())
        rec = frame_list(run_dir, meta)
        for label in ("hover", "other"):
            segs = segments(labels, pairs, label)
            if not segs or len(max(segs, key=len)) < WINDOW - 1:
                print(f"  {tag}: no {label} segment of {WINDOW - 1} pairs")
                continue
            seg = max(segs, key=len)
            mid = len(seg) // 2
            sel = seg[max(0, mid - (WINDOW - 1) // 2) :][: WINDOW - 1]
            chain = [pairs[sel[0]]["prev"]] + [pairs[i]["k"] for i in sel]

            fidx = []
            for k in chain:
                if (run_dir.name, k) not in fkey:
                    fkey[(run_dir.name, k)] = len(frames)
                    frames.append(frame_entry(run_dir, tag, meta, chunks, ticks, rec, k))
                fidx.append(fkey[(run_dir.name, k)])
            wp = []
            for n, i in enumerate(sel):
                p = pairs[i]
                wp.append({
                    "k": p["k"], "prev": p["prev"], "q": p["q"], "drop": p["drop"],
                    "delay": int(chunks["inference_delay"][p["k"]]),
                    "fp": fidx[n], "fk": fidx[n + 1],
                    "measured_json": float(np.sqrt(np.mean(np.square(p["unguided"])))),
                })  # fmt: skip
            windows.append({"run": run_dir.name, "tag": tag, "label": label, "frames": fidx, "pairs": wp,
                            "t_from_start": float(pairs[sel[0]]["t"])})  # fmt: skip
    for w in windows:
        for p in w["pairs"]:
            frames[p["fk"]].setdefault("pair_of", []).append(
                [w["label"], p["prev"], p["q"], p["delay"], p["fp"]]
            )
            if w["label"] == "hover":
                frames[p["fk"]]["want_next"] = True
    return frames, windows


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class Sampler(ChunkSampler):
    """ChunkSampler with the VLM pass split from the flow steps, x0 given, and the RTC-guided path."""

    def __init__(self, checkpoint: Path):
        """Load as ChunkSampler; no parameter needs a gradient (RTC's autograd only touches x_t)."""
        super().__init__(checkpoint)
        for prm in self.policy.parameters():
            prm.requires_grad_(False)

    def batch_ng(self, image: np.ndarray, state: np.ndarray, task: str) -> dict:
        """Preprocessed batch built under no_grad (not inference_mode: RTC guidance needs autograd)."""
        from molmo_common import preprocess
        from prompt_counterfactual import IMAGE_KEY

        obs = {IMAGE_KEY: image, "observation.state": np.asarray(state, np.float32)}
        with self.torch.no_grad():
            return preprocess(self.pre, obs, task, self.device)

    def encode(self, batch: dict) -> dict:
        """One VLM pass: depth-gated KV cache and encoder mask, as _generate_actions_from_inputs_with_rtc."""
        torch, p = self.torch, self.policy
        model_inputs = p._model_inputs(batch)
        with torch.no_grad(), p._autocast_context():
            backbone = p._backbone()
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
        return {"kv": kv, "mask": enc_mask, "pad": batch.get("action_dim_is_pad"),
                "action_dim": p._output_action_dim(batch)}  # fmt: skip

    def noise(self, seeds: list[int]):
        """(len(seeds), H, max_dim) float32: what a stock batch-1 call with Generator(seed) draws."""
        torch = self.torch
        bb = self.policy._backbone()
        h, d = self.policy._generation_action_horizon(), int(bb.config.max_action_dim)
        return torch.cat([
            torch.randn(1, h, d, device=self.device, dtype=torch.float32,
                        generator=torch.Generator(device=self.device).manual_seed(int(s)))
            for s in seeds
        ])  # fmt: skip

    def flow(
        self, enc: dict, x0, prev=None, delay: int = 0, horizon: int | None = None, seed_batch: int = 16
    ):
        """Raw normalised chunks (B, n_action_steps, action_dim) from initial noise x0 (B, H, max_dim).

        The loop of MolmoAct2Policy._generate_actions_from_inputs_with_rtc, the rtc_processor included:
        with prev=None its denoise_step returns the plain velocity.
        """
        from lerobot.policies.molmoact2.modeling_molmoact2 import _mask_action_dim_tensor

        torch, p = self.torch, self.policy
        expert = p._action_expert()
        steps = int(
            getattr(p.config, "num_inference_steps", None) or p._backbone().config.flow_matching_num_steps
        )
        mask_pad, pad = p.config.mask_action_dim_padding, enc["pad"]
        outs = []
        with torch.no_grad(), p._autocast_context():
            for i in range(0, x0.shape[0], seed_batch):
                traj = x0[i : i + seed_batch].clone()
                b = traj.shape[0]
                kv_b = [(k.expand(b, *k.shape[1:]), v.expand(b, *v.shape[1:])) for k, v in enc["kv"]]
                mask_b = None if enc["mask"] is None else enc["mask"].expand(b, *enc["mask"].shape[1:])
                if mask_pad:
                    traj = _mask_action_dim_tensor(traj, pad)
                ctx = expert.prepare_context(encoder_kv_states=kv_b, encoder_attention_mask=mask_b,
                                             state_embeddings=None, batch_size=b, seq_len=traj.shape[1],
                                             device=traj.device, dtype=traj.dtype)  # fmt: skip
                ts = [torch.full((b,), k / steps, device=traj.device, dtype=traj.dtype) for k in range(steps)]
                mods = expert.get_or_prepare_modulation_cache(
                    ts, cache_key=(steps, b, traj.device, traj.dtype)
                )
                for k in range(steps):

                    def denoise(x, m=mods[k], ctx=ctx):  # noqa: ANN001, ANN202
                        vel = expert.forward_with_context(x, m.conditioning, context=ctx, modulation=m)
                        return _mask_action_dim_tensor(vel, pad) if mask_pad else vel

                    rtc_v = p.rtc_processor.denoise_step(
                        x_t=traj, prev_chunk_left_over=prev, inference_delay=int(delay),
                        time=1.0 - float(ts[k][0].item()), original_denoise_step_partial=lambda x: -denoise(x),
                        execution_horizon=horizon,
                    )  # fmt: skip
                    traj = traj + (1.0 / steps) * (-rtc_v)
                    if mask_pad:
                        traj = _mask_action_dim_tensor(traj, pad)
                outs.append(traj[:, : p.config.n_action_steps, : enc["action_dim"]].to(dtype=torch.float32))
        return torch.cat(outs)

    def stock_rtc(self, batch: dict, seed: int, prev=None, delay: int = 0, horizon: int | None = None):
        """Stock predict_action_chunk at batch 1 (RTC path), under no_grad as the RTC engine allows."""
        torch = self.torch
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        kw: dict[str, Any] = {"generator": g}
        if prev is not None:
            kw.update(inference_delay=delay, prev_chunk_left_over=prev, execution_horizon=horizon)
        with torch.no_grad():
            return self.policy.predict_action_chunk(batch, **kw)


def leftover(sampler: Sampler, chunk_norm: np.ndarray, q: int, horizon: int):
    """The RTC engine's prev_chunk_left_over: the queue's raw remainder, padded/truncated to the horizon."""
    from lerobot.rollout.inference.rtc import _normalize_prev_actions_length

    t = sampler.torch.as_tensor(chunk_norm[q:], dtype=sampler.torch.float32, device=sampler.device)
    return _normalize_prev_actions_length(t, target_steps=horizon)


def shifted(sampler: Sampler, seeds: list[int], q: int):
    """Tick-aligned noise: seed s's x0 advanced by q steps, fresh noise (seed TAIL_SEED + s) at the end."""
    base = sampler.noise(seeds)
    if q <= 0:
        return base
    tail = sampler.noise([TAIL_SEED + s for s in seeds])
    return sampler.torch.cat([base[:, q:], tail[:, :q]], dim=1)


def validate(sampler: Sampler, frames: list[dict], windows: list[dict]) -> dict:
    """Batch-1 flows against stock predict_action_chunk; seed-batching floor."""
    torch = sampler.torch
    w = next(w for w in windows if w["label"] == "hover")
    pr = w["pairs"][0]
    f = frames[pr["fk"]]
    img = load_image(Path(f["run_dir"]), f)
    batch = sampler.batch_ng(img, np.asarray(f["state"], np.float32), f["task"])
    enc = sampler.encode(batch)
    cn = np.load(Path(f["run_dir"]) / "chunks.npz")["chunk_norm"]
    out: dict[str, Any] = {}
    seeds = [0, 1, 2]
    for name, h in (("no_prefix", None), ("rtc_h10", 10), ("rtc_h20", 20)):
        prev = None if h is None else leftover(sampler, cn[pr["prev"]], pr["q"], h)
        mine = torch.cat([sampler.flow(enc, sampler.noise([s]), prev, pr["delay"], h, 1) for s in seeds])
        stock = torch.cat([sampler.stock_rtc(batch, s, prev, pr["delay"], h) for s in seeds])
        out[f"{name}_batch1_vs_stock_max_abs_raw"] = float((mine - stock).abs().max())
    x0 = sampler.noise(list(range(16)))
    a = sampler.postprocess(sampler.flow(enc, x0, seed_batch=16))
    b = sampler.postprocess(sampler.flow(enc, x0, seed_batch=8))
    c = sampler.postprocess(sampler.flow(enc, x0, seed_batch=16))
    out["seed_batch_16_vs_8_D_deg_median"] = float(np.median(dmat_same(a, b, G0, H - pr["q"]).diagonal()))
    out["seed_batch_16_vs_8_max_abs_deg"] = float(np.abs(a - b).max())
    out["repeat_identical"] = bool(np.array_equal(a, c))
    print("validation:", out)
    return out


def sample_all(sampler: Sampler, frames: list[dict], seed_batch: int) -> dict[str, np.ndarray]:
    """Every sample set of every frame (arm frame, deg); NaN where a set does not apply."""
    nf = len(frames)
    sets = {"A": N_A, "T7": N_T, "T5": N_T, "S": N_T, "R10": N_R, "R20": N_R, "B": N_T, "Bs": N_T}
    res = {k: np.full((nf, n, H, 6), np.nan, np.float32) for k, n in sets.items()}
    shift_b = np.zeros(nf, np.int64)
    torch = sampler.torch
    chunk_norm: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        run_dir = Path(f["run_dir"])
        if f["run"] not in chunk_norm:
            chunk_norm[f["run"]] = np.load(run_dir / "chunks.npz")["chunk_norm"]
        batch = sampler.batch_ng(load_image(run_dir, f), np.asarray(f["state"], np.float32), f["task"])
        enc = sampler.encode(batch)
        x0 = sampler.noise(list(range(N_A)))
        res["A"][i] = sampler.postprocess(sampler.flow(enc, x0, seed_batch=seed_batch))
        for name, tau in TEMPS.items():
            res[name][i] = sampler.postprocess(sampler.flow(enc, tau * x0[:N_T], seed_batch=seed_batch))
        if f.get("pair_of"):
            _, prev, q, delay, _ = f["pair_of"][0]
            res["S"][i] = sampler.postprocess(sampler.flow(enc, shifted(sampler, list(range(N_T)), q),
                                                           seed_batch=seed_batch))  # fmt: skip
            for name, h in (("R10", 10), ("R20", 20)):
                lo = leftover(sampler, chunk_norm[f["run"]][prev], q, h)
                res[name][i] = sampler.postprocess(sampler.flow(enc, x0[:N_R], lo, delay, h, seed_batch))
        if f.get("want_next") and f.get("next_file"):
            nb = sampler.batch_ng(load_image(run_dir, {"file": f["next_file"]}),
                                  np.asarray(f["next_state"], np.float32), f["task"])  # fmt: skip
            ne = sampler.encode(nb)
            sh = int(round((f["next_t"] - f["t"]) * 30.0))
            shift_b[i] = sh
            res["B"][i] = sampler.postprocess(sampler.flow(ne, x0[:N_T], seed_batch=seed_batch))
            res["Bs"][i] = sampler.postprocess(sampler.flow(ne, shifted(sampler, list(range(N_T)), sh),
                                                            seed_batch=seed_batch))  # fmt: skip
        del enc
        torch.cuda.empty_cache() if i % 20 == 19 else None
        print(f"  frame {i + 1}/{nf} {f['tag'][-24:]} chunk {f['chunk']} ({time.perf_counter() - t0:.0f} s)")
    res["shift_b"] = shift_b
    return res


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def dmat(a: np.ndarray, b: np.ndarray, q: int, j1: int, per_joint: bool = False) -> np.ndarray:
    """D between every row of a (frame k) and every row of b (frame p, its step j + q), j in [G0, j1)."""
    x = a[:, G0:j1, :5].astype(np.float64)
    y = b[:, G0 + q : j1 + q, :5].astype(np.float64)
    d2 = np.square(x[:, None] - y[None])  # (na, nb, steps, 5)
    return np.sqrt(d2.mean(axis=2)) if per_joint else np.sqrt(d2.mean(axis=(2, 3)))


def dmat_same(a: np.ndarray, b: np.ndarray, j0: int, j1: int) -> np.ndarray:
    """D between rows of two sets from the same observation, steps [j0, j1)."""
    x, y = a[:, j0:j1, :5].astype(np.float64), b[:, j0:j1, :5].astype(np.float64)
    return np.sqrt(np.square(x[:, None] - y[None]).mean(axis=(2, 3)))


def offdiag(m: np.ndarray) -> np.ndarray:
    """Off-diagonal entries (all of them for a non-square matrix)."""
    if m.shape[0] != m.shape[1]:
        return m.reshape(-1, *m.shape[2:])
    keep = ~np.eye(m.shape[0], dtype=bool)
    return m[keep]


def pair_readouts(s: dict, frames: list[dict], pr: dict, rec: np.ndarray) -> dict:
    """All readouts of one pair p -> k (see the module docstring)."""
    fp, fk, q = pr["fp"], pr["fk"], pr["q"]
    j1 = H - q
    a_k, a_p = s["A"][fk], s["A"][fp]
    rk, rp = rec[pr["k"]], rec[pr["prev"]]
    out: dict[str, Any] = {"q": q, "drop": pr["drop"], "steps": j1 - G0}
    out["measured"] = float(dmat(rk[None], rp[None], q, j1)[0, 0])
    out["measured_per_joint"] = dmat(rk[None], rp[None], q, j1, True)[0, 0]
    same = dmat(a_k, a_k, 0, j1)
    out["seed_spread"] = float(np.median(same[np.triu_indices(N_A, 1)]))
    pj = dmat(a_k, a_k, 0, j1, True)
    out["seed_spread_per_joint"] = np.median(pj[np.triu_indices(N_A, 1)], axis=0)
    cross = dmat(a_k, a_p, q, j1)
    out["fixed_index"] = float(np.median(cross.diagonal()))
    out["fresh"] = float(np.median(offdiag(cross)))
    out["fixed_tick"] = float(np.median(dmat(s["S"][fk], a_p[:N_T], q, j1).diagonal()))
    pj = dmat(s["S"][fk], a_p[:N_T], q, j1, True)
    out["fixed_tick_per_joint"] = np.median(pj[np.arange(N_T), np.arange(N_T)], axis=0)
    out["fresh_per_joint"] = np.median(offdiag(dmat(a_k, a_p, q, j1, True)), axis=0)
    for name in TEMPS:
        c = dmat(s[name][fk], s[name][fp], q, j1)
        out[f"fresh_{name}"] = float(np.median(offdiag(c)))
        out[f"spread_{name}"] = float(
            np.median(dmat(s[name][fk], s[name][fk], 0, j1)[np.triu_indices(N_T, 1)])
        )
    out["free_vs_rec"] = float(np.median(dmat(a_k, rp[None], q, j1)[:, 0]))
    for name in ("R10", "R20"):
        d = dmat(s[name][fk], rp[None], q, j1)[:, 0]
        out[f"{name.lower()}_vs_rec"] = float(np.median(d))
        out[f"{name.lower()}_vs_rec_all"] = d
        # best of the guided seeds by the same overlap criterion as the chains
        full = np.sqrt(np.square(s[name][fk][:, :j1, :5] - rp[None, q:, :5]).mean(axis=(1, 2)))
        out[f"{name.lower()}_best16_vs_rec"] = float(d[np.argmin(full)])
    g = dmat(s["R10"][fk], s["R10"][fk], 0, j1)
    out["guided_spread"] = float(np.median(g[np.triu_indices(N_R, 1)]))
    out["measured_pct_in_rtc10"] = float(100 * np.mean(out["r10_vs_rec_all"] <= out["measured"]))
    # cost of h20: the new steps beyond the old plan (j >= 30 - q), mean(R20) - mean(R10), against the
    # split-half noise of R10 (8 vs 8 seeds); and the commanded speed over the steps h20 pads by holding
    # the last leftover action (j in [30 - q, 20)), deg per step, arm joints 0-4.
    r10, r20 = s["R10"][fk][:, :, :5], s["R20"][fk][:, :, :5]
    out["tail_shift_r20_vs_r10"] = float(
        np.sqrt(np.mean(np.square(r20[:, j1:].mean(0) - r10[:, j1:].mean(0))))
    )
    out["tail_split_half_r10"] = float(
        np.sqrt(np.mean(np.square(r10[::2, j1:].mean(0) - r10[1::2, j1:].mean(0))))
    )
    # largest step-to-step wrist_roll change inside the executed window [drop, drop + 14): where h20 moves the seam
    dd = pr["drop"]
    for name, arr in (("r10", r10), ("r20", r20), ("free", a_k[:, :, :5])):
        out[f"max_exec_step_wroll_{name}"] = float(
            np.median(np.abs(np.diff(arr[:, dd : dd + 14, 4], axis=1)).max(1))
        )
    if j1 < 20:
        for name, arr in (("r10", r10), ("r20", r20), ("free", a_k[:, :, :5])):
            out[f"speed_hold_{name}"] = float(
                np.median(np.abs(np.diff(arr[:, j1 - 1 : 21], axis=1)).mean(axis=(1, 2)))
            )
    d = pr["drop"]
    if q + d < H:
        out["seam_wroll_measured"] = float(abs(rk[d, 4] - rp[q + d, 4]))
        for name, arr in (("free", a_k), ("r10", s["R10"][fk]), ("r20", s["R20"][fk])):
            out[f"seam_wroll_{name}"] = float(np.median(np.abs(arr[:, d, 4] - rp[q + d, 4])))
    if np.isfinite(s["B"][fk]).all():
        sh = int(s["shift_b"][fk])
        out["fixed_tick_0.1s"] = float(np.median(dmat(s["Bs"][fk], a_k[:N_T], sh, j1 - sh).diagonal()))
        out["fixed_index_0.1s"] = float(np.median(dmat(s["B"][fk], a_k[:N_T], sh, j1 - sh).diagonal()))
        out["seed_spread_0.1s_steps"] = float(
            np.median(dmat(a_k, a_k, 0, j1 - sh)[np.triu_indices(N_A, 1)])
        )  # same steps as the 0.1 s readouts, for scale
    # temperature plan shift (all steps): mean(T) - mean(A[:32]) against the difference of two independent
    # 32-seed means of A (an upper bound on the sampling noise of the paired shift)
    rng = np.random.default_rng(pr["k"])
    perm = rng.permutation(N_A)
    half = a_k[perm[: N_A // 2], :, :5].mean(0) - a_k[perm[N_A // 2 :], :, :5].mean(0)
    out["mean_split_half_rms"] = float(np.sqrt(np.mean(np.square(half))))
    for name in TEMPS:
        dm = s[name][fk][:, :, :5].mean(0) - a_k[:N_T, :, :5].mean(0)
        out[f"shift_{name}"] = float(np.sqrt(np.mean(np.square(dm))))
        out[f"shift_{name}_per_joint"] = np.sqrt(np.mean(np.square(dm), axis=0))
    return out


def best_of_n(s: dict, w: dict, n: int, trials: int, seed: int) -> dict:
    """Chains over one window: at each frame pick, among n random seeds, the one closest to the last pick."""
    rng = np.random.default_rng(seed)
    pools = [s["A"][f] for f in w["frames"]]
    dis, seam, shift_sel, shift_rnd = [], [], [], []
    bias = []
    for _ in range(trials):
        cur = pools[0][rng.integers(N_A)]
        for pr, pool in zip(w["pairs"], pools[1:], strict=True):
            q, j1 = pr["q"], H - pr["q"]
            idx = rng.choice(N_A, size=n, replace=False)
            cand = pool[idx]
            crit = np.sqrt(np.square(cand[:, :j1, :5] - cur[None, q:, :5]).mean(axis=(1, 2)))
            b = int(np.argmin(crit))
            sel = cand[b]
            dis.append(float(dmat(sel[None], cur[None], q, j1)[0, 0]))
            if q + pr["drop"] < H:
                seam.append(abs(float(sel[pr["drop"], 4] - cur[q + pr["drop"], 4])))
            if n > 1:
                rest = np.delete(cand, b, axis=0)[:, j1:, :5].mean(0)
                r = int(rng.integers(n))
                rest_r = np.delete(cand, r, axis=0)[:, j1:, :5].mean(0)
                shift_sel.append(float(np.sqrt(np.mean(np.square(sel[j1:, :5] - rest)))))
                shift_rnd.append(float(np.sqrt(np.mean(np.square(cand[r, j1:, :5] - rest_r)))))
                bias.append(sel[j1:, :5].mean(0) - pool[:, j1:, :5].mean(axis=(0, 1)))
            cur = sel
    out = {"D_median": float(np.median(dis)), "D_mean": float(np.mean(dis)),
           "seam_wroll_median": float(np.median(seam)) if seam else None}  # fmt: skip
    if n > 1:
        out["shift_selected_rms"] = float(np.median(shift_sel))
        out["shift_random_pick_rms"] = float(np.median(shift_rnd))
        out["bias_per_joint_abs_mean"] = np.abs(np.mean(bias, axis=0)).tolist()
    return out


def gmm2(x: np.ndarray, iters: int = 200) -> dict:
    """1- vs 2-component 1-D Gaussian mixture (EM); delta BIC (positive favours 2), weights, separation."""
    x = np.asarray(x, np.float64)
    n = len(x)
    sd1 = max(x.std(), 1e-3)
    ll1 = np.sum(-0.5 * np.log(2 * np.pi * sd1**2) - 0.5 * ((x - x.mean()) / sd1) ** 2)
    best = None
    for init in (np.percentile(x, [25, 75]), np.percentile(x, [10, 90]), np.percentile(x, [40, 60])):
        mu = init.astype(np.float64)
        sd = np.array([sd1, sd1]) / 2
        wgt = np.array([0.5, 0.5])
        for _ in range(iters):
            pdf = wgt / np.sqrt(2 * np.pi * sd**2) * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2)
            tot = pdf.sum(1, keepdims=True) + 1e-300
            r = pdf / tot
            nk = r.sum(0) + 1e-9
            wgt = nk / n
            mu = (r * x[:, None]).sum(0) / nk
            sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk)
            sd = np.maximum(sd, 0.05 * sd1 + 1e-3)
        pdf = wgt / np.sqrt(2 * np.pi * sd**2) * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2)
        ll2 = float(np.sum(np.log(pdf.sum(1) + 1e-300)))
        if best is None or ll2 > best[0]:
            best = (ll2, mu.copy(), sd.copy(), wgt.copy())
    ll2, mu, sd, wgt = best
    dbic = (2 * np.log(n) - 2 * ll1) - (5 * np.log(n) - 2 * ll2)
    sep = abs(mu[0] - mu[1]) / np.sqrt(np.mean(sd**2))
    return {"delta_bic": float(dbic), "weights": wgt.tolist(), "means": mu.tolist(), "sds": sd.tolist(),
            "separation": float(sep), "bimodal": bool(dbic > 10 and wgt.min() > 0.1 and sep > 2)}  # fmt: skip


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def med(xs: list) -> float | None:
    """Median ignoring None/NaN."""
    v = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def analyse_all(s: dict, frames: list[dict], windows: list[dict]) -> dict:
    """Per pair, per window and per label summaries."""
    recs: dict[str, np.ndarray] = {}
    per_pair = []
    for w in windows:
        if w["run"] not in recs:
            recs[w["run"]] = np.load(RUNS / w["run"] / "chunks.npz")["chunk_arm"]
        for pr in w["pairs"]:
            r = pair_readouts(s, frames, pr, recs[w["run"]])
            r.update(run=w["run"], tag=w["tag"], label=w["label"], k=pr["k"], prev=pr["prev"],
                     measured_json=pr["measured_json"])  # fmt: skip
            per_pair.append(r)
    keys = ["measured", "r10_vs_rec", "r20_vs_rec", "free_vs_rec", "seed_spread", "guided_spread", "fixed_index",
            "fixed_tick", "fixed_tick_0.1s", "fixed_index_0.1s", "seed_spread_0.1s_steps", "fresh", "fresh_T7",
            "fresh_T5", "spread_T7", "spread_T5", "r10_best16_vs_rec", "r20_best16_vs_rec", "measured_pct_in_rtc10",
            "seam_wroll_measured", "seam_wroll_free", "seam_wroll_r10", "seam_wroll_r20", "shift_T7", "shift_T5",
            "mean_split_half_rms", "tail_shift_r20_vs_r10", "tail_split_half_r10", "speed_hold_r10",
            "speed_hold_r20", "speed_hold_free", "max_exec_step_wroll_r10", "max_exec_step_wroll_r20",
            "max_exec_step_wroll_free"]  # fmt: skip
    by_label: dict[str, Any] = {}
    for lab in ("hover", "other"):
        pp = [p for p in per_pair if p["label"] == lab]
        entry: dict[str, Any] = {"n_pairs": len(pp), "n_runs": len({p["run"] for p in pp})}
        for k in keys:
            entry[k] = med([p.get(k) for p in pp])
            # run-level: median of per-run medians (runs are the independent units)
            runs = sorted({p["run"] for p in pp})
            entry[k + "_run_medians"] = [med([p.get(k) for p in pp if p["run"] == r]) for r in runs]
        for k in ("measured_per_joint", "seed_spread_per_joint", "fixed_tick_per_joint", "fresh_per_joint",
                  "shift_T7_per_joint", "shift_T5_per_joint"):  # fmt: skip
            entry[k] = dict(zip(SHORT, np.median([p[k] for p in pp], axis=0).tolist(), strict=True))
        entry["ratio_seed_spread_to_measured"] = med([p["seed_spread"] / p["measured"] for p in pp])
        entry["ratio_fixed_tick_to_measured"] = med([p["fixed_tick"] / p["measured"] for p in pp])
        entry["ratio_fixed_tick_to_seed_spread"] = med([p["fixed_tick"] / p["seed_spread"] for p in pp])
        by_label[lab] = entry
    # best-of-N chains
    chains: dict[str, Any] = {}
    for lab in ("hover", "other"):
        ws = [w for w in windows if w["label"] == lab]
        chains[lab] = {}
        for n in BEST_N:
            res = [best_of_n(s, w, n, N_TRIALS, seed=1000 * n + i) for i, w in enumerate(ws)]
            entry = {"D_median_over_windows": med([r["D_median"] for r in res]),
                     "seam_wroll_median_over_windows": med([r["seam_wroll_median"] for r in res]),
                     "per_window_D": [r["D_median"] for r in res]}  # fmt: skip
            if n > 1:
                entry["shift_selected_rms"] = med([r["shift_selected_rms"] for r in res])
                entry["shift_random_pick_rms"] = med([r["shift_random_pick_rms"] for r in res])
                entry["bias_per_joint_abs_mean"] = dict(zip(SHORT, np.median([r["bias_per_joint_abs_mean"]
                                                                              for r in res], axis=0).tolist(),
                                                            strict=True))  # fmt: skip
            chains[lab][f"N{n}"] = entry
    # multimodality of the endpoint across seeds
    modes: dict[str, Any] = {}
    frame_label: dict[int, str] = {}
    for w in windows:
        for f in w["frames"]:
            frame_label.setdefault(f, w["label"])
    per_frame_modes = {}
    for f in frame_label:
        per_frame_modes[f] = {j: gmm2(s["A"][f][:, -1, j]) for j in range(5)}
    for lab in ("hover", "other"):
        fs = [f for f, la in frame_label.items() if la == lab]
        modes[lab] = {
            "n_frames": len(fs),
            "bimodal_fraction_endpoint": {
                SHORT[j]: float(np.mean([per_frame_modes[f][j]["bimodal"] for f in fs])) for j in range(5)
            },  # fmt: skip
            "endpoint_sd_median": {
                SHORT[j]: float(np.median([s["A"][f][:, -1, j].std() for f in fs])) for j in range(5)
            },  # fmt: skip
            "wroll_mode_gap_median_bimodal": med(
                [
                    abs(np.diff(per_frame_modes[f][4]["means"])[0])
                    for f in fs
                    if per_frame_modes[f][4]["bimodal"]
                ]
            ),  # fmt: skip
        }
    return {"by_label": by_label, "best_of_n": chains, "multimodality": modes, "per_pair": per_pair,
            "per_frame_modes": {str(f): {SHORT[j]: m for j, m in v.items()} for f, v in per_frame_modes.items()}}  # fmt: skip


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", color=C_GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C_MUTED)
    ax.tick_params(colors=C_INK, labelsize=8)


def strip(
    ax: plt.Axes, res: dict, rows: list[tuple[str, str]], title: str, extra: dict | None = None
) -> None:
    """Per pair dots, run medians and the overall median per condition; hover vs other offset."""
    pp = res["per_pair"]
    rng = np.random.default_rng(0)
    for y, (key, _name) in enumerate(rows):
        for off, lab, col in ((-0.17, "hover", C_HOVER), (0.17, "other", C_OTHER)):
            if extra is not None and key in extra:
                vals = extra[key][lab]
            else:
                vals = [p[key] for p in pp if p["label"] == lab and p.get(key) is not None]
            if not vals:
                continue
            ax.scatter(vals, y + off + rng.uniform(-0.06, 0.06, len(vals)), s=9, color=col, alpha=0.35, lw=0)
            m = float(np.median(vals))
            ax.plot(m, y + off, "D", ms=8, color=col, mec="white", mew=1.2)
            ax.text(m, y + off - 0.2, f"{m:.1f}", fontsize=7, color=C_INK, ha="center", va="bottom")
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("disagreement: RMS over steps 10..29-q and arm joints 0-4 (deg)", fontsize=8, color=C_MUTED)
    ax.set_title(title, fontsize=10, loc="left", color=C_INK)
    ax.plot([], [], "D", color=C_HOVER, mec="white", label="hover windows (median, dots = pairs)")
    ax.plot([], [], "D", color=C_OTHER, mec="white", label="other moving windows")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)


def plot_causes(res: dict, out: Path) -> None:
    """H1 vs H2: measured, reproduction, seed spread, noise held fixed."""
    rows = [
        ("measured", "measured on the arm (chunk k vs chunk p)"),
        ("r10_vs_rec", "offline RTC h10 on frame k vs recorded chunk p"),
        ("free_vs_rec", "offline unguided on frame k vs recorded chunk p"),
        ("seed_spread", "H1: two seeds, same frame (unguided)"),
        ("guided_spread", "H1: two seeds, same frame (RTC h10)"),
        ("fresh", "fresh noise, frames p -> k (unguided)"),
        ("fixed_index", "same seed, frames p -> k, index-aligned noise"),
        ("fixed_tick", "H2: same noise, tick-aligned, frames p -> k"),
        ("fixed_tick_0.1s", "H2 control: same noise, frame k -> +0.1 s"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    strip(ax, res, rows, "Where the chunk-to-chunk disagreement comes from")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_remedies(res: dict, out: Path) -> None:
    """Remedies: disagreement (left) and plan shift (right)."""
    chains = res["best_of_n"]
    extra = {
        f"bo{n}": {lab: chains[lab][f"N{n}"]["per_window_D"] for lab in ("hover", "other")} for n in BEST_N
    }
    rows = [
        ("measured", "measured on the arm (RTC h10, fresh noise)"),
        ("bo1", "fresh noise, unguided chain (N=1)"),
        ("fixed_index", "(a) one noise sample per episode"),
        ("fresh_T7", "(b) temperature 0.7, fresh noise"),
        ("fresh_T5", "(b) temperature 0.5, fresh noise"),
        ("bo8", "(c) best-of-8 chain (unguided)"),
        ("bo16", "(c) best-of-16 chain (unguided)"),
        ("r10_vs_rec", "(d) RTC h10 offline vs recorded p"),
        ("r20_vs_rec", "(d) RTC h20 offline vs recorded p"),
        ("r10_best16_vs_rec", "(c+d) RTC h10 + best-of-16 vs recorded p"),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
    strip(ax, res, rows, "Remedies, offline: disagreement (best-of-N dots = windows)", extra)
    bl = res["by_label"]
    costs = [
        ("T7: |mean(T0.7) - mean(T1)|", "shift_T7", None),
        ("T5: |mean(T0.5) - mean(T1)|", "shift_T5", None),
        ("two independent 32-seed means (noise)", "mean_split_half_rms", None),
        ("best-of-8 pick vs unpicked", None, ("N8", "shift_selected_rms")),
        ("random pick of 8 vs the rest", None, ("N8", "shift_random_pick_rms")),
        ("best-of-16 pick vs unpicked", None, ("N16", "shift_selected_rms")),
        ("random pick of 16 vs the rest", None, ("N16", "shift_random_pick_rms")),
    ]
    for y, (_name, key, ch) in enumerate(costs):
        for off, lab, col in ((-0.15, "hover", C_HOVER), (0.15, "other", C_OTHER)):
            v = bl[lab][key] if key else chains[lab][ch[0]][ch[1]]
            if v is None:
                continue
            bx.barh(y + off, v, height=0.28, color=col)
            bx.text(v + 0.05, y + off, f"{v:.2f}", fontsize=7, va="center", color=C_INK)
    bx.set_yticks(range(len(costs)), [c[0] for c in costs], fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel("RMS (deg): temperature over all 30 steps;\nbest-of-N over the new steps j >= 30-q",
                  fontsize=8, color=C_MUTED)  # fmt: skip
    bx.set_title("Cost: how far the remedy moves the plan", fontsize=10, loc="left", color=C_INK)
    _style(bx)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_joints(res: dict, out: Path) -> None:
    """Per joint: measured, seed spread, fixed-tick, fresh; hover and other."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    keys = (("measured_per_joint", "measured on the arm"), ("seed_spread_per_joint", "two seeds, same frame"),
            ("fixed_tick_per_joint", "same noise, frames p -> k"), ("fresh_per_joint", "fresh noise, p -> k"))  # fmt: skip
    x = np.arange(5)
    for ax, lab in zip(axes, ("hover", "other"), strict=True):
        e = res["by_label"][lab]
        for i, (k, name) in enumerate(keys):
            v = [e[k][j] for j in SHORT]
            ax.bar(x + (i - 1.5) * 0.2, v, width=0.18, color=C_COND[i], label=name)
        ax.set_xticks(x, SHORT, fontsize=8)
        ax.set_title(f"{lab} windows ({e['n_pairs']} pairs): per-joint RMS over steps 10..29-q", fontsize=10,
                     loc="left")  # fmt: skip
        ax.grid(True, axis="y", color=C_GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("deg (median over pairs)", fontsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_wroll(s: dict, frames: list[dict], windows: list[dict], res: dict, out: Path) -> None:
    """wrist_roll endpoint across 64 seeds: one hover window (8 frames) and one other window."""
    hw = next(w for w in windows if w["label"] == "hover" and w["tag"] == "median_rtc_cap6_1")
    ow = next(w for w in windows if w["label"] == "other" and w["tag"] == "median_rtc_cap6_1")
    fig, axes = plt.subplots(2, 8, figsize=(17, 4.8), sharey="row")
    for row, w in enumerate((hw, ow)):
        rec = np.load(RUNS / w["run"] / "chunks.npz")["chunk_arm"]
        for c, f in enumerate(w["frames"]):
            ax = axes[row, c]
            v = s["A"][f][:, -1, 4]
            m = res["per_frame_modes"][str(f)]["wroll"]
            ax.hist(v, bins=16, color=C_HOVER if row == 0 else C_OTHER, alpha=0.85)
            ax.axvline(rec[frames[f]["chunk"]][-1, 4], color=C_INK, lw=1.5)
            ax.set_title(f"chunk {frames[f]['chunk']}{'  bimodal' if m['bimodal'] else ''}\ndBIC {m['delta_bic']:.0f}",
                         fontsize=8, loc="left")  # fmt: skip
            ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[row, 0].set_ylabel(f"{w['label']}\nseeds", fontsize=8)
    fig.suptitle("median_rtc_cap6_1: wrist_roll at step 29 across 64 seeds per observation (black line: the "
                 "chunk the arm actually predicted)", fontsize=10, x=0.01, ha="left")  # fmt: skip
    fig.supxlabel("wrist_roll command, deg (arm frame)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Plan, sample, analyse, plot."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="sample only the first N frames (timing probe)")
    ap.add_argument("--seed-batch", type=int, default=16)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.analyse_only:
        plan = json.loads((out / "windows.json").read_text())
        frames, windows = plan["frames"], plan["windows"]
        data = np.load(out / "samples.npz")
        s = {k: data[k] for k in data.files}
        extra = json.loads((out / "results.json").read_text()).get("run_info", {}) if (out / "results.json").is_file() \
            else {}  # fmt: skip
    else:
        frames, windows = plan_windows()
        print(f"{len(windows)} windows ({sum(w['label'] == 'hover' for w in windows)} hover), {len(frames)} frames, "
              f"{sum(len(w['pairs']) for w in windows)} pairs")  # fmt: skip
        for w in windows:
            print(f"  {w['tag']:<42} {w['label']:<6} chunks {[frames[f]['chunk'] for f in w['frames']]} "
                  f"measured {[round(p['measured_json'], 1) for p in w['pairs']]}")  # fmt: skip
        (out / "windows.json").write_text(json.dumps({"frames": frames, "windows": windows}, indent=1) + "\n")
        if args.plan_only:
            return
        sampler = Sampler(args.checkpoint)
        t0 = time.perf_counter()
        validation = validate(sampler, frames, windows)
        todo = frames[: args.max_frames] if args.max_frames else frames
        t1 = time.perf_counter()
        s = sample_all(sampler, todo, args.seed_batch)
        t_s = time.perf_counter() - t1
        extra = {"validation": validation, "model": sampler.load_info, "validation_s": round(t1 - t0, 1),
                 "sampling_s": round(t_s, 1), "per_frame_s": round(t_s / len(todo), 2)}  # fmt: skip
        # flow cost per seed batch (latency of extra candidates), measured on the first frame
        f = frames[0]
        enc = sampler.encode(sampler.batch_ng(load_image(Path(f["run_dir"]), f), np.asarray(f["state"], np.float32),
                                              f["task"]))  # fmt: skip
        torch = sampler.torch
        timing = {}
        for b in (1, 8, 16):
            x0 = sampler.noise(list(range(b)))
            sampler.flow(enc, x0, seed_batch=b)
            torch.cuda.synchronize()
            ta = time.perf_counter()
            for _ in range(5):
                sampler.flow(enc, x0, seed_batch=b)
            torch.cuda.synchronize()
            timing[f"flow_batch{b}_ms"] = round((time.perf_counter() - ta) / 5 * 1000, 1)
        ta = time.perf_counter()
        for _ in range(3):
            sampler.encode(sampler.batch_ng(load_image(Path(f["run_dir"]), f), np.asarray(f["state"], np.float32),
                                            f["task"]))  # fmt: skip
        torch.cuda.synchronize()
        timing["preprocess_plus_vlm_ms"] = round((time.perf_counter() - ta) / 3 * 1000, 1)
        extra["timing"] = timing
        print("timing:", timing)
        if args.max_frames:
            print(extra)
            return
        np.savez_compressed(out / "samples.npz", **s)
    res = analyse_all(s, frames, windows)
    result = {
        "question": "why do consecutive RTC chunks disagree while hovering: sampling noise (H1) or observation "
        "sensitivity (H2)?",
        "metric": "RMS over steps j in [10, 30-q) and arm joints 0-4 of chunk_k[j] - chunk_p[j+q], deg (arm frame); "
        "plan_consistency's unguided arm RMS",
        "run_info": extra,
        "n_windows": len(windows),
        "windows": [
            {k: w[k] for k in ("tag", "label", "t_from_start")}
            | {"chunks": [frames[f]["chunk"] for f in w["frames"]]}
            for w in windows
        ],  # fmt: skip
        **{k: v for k, v in res.items() if k != "per_frame_modes"},
    }
    (out / "results.json").write_text(json.dumps(r3(result), indent=1) + "\n")
    plot_causes(res, out / "causes.png")
    plot_remedies(res, out / "remedies.png")
    plot_joints(res, out / "joints.png")
    plot_wroll(s, frames, windows, res, out / "wroll_hist.png")
    for lab, e in res["by_label"].items():
        print(lab, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()
                    if not k.endswith("_run_medians") and not isinstance(v, dict)})  # fmt: skip
    for lab, c in res["best_of_n"].items():
        print("best-of-N", lab, {n: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()
                                     if k != "per_window_D"} for n, e in c.items()})  # fmt: skip
    print("modes", json.dumps(r3(res["multimodality"])))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
