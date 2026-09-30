"""Step 5 precursor: do the predicted action chunks separate by instruction on a fixed frame?

Same observation, several instructions, the same noise seeds for every instruction (so
differences between conditions are paired, not confounded by flow-matching noise).

    uv run python steering/frame_diagnostic.py                      # Ai2 sample frame (fruit + bowl)
    uv run python steering/frame_diagnostic.py --cam0 a.png --cam1 b.png --state s.json \
        --condition blue="pick up the blue pen" --condition red="pick up the red pen" --condition null=""

Reports, per pair of conditions, the paired between-condition distance against the
within-condition seed spread, a label-permutation p-value, and how separation grows along the chunk.
A null result is a separation ratio near 1 with a large p-value: the instruction moves the
prediction no more than resampling the noise does.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from molmo_common import load_config, load_policy, load_sample_observation, preprocess
from PIL import Image

RESULTS = Path(__file__).parent / "results"
# Ai2's sample frame holds an apple, lemon, strawberry, peach and a red bowl at distinct positions.
DEFAULT_CONDITIONS = {
    "lemon": "pick up the lemon",
    "apple": "pick up the apple",
    "strawberry": "pick up the strawberry",
    "peach": "pick up the peach",
    "null": "",
}


def rms(x: np.ndarray, axis=None) -> np.ndarray:
    """Root mean square over ``axis``."""
    return np.sqrt(np.mean(np.square(x), axis=axis))


def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sample test on the distance between condition means; a, b are (seeds, T, D)."""
    observed = rms(a.mean(0) - b.mean(0))
    pooled = np.concatenate([a, b])
    n = len(a)
    hits = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        hits += rms(pooled[idx[:n]].mean(0) - pooled[idx[n:]].mean(0)) >= observed
    return (hits + 1) / (n_perm + 1)


def parse_conditions(items: list[str] | None) -> dict[str, str]:
    """Parse repeated ``name=instruction`` arguments, falling back to the defaults."""
    if not items:
        return DEFAULT_CONDITIONS
    out = {}
    for item in items:
        name, _, text = item.partition("=")
        out[name] = text
    return out


def main() -> None:
    """Predict chunks for every condition with shared seeds and compare them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", help='name="instruction" (repeatable)')
    parser.add_argument("--cam0")
    parser.add_argument("--cam1")
    parser.add_argument("--state", help="JSON list of 6 joint positions, arm (LeRobot >= 0.5) frame")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--tag", default="sample")
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for mover direction")
    parser.add_argument(
        "--move-threshold", type=float, default=5.0, help="deg, max |joint change| over the chunk"
    )
    args = parser.parse_args()

    device = "cuda"
    conditions = parse_conditions(args.condition)
    config = load_config(device=device)
    policy, preprocessor, postprocessor = load_policy(config)
    observation = load_sample_observation(config)
    if args.cam0:
        observation["observation.images.cam0"] = np.asarray(Image.open(args.cam0).convert("RGB"))
    if args.cam1:
        observation["observation.images.cam1"] = np.asarray(Image.open(args.cam1).convert("RGB"))
    if args.state:
        observation["observation.state"] = np.asarray(json.loads(Path(args.state).read_text()), np.float32)

    def chunk(task: str, seed: int) -> np.ndarray:
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            batch = preprocess(preprocessor, observation, task, device)
            actions = postprocessor(policy.predict_action_chunk(batch, generator=generator))
        return torch.as_tensor(actions).squeeze(0).float().cpu().numpy()

    first = next(iter(conditions.values()))
    for s in range(3):  # CUDA-graph capture and warm-up
        chunk(first, 10_000 + s)
    deterministic = bool(np.allclose(chunk(first, 0), chunk(first, 0)))

    chunks = {
        name: np.stack([chunk(text, s) for s in range(args.seeds)]) for name, text in conditions.items()
    }
    rng = np.random.default_rng(0)

    within = {}
    for name, c in chunks.items():
        pairs = [rms(c[i] - c[j]) for i, j in itertools.combinations(range(len(c)), 2)]
        within[name] = float(np.mean(pairs))

    pairs_out = {}
    for a, b in itertools.combinations(chunks, 2):
        paired = rms(chunks[a] - chunks[b], axis=(1, 2))  # same seed, different instruction
        per_t = rms(chunks[a] - chunks[b], axis=(0, 2))  # separation along the chunk
        noise = 0.5 * (within[a] + within[b])
        pairs_out[f"{a}_vs_{b}"] = {
            "paired_rms_deg": round(float(paired.mean()), 3),
            "within_seed_spread_deg": round(noise, 3),
            "separation_ratio": round(float(paired.mean()) / noise, 2) if noise > 0 else None,
            "permutation_p": round(float(permutation_p(chunks[a], chunks[b], args.n_perm, rng)), 4),
            "per_timestep_rms_deg": [round(float(x), 2) for x in per_t],
            # Separation grows along the chunk, so the whole-chunk RMS above dilutes it. Per joint at the
            # last step: mean difference over the pooled seed std (Cohen's d).
            "final_step_effect_size": [
                round(float(x), 2)
                for x in (chunks[a][:, -1].mean(0) - chunks[b][:, -1].mean(0))
                / np.sqrt(0.5 * (chunks[a][:, -1].var(0) + chunks[b][:, -1].var(0)) + 1e-9)
            ],
            "final_action_mean_diff_deg": [
                round(float(x), 2) for x in chunks[a][:, -1].mean(0) - chunks[b][:, -1].mean(0)
            ],
        }

    # The chunk is bimodal per seed: the arm either holds still or starts a reach. Mean-based
    # statistics above then mostly measure how often it starts, so report that directly, and where
    # the seeds that do move are heading.
    movers = {}
    for name, c in chunks.items():
        delta = c[:, -1] - c[:, 0]  # (seeds, joints) displacement over the chunk
        moving = np.abs(delta[:, :5]).max(1) > args.move_threshold
        movers[name] = {
            "fraction_moving": round(float(moving.mean()), 3),
            "n_moving": int(moving.sum()),
            "mover_mean_delta_deg": [round(float(x), 2) for x in delta[moving].mean(0)]
            if moving.any()
            else None,
            "mover_std_delta_deg": [round(float(x), 2) for x in delta[moving].std(0)]
            if moving.any()
            else None,
        }

    # Directional separation between two instructions, among seeds that move: the distance between
    # their mean displacement vectors (arm joints only) and the shoulder_pan difference, each with a
    # bootstrap 95% CI. This is the pose-selection criterion (Appendix G), because "moves more often"
    # is not "moves towards the named object". The distance is descriptive only: a norm of noisy means
    # is biased upwards and its CI never reaches 0. The pass condition for the pen frame is the pan CI
    # excluding 0 in both counterbalanced layouts, with the sign flipping when the pens swap sides.
    direction = {}
    for a, b in itertools.combinations(chunks, 2):
        da = (chunks[a][:, -1] - chunks[a][:, 0])[:, :5]
        db = (chunks[b][:, -1] - chunks[b][:, 0])[:, :5]
        ma = np.abs(da).max(1) > args.move_threshold
        mb = np.abs(db).max(1) > args.move_threshold
        if ma.sum() < 5 or mb.sum() < 5:
            direction[f"{a}_vs_{b}"] = None
            continue
        da, db = da[ma], db[mb]
        boot_dist, boot_pan = [], []
        for _ in range(args.n_boot):
            sa = da[rng.integers(len(da), size=len(da))].mean(0)
            sb = db[rng.integers(len(db), size=len(db))].mean(0)
            boot_dist.append(np.linalg.norm(sa - sb))
            boot_pan.append(sa[0] - sb[0])
        direction[f"{a}_vs_{b}"] = {
            "n_moving": [int(len(da)), int(len(db))],
            "mover_vector_distance_deg": round(float(np.linalg.norm(da.mean(0) - db.mean(0))), 2),
            "mover_vector_distance_ci95": [round(float(x), 2) for x in np.percentile(boot_dist, [2.5, 97.5])],
            "mover_pan_diff_deg": round(float(da[:, 0].mean() - db[:, 0].mean()), 2),
            "mover_pan_diff_ci95": [round(float(x), 2) for x in np.percentile(boot_pan, [2.5, 97.5])],
        }

    result = {
        "frame": args.tag,
        "move_threshold_deg": args.move_threshold,
        "movers": movers,
        "mover_direction": direction,
        "conditions": conditions,
        "seeds": args.seeds,
        "same_seed_deterministic": deterministic,
        "within_condition_spread_deg": {k: round(v, 3) for k, v in within.items()},
        "pairs": pairs_out,
        "mean_final_action_deg": {
            k: [round(float(x), 2) for x in v[:, -1].mean(0)] for k, v in chunks.items()
        },
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"frame_diagnostic_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    np.savez(RESULTS / f"frame_diagnostic_{args.tag}_chunks.npz", **chunks)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
    for name, p in pairs_out.items():
        print(name, {k: v for k, v in p.items() if k != "per_timestep_rms_deg"})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
