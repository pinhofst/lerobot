"""Step 5 precursor: do the predicted action chunks separate by instruction on a fixed frame?

Same observation, several instructions, the same noise seeds for every instruction (so
differences between conditions are paired, not confounded by flow-matching noise).

    uv run python steering/frame_diagnostic.py                      # Ai2 sample frame (fruit + bowl)
    uv run python steering/frame_diagnostic.py --cam0 a.png --cam1 b.png --state s.json \
        --condition blue="pick up the blue pen" --condition red="pick up the red pen" --condition null=""

Reports, per pair of conditions, the paired between-condition distance against the
within-condition seed spread, a label-permutation p-value, and how separation grows along the chunk.
A null result is a separation ratio near 1 with a large p-value: the instruction moves the
prediction no more than resampling the noise does. `movers` and `mover_direction` are the readouts
that matter (see RUNLOG). `--cag-weights 1 1.5 2 3` adds counterfactual action guidance (final-action
variant, a = (1 - w) * a_null + w * a_cond on the raw normalised output) from the same model calls;
w = 1 reproduces the plain instruction bit for bit.

Clamping. The postprocessor clamps the normalised action to [-1, 1] before unnormalising. On Ai2's
sample frame that clamp is not what holds the arm still: in the plain run it changes arm actions by
at most ~1.2 deg and leaves the mover counts unchanged but for one peach seed (26 vs 27). The ~7.8 deg elbow jump from
observation.state to chunk[0] comes from the preprocessor instead: `molmoact2_clamp_normalized` clips
the normalised state to [-1, 1] (elbow_flex 1.123 -> 1, shoulder_lift 1.055 -> 1), so the model sees
the arm at the edge of its trained range, not where it is. As a sanity check every readout is still
computed twice from the same raw chunks: `clamped` (the stock postprocessor: what the robot receives)
and `unclamped` (the same pipeline without the action clamp step). A joint is "pinned" when
|raw normalised| >= 1.

The "null" condition is not an empty prompt: the processor still builds "The task is to . The setup
is <setup_start>...<setup_end>. The current state of the robot is ... Given these, what action
should the robot take to complete the task?", a sentence never seen in training. It is an
out-of-distribution null, which CAG then extrapolates away from.

Output JSON, top-level keys:
    clamp                     per condition (and per CAG weight): fraction of steps / first actions
                              with any arm joint (0-4) pinned, and per-joint step fractions
    clamped, unclamped        each {movers, mover_direction, steps, within_condition_spread_deg,
                              pairs, mean_final_action_deg, cag: {w: {movers, mover_direction, steps}}}
    frame, conditions, seeds, move_threshold_deg, state_arm_frame, same_seed_deterministic
A seed "moves" when |chunk[-1] - chunk[0]| exceeds the threshold on any of joints 0-3 (wrist_roll is
excluded); `trigger_joint_counts` says which joint had the largest displacement among movers.
`steps` gives, over arm joints 0-4, the largest step inside the chunk and the largest jump from
observation.state to chunk[0], per condition and as `_all`, the maximum over the non-null ones (so
plain and CAG runs compare like for like). Bootstrap and permutation resampling use generators seeded per pair,
so CAG at w = 1 reproduces the plain CIs. The npz holds `raw_<cond>` (normalised policy output),
`clamped_<cond>`, `unclamped_<cond>`, `cag_w<w>_raw_<cond>`, `cag_w<w>_{clamped,unclamped}_<cond>`
and `state`.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from molmo_common import load_config, load_policy, load_sample_observation, preprocess
from PIL import Image

from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2ClampActionProcessorStep

RESULTS = Path(__file__).parent / "results"
# Ai2's sample frame holds an apple, lemon, strawberry, peach and a red bowl at distinct positions.
DEFAULT_CONDITIONS = {
    "lemon": "pick up the lemon",
    "apple": "pick up the apple",
    "strawberry": "pick up the strawberry",
    "peach": "pick up the peach",
    "null": "",
}
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = 5  # joints 0-4; arm readouts exclude the gripper
MOVE = 4  # joints 0-3; the moving test also excludes wrist_roll
SEED = 0  # base seed for every bootstrap / permutation generator


def rms(x: np.ndarray, axis=None) -> np.ndarray:
    """Root mean square over ``axis``."""
    return np.sqrt(np.mean(np.square(x), axis=axis))


def pair_rng(seed: int, key: str) -> np.random.Generator:
    """Generator for one pair, independent of which other pairs are analysed."""
    return np.random.default_rng([seed, zlib.crc32(key.encode())])


def rounded(x) -> list[float]:
    """Array to a JSON list, 2 decimals."""
    return [round(float(v), 2) for v in x]


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


def displacement(c: np.ndarray) -> np.ndarray:
    """(seeds, joints) change over the chunk, chunk[-1] - chunk[0]."""
    return c[:, -1] - c[:, 0]


def is_moving(delta: np.ndarray, threshold: float) -> np.ndarray:
    """Seeds whose displacement exceeds ``threshold`` on any of joints 0-3."""
    return np.abs(delta[:, :MOVE]).max(1) > threshold


def mover_stats(chunks: dict[str, np.ndarray], threshold: float) -> dict:
    """Fraction of seeds that move, and where the moving ones go.

    The chunk is bimodal per seed: the arm either holds still or starts a reach. Mean-based statistics
    then mostly measure how often it starts, so report that directly.
    """
    out = {}
    for name, c in chunks.items():
        delta = displacement(c)
        moving = is_moving(delta, threshold)
        trigger = np.abs(delta[moving, :MOVE]).argmax(1)
        out[name] = {
            "fraction_moving": round(float(moving.mean()), 3),
            "n_moving": int(moving.sum()),
            "trigger_joint_counts": {JOINTS[j]: int((trigger == j).sum()) for j in range(MOVE)},
            # Arm joints only: the gripper is not in degrees.
            "mover_mean_delta_deg": rounded(delta[moving, :ARM].mean(0)) if moving.any() else None,
            "mover_std_delta_deg": rounded(delta[moving, :ARM].std(0)) if moving.any() else None,
        }
    return out


def mover_direction(chunks: dict[str, np.ndarray], threshold: float, n_boot: int, seed: int) -> dict:
    """Directional separation between two instructions, among seeds that move.

    Distance between mean displacement vectors (arm joints only) and the shoulder_pan difference, each
    with a bootstrap 95% CI. "Moves more often" is not "moves towards the named object", so this is the
    pass condition for the pen frame: the pan CI excludes 0 in both counterbalanced layouts, with the
    sign flipping when the pens swap sides. The distance is descriptive only: a norm of noisy means is
    biased upwards and its CI never reaches 0.
    """
    out = {}
    for a, b in itertools.combinations(chunks, 2):
        key = f"{a}_vs_{b}"
        da, db = displacement(chunks[a]), displacement(chunks[b])
        da = da[is_moving(da, threshold), :ARM]
        db = db[is_moving(db, threshold), :ARM]
        if len(da) < 5 or len(db) < 5:
            out[key] = None
            continue
        rng = pair_rng(seed, key)
        boot_dist, boot_pan = [], []
        for _ in range(n_boot):
            sa = da[rng.integers(len(da), size=len(da))].mean(0)
            sb = db[rng.integers(len(db), size=len(db))].mean(0)
            boot_dist.append(np.linalg.norm(sa - sb))
            boot_pan.append(sa[0] - sb[0])
        out[key] = {
            "n_moving": [len(da), len(db)],
            "mover_vector_distance_deg": round(float(np.linalg.norm(da.mean(0) - db.mean(0))), 2),
            "mover_vector_distance_ci95": rounded(np.percentile(boot_dist, [2.5, 97.5])),
            "mover_pan_diff_deg": round(float(da[:, 0].mean() - db[:, 0].mean()), 2),
            "mover_pan_diff_ci95": rounded(np.percentile(boot_pan, [2.5, 97.5])),
        }
    return out


def step_stats(chunks: dict[str, np.ndarray], state: np.ndarray) -> dict:
    """Largest in-chunk step and largest state -> chunk[0] jump, arm joints 0-4, in degrees.

    ``_all`` is the maximum over every condition except "null", which CAG runs do not have.
    """
    out = {}
    for name, c in chunks.items():
        out[name] = {
            "max_in_chunk_step_deg": round(float(np.abs(np.diff(c[:, :, :ARM], axis=1)).max()), 2),
            "max_state_to_first_jump_deg": round(float(np.abs(c[:, 0, :ARM] - state[:ARM]).max()), 2),
        }
    kept = [v for name, v in out.items() if name != "null"] or list(out.values())
    out["_all"] = {k: max(v[k] for v in kept) for k in kept[0]}
    return out


def clamp_stats(raw: dict[str, np.ndarray]) -> dict:
    """How often arm joints sit at the [-1, 1] clamp; raw is normalised, (seeds, T, D)."""
    out = {}
    for name, r in raw.items():
        pinned = np.abs(r[..., :ARM]) >= 1.0  # (seeds, T, joints)
        out[name] = {
            "steps_pinned_fraction": round(float(pinned.any(-1).mean()), 3),
            "first_action_pinned_fraction": round(float(pinned[:, 0].any(-1).mean()), 3),
            "per_joint_steps_pinned_fraction": {
                JOINTS[j]: round(float(pinned[..., j].mean()), 3) for j in range(ARM)
            },
        }
    return out


def pair_stats(chunks: dict[str, np.ndarray], n_perm: int, seed: int) -> tuple[dict, dict]:
    """Within-condition seed spread, and paired separation / permutation test per condition pair."""
    within = {}
    for name, c in chunks.items():
        pairs = [rms(c[i] - c[j]) for i, j in itertools.combinations(range(len(c)), 2)]
        within[name] = round(float(np.mean(pairs)), 3)

    out = {}
    for a, b in itertools.combinations(chunks, 2):
        key = f"{a}_vs_{b}"
        paired = rms(chunks[a] - chunks[b], axis=(1, 2))  # same seed, different instruction
        per_t = rms(chunks[a] - chunks[b], axis=(0, 2))  # separation along the chunk
        noise = 0.5 * (within[a] + within[b])
        final_diff = chunks[a][:, -1].mean(0) - chunks[b][:, -1].mean(0)
        pooled_sd = np.sqrt(0.5 * (chunks[a][:, -1].var(0) + chunks[b][:, -1].var(0)) + 1e-9)
        out[key] = {
            "paired_rms_deg": round(float(paired.mean()), 3),
            "within_seed_spread_deg": round(noise, 3),
            "separation_ratio": round(float(paired.mean()) / noise, 2) if noise > 0 else None,
            "permutation_p": round(
                float(permutation_p(chunks[a], chunks[b], n_perm, pair_rng(seed, key))), 4
            ),
            "per_timestep_rms_deg": rounded(per_t),
            # Separation grows along the chunk, so the whole-chunk RMS above dilutes it. Per joint at the
            # last step: mean difference over the pooled seed std (Cohen's d).
            "final_step_effect_size": rounded(final_diff / pooled_sd),
            "final_action_mean_diff_deg": rounded(final_diff),
        }
    return within, out


def parse_conditions(items: list[str] | None) -> dict[str, str]:
    """Parse repeated ``name=instruction`` arguments, falling back to the defaults."""
    if not items:
        return DEFAULT_CONDITIONS
    out = {}
    for item in items:
        name, _, text = item.partition("=")
        if name == "_all":
            raise ValueError("'_all' is reserved for the aggregate in step_stats; rename the condition")
        out[name] = text
    return out


def without_clamp(postprocessor):
    """A copy of ``postprocessor`` without the clamp step; the original is left untouched."""
    steps = [s for s in postprocessor.steps if not isinstance(s, MolmoAct2ClampActionProcessorStep)]
    if len(steps) != len(postprocessor.steps) - 1:
        raise RuntimeError(f"expected one clamp step in {[type(s).__name__ for s in postprocessor.steps]}")
    return dataclasses.replace(
        postprocessor,
        steps=steps,
        before_step_hooks=list(postprocessor.before_step_hooks),
        after_step_hooks=list(postprocessor.after_step_hooks),
    )


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
    parser.add_argument(
        "--cag-weights",
        type=float,
        nargs="*",
        default=[],
        help="CAG guidance weights w (needs a 'null' condition)",
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for mover direction")
    parser.add_argument(
        "--move-threshold", type=float, default=5.0, help="deg, max |joint 0-3 change| over the chunk"
    )
    args = parser.parse_args()

    if args.seeds < 2:
        parser.error("--seeds must be at least 2 (the within-condition spread needs a pair of seeds)")
    try:
        conditions = parse_conditions(args.condition)
    except ValueError as e:
        parser.error(str(e))
    if args.cag_weights and "null" not in conditions:
        parser.error(f"--cag-weights needs a condition named 'null'; got {sorted(conditions)}")
    if args.cag_weights and len(conditions) < 2:
        parser.error("--cag-weights needs at least one condition besides 'null'")

    device = "cuda"
    config = load_config(device=device)
    policy, preprocessor, postprocessor = load_policy(config)
    pipelines = {"clamped": postprocessor, "unclamped": without_clamp(postprocessor)}
    observation = load_sample_observation(config)
    if args.cam0:
        observation["observation.images.cam0"] = np.asarray(Image.open(args.cam0).convert("RGB"))
    if args.cam1:
        observation["observation.images.cam1"] = np.asarray(Image.open(args.cam1).convert("RGB"))
    if args.state:
        observation["observation.state"] = np.asarray(json.loads(Path(args.state).read_text()), np.float32)
    state = np.asarray(observation["observation.state"], np.float32)

    def raw_chunk(task: str, seed: int) -> torch.Tensor:
        """Policy output before the postprocessor: normalised actions, (1, T, D)."""
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            batch = preprocess(preprocessor, observation, task, device)
            return policy.predict_action_chunk(batch, generator=generator)

    def postprocess(raw_actions: torch.Tensor, variant: str = "clamped") -> np.ndarray:
        with torch.inference_mode():
            actions = pipelines[variant](raw_actions.clone())
        return torch.as_tensor(actions).squeeze(0).float().cpu().numpy()

    def to_numpy(rs: list[torch.Tensor]) -> np.ndarray:
        return torch.cat(rs).float().cpu().numpy()

    first = next(iter(conditions.values()))
    for s in range(3):  # CUDA-graph capture and warm-up
        postprocess(raw_chunk(first, 10_000 + s))
    deterministic = bool(np.allclose(postprocess(raw_chunk(first, 0)), postprocess(raw_chunk(first, 0))))

    raw = {name: [raw_chunk(text, s) for s in range(args.seeds)] for name, text in conditions.items()}
    arrays = {f"raw_{name}": to_numpy(rs) for name, rs in raw.items()}
    arrays["state"] = state
    clamp = {"plain": clamp_stats({name: to_numpy(rs) for name, rs in raw.items()}), "cag": {}}
    readouts = {}
    for variant in pipelines:
        chunks = {name: np.stack([postprocess(r, variant) for r in rs]) for name, rs in raw.items()}
        arrays.update({f"{variant}_{name}": c for name, c in chunks.items()})
        within, pairs = pair_stats(chunks, args.n_perm, SEED)
        readouts[variant] = {
            "movers": mover_stats(chunks, args.move_threshold),
            "mover_direction": mover_direction(chunks, args.move_threshold, args.n_boot, SEED),
            "steps": step_stats(chunks, state),
            "within_condition_spread_deg": within,
            "pairs": pairs,
            "mean_final_action_deg": {k: rounded(v[:, -1].mean(0)) for k, v in chunks.items()},
            "cag": {},
        }

    # CAG, final-action variant (Appendix E): a = (1 - w) * a_null + w * a_cond, same seed for both,
    # mixed in the policy's normalised action space before the postprocessor. No extra model calls:
    # it reuses the raw chunks above. Written this way, w = 1 gives 0 * a_null + a_cond, which is
    # a_cond bit for bit (a_null + 1 * (a_cond - a_null) is not, in floating point).
    for w in args.cag_weights:
        guided_raw = {
            name: [(1 - w) * raw["null"][i] + w * rs[i] for i in range(args.seeds)]
            for name, rs in raw.items()
            if name != "null"
        }
        guided_raw_np = {name: to_numpy(rs) for name, rs in guided_raw.items()}
        arrays.update({f"cag_w{w}_raw_{name}": r for name, r in guided_raw_np.items()})
        clamp["cag"][str(w)] = clamp_stats(guided_raw_np)
        for variant in pipelines:
            guided = {
                name: np.stack([postprocess(r, variant) for r in rs]) for name, rs in guided_raw.items()
            }
            arrays.update({f"cag_w{w}_{variant}_{name}": c for name, c in guided.items()})
            readouts[variant]["cag"][str(w)] = {
                "movers": mover_stats(guided, args.move_threshold),
                "mover_direction": mover_direction(guided, args.move_threshold, args.n_boot, SEED),
                "steps": step_stats(guided, state),
            }

    result = {
        "frame": args.tag,
        "conditions": conditions,
        "seeds": args.seeds,
        "move_threshold_deg": args.move_threshold,
        "state_arm_frame": rounded(state),
        "same_seed_deterministic": deterministic,
        "clamp": clamp,
        **readouts,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"frame_diagnostic_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    np.savez(RESULTS / f"frame_diagnostic_{args.tag}_chunks.npz", **arrays)

    print("clamp:", json.dumps(clamp, indent=2))
    for variant, block in readouts.items():
        print(f"=== {variant}")
        print("movers:", json.dumps(block["movers"], indent=2))
        print("mover_direction:", json.dumps(block["mover_direction"], indent=2))
        print("steps:", block["steps"]["_all"])
        for name, p in block["pairs"].items():
            print(name, {k: v for k, v in p.items() if k != "per_timestep_rms_deg"})
        for w, cag in block["cag"].items():
            print(
                f"CAG w={w}:",
                {k: v["n_moving"] for k, v in cag["movers"].items()},
                "steps",
                cag["steps"]["_all"],
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
