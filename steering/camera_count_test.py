"""How does the checkpoint behave with one camera instead of the two it was trained with?

Same Ai2 sample frame, task and 64 noise seeds for every variant, one policy load:
    A two_views   cam0 + cam1 (baseline)
    B one_cam0    cam0 only (table-height view)
    C cam0_twice  cam1 := copy of cam0
    D one_cam1    cam1 only (overhead view)

One view works by overriding the saved preprocessor's pack step, which otherwise raises on the
missing key: ``preprocessor_overrides={"molmoact2_pack_inputs": {"image_keys": [key]}}``. On the
pretrained path the processors come from policy_preprocessor.json, so ``config.image_keys`` is not
read. The policy itself has no image keys: it takes whatever pixel_values / image_num_crops it gets.

    uv run python steering/camera_count_test.py   # writes results/camera_count_test.json
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from molmo_common import SAMPLE_TASK, load_config, load_policy, load_sample_observation, preprocess

from lerobot.policies import make_pre_post_processors

CAM0, CAM1 = "observation.images.cam0", "observation.images.cam1"
SEEDS = range(64)
MOVE_DEG = 5.0  # a seed moves when any of joints 0-3 changes by more than this over the chunk
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
OUT = Path(__file__).parent / "results" / "camera_count_test.json"


def main() -> None:
    """Run every variant on the same seeds and write the JSON report."""
    device = "cuda"
    config = load_config(device=device)
    policy, two_view_pre, postprocessor = load_policy(config)
    sample = load_sample_observation(config)
    state = sample["observation.state"]

    def one_view_pre(key: str):
        pre, _ = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(config.pretrained_path),
            preprocessor_overrides={
                "device_processor": {"device": device},
                "molmoact2_pack_inputs": {"image_keys": [key]},
            },
        )
        return pre

    variants = {
        "two_views": (two_view_pre, {CAM0: sample[CAM0], CAM1: sample[CAM1]}),
        "one_cam0": (one_view_pre(CAM0), {CAM0: sample[CAM0]}),
        "cam0_twice": (two_view_pre, {CAM0: sample[CAM0], CAM1: sample[CAM0].copy()}),
        "one_cam1": (one_view_pre(CAM1), {CAM1: sample[CAM1]}),
    }

    def chunk(pre, obs, seed: int) -> np.ndarray:
        generator = torch.Generator(device).manual_seed(seed)
        with torch.inference_mode():
            batch = preprocess(pre, {**obs, "observation.state": state}, SAMPLE_TASK, device)
            actions = postprocessor(policy.predict_action_chunk(batch, generator=generator))
        return torch.as_tensor(actions).squeeze(0).float().cpu().numpy()

    chunks, report = {}, {}
    for name, (pre, obs) in variants.items():
        try:
            with torch.inference_mode():
                batch = preprocess(pre, {**obs, "observation.state": state}, SAMPLE_TASK, device)
            shapes = {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v) and "image" in k}
            shapes["pixel_values"] = list(batch["pixel_values"].shape)
            shapes["image_num_crops_values"] = batch["image_num_crops"].tolist()
            shapes["valid_tokens"] = int(batch["attention_mask"].sum())
            for s in range(3):  # CUDA-graph capture for this sequence length
                chunk(pre, obs, 10_000 + s)
            out, ms = [], []
            for s in SEEDS:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out.append(chunk(pre, obs, s))
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t0) * 1000)
            chunks[name] = np.stack(out).astype(np.float64)  # (seeds, T, 6), arm frame, degrees
            report[name] = {"error": None, "batch": shapes, "latency_ms_median": float(np.median(ms))}
        except Exception as exc:  # noqa: BLE001 - an error is a result here
            report[name] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        print(name, {k: v for k, v in report[name].items() if k != "traceback"}, flush=True)

    base = chunks.get("two_views")
    for name, c in chunks.items():
        disp = c[:, -1] - c[:, 0]
        moving = np.abs(disp[:, :4]).max(axis=1) > MOVE_DEG
        r = report[name]
        r["moving_seeds"] = f"{int(moving.sum())}/{len(SEEDS)}"
        r["moving_mean_displacement_deg"] = (
            dict(zip(JOINTS[:3], disp[moving, :3].mean(axis=0).round(2).tolist(), strict=True))
            if moving.any()
            else None
        )
        r["first_action_mean_deg"] = dict(zip(JOINTS, c[:, 0].mean(axis=0).round(2).tolist(), strict=True))
        if base is not None and name != "two_views":
            # Per-step RMS over arm joints 0-4, same seed as the baseline, averaged over seeds.
            step_rms = np.sqrt(((c[..., :5] - base[..., :5]) ** 2).mean(axis=-1))
            r["rms_vs_two_views_deg"] = {
                "mean_over_chunk": round(float(step_rms.mean()), 2),
                "last_step": round(float(step_rms[:, -1].mean()), 2),
            }
    report["_meta"] = {
        "task": SAMPLE_TASK,
        "seeds": len(SEEDS),
        "move_threshold_deg": MOVE_DEG,
        "state_arm_frame": state.astype(float).round(3).tolist(),
        "one_view_override": {"molmoact2_pack_inputs": {"image_keys": ["<the one camera key>"]}},
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(
        json.dumps({k: {i: j for i, j in v.items() if i != "traceback"} for k, v in report.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
