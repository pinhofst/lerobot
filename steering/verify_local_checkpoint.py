"""Check that the local checkpoint loads through the stock rollout path and matches the streamed load.

    uv run python steering/verify_local_checkpoint.py

Uses the same calls as `lerobot-rollout` (PreTrainedConfig.from_pretrained, policy_class.from_pretrained,
make_pre_post_processors) and compares the full REFERENCE_SEED (29) chunk on Ai2's sample frame with
the one bench_latency.py --cuda-graph on saves (results/latency_bfloat16_graph-on_seed29.npy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from make_local_checkpoint import OUT
from molmo_common import REFERENCE_SEED, SAMPLE_TASK, load_sample_observation, preprocess

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    """Load the local checkpoint the stock way and compare seed-matched actions."""
    config = PreTrainedConfig.from_pretrained(OUT)
    config.pretrained_path = OUT
    policy = MolmoAct2Policy.from_pretrained(OUT, config=config)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=OUT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    print("checkpoint action mode seen by policy:", policy._checkpoint_action_mode)
    print("peak allocated after load GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))
    observation = load_sample_observation(config)
    with torch.inference_mode():
        for s in range(3):
            policy.predict_action_chunk(
                preprocess(preprocessor, observation, SAMPLE_TASK, "cuda"),
                generator=torch.Generator("cuda").manual_seed(1000 + s),
            )
        raw = policy.predict_action_chunk(
            preprocess(preprocessor, observation, SAMPLE_TASK, "cuda"),
            generator=torch.Generator("cuda").manual_seed(REFERENCE_SEED),
        )
        chunk = postprocessor(raw.clone())
    chunk = torch.as_tensor(chunk).squeeze(0).float().cpu().numpy()
    raw = raw.squeeze(0).float().cpu().numpy()
    np.save(RESULTS / f"verify_local_checkpoint_seed{REFERENCE_SEED}.npy", chunk)
    ref = np.load(RESULTS / f"latency_bfloat16_graph-on_seed{REFERENCE_SEED}.npy")
    diff = np.abs(chunk - ref)
    # The postprocessor clamps normalised actions to [-1, 1]; clamped entries match trivially.
    pinned = np.abs(raw[:, : chunk.shape[1]]) >= 1.0
    print(f"chunk {chunk.shape}: max|diff|={diff.max():.6f} mean|diff|={diff.mean():.6f}")
    print(f"max|diff| over entries not at the clamp: {diff[~pinned].max() if (~pinned).any() else None}")
    print(
        f"entries at the clamp bounds: {int(pinned.sum())}/{pinned.size}, per joint {pinned.sum(0).tolist()}"
    )
    print(f"first row at the clamp: {int(pinned[0].sum())}/{pinned.shape[1]}")


if __name__ == "__main__":
    main()
