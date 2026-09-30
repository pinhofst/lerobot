"""Check that the local checkpoint loads through the stock rollout path and matches the streamed load.

    uv run python steering/verify_local_checkpoint.py

Uses the same calls as `lerobot-rollout` (PreTrainedConfig.from_pretrained, policy_class.from_pretrained,
make_pre_post_processors) and compares seed-0 actions on Ai2's sample frame with bench_latency.py's output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from make_local_checkpoint import OUT
from molmo_common import SAMPLE_TASK, load_sample_observation, preprocess

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
        chunk = postprocessor(
            policy.predict_action_chunk(
                preprocess(preprocessor, observation, SAMPLE_TASK, "cuda"),
                generator=torch.Generator("cuda").manual_seed(29),
            )
        )
    chunk = torch.as_tensor(chunk).squeeze(0).float().cpu().numpy()
    ref = json.loads((RESULTS / "latency_bfloat16_graph-on.json").read_text())
    for name, row in (("first_action", chunk[0]), ("last_action", chunk[-1])):
        diff = np.abs(np.round(row, 2) - np.asarray(ref[name])).max()
        print(f"{name}: local={np.round(row, 2).tolist()} bench={ref[name]} max|diff|={diff:.2f}")


if __name__ == "__main__":
    main()
