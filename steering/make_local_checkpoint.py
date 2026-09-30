"""Build a local checkpoint that stock `lerobot-rollout` can load on a 16 GB GPU.

    uv run python steering/make_local_checkpoint.py
    uv run lerobot-rollout --policy.path=steering/checkpoints/MolmoAct2-SO100_101-LeRobot ...

Two problems with `--policy.path=lerobot/MolmoAct2-SO100_101-LeRobot` on this commit:

1. Its config.json predates #4249 and does not parse (see molmo_common.translate_legacy_config).
2. The stock loader puts the whole fp32 `model.safetensors` (21.8 GB) on the GPU before copying it
   into the bf16 model, which runs out of memory on 16 GB.

This script loads the policy on CPU with the streaming loader and re-saves it with
`save_pretrained`. The config is then in the current format, and the weights are in the bf16
storage tree the model actually runs with (VLM bf16, action expert fp32), so they are the same
values the stock loader would have produced. The processor files are symlinked from the HF cache.

`action_mode` is dropped from the saved config.json (the dataclass default is the same
"continuous"). MolmoAct2Policy reads that key back from `<pretrained_path>/config.json` only when
the path is a local directory, and switches on a continuous-training attention mask when it finds
it. The Hub-id path never finds it, so without this the local copy would run a different
inference path from `--policy.path=lerobot/MolmoAct2-SO100_101-LeRobot`.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from molmo_common import POLICY_REPO, load_config, load_policy

OUT = Path(__file__).parent / "checkpoints" / POLICY_REPO.split("/")[-1]


def main() -> None:
    """Load on CPU, re-save in bf16 with a current-format config, link the processor files."""
    config = load_config(device="cpu")
    policy, _, _ = load_policy(config)
    OUT.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(OUT)

    raw = json.loads((OUT / "config.json").read_text())
    raw.pop("action_mode", None)
    raw["device"] = "cuda"
    raw["pretrained_path"] = None
    (OUT / "config.json").write_text(json.dumps(raw, indent=4))

    src = Path(snapshot_download(POLICY_REPO, allow_patterns=["policy_*"]))
    for f in src.glob("policy_*"):
        link = OUT / f.name
        link.unlink(missing_ok=True)
        link.symlink_to(f.resolve())
    print(OUT)


if __name__ == "__main__":
    main()
