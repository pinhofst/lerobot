"""Shared loading helpers for the MolmoAct2 SO-100/101 experiments.

Loads ``lerobot/MolmoAct2-SO100_101-LeRobot`` the same way ``lerobot-rollout --policy.path=<hub id>``
does, except for one workaround: the Hub ``config.json`` (26 Jun 2026) predates #4249 (21 Aug 2026),
which removed four MolmoAct2 config fields, so draccus refuses to parse it on current main. We drop
those fields and translate them into the ``train_mode_vlm`` switch that replaced them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open

from lerobot.policies import make_pre_post_processors
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.utils import prepare_observation_for_inference

POLICY_REPO = "lerobot/MolmoAct2-SO100_101-LeRobot"
BASE_REPO = "allenai/MolmoAct2-SO100_101"
LEGACY_KEYS = ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only", "model_dtype")

# Ai2's model-card sample: Beegbrain/pick_lemon_and_drop_in_bowl, episode 0, frame 0.
SAMPLE_TASK = "Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl."
# The model card gives the state in the *model* (pre-v0.5 calibration) frame.
SAMPLE_STATE_MODEL_FRAME = np.array(
    [-0.52734375, 189.140625, 181.40625, 60.64453125, -3.603515625, 1.0971786975860596], dtype=np.float32
)


def translate_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a pre-#4249 MolmoAct2 config.json onto the current fields."""
    raw = dict(raw)
    legacy = {key: raw.pop(key) for key in LEGACY_KEYS if key in raw}
    if "train_mode_vlm" not in raw:
        if legacy.get("train_action_expert_only"):
            raw["train_mode_vlm"] = "freeze"
        elif legacy.get("enable_lora_vlm"):
            raw["train_mode_vlm"] = "lora"
        else:
            # No adapters. The new default ("lora") would wrap the VLM in fresh LoRA layers at load time.
            raw["train_mode_vlm"] = "fft"
    return raw


def load_config(
    *, dtype: str = "bfloat16", cuda_graph: bool = True, device: str = "cuda", repo: str = POLICY_REPO
) -> MolmoAct2Config:
    """Build the policy config from the Hub config.json, with dtype, CUDA graphs and device set."""
    raw = translate_legacy_config(json.loads(Path(hf_hub_download(repo, "config.json")).read_text()))
    raw.pop("type")
    raw.update(dtype=dtype, enable_inference_cuda_graph=cuda_graph, device=device)
    config = draccus.decode(MolmoAct2Config, raw)
    # Mirror `--policy.path=<hub id>`: a relative Path that does not exist locally.
    config.pretrained_path = Path(repo)
    return config


def _stream_weights(policy: torch.nn.Module, weights_file: str) -> None:
    """Copy a safetensors file into ``policy`` one tensor at a time.

    ``PreTrainedPolicy.from_pretrained`` instead calls ``safetensors.torch.load_model(device="cuda:0")``,
    which materialises the whole 21.8 GB fp32 state dict on the GPU before copying it into the bf16
    model: that cannot fit a 16 GB card. Streaming keeps peak GPU memory at the model itself and
    peak host memory at one tensor. fp32 values are rounded into the bf16 parameters exactly as
    ``load_state_dict`` would.
    """
    targets = policy.state_dict()
    with safe_open(weights_file, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        unexpected = sorted(keys - set(targets))
        # Tied weights are saved once; their alias is not missing if it shares storage with a saved key.
        saved_ptrs = {targets[k].data_ptr() for k in keys if k in targets}
        missing = sorted(k for k in set(targets) - keys if targets[k].data_ptr() not in saved_ptrs)
        if unexpected or missing:
            raise RuntimeError(f"weights mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        with torch.no_grad():
            for key in keys:
                targets[key].copy_(f.get_tensor(key))


def load_policy(config: MolmoAct2Config, repo: str = POLICY_REPO):
    """Load policy and pre/post-processors, streaming weights so the load fits a 16 GB GPU."""
    device = config.device
    # Builds the model from the base HF checkpoint (bf16 storage tree, on CPU).
    policy = MolmoAct2Policy(config)
    policy.to(device)
    _stream_weights(policy, hf_hub_download(repo, "model.safetensors"))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=repo,
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    return policy, preprocessor, postprocessor


def model_to_arm_frame(state: np.ndarray, config: MolmoAct2Config) -> np.ndarray:
    """Invert the checkpoint's ``state_model = signs * arm_state + offsets``."""
    signs = np.asarray(config.joint_signs, dtype=np.float32)
    offsets = np.asarray(config.joint_offsets, dtype=np.float32)
    out = state.copy()
    out[: len(signs)] = signs * (state[: len(signs)] - offsets)
    return out


def load_sample_observation(config: MolmoAct2Config) -> dict[str, np.ndarray]:
    """Ai2's sample frame as a raw robot observation: cam0=top, cam1=side, state in arm frame."""

    def image(name: str) -> np.ndarray:
        return np.asarray(Image.open(hf_hub_download(BASE_REPO, f"assets/{name}")).convert("RGB"))

    return {
        "observation.images.cam0": image("sample_realsense_top_rgb.png"),
        "observation.images.cam1": image("sample_realsense_side_rgb.png"),
        "observation.state": model_to_arm_frame(SAMPLE_STATE_MODEL_FRAME, config),
    }


def preprocess(preprocessor, observation: dict[str, np.ndarray], task: str, device: str) -> dict[str, Any]:
    """Turn a raw robot observation into a preprocessed policy batch, as lerobot-rollout does."""
    obs = {key: value.copy() for key, value in observation.items()}
    batch = prepare_observation_for_inference(obs, torch.device(device), task, "so101_follower")
    return preprocessor(batch)
