"""Activation steering for MolmoAct2: add ``alpha * vector`` to one VLM decoder block during the prompt pass.

Self-contained port of ``steervla/steering_molmo/hooks.py`` (the LIBERO experiments), so this folder
stays standalone. ``src/`` is untouched.

Where it sits (``MolmoAct2Policy``):

    policy._backbone().transformer.blocks[l]   decoder block l of the VLM

Each policy call runs ONE VLM forward over the prompt, then the action expert denoises 10 steps reading
the cached keys/values of every VLM block. A forward hook on block ``l`` therefore fires once per chunk,
and its edit changes what the expert reads from blocks l+1 onward. The LAST block cannot affect actions
(only the LM head reads its output), so ``layer`` must be below ``len(blocks) - 1``.

Which tokens are edited (``positions``): ``text`` (default) = valid tokens after the last image patch,
that is the chat prompt including the instruction and the state tokens; ``all``, ``image``, ``last``
are also available. Masks come from ``input_ids`` captured by a pre-hook on the VLM forward.

The sum is formed in float32 and cast once, so ``alpha = 0`` is bitwise identical to no hook.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

POSITIONS = ("all", "image", "text", "last")


def vlm(policy) -> nn.Module:
    return policy._backbone()


def vlm_blocks(policy) -> nn.ModuleList:
    return vlm(policy).transformer.blocks


def token_masks(input_ids: Tensor, attention_mask: Tensor | None, image_patch_id: int) -> dict[str, Tensor]:
    """Bool masks [B, L]: all valid tokens, image patches, text after the last image patch, last token."""
    valid = input_ids != -1
    if attention_mask is not None:
        valid = valid & attention_mask.to(torch.bool)
    if not valid.any(dim=1).all():
        raise ValueError("a batch row has no valid token")
    B, L = input_ids.shape
    idx = torch.arange(L, device=input_ids.device)[None].expand(B, L)
    image = valid & (input_ids == image_patch_id)
    last_img = torch.where(image, idx, torch.full_like(idx, -1)).max(dim=1).values
    last_idx = torch.where(valid, idx, torch.full_like(idx, -1)).max(dim=1).values
    return {
        "all": valid,
        "image": image,
        "text": valid & (idx > last_img[:, None]),
        "last": idx == last_idx[:, None],
    }


class ActivationSteerer:
    """Add ``alpha * vector`` to the output of VLM block ``layer`` at ``positions``, prompt pass only."""

    def __init__(self, policy, layer: int, vector, alpha: float, positions: str = "text"):
        if positions not in POSITIONS:
            raise ValueError(f"unknown positions {positions!r}; choose from {POSITIONS}")
        n_blocks = len(vlm_blocks(policy))
        if not 0 <= layer < n_blocks - 1:
            raise ValueError(f"layer {layer} out of range: use 0..{n_blocks - 2} (the last block cannot steer)")
        v = torch.as_tensor(vector, dtype=torch.float32)
        if v.ndim != 1:
            raise ValueError(f"vector must be 1-D [hidden_dim], got {tuple(v.shape)}")
        self.policy, self.layer, self.vector = policy, int(layer), v
        self.alpha, self.positions = float(alpha), positions
        self.image_patch_id = int(vlm(policy).config.image_patch_id)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.masks: dict[str, Tensor] | None = None
        self.prefix_len: int | None = None
        self.n_calls = 0  # VLM forwards seen
        self.n_fires = 0  # block calls actually edited (should equal n_calls)

    def _on_vlm_forward(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise RuntimeError("VLM forward called without input_ids; cannot locate tokens")
        self.masks = token_masks(ids, kwargs.get("attention_mask"), self.image_patch_id)
        self.prefix_len = ids.shape[1]
        self.n_calls += 1

    def _hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if self.masks is None or hidden.shape[1] != self.prefix_len:
            return None  # not the prompt pass
        if hidden.shape[-1] != self.vector.shape[0]:
            raise ValueError(f"vector dim {self.vector.shape[0]} != hidden dim {hidden.shape[-1]}")
        mask = self.masks[self.positions]
        if mask.shape[0] == 1 and hidden.shape[0] > 1:
            mask = mask.expand(hidden.shape[0], -1)
        self.n_fires += 1
        delta = self.alpha * self.vector.to(hidden.device)
        steered = torch.where(mask[..., None], (hidden.to(torch.float32) + delta).to(hidden.dtype), hidden)
        return (steered, *output[1:]) if is_tuple else steered

    def attach(self):
        if self.handles:
            raise RuntimeError("already attached")
        self.handles.append(vlm(self.policy).register_forward_pre_hook(self._on_vlm_forward, with_kwargs=True))
        self.handles.append(vlm_blocks(self.policy)[self.layer].register_forward_hook(self._hook, prepend=True))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.remove()
        return False


def load_vector(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load ``<name>.npy`` and its ``<name>.json`` sidecar (``{}`` if absent)."""
    path = Path(path)
    v = np.load(path)
    side = path.with_suffix(".json")
    return v, (json.loads(side.read_text()) if side.exists() else {})
