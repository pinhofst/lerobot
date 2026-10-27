"""Activation steering for MolmoAct2: edit one VLM decoder block's output during the prompt pass.

Self-contained port of ``steervla/steering_molmo/hooks.py`` (constant vector) and
``steervla/steering_dial/hooks.py`` (trained barrier), so this folder stays standalone. ``src/`` is untouched.

Two steerers, both on the output of ``policy._backbone().transformer.blocks[layer]``:

- ``ActivationSteerer``: add the fixed ``alpha * vector`` at the selected tokens (constant vector).
- ``BarrierSteerer``: a trained module (``steer_barrier.py``) reads the selected tokens' states and adds
  ``u = ReLU(k * tau - <a, h>) a`` (unit a), the smallest edit that brings ``<a, h>`` up to ``k * tau``; zero where
  the state already satisfies it. ``k = 0`` is "off" (bitwise no edit; see steer_barrier.py for why the law itself
  is not zero there).

Each policy call runs ONE VLM forward over the prompt, then the action expert denoises 10 steps reading
the cached keys/values of every VLM block. A forward hook on block ``l`` therefore fires once per chunk,
and its edit changes what the expert reads from blocks l+1 onward. The LAST block cannot affect actions
(only the LM head reads its output), so ``layer`` must be below ``len(blocks) - 1``. The CUDA graph of the
policy (``enable_inference_cuda_graph``) captures only the action expert, so the eager prompt pass is hooked.

Which tokens are edited (``positions``): ``text`` = valid tokens after the last image patch, that is the chat
prompt including the instruction and the state tokens; ``task`` = the instruction span only (the tokens of
" pick up the red marker" between "The task is to" and ". The setup is"), which needs a ``task_span``
function (``make_task_span_fn(tokenizer)``); ``all``, ``image``, ``last`` are also available. Masks come from
``input_ids`` captured by a pre-hook on the VLM forward.

The sum is formed in float32 and cast once, so ``alpha = 0`` (vector) or ``k = 0`` (barrier) is bitwise
identical to no hook.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

import steer_barrier

POSITIONS = ("all", "image", "text", "last", "task")
TASK_START, TASK_END = "The task is to ", ". The setup is"  # processor_molmoact2._build_robot_text


def vlm(policy) -> nn.Module:
    return policy._backbone()


def vlm_blocks(policy) -> nn.ModuleList:
    return vlm(policy).transformer.blocks


def token_masks(input_ids: Tensor, attention_mask: Tensor | None, image_patch_id: int,
                task: Tensor | None = None) -> dict[str, Tensor]:
    """Bool masks [B, L]: all valid tokens, image patches, text after the last image patch, last token, and
    ``task`` (the given span intersected with valid) when ``task`` is given."""
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
    out = {
        "all": valid,
        "image": image,
        "text": valid & (idx > last_img[:, None]),
        "last": idx == last_idx[:, None],
    }
    if task is not None:
        out["task"] = task.to(device=input_ids.device, dtype=torch.bool) & valid
        if not out["task"].any(dim=1).all():
            raise RuntimeError("task span not found in a batch row (no valid token in it)")
    return out


# ----------------------------------------------------------------------------- task span


def span_mask(token_strs: list[str], start_marker: str = TASK_START, end_marker: str = TASK_END) -> np.ndarray:
    """bool [L]: tokens overlapping the text strictly between the first ``start_marker`` and the next
    ``end_marker`` in the concatenation of ``token_strs``. All False if a marker is missing or the text between
    them is blank."""
    offs, pos = [], 0
    for s in token_strs:
        offs.append((pos, pos + len(s)))
        pos += len(s)
    text = "".join(token_strs)
    out = np.zeros(len(token_strs), dtype=bool)
    a = text.find(start_marker)
    if a < 0:
        return out
    a += len(start_marker)
    b = text.find(end_marker, a)
    if b < 0 or not text[a:b].strip():
        return out  # no end marker, or an empty instruction (a token straddling the markers is not a span)
    for i, (s, e) in enumerate(offs):
        out[i] = s < b and e > a
    return out


def make_task_span_fn(tokenizer) -> Callable[[Tensor], Tensor]:
    """input_ids [B, L] -> bool [B, L] mask of the instruction tokens; raises if a row has no span.

    Token strings are cached per token id (the state tokens change on every call, so a per-sequence cache would
    miss every time). ``fn.last_text`` holds the decoded span of the last row, for the run's meta.json."""
    strs: dict[int, str] = {}

    def tok_str(t: int) -> str:
        if t < 0:
            return ""
        s = strs.get(t)
        if s is None:
            s = strs[t] = tokenizer.decode([t])
        return s

    def fn(ids: Tensor) -> Tensor:
        rows = []
        for row in ids.detach().cpu().tolist():
            ts = [tok_str(t) for t in row]
            m = span_mask(ts)
            if not m.any():
                raise RuntimeError(
                    f"task span not found in the prompt (looked for {TASK_START!r} ... {TASK_END!r}); "
                    "positions='task' needs the MolmoAct2 robot prompt")
            fn.last_text = "".join(s for s, keep in zip(ts, m) if keep)
            fn.last_n_tok = int(m.sum())
            rows.append(torch.from_numpy(m))
        return torch.stack(rows).to(ids.device)

    fn.last_text, fn.last_n_tok = None, 0
    return fn


def gather_rows(hidden: Tensor, sel: Tensor):
    """hidden [R, L, D], sel [R, L] bool -> (h [R, T, D] float32 padded, m [R, T] bool, row idx, col idx, slot)."""
    R, L, D = hidden.shape
    counts = sel.sum(1)
    T = int(counts.max()) if R else 0
    rr, cc = sel.nonzero(as_tuple=True)  # row-major order
    start = torch.cumsum(counts, 0) - counts
    slot = torch.arange(rr.shape[0], device=hidden.device) - start[rr]
    h = torch.zeros(R, max(T, 1), D, dtype=torch.float32, device=hidden.device)
    h = h.index_put((rr, slot), hidden[rr, cc].to(torch.float32))
    m = torch.zeros(R, max(T, 1), dtype=torch.bool, device=hidden.device)
    m[rr, slot] = True
    return h, m, rr, cc, slot


# ----------------------------------------------------------------------------- steerers


class _PromptHook:
    """Plumbing shared by the steerers: per-call masks from a pre-hook on the VLM forward, one block hook."""

    def __init__(self, policy, layer: int, positions: str, task_span: Callable[[Tensor], Tensor] | None):
        if positions not in POSITIONS:
            raise ValueError(f"unknown positions {positions!r}; choose from {POSITIONS}")
        if positions == "task" and task_span is None:
            raise ValueError("positions='task' needs task_span=make_task_span_fn(tokenizer)")
        n_blocks = len(vlm_blocks(policy))
        if not 0 <= layer < n_blocks - 1:
            raise ValueError(f"layer {layer} out of range: use 0..{n_blocks - 2} (the last block cannot steer)")
        self.policy, self.layer, self.positions, self.task_span = policy, int(layer), positions, task_span
        self.image_patch_id = int(vlm(policy).config.image_patch_id)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.masks: dict[str, Tensor] | None = None
        self.prefix_len: int | None = None
        self.n_calls = 0  # VLM forwards seen
        self.n_fires = 0  # block calls on the prompt pass (should equal n_calls)

    def _on_vlm_forward(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise RuntimeError("VLM forward called without input_ids; cannot locate tokens")
        task = self.task_span(ids) if self.positions == "task" else None
        self.masks = token_masks(ids, kwargs.get("attention_mask"), self.image_patch_id, task)
        self.prefix_len = ids.shape[1]
        self.n_calls += 1

    def _mask(self, hidden: Tensor) -> Tensor | None:
        """The positions mask for this block call, or None if it is not the prompt pass."""
        if self.masks is None or hidden.shape[1] != self.prefix_len:
            return None
        mask = self.masks[self.positions]
        if mask.shape[0] == 1 and hidden.shape[0] > 1:
            mask = mask.expand(hidden.shape[0], -1)
        return mask

    def _hook(self, module, args, output):
        raise NotImplementedError

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


class ActivationSteerer(_PromptHook):
    """Add ``alpha * vector`` to the output of VLM block ``layer`` at ``positions``, prompt pass only."""

    def __init__(self, policy, layer: int, vector, alpha: float, positions: str = "text",
                 task_span: Callable[[Tensor], Tensor] | None = None):
        super().__init__(policy, layer, positions, task_span)
        v = torch.as_tensor(vector, dtype=torch.float32)
        if v.ndim != 1:
            raise ValueError(f"vector must be 1-D [hidden_dim], got {tuple(v.shape)}")
        self.vector, self.alpha = v, float(alpha)

    def _hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        mask = self._mask(hidden)
        if mask is None:
            return None  # not the prompt pass
        if hidden.shape[-1] != self.vector.shape[0]:
            raise ValueError(f"vector dim {self.vector.shape[0]} != hidden dim {hidden.shape[-1]}")
        self.n_fires += 1
        delta = self.alpha * self.vector.to(hidden.device)
        steered = torch.where(mask[..., None], (hidden.to(torch.float32) + delta).to(hidden.dtype), hidden)
        return (steered, *output[1:]) if is_tuple else steered


class BarrierSteerer(_PromptHook):
    """Apply a trained steering module (``steer_barrier.DialBarrier`` or ``ConstVec``) at strength ``k`` to the
    output of VLM block ``layer`` at ``positions`` (default the task span), prompt pass only, gradient-free.

    Per call (``self.calls``): n_tok, n_zero (tokens whose edit is exactly 0), zero_frac, mean edit norm, and for a
    barrier the mean <a, h> and k * tau. ``k = 0`` edits nothing (bitwise) but still locates the span and counts."""

    def __init__(self, policy, layer: int, module: nn.Module, k: float, positions: str = "task",
                 task_span: Callable[[Tensor], Tensor] | None = None, max_logged_calls: int = 10000):
        super().__init__(policy, layer, positions, task_span)
        self.module = module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.dim = int(module.bank.shape[-1]) if hasattr(module, "bank") else int(module.w.shape[0])
        self.k = float(k)
        self.calls: list[dict] = []
        self.max_logged_calls = int(max_logged_calls)
        self._device = None

    def _hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        sel = self._mask(hidden)
        if sel is None:
            return None  # not the prompt pass
        if hidden.shape[-1] != self.dim:
            raise ValueError(f"module dim {self.dim} != hidden dim {hidden.shape[-1]}")
        if not bool(sel.any(1).all()):
            raise RuntimeError("a steered row has no selected token")
        self.n_fires += 1
        if self.k == 0.0:
            self._log({"n_tok": int(sel.sum()), "n_zero": int(sel.sum()), "zero_frac": 1.0, "edit_norm": 0.0})
            return None
        with torch.no_grad():
            if self._device != hidden.device:
                self.module.to(hidden.device)
                self._device = hidden.device
            B, L, D = hidden.shape
            h, m, rr, cc, slot = gather_rows(hidden, sel)
            u, info = steer_barrier.apply_k(self.module, h, m, self.k)
            delta = torch.zeros(B * L, D, dtype=torch.float32, device=hidden.device)
            delta = delta.index_add(0, rr * L + cc, u[rr, slot]).view(B, L, D)
            steered = torch.where(sel[..., None], (hidden.to(torch.float32) + delta).to(hidden.dtype), hidden)
            gap = info["gap"][m]
            n_tok, n_zero = int(m.sum()), int((gap == 0).sum())
            rec = {"n_tok": n_tok, "n_zero": n_zero, "zero_frac": n_zero / n_tok,
                   "edit_norm": float(u[m].norm(dim=-1).mean())}
            if "b" in info:
                rec.update(b_mean=float(info["b"][m].mean()), tau_mean=float(info["tau"][m].mean()))
        self._log(rec)
        return (steered, *output[1:]) if is_tuple else steered

    def _log(self, rec: dict):
        if len(self.calls) < self.max_logged_calls:
            self.calls.append(rec)

    def stats(self) -> dict:
        """Totals for meta.json."""
        out = {"n_calls": self.n_calls, "n_fires": self.n_fires}
        if self.calls:
            for key in ("zero_frac", "edit_norm", "b_mean", "tau_mean", "n_tok"):
                vals = [c[key] for c in self.calls if key in c]
                if vals:
                    out[f"mean_{key}"] = float(np.mean(vals))
        return out


def load_vector(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load ``<name>.npy`` and its ``<name>.json`` sidecar (``{}`` if absent)."""
    path = Path(path)
    v = np.load(path)
    side = path.with_suffix(".json")
    return v, (json.loads(side.read_text()) if side.exists() else {})
