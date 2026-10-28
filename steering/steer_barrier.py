"""Trained steering modules for MolmoAct2: the barrier and the constant vector, plus load / export / k knob.

Standalone copy of ``steervla/steering_dial/barrier.py`` (modules, checkpoint format) and
``steervla/steering_wrist/kmod.py`` (the strength knob k), so this folder stays standalone. The math and the op
order are the same, so the outputs are bitwise those of steervla (``test_steer_barrier.py`` checks it when
steervla can be imported).

Both modules act on the token states selected at one VLM layer in the one prompt pass of a policy call:
    h [R, T, D] float32 (T = padded number of selected tokens per row), m [R, T] bool (valid tokens)
and return the edit u [R, T, D] (exactly 0 at invalid tokens) and per-token diagnostics.

Barrier (``DialBarrier``, an adaptation of AdaBarrier's AttnConcatKVBarrier from ControlLLM / barriersteer):

    kv_i  = [ E_1 .. E_Td ; SEP ; h_1 .. h_T ]          one softmax over the frozen description bank and the states
    ctx_i = attn(q = Wq n(h_i), kv)                     n = parameter-free RMS normalisation
    g_i   = GELU(ctx_proj(ctx_i))
    a_i   = unit(out(g_i) + bias_dir)
    tau_i = scale * (tau0 + thr(g_i))
    u_i   = k * ReLU(tau_i - <a_i, h_i>) a_i            k = 1 is the trained module

At k = 1 the edit is the minimum-norm step onto {<a_i, h> >= tau_i}, exactly zero where the state already satisfies
it. k SCALES THE EDIT (ControlLLM's strength form), continuously from 0 (no edit) through 1 (trained) and beyond; it
does not move the threshold. Note: steervla/steering_wrist (kmod.py) used the threshold form ReLU(k * tau - <a, h>)
in its sim k sweep, so its k values do not carry over numerically.

Constant vector (``ConstVec``): u_i = k * scale * w at every selected token. For deployment it is exported in the
``vectors/`` format instead (``vector = w``, ``alpha_unit_value = scale``), so ``--steer-vector ... --steer-k 1``
reproduces it bitwise; ``load_barrier`` still reads a ConstVec ``.pt``.

File format (``.pt``, the steervla ``save_module`` format, tensors only so ``weights_only=True`` loads it):
    {"config": {"kind", "d_attn", "n_heads", "d_hidden", "scale"}, "state_dict": {... "bank": [Td, D] ...},
     "extra": {...}}
The description bank is a buffer in the state dict, so loading needs no model pass. Next to it,
``<name>.json`` (sidecar): layer, positions, k_default, policy_checkpoint (the ``config.checkpoint_path`` of the
policy the module was trained on), trained_on, bank description, alpha unit, source checkpoint and its sha256.

Export a steervla checkpoint (no steervla import needed):
    python steering/steer_barrier.py --src <ckpt.pt> --out steering/barriers/<name>.pt --layer 6 \
        --policy-checkpoint allenai/MolmoAct2-LIBERO --trained-on "..." [--bank-note "..."]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class DialBarrier(nn.Module):
    kind = "barrier"

    def __init__(self, bank: Tensor, v0: Tensor, scale: float, tau0: float = 0.0, d_attn: int = 256,
                 n_heads: int = 4, d_hidden: int = 512):
        super().__init__()
        bank = torch.as_tensor(bank, dtype=torch.float32)
        d_model = bank.shape[-1]
        if tuple(v0.shape) != (d_model,):
            raise ValueError(f"v0 shape {tuple(v0.shape)} != ({d_model},)")
        self.register_buffer("bank", bank.clone())  # [Td, D] frozen description token states
        self.d_model, self.d_attn, self.n_heads, self.d_hidden = d_model, d_attn, n_heads, d_hidden
        self.d_head = d_attn // n_heads
        self.scale = float(scale)
        self.q_proj = nn.Linear(d_model, d_attn, bias=False)
        self.k_proj = nn.Linear(d_model, d_attn, bias=False)
        self.v_proj = nn.Linear(d_model, d_attn, bias=False)
        self.sep_k = nn.Parameter(torch.randn(d_attn) * d_attn ** -0.5)
        self.sep_v = nn.Parameter(torch.randn(d_attn) * d_attn ** -0.5)
        self.ctx_proj = nn.Linear(d_attn, d_hidden)
        self.out = nn.Linear(d_hidden, d_model)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.bias_dir = nn.Parameter(torch.as_tensor(v0, dtype=torch.float32).clone() / float(v0.norm()))
        self.thr = nn.Linear(d_hidden, 1)
        nn.init.zeros_(self.thr.weight)
        nn.init.zeros_(self.thr.bias)
        self.tau0 = nn.Parameter(torch.tensor(float(tau0) / self.scale))

    def config(self) -> dict:
        return {"kind": self.kind, "d_attn": self.d_attn, "n_heads": self.n_heads, "d_hidden": self.d_hidden,
                "scale": self.scale}

    def _heads(self, x: Tensor) -> Tensor:  # [R, N, d_attn] -> [R, H, N, dh]
        R, N, _ = x.shape
        return x.view(R, N, self.n_heads, self.d_head).transpose(1, 2)

    def direction_thr(self, h: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
        R, T, D = h.shape
        hn = rms_norm(h)
        en = rms_norm(self.bank)[None].expand(R, -1, -1)
        Td = en.shape[1]
        q = self._heads(self.q_proj(hn))
        k = torch.cat([self.k_proj(en), self.sep_k.view(1, 1, -1).expand(R, 1, -1), self.k_proj(hn)], 1)
        v = torch.cat([self.v_proj(en), self.sep_v.view(1, 1, -1).expand(R, 1, -1), self.v_proj(hn)], 1)
        keep = torch.cat([torch.ones(R, Td + 1, dtype=torch.bool, device=h.device), m], 1)  # [R, Td+1+T]
        bias = torch.zeros(R, 1, 1, Td + 1 + T, device=h.device, dtype=q.dtype)
        bias = bias.masked_fill(~keep[:, None, None, :], float("-inf"))
        ctx = F.scaled_dot_product_attention(q, self._heads(k), self._heads(v), attn_mask=bias)
        ctx = ctx.transpose(1, 2).reshape(R, T, self.d_attn)
        g = F.gelu(self.ctx_proj(ctx))
        d = self.out(g) + self.bias_dir
        a = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        tau = self.scale * (self.tau0 + self.thr(g).squeeze(-1))
        return a, tau

    def forward(self, h: Tensor, m: Tensor) -> tuple[Tensor, dict]:
        a, tau = self.direction_thr(h, m)
        b = (a * h).sum(-1)
        gap = F.relu(tau - b) * m.to(h.dtype)
        u = gap.unsqueeze(-1) * a
        return u, {"gap": gap, "b": b, "tau": tau}


class ConstVec(nn.Module):
    kind = "const"

    def __init__(self, w0: Tensor, scale: float = 1.0):
        super().__init__()
        self.w = nn.Parameter(torch.as_tensor(w0, dtype=torch.float32).clone())
        self.scale = float(scale)

    def config(self) -> dict:
        return {"kind": self.kind, "scale": self.scale}

    def delta(self) -> Tensor:
        return self.w if self.scale == 1.0 else self.w * self.scale

    def forward(self, h: Tensor, m: Tensor) -> tuple[Tensor, dict]:
        d = self.delta()
        u = torch.where(m[..., None], d.view(1, 1, -1).expand_as(h), torch.zeros((), device=h.device, dtype=h.dtype))
        return u, {"gap": m.to(h.dtype) * d.norm()}


def apply_k(mod: nn.Module, h: Tensor, m: Tensor, k: float) -> tuple[Tensor, dict]:
    """The module's edit at strength k: k times the trained edit. k = 1 calls the module itself (bitwise).

    Barrier: u = k * ReLU(tau - <a, h>) a (``gap`` is scaled too; ``tau`` and ``b`` are k-independent). Constant
    vector: u = k * scale * w (as steervla.steering_wrist.kmod.KConst)."""
    k = float(k)
    if k == 1.0:
        return mod(h, m)
    if isinstance(mod, DialBarrier):
        u, info = mod(h, m)
        return k * u, {**info, "gap": k * info["gap"]}
    if isinstance(mod, ConstVec):
        d = mod.delta() * k
        u = torch.where(m[..., None], d.view(1, 1, -1).expand_as(h), torch.zeros((), device=h.device, dtype=h.dtype))
        return u, {"gap": m.to(h.dtype) * d.norm()}
    raise TypeError(type(mod))


# ----------------------------------------------------------------------------- files


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_module(mod: nn.Module, path, extra: dict | None = None) -> None:
    torch.save({"config": mod.config(), "state_dict": {k: v.detach().cpu() for k, v in mod.state_dict().items()},
                "extra": extra or {}}, path)


def module_from_checkpoint(ck: dict) -> nn.Module:
    cfg, sd = ck["config"], ck["state_dict"]
    if cfg["kind"] == "barrier":
        mod = DialBarrier(sd["bank"], sd["bias_dir"], cfg["scale"], d_attn=cfg["d_attn"], n_heads=cfg["n_heads"],
                          d_hidden=cfg["d_hidden"])
    elif cfg["kind"] == "const":
        mod = ConstVec(sd["w"], cfg["scale"])
    else:
        raise ValueError(f"unknown module kind {cfg['kind']!r}")
    mod.load_state_dict(sd)
    return mod.eval()


def load_barrier(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Load ``<name>.pt`` (steervla save_module format) and its ``<name>.json`` sidecar (``{}`` if absent)."""
    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    mod = module_from_checkpoint(ck).to(device)
    for p in mod.parameters():
        p.requires_grad_(False)
    side = path.with_suffix(".json")
    meta = json.loads(side.read_text()) if side.exists() else {}
    return mod, meta


def check_policy(meta: dict, policy_checkpoint: str | None) -> bool:
    """Warn when the policy differs from the one the module was trained on. Returns True on a mismatch."""
    trained = meta.get("policy_checkpoint")
    if not trained or not policy_checkpoint:
        return False
    if str(trained).rstrip("/") == str(policy_checkpoint).rstrip("/"):
        return False
    msg = (f"steering module was trained on {trained!r} but the policy is {policy_checkpoint!r} "
           f"({meta.get('trained_on', 'no trained_on note')}); it may not transfer, run --steer-k 0 first")
    warnings.warn(msg, stacklevel=2)
    print(f"WARNING: {msg}", flush=True)
    return True


def _slim_extra(extra: dict) -> dict:
    """Keep the training args and the last history entry (plain data, small)."""
    out = {k: v for k, v in extra.items() if k in ("args", "init", "steps_done", "train_s")}
    if extra.get("hist"):
        out["hist_last"] = extra["hist"][-1]
    return out


def export(src: str | Path, out: str | Path, *, layer: int, positions: str, policy_checkpoint: str,
           trained_on: str, k_default: float = 1.0, bank_note: str = "", notes: dict | None = None,
           src_id: str | None = None, target: str | None = None) -> dict:
    """Copy a steervla checkpoint to ``out`` and write the sidecar ``out.json``. Returns the sidecar.

    ``out`` ending in ``.pt``: the module, tensors only (``load_barrier`` / ``--steer-barrier``). ``out`` ending in
    ``.npy`` (ConstVec only): the vector ``w`` with ``alpha_unit_value = scale``, so ``--steer-vector out.npy
    --steer-k 1`` adds ``scale * w``, the trained constant vector, bitwise."""
    import numpy as np

    src, out = Path(src), Path(out)
    ck = torch.load(src, map_location="cpu", weights_only=False)  # steervla's own file: its extra holds plain data
    mod = module_from_checkpoint(ck)
    extra = _slim_extra(ck.get("extra") or {})
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".npy":
        if not isinstance(mod, ConstVec):
            raise ValueError("only a constant vector can be exported as .npy")
        np.save(out, mod.w.detach().cpu().numpy().astype(np.float32))
    else:
        save_module(mod, out, extra)
        module_from_checkpoint(torch.load(out, map_location="cpu", weights_only=True))  # loadable without pickle
    side: dict[str, Any] = {
        "name": out.stem,
        "kind": mod.kind,
        "target": target,
        "layer": int(layer),
        "positions": positions,
        "k_default": float(k_default),
        "policy_checkpoint": policy_checkpoint,
        "trained_on": trained_on,
        "dim": int(mod.bank.shape[-1]) if isinstance(mod, DialBarrier) else int(mod.w.shape[0]),
        "alpha_unit_value": mod.scale,
        "alpha_unit": "module scale: median L2 norm of text-token activations at this layer (fit set)",
        "source": {"file": src_id or src.name, "sha256": sha256(src), "steps_done": extra.get("steps_done"),
                   "args": extra.get("args", {})},
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if isinstance(mod, ConstVec):
        side["vector_norm"] = float(mod.w.detach().norm())
        side["k_trained"] = 1.0
    if isinstance(mod, DialBarrier):
        side["bank"] = {"included": True, "shape": list(mod.bank.shape), "how": bank_note}
        side["n_params"] = sum(p.numel() for p in mod.parameters())
    if notes:
        side.update(notes)
    out.with_suffix(".json").write_text(json.dumps(side, indent=1) + "\n")
    return side


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export a steervla steering_dial checkpoint for steering/rollout.py.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--positions", default="task")
    ap.add_argument("--policy-checkpoint", required=True, help="config.checkpoint_path of the training policy")
    ap.add_argument("--trained-on", required=True)
    ap.add_argument("--k-default", type=float, default=1.0)
    ap.add_argument("--bank-note", default="")
    ap.add_argument("--target", default=None, help="colour the module steers toward (red / green)")
    ap.add_argument("--src-id", default=None, help="how the sidecar names the source (default: its file name)")
    ap.add_argument("--notes-json", default="", help="extra sidecar fields as a JSON object")
    a = ap.parse_args(argv)
    side = export(a.src, a.out, layer=a.layer, positions=a.positions, policy_checkpoint=a.policy_checkpoint,
                  trained_on=a.trained_on, k_default=a.k_default, bank_note=a.bank_note,
                  notes=json.loads(a.notes_json) if a.notes_json else None, src_id=a.src_id, target=a.target)
    print(json.dumps(side, indent=1))


if __name__ == "__main__":
    main()
