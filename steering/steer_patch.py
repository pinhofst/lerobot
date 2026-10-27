"""Attach a steerer to every MolmoAct2 policy that ``lerobot-rollout`` loads.

``steering/rollout.py`` calls ``install(...)`` when ``--steer-vector`` or ``--steer-barrier`` is given (exactly one;
both is an error). The patch wraps ``MolmoAct2Policy.from_pretrained`` (the same entry point ``fast_load.py``
wraps, so the two compose in either order) and attaches the steerer to the policy it returns. Nothing in ``src/``
changes.

Constant vector (``--steer-vector <name>.npy``): ``<name>.npy`` (1-D, width of the VLM) with an optional
``<name>.json`` sidecar. With a sidecar that has ``alpha_unit_value`` (the steervla format), the applied strength is
``alpha = k * alpha_unit_value`` and ``layer`` / ``positions`` default to the sidecar's. Without one, pass
``--steer-alpha`` (absolute) and ``--steer-layer``.

Barrier (``--steer-barrier <name>.pt``, see steer_barrier.py): a trained module with its sidecar. ``--steer-k``
scales its threshold (k = 1 is the trained module; k = 0 is off, bitwise); default the sidecar's ``k_default``.
Layer and positions come from the sidecar; a different ``--steer-layer`` is refused. ``--steer-alpha`` is refused.

``positions = task`` (the instruction span) needs the tokenizer: it is loaded from the policy's
``config.checkpoint_path`` the same way the MolmoAct2 processor loads it.

A vector or barrier is specific to one checkpoint, one camera setup and one layer. The sidecar's
``policy_checkpoint`` is compared with the policy's ``config.checkpoint_path``; a mismatch prints a warning and is
recorded (``policy_mismatch``) in meta.json. The LIBERO files in ``vectors/`` and ``barriers/`` may not transfer to
``MolmoAct2-SO100_101``; run ``--steer-k 0`` first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import steer_hooks

STEER: dict[str, Any] = {"installed": False}
_ORIG: dict[str, Any] = {}


def resolve(vector: str, layer: int | None, k: float | None, alpha: float | None, positions: str | None):
    """Vector, layer, alpha and positions from the CLI values and the sidecar; raise on missing pieces."""
    v, meta = steer_hooks.load_vector(vector)
    layer = layer if layer is not None else meta.get("layer")
    if layer is None:
        raise SystemExit("--steer-layer is required (the vector has no sidecar layer)")
    if alpha is None:
        if k is None:
            raise SystemExit("give --steer-k (needs a sidecar with alpha_unit_value) or --steer-alpha")
        if "alpha_unit_value" not in meta:
            raise SystemExit("--steer-k needs a sidecar with alpha_unit_value; use --steer-alpha instead")
        alpha = k * float(meta["alpha_unit_value"])
    return v, int(layer), float(alpha), positions or meta.get("positions", "text"), meta


def resolve_barrier(barrier: str, layer: int | None, k: float | None, alpha: float | None, positions: str | None):
    """Module, layer, k and positions for ``--steer-barrier``; raise on conflicting flags."""
    import steer_barrier

    if alpha is not None:
        raise SystemExit("--steer-alpha is for --steer-vector; a barrier's strength is --steer-k (threshold scale)")
    mod, meta = steer_barrier.load_barrier(barrier)
    side_layer = meta.get("layer")
    if side_layer is None and layer is None:
        raise SystemExit("--steer-layer is required (the barrier has no sidecar layer)")
    if side_layer is not None and layer is not None and int(layer) != int(side_layer):
        raise SystemExit(f"--steer-layer {layer} != the barrier's trained layer {side_layer}")
    layer = int(side_layer if side_layer is not None else layer)
    if k is None:
        k = float(meta.get("k_default", 1.0))
        print(f"steer: no --steer-k given, using the sidecar's k_default {k}", flush=True)
    if k < 0:
        raise SystemExit("--steer-k must be >= 0 for a barrier")
    return mod, layer, float(k), positions or meta.get("positions", "task"), meta


def load_tokenizer(policy):
    """The policy's own tokenizer (as processor_molmoact2 loads it) from ``config.checkpoint_path``."""
    from transformers import Qwen2Tokenizer

    from lerobot.policies.molmoact2.utils_molmoact2 import hf_token, resolve_checkpoint_location

    loc = resolve_checkpoint_location(policy.config.checkpoint_path,
                                      revision=getattr(policy.config, "checkpoint_revision", None))
    return Qwen2Tokenizer.from_pretrained(loc, token=hf_token())


def install(vector: str | None, layer: int | None, k: float | None, alpha: float | None, positions: str | None,
            barrier: str | None = None) -> None:
    """Patch ``MolmoAct2Policy.from_pretrained`` to attach the steerer (idempotent)."""
    if _ORIG:
        return
    if (vector is None) == (barrier is None):
        raise SystemExit("give exactly one of --steer-vector and --steer-barrier (they are mutually exclusive)")
    from lerobot.policies.molmoact2 import modeling_molmoact2 as mm

    import steer_barrier

    if barrier is not None:
        mod, layer, k_val, pos, meta = resolve_barrier(barrier, layer, k, alpha, positions)
        STEER.update(mode="barrier", barrier=str(barrier), barrier_sha256=steer_barrier.sha256(barrier),
                     kind=mod.kind, layer=layer, k=k_val, positions=pos, sidecar=meta)
    else:
        v, layer, alpha_val, pos, meta = resolve(vector, layer, k, alpha, positions)
        STEER.update(mode="vector", vector=str(vector), vector_sha256=steer_barrier.sha256(vector), layer=layer,
                     alpha=alpha_val, k=k, positions=pos, vector_norm=float(abs(v).sum()),
                     edit_norm=abs(alpha_val) * float((v.astype("float64") ** 2).sum() ** 0.5), sidecar=meta)
    if pos not in steer_hooks.POSITIONS:
        raise SystemExit(f"--steer-positions {pos!r}: choose from {steer_hooks.POSITIONS}")
    cls = mm.MolmoAct2Policy
    _ORIG["from_pretrained"] = cls.__dict__["from_pretrained"]
    original = _ORIG["from_pretrained"].__func__

    def from_pretrained(klass, *args, **kwargs):
        policy = original(klass, *args, **kwargs)
        ckpt = getattr(policy.config, "checkpoint_path", None)
        STEER.update(policy_checkpoint=ckpt, policy_mismatch=steer_barrier.check_policy(meta, ckpt))
        span = steer_hooks.make_task_span_fn(load_tokenizer(policy)) if pos == "task" else None
        STEER["task_span_fn"] = span
        if barrier is not None:
            steerer = steer_hooks.BarrierSteerer(policy, layer, mod, k_val, pos, task_span=span).attach()
            what = f"barrier {Path(barrier).name} ({mod.kind}), k {k_val:g}"
        else:
            steerer = steer_hooks.ActivationSteerer(policy, layer, v, alpha_val, pos, task_span=span).attach()
            what = f"vector {Path(vector).name}, alpha {alpha_val:.4f}"
        STEER.update(installed=True, steerer=steerer)
        print(f"steer: {what}, layer {layer}, positions {pos}", flush=True)
        return policy

    cls.from_pretrained = classmethod(from_pretrained)


def summary() -> dict[str, Any]:
    """For meta.json: what was applied and how often it fired (fires must equal calls)."""
    out = {key: val for key, val in STEER.items() if key not in ("steerer", "task_span_fn")}
    s = STEER.get("steerer")
    if s is not None:
        out.update(s.stats() if hasattr(s, "stats") else {"n_calls": s.n_calls, "n_fires": s.n_fires})
    span = STEER.get("task_span_fn")
    if span is not None:
        out.update(task_span_text=span.last_text, task_span_n_tok=span.last_n_tok)
    return out
