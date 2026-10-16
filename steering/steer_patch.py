"""Attach an activation steerer to every MolmoAct2 policy that ``lerobot-rollout`` loads.

``steering/rollout.py`` calls ``install(...)`` when ``--steer-vector`` is given. The patch wraps
``MolmoAct2Policy.from_pretrained`` (the same entry point ``fast_load.py`` wraps, so the two compose
in either order) and attaches an ``ActivationSteerer`` to the policy it returns. Nothing in ``src/``
changes.

Vector files: ``<name>.npy`` (1-D, width of the VLM) with an optional ``<name>.json`` sidecar. With a
sidecar that has ``alpha_unit_value`` (the steervla format), the applied strength is
``alpha = k * alpha_unit_value`` and ``layer`` defaults to the sidecar's. Without one, pass
``--steer-alpha`` (absolute) and ``--steer-layer``.

A vector is specific to one checkpoint, one camera setup and one layer: the LIBERO vectors in
steervla do not carry over to ``MolmoAct2-SO100_101``. Fit one on SO-101 frames first.
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


def install(vector: str, layer: int | None, k: float | None, alpha: float | None, positions: str | None) -> None:
    """Patch ``MolmoAct2Policy.from_pretrained`` to attach the steerer (idempotent)."""
    if _ORIG:
        return
    from lerobot.policies.molmoact2 import modeling_molmoact2 as mm

    v, layer, alpha_val, pos, meta = resolve(vector, layer, k, alpha, positions)
    cls = mm.MolmoAct2Policy
    _ORIG["from_pretrained"] = cls.__dict__["from_pretrained"]
    original = _ORIG["from_pretrained"].__func__

    def from_pretrained(klass, *args, **kwargs):
        policy = original(klass, *args, **kwargs)
        steerer = steer_hooks.ActivationSteerer(policy, layer, v, alpha_val, pos).attach()
        STEER.update(installed=True, steerer=steerer)
        print(f"steer: layer {layer}, alpha {alpha_val:.4f}, positions {pos}, vector {Path(vector).name}", flush=True)
        return policy

    cls.from_pretrained = classmethod(from_pretrained)
    STEER.update(
        vector=str(vector), layer=layer, alpha=alpha_val, k=k, positions=pos, vector_norm=float(abs(v).sum()),
        sidecar=meta,
    )


def summary() -> dict[str, Any]:
    """For meta.json: what was applied and how often it fired (fires must equal calls)."""
    out = {key: val for key, val in STEER.items() if key != "steerer"}
    s = STEER.get("steerer")
    if s is not None:
        out.update(n_calls=s.n_calls, n_fires=s.n_fires)
    return out
