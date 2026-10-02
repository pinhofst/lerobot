"""Load MolmoAct2 from a local re-saved checkpoint without reading the allenai weight shards.

    uv run python steering/fast_load.py                 # self-check: fast load + seed-29 action comparison
    uv run python steering/fast_load.py --no-file-check # skip the tensor-by-tensor check against the file

Use from a script (before ``lerobot_rollout.main()``)::

    import fast_load

    fast_load.install()

or, scoped, ``with fast_load.enabled(): ...``. ``install()`` is idempotent; ``uninstall()`` undoes it,
restoring each patched attribute only if it is still this module's own wrapper (one that somebody
else replaced since is left alone, with a warning).

Why. ``MolmoAct2Policy.from_pretrained(<local dir>)`` builds the policy, and ``MolmoAct2Policy.__init__``
calls ``_load_hf_model``, which

1. builds the HF model with ``MolmoAct2ForConditionalGeneration.from_pretrained(<allenai snapshot>)``,
   reading the 21.8 GB of fp32 shards into a bf16 model,
2. applies ``_apply_bfloat16_parameter_policy`` (action expert, norms and RoPE back to fp32),
3. re-reads the same shards with ``_strict_load_safetensors_weights`` to get exact fp32 values,

and only then does ``PreTrainedPolicy.from_pretrained`` load ``<local dir>/model.safetensors``, which
overwrites every one of those weights. Steps 1 and 3 are wasted work for a full re-save.

What this patch does. Only inside ``MolmoAct2Policy.from_pretrained`` called with a local directory
that holds a ``model.safetensors``:

* ``MolmoAct2ForConditionalGeneration.from_pretrained`` runs unchanged (config and generation config
  from the allenai snapshot's JSON files, the same meta-device construction under the same dtype, the
  same attention implementation), except that its weight-loading step receives an empty state dict.
  Transformers then materialises every parameter and buffer from meta as ``torch.empty_like`` on CPU,
  exactly as it does for non-persistent buffers in a normal load. The init pass
  (``_initialize_missing_keys``, which in a stock load runs the vendored ``_init_weights`` on every
  module) is replaced by one that only sets the ``_is_hf_initialized`` flags a stock load leaves on
  every module and every state_dict tensor, and the load report that would list every weight as
  missing is muted. No shard is opened.
* ``_apply_bfloat16_parameter_policy`` runs unchanged, so the dtype tree is the stock one.
* ``_strict_load_safetensors_weights`` is skipped, after re-checking that the local file holds every
  key of the HF model (under the policy's ``model.`` prefix) with the same shape and the same dtype as
  the final dtype tree.
* ``PreTrainedPolicy.from_pretrained`` then loads the local ``model.safetensors`` strictly, as usual.

Fallbacks, all to the exact stock values:

* Not a local directory, or no ``model.safetensors`` in it (Hub id, PEFT adapter directory): nothing
  is patched for that call.
* The file's keys or shapes do not cover the HF model (a LoRA re-save, a different model): the
  weight load runs on the allenai shards as usual.

A dtype difference between the local file and the final dtype tree (keys and shapes matching) is
logged as a warning and the shard re-read is still skipped: re-reading would be redundant, because
``PreTrainedPolicy.from_pretrained`` then loads the local file with ``load_state_dict``, which copies
into the existing parameters and keeps their dtypes, so every value ends up as the file's value cast
to the stock dtype tree either way. ``LAST_LOAD.reason`` then names the mismatch.

Prerequisite: the allenai snapshot must still be in the HF cache or downloadable. The shards are not
read, but ``_load_hf_model`` still resolves the snapshot (``_resolve_checkpoint_location``) and reads
its config, and transformers resolves the shard files before the (empty) weight-loading step.

RNG state: a fast load does not run ``_init_weights`` (no ``normal_``/``uniform_`` draws from the
default CPU generator), so the global CPU RNG state after loading differs from a stock load; code that
draws from the global CPU generator afterwards sees different numbers. The action noise is unaffected
on a CUDA policy: ``torch.randn`` draws it on the policy device, from the seeded generator when one is
passed (per_episode_seed) or else the CUDA default generator, which the CPU init draws never touch
(the self-check below reproduces the seed-29 reference chunk bit for bit). With
``--policy.device=cpu`` and no generator, the noise would follow the CPU RNG state.

Every decision is logged at INFO level by the ``fast_load`` logger, and the last one is kept in
``LAST_LOAD`` (``fast`` is True when no shard was read).

Non-persistent buffers: the only ones in this model are the RoPE sin/cos caches
(``_pos_sin_cache``/``_pos_cos_cache``), which ``_apply_bfloat16_parameter_policy`` resets to empty
fp32 tensors in both paths. ``inv_freq`` is persistent and comes from the local file. The module
``original_inv_freq`` alias is re-pointed to ``inv_freq`` by the same policy step in both paths.
"""

from __future__ import annotations

import argparse
import contextvars
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from lerobot.policies.molmoact2 import modeling_molmoact2 as _mm

log = logging.getLogger("fast_load")

_POLICY_PREFIX = "model."  # MolmoAct2Policy.model is the HF model
_WEIGHTS_NAME = "model.safetensors"
_TORCH_TO_ST = {
    torch.float32: "F32",
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float64: "F64",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


@dataclass
class LoadInfo:
    """What the patched ``from_pretrained`` decided for one call."""

    path: str
    weights_file: str | None = None
    fast: bool = False  # True when the HF model was built without reading the allenai shards
    reason: str = ""
    seconds: float = 0.0
    header: dict[str, tuple[str, tuple[int, ...]]] = field(default_factory=dict, repr=False)
    hf_model: Any = field(default=None, repr=False)


LAST_LOAD: LoadInfo | None = None
_ACTIVE: contextvars.ContextVar[LoadInfo | None] = contextvars.ContextVar("fast_load_active", default=None)
_ORIGINALS: dict[str, Any] = {}


def _read_header(weights_file: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Return {key: (safetensors dtype, shape)} from the file header, without reading tensor data."""
    with safe_open(str(weights_file), framework="pt", device="cpu") as f:
        out = {}
        for key in f.keys():  # noqa: SIM118  (safe_open is not a dict)
            sl = f.get_slice(key)
            out[key] = (sl.get_dtype(), tuple(sl.get_shape()))
        return out


def _header_mismatch(model: torch.nn.Module, info: LoadInfo, *, check_dtype: bool) -> str:
    """Empty string if the local file covers every key of the HF model with the same shape (and dtype)."""
    problems: list[str] = []
    for key, tensor in model.state_dict().items():
        entry = info.header.get(_POLICY_PREFIX + key)
        if entry is None:
            problems.append(f"missing {key}")
        elif entry[1] != tuple(tensor.shape):
            problems.append(f"shape {key}: file {entry[1]} model {tuple(tensor.shape)}")
        elif check_dtype and entry[0] != _TORCH_TO_ST.get(tensor.dtype):
            problems.append(f"dtype {key}: file {entry[0]} model {tensor.dtype}")
        if len(problems) >= 5:
            break
    return "; ".join(problems)


def _probe(pretrained_name_or_path: Any) -> LoadInfo:
    """Decide whether this from_pretrained call may use the fast path (header check only)."""
    info = LoadInfo(path=str(pretrained_name_or_path))
    path = Path(str(pretrained_name_or_path))
    weights_file = path / _WEIGHTS_NAME
    # PreTrainedPolicy.from_pretrained loads <dir>/model.safetensors exactly when os.path.isdir(path).
    if not path.is_dir():
        info.reason = "not a local directory"
        return info
    if not weights_file.is_file():
        info.reason = f"no {_WEIGHTS_NAME} in the directory"
        return info
    try:
        info.header = _read_header(weights_file)
    except Exception as e:  # noqa: BLE001  (unreadable file: let the stock path raise its own error)
        info.reason = f"cannot read the {_WEIGHTS_NAME} header ({e!r})"
        return info
    info.weights_file = str(weights_file)
    return info


# --- patched functions -------------------------------------------------------------------------


def _policy_from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
    """``MolmoAct2Policy.from_pretrained`` with the fast HF-model build when the checkpoint allows it."""
    global LAST_LOAD
    original = _ORIGINALS["policy_from_pretrained"].__func__
    info = _probe(pretrained_name_or_path)
    LAST_LOAD = info
    if info.weights_file is None:
        log.info("fast_load: stock load for %s (%s)", info.path, info.reason)
        return original(cls, pretrained_name_or_path, *args, **kwargs)
    t0 = time.perf_counter()
    token = _ACTIVE.set(info)
    try:
        policy = original(cls, pretrained_name_or_path, *args, **kwargs)
    finally:
        _ACTIVE.reset(token)
        info.seconds = time.perf_counter() - t0
        info.hf_model = None
        info.header = {}
    if info.fast:
        log.info("fast_load: built without the allenai shards; weights from %s", info.weights_file)
    else:
        log.info("fast_load: stock load for %s (%s)", info.path, info.reason)
    return policy


def _load_pretrained_model(model, state_dict, checkpoint_files, load_config, *args, **kwargs):
    """Give transformers an empty state dict when the local file will supply every weight."""
    original = _ORIGINALS["load_pretrained_model"]
    info = _ACTIVE.get()
    if (
        info is not None
        and info.hf_model is None
        and type(model) is _mm.MolmoAct2ForConditionalGeneration
        and state_dict is None
    ):
        info.hf_model = model
        mismatch = _header_mismatch(model, info, check_dtype=False)
        if mismatch:
            info.reason = f"local file does not cover the HF model: {mismatch}"
        else:
            info.fast = True
            return original(model, {}, None, load_config, *args, **kwargs)
    return original(model, state_dict, checkpoint_files, load_config, *args, **kwargs)


def _mark_initialized(model: torch.nn.Module) -> None:
    """Set the ``_is_hf_initialized`` flags a stock load leaves, without initialising any tensor.

    Stock: the weight load flags every loaded state_dict tensor, and ``initialize_weights`` visits every
    module (``smart_apply``) and flags it after running ``_init_weights`` on it.
    """
    for tensor in model.state_dict(keep_vars=True).values():
        tensor._is_hf_initialized = True
    for module in model.modules():
        module._is_hf_initialized = True


def _finalize_model_loading(model, load_config, loading_info):
    """Replace the init pass over the (deliberately) missing weights; mute the all-missing report."""
    original = _ORIGINALS["finalize_model_loading"]
    info = _ACTIVE.get()
    if info is None or not info.fast or model is not info.hf_model:
        return original(model, load_config, loading_info)
    report_logger = logging.getLogger("transformers.modeling_utils")
    level = report_logger.level
    # Instance attribute shadows the method for this one call.
    model._initialize_missing_keys = lambda *a, **k: _mark_initialized(model)
    report_logger.setLevel(logging.ERROR)
    try:
        return original(model, load_config, loading_info)
    finally:
        report_logger.setLevel(level)
        del model._initialize_missing_keys


def _strict_load_safetensors_weights(model: torch.nn.Module, checkpoint_location: str) -> None:
    """Skip the second shard read; log (only) if the local file's dtypes differ from the final tree."""
    original = _ORIGINALS["strict_load"]
    info = _ACTIVE.get()
    if info is None or not info.fast or model is not info.hf_model:
        return original(model, checkpoint_location)
    mismatch = _header_mismatch(model, info, check_dtype=True)
    if mismatch:
        # load_state_dict keeps the parameter dtypes, so the local load casts to the stock tree anyway.
        info.reason = f"local file dtypes differ from the final dtype tree (cast on load): {mismatch}"
        log.warning("fast_load: %s; shard re-read skipped", info.reason)
    return None


# --- install / uninstall -----------------------------------------------------------------------


def install() -> None:
    """Patch MolmoAct2 loading in this process (idempotent). See the module docstring."""
    if _ORIGINALS:
        return
    policy_cls = _mm.MolmoAct2Policy
    hf_cls = _mm.MolmoAct2ForConditionalGeneration
    if hf_cls is None:
        raise RuntimeError("fast_load needs transformers (the molmoact2 extra)")
    _ORIGINALS["policy_from_pretrained"] = policy_cls.__dict__["from_pretrained"]
    _ORIGINALS["load_pretrained_model"] = hf_cls._load_pretrained_model
    _ORIGINALS["finalize_model_loading"] = hf_cls._finalize_model_loading
    _ORIGINALS["hf_own"] = {
        name: hf_cls.__dict__[name]
        for name in ("_load_pretrained_model", "_finalize_model_loading")
        if name in hf_cls.__dict__
    }
    _ORIGINALS["strict_load"] = _mm._strict_load_safetensors_weights
    policy_cls.from_pretrained = classmethod(_policy_from_pretrained)
    hf_cls._load_pretrained_model = staticmethod(_load_pretrained_model)
    hf_cls._finalize_model_loading = staticmethod(_finalize_model_loading)
    _mm._strict_load_safetensors_weights = _strict_load_safetensors_weights


def _is_ours(owner: Any, name: str, wrapper: Any) -> bool:
    """True if ``owner.<name>`` (own attribute, unwrapping class/staticmethod) is ``wrapper``."""
    attr = owner.__dict__.get(name) if isinstance(owner, type) else getattr(owner, name, None)
    attr = getattr(attr, "__func__", attr)
    if attr is wrapper:
        return True
    log.warning("fast_load: %s.%s was replaced after install(); not restoring it", owner.__name__, name)
    return False


def uninstall() -> None:
    """Undo ``install()``: restore each attribute that still holds this module's wrapper."""
    if not _ORIGINALS:
        return
    policy_cls = _mm.MolmoAct2Policy
    hf_cls = _mm.MolmoAct2ForConditionalGeneration
    if _is_ours(policy_cls, "from_pretrained", _policy_from_pretrained):
        policy_cls.from_pretrained = _ORIGINALS["policy_from_pretrained"]
    wrappers = {
        "_load_pretrained_model": _load_pretrained_model,
        "_finalize_model_loading": _finalize_model_loading,
    }
    for name, wrapper in wrappers.items():
        if not _is_ours(hf_cls, name, wrapper):
            continue
        if name in _ORIGINALS["hf_own"]:
            setattr(hf_cls, name, _ORIGINALS["hf_own"][name])
        else:
            delattr(hf_cls, name)
    if _is_ours(_mm, "_strict_load_safetensors_weights", _strict_load_safetensors_weights):
        _mm._strict_load_safetensors_weights = _ORIGINALS["strict_load"]
    _ORIGINALS.clear()


@contextmanager
def enabled() -> Iterator[None]:
    """``install()`` for the duration of the block (only if it was not installed already)."""
    was_installed = bool(_ORIGINALS)
    install()
    try:
        yield
    finally:
        if not was_installed:
            uninstall()


# --- self-check --------------------------------------------------------------------------------


def _bits(t: torch.Tensor) -> torch.Tensor:
    """View a tensor as integers so that equality is bitwise (NaN == NaN, -0.0 != 0.0)."""
    views: dict[int, torch.dtype] = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    if t.dtype == torch.bool:
        return t
    return t.contiguous().view(views[t.element_size()])


def check_against_file(policy: torch.nn.Module, weights_file: str, device: str = "cuda") -> int:
    """Compare every state_dict tensor of ``policy`` with the file, bitwise and dtype-exact.

    Returns the number of tensors compared; raises on the first difference.
    """
    state = policy.state_dict()
    with safe_open(weights_file, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if keys != set(state):
            raise AssertionError(f"key sets differ: {sorted(keys ^ set(state))[:8]}")
        for key in sorted(keys):
            ref = f.get_tensor(key).to(device)
            got = state[key].to(device)
            if ref.dtype != got.dtype or ref.shape != got.shape or not torch.equal(_bits(ref), _bits(got)):
                raise AssertionError(
                    f"{key}: file {ref.dtype}{tuple(ref.shape)} model {got.dtype}{tuple(got.shape)}"
                )
    return len(keys)


def _self_check(argv: list[str] | None = None) -> None:
    import numpy as np
    from make_local_checkpoint import OUT
    from molmo_common import REFERENCE_SEED, SAMPLE_TASK, load_sample_observation, preprocess

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=OUT)
    parser.add_argument(
        "--no-file-check", action="store_true", help="skip the per-tensor check against the file"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    install()
    t0 = time.perf_counter()
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = args.checkpoint
    policy = _mm.MolmoAct2Policy.from_pretrained(args.checkpoint, config=config)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    load_s = time.perf_counter() - t0
    assert LAST_LOAD is not None
    print(f"fast path: {LAST_LOAD.fast} ({LAST_LOAD.reason or 'full re-save'})")
    print(f"load to CUDA incl. processors: {load_s:.1f} s")
    if not LAST_LOAD.fast:
        raise SystemExit("the fast path was not taken")

    if not args.no_file_check:
        n = check_against_file(policy, str(args.checkpoint / _WEIGHTS_NAME))
        print(f"state_dict vs {_WEIGHTS_NAME}: {n} tensors, keys, dtypes and bits identical")

    observation = load_sample_observation(config)
    with torch.inference_mode():
        for s in range(3):  # same warm-up as verify_local_checkpoint.py
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
    ref_file = Path(__file__).parent / "results" / f"latency_bfloat16_graph-on_seed{REFERENCE_SEED}.npy"
    ref = np.load(ref_file)
    diff = float(np.abs(chunk - ref).max())
    print(f"seed {REFERENCE_SEED} chunk {chunk.shape} vs {ref_file.name}: max|diff| = {diff!r}")
    if diff != 0.0:
        raise SystemExit("actions differ from the reference")


if __name__ == "__main__":
    _self_check()
