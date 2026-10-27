"""CPU checks of the barrier / task-span steering (no checkpoint download): k=0 identity, the barrier law, the
task-span mask on SO-101 prompts, save/load, equivalence with steervla, the exported LIBERO files, and the
rollout.py flag parsing."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import rollout
import steer_barrier
import steer_hooks

HERE = Path(__file__).parent
STEERVLA = Path("/home/thanhtran/workspace/steervla")
D = 8
IMG = 7  # image patch id of the toy VLM


# ----------------------------------------------------------------------------- toy policy


class Block(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def forward(self, x):
        return (x * 0.9 + 0.1 * (self.i + 1),)


class VLM(nn.Module):
    def __init__(self, n=4, dtype=torch.float32):
        super().__init__()
        self.config = types.SimpleNamespace(image_patch_id=IMG)
        self.transformer = types.SimpleNamespace(blocks=nn.ModuleList([Block(i) for i in range(n)]))
        self.emb = torch.randn(64, D, generator=torch.Generator().manual_seed(0)).to(dtype)

    def forward(self, input_ids=None, attention_mask=None):
        h = self.emb[input_ids.clamp_min(0)]
        for b in self.transformer.blocks:
            h = b(h)[0]
        return h


class Policy:
    def __init__(self, dtype=torch.float32):
        self.m = VLM(dtype=dtype)

    def _backbone(self):
        return self.m


# toy vocabulary: token id -> string. The prompt mimics processor_molmoact2._build_robot_text.
VOCAB = {7: "<im>", 1: "<|im_start|>user\n", 2: "The", 3: " task", 4: " is", 5: " to", 10: " pick", 11: " up",
         12: " the", 13: " red", 14: " marker", 20: ".", 21: " The", 22: " setup", 23: " is", 24: " so101",
         25: "<s1>", 26: "<s2>"}


class StubTok:
    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)


# 3 image patches, then "The task is to pick up the red marker. The setup is so101<s1><s2>"
IDS = torch.tensor([[7, 7, 7, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 26]])
TASK_COLS = [8, 9, 10, 11, 12]


def _barrier(seed=0, tau0=0.0):
    torch.manual_seed(seed)
    bar = steer_barrier.DialBarrier(torch.randn(3, D), torch.randn(D), scale=10.0, tau0=tau0, d_attn=8, n_heads=2,
                                    d_hidden=16)
    with torch.no_grad():  # make the direction and threshold depend on the tokens (zero-init heads otherwise)
        bar.out.weight.normal_(0, 0.3)
        bar.thr.weight.normal_(0, 0.3)
    return bar.eval()


def _span():
    return steer_hooks.make_task_span_fn(StubTok())


# ----------------------------------------------------------------------------- tests


def test_task_span_stub():
    fn = _span()
    m = fn(IDS)
    assert m[0].nonzero().flatten().tolist() == TASK_COLS
    assert fn.last_text == " pick up the red marker" and fn.last_n_tok == 5
    with pytest.raises(RuntimeError, match="task span not found"):
        fn(torch.tensor([[7, 7, 1, 10, 11, 12, 13, 14]]))
    assert not steer_hooks.span_mask(["The task is to", " .", " The setup is"]).any()  # empty task: no span


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["barrier", "const"])
def test_k_zero_bitwise_identity(dtype, kind):
    p = Policy(dtype)
    ref = p.m(input_ids=IDS)
    mod = _barrier() if kind == "barrier" else steer_barrier.ConstVec(torch.ones(D), 3.0)
    with steer_hooks.BarrierSteerer(p, 1, mod, 0.0, "task", task_span=_span()) as s:
        out = p.m(input_ids=IDS)
    assert torch.equal(out, ref)
    assert s.n_calls == 1 and s.n_fires == 1 and s.stats()["mean_zero_frac"] == 1.0


def test_barrier_edits_task_span_only_reaches_threshold_and_scales_with_k():
    p = Policy()
    bar = _barrier(tau0=5.0)
    states = {}
    p.m.transformer.blocks[2].register_forward_pre_hook(lambda mod, a: states.__setitem__("in2", a[0].clone()))
    ref = p.m(input_ids=IDS)
    h0 = states["in2"][0]  # block 1 output without steering
    deltas, norms = {}, {}
    for k in (1.0, 0.5, 2.0):
        with steer_hooks.BarrierSteerer(p, 1, bar, k, "task", task_span=_span()) as s:
            out = p.m(input_ids=IDS)
        assert not torch.equal(out, ref)
        h1 = states["in2"][0]
        other = [c for c in range(IDS.shape[1]) if c not in TASK_COLS]
        assert torch.equal(h1[other], h0[other])  # only the task span is edited
        deltas[k], norms[k] = h1 - h0, s.stats()["mean_edit_norm"]
        st = s.stats()
        assert st["n_calls"] == st["n_fires"] == 1 and st["mean_n_tok"] == 5
    h = h0[TASK_COLS][None]
    m = torch.ones(1, len(TASK_COLS), dtype=torch.bool)
    a, tau = bar.direction_thr(h, m)
    after = (a[0] * (h0 + deltas[1.0])[TASK_COLS]).sum(-1)
    assert torch.all(after >= tau[0] - 1e-4)  # k = 1: <a, h + u> >= tau on the masked tokens
    for k in (0.5, 2.0):  # k scales the edit: u_k = k * u_1
        assert torch.allclose(deltas[k], k * deltas[1.0], atol=1e-5)
        assert norms[k] == pytest.approx(k * norms[1.0], rel=1e-5)


def test_barrier_zero_edit_where_constraint_holds():
    bar = _barrier()
    h = torch.randn(2, 6, D) * 3
    m = torch.ones(2, 6, dtype=torch.bool)
    a, tau = bar.direction_thr(h, m)
    b = (a * h).sum(-1)
    holds = b >= tau
    assert holds.any() and (~holds).any()
    u1, _ = steer_barrier.apply_k(bar, h, m, 1.0)
    assert torch.all((a * (h + u1)).sum(-1) >= tau - 1e-4)  # k = 1 reaches the threshold
    for k in (0.0, 0.5, 1.0, 2.0):
        u, info = steer_barrier.apply_k(bar, h, m, k)
        assert torch.equal(u[holds], torch.zeros_like(u[holds]))  # zero where <a, h> >= tau, for every k
        assert torch.equal(u, k * u1)  # k scales the edit
        assert torch.equal(info["gap"] == 0, holds | (k == 0.0))
    with torch.no_grad():
        bar.tau0.fill_(-1e3)  # threshold far below <a, h>: no edit anywhere, the steered output is the input
    p = Policy()
    ref = p.m(input_ids=IDS)
    with steer_hooks.BarrierSteerer(p, 1, bar, 1.0, "task", task_span=_span()) as s:
        assert torch.equal(p.m(input_ids=IDS), ref)
    assert s.stats()["mean_zero_frac"] == 1.0


def test_positions_task_needs_span_and_prompt_pass_only():
    with pytest.raises(ValueError, match="task_span"):
        steer_hooks.BarrierSteerer(Policy(), 1, _barrier(), 1.0, "task")
    with pytest.raises(ValueError, match="task_span"):
        steer_hooks.ActivationSteerer(Policy(), 1, np.ones(D), 1.0, "task")
    p = Policy()
    with steer_hooks.BarrierSteerer(p, 1, _barrier(), 1.0, "task", task_span=_span()) as s:
        p.m(input_ids=IDS)
        p.m.transformer.blocks[1](torch.zeros(1, 3, D))  # a cached decode step: different length, not edited
    assert s.n_calls == 1 and s.n_fires == 1


def test_save_load_roundtrip(tmp_path):
    bar = _barrier(seed=3)
    path = tmp_path / "b.pt"
    steer_barrier.save_module(bar, path, {"note": "x"})
    (tmp_path / "b.json").write_text(json.dumps({"layer": 1, "positions": "task", "policy_checkpoint": "a/B"}))
    mod, meta = steer_barrier.load_barrier(path)
    assert meta["layer"] == 1
    sd0, sd1 = bar.state_dict(), mod.state_dict()
    assert sd0.keys() == sd1.keys() and all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    h, m = torch.randn(2, 4, D) * 3, torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():  # same weights; BLAS may differ by an ulp with the tensors' memory alignment
        assert torch.allclose(bar(h, m)[0], mod(h, m)[0], atol=1e-5)
    with pytest.warns(UserWarning, match="trained on"):
        assert steer_barrier.check_policy(meta, "allenai/MolmoAct2-SO100_101")
    assert not steer_barrier.check_policy(meta, "a/B")


def test_exported_const_vector_matches_constvec_through_activation_steerer(tmp_path):
    """--steer-vector const.npy --steer-k 1 (ActivationSteerer) == the trained ConstVec (BarrierSteerer), bitwise."""
    w = torch.randn(D)
    cv = steer_barrier.ConstVec(w, 36.6)
    for k in (1.0, 0.5):
        p = Policy(torch.bfloat16)
        with steer_hooks.BarrierSteerer(p, 1, cv, k, "task", task_span=_span()):
            a = p.m(input_ids=IDS)
        with steer_hooks.ActivationSteerer(p, 1, w.numpy(), k * 36.6, "task", task_span=_span()):
            b = p.m(input_ids=IDS)
        assert torch.equal(a, b) if k == 1.0 else torch.allclose(a.float(), b.float())


@pytest.mark.parametrize("name", ["barrier_r3_L06.pt"])
def test_exported_libero_barrier_loads(name):
    path = HERE / "barriers" / name
    mod, meta = steer_barrier.load_barrier(path)
    assert meta["layer"] == 6 and meta["positions"] == "task" and meta["policy_checkpoint"] == "allenai/MolmoAct2-LIBERO"
    assert tuple(mod.bank.shape) == (5, 2560) and meta["source"]["sha256"]
    v, vmeta = steer_hooks.load_vector(HERE / "vectors" / "const_r3_L06.npy")
    assert v.shape == (2560,) and vmeta["alpha_unit_value"] == mod.scale


# ----------------------------------------------------------------------------- real tokenizer


def _real_tokenizer():
    try:
        from huggingface_hub import snapshot_download
        from transformers import Qwen2Tokenizer

        loc = snapshot_download("allenai/MolmoAct2-SO100_101", local_files_only=True,
                                allow_patterns=["tokenizer*", "*.json", "*.jinja"])
        return Qwen2Tokenizer.from_pretrained(loc, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"MolmoAct2-SO100_101 tokenizer not in the local HF cache ({e!r})")


@pytest.mark.parametrize("task,n_tok", [("pick up the marker", 4), ("pick up the red marker", 5),
                                        ("pick up the pen", 4), ("pick up the blue marker", 5)])
def test_task_span_real_tokenizer(task, n_tok):
    tok = _real_tokenizer()
    state = "".join(f"<state_{i}>" for i in (3, 200, 17))
    prompt = (f"<|image|><|im_start|>user\nThe task is to {task}. The setup is so100_101_single_arm. "
              f"The current state of the robot is {state}. The expected control mode is joint_pos. Given these, "
              "what action should the robot take to complete the task?<|im_end|>\n<|im_start|>assistant\n")
    ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]])
    fn = steer_hooks.make_task_span_fn(tok)
    m = fn(ids)
    assert fn.last_text == f" {task}" and int(m.sum()) == n_tok == fn.last_n_tok
    cols = m[0].nonzero().flatten()
    assert bool((cols.diff() == 1).all())  # one contiguous span


# ----------------------------------------------------------------------------- steervla equivalence


def _steervla():
    if not (STEERVLA / "steervla").is_dir():
        pytest.skip(f"steervla not found at {STEERVLA}")
    sys.path.insert(0, str(STEERVLA))
    try:
        from steervla.steering_dial import barrier as sb
        from steervla.steering_dial.hooks import DialHooks
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"steervla cannot be imported here ({e!r})")
    return sb, DialHooks


class _KEdit(nn.Module):
    """k times the steervla module's edit (the reference for k != 1)."""

    def __init__(self, base, k):
        super().__init__()
        self.base, self.k = base, k

    def forward(self, h, m):
        u, info = self.base(h, m)
        return self.k * u, {**info, "gap": self.k * info["gap"]}


@pytest.mark.parametrize("k", [1.0, 0.5, 2.0, 0.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_equivalent_to_steervla(k, dtype, tmp_path):
    """k = 1: bitwise steervla's DialHooks + DialBarrier. Other k: bitwise k times steervla's k = 1 edit."""
    sb, DialHooks = _steervla()
    steer_barrier.save_module(_barrier(seed=5, tau0=3.0), tmp_path / "b.pt")
    ours, _ = steer_barrier.load_barrier(tmp_path / "b.pt")
    theirs = sb.load_module(tmp_path / "b.pt")  # their loader reads our file
    p = Policy(dtype)
    with steer_hooks.BarrierSteerer(p, 2, ours, k, "task", task_span=_span()):
        a = p.m(input_ids=IDS)
    dh = DialHooks(p, 2, positions="task")
    dh.task_span = _span()
    dh.set_plans([theirs if k == 1.0 else _KEdit(theirs, k)])
    with dh, torch.no_grad():
        b = p.m(input_ids=IDS)
    assert torch.equal(a, b)


# ----------------------------------------------------------------------------- rollout.py flags


def test_rollout_split_argv_strips_steer_flags():
    argv = ["--tag", "t", "--steer-barrier", "steering/barriers/barrier_r3_L06.pt", "--steer-k=0.5",
            "--steer-positions", "task", "--steer-layer=6", "--robot.max_relative_target=4", "--task=pick up the pen"]
    tag, rest, flags = rollout.split_argv(argv)
    assert tag == "t"
    assert flags["steer"] == {"barrier": "steering/barriers/barrier_r3_L06.pt", "k": "0.5", "positions": "task",
                              "layer": "6"}
    assert not any(a.startswith("--steer") or a.endswith(".pt") for a in rest)
    assert rest[:2] == ["--robot.max_relative_target=4", "--task=pick up the pen"]
    _, rest, flags = rollout.split_argv(["--steer-vector=v.npy", "--steer-alpha", "2", "--robot.max_relative_target=4"])
    assert flags["steer"] == {"vector": "v.npy", "alpha": "2"} and not any("steer" in a for a in rest)
    with pytest.raises(SystemExit, match="needs a value"):
        rollout.split_argv(["--robot.max_relative_target=4", "--steer-k"])


def test_rollout_task_arg():
    assert rollout._task_arg(["--robot.x=1", "--task=pick up the pen"]) == "pick up the pen"
    assert rollout._task_arg(["--dataset.single_task="]) == ""
    assert rollout._task_arg(["--robot.x=1"]) is None


def test_preflight_refuses_empty_task():
    tok = _real_tokenizer()
    import steer_patch

    fn = steer_hooks.make_task_span_fn(tok)
    steer_patch.preflight_task_span(fn, tok, "pick up the red marker")
    with pytest.raises(RuntimeError, match="task span not found"):
        steer_patch.preflight_task_span(fn, tok, "")


def test_rollout_install_flag_conflicts():
    with pytest.raises(SystemExit, match="mutually exclusive"):
        rollout._install_steering({"vector": "v.npy", "barrier": "b.pt"})
    with pytest.raises(SystemExit, match="need --steer-vector or --steer-barrier"):
        rollout._install_steering({"k": "1"})
