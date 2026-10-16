"""CPU check of steer_hooks with a toy VLM (no checkpoint): identity at alpha=0, edit only on text tokens, last block refused."""
import types

import numpy as np
import pytest
import torch
from torch import nn

import steer_hooks


class Block(nn.Module):
    def forward(self, x):
        return (x + 1.0,)


class VLM(nn.Module):
    def __init__(self, n=4, d=8):
        super().__init__()
        self.config = types.SimpleNamespace(image_patch_id=7)
        self.transformer = types.SimpleNamespace(blocks=nn.ModuleList([Block() for _ in range(n)]))

    def forward(self, input_ids=None, attention_mask=None):
        h = torch.zeros(*input_ids.shape, 8)
        for b in self.transformer.blocks:
            h = b(h)[0]
        return h


class Policy:
    def __init__(self):
        self.m = VLM()

    def _backbone(self):
        return self.m


IDS = torch.tensor([[7, 7, 1, 2, 3]])  # 2 image patches, 3 text tokens


def test_alpha_zero_identity():
    p = Policy()
    ref = p.m(input_ids=IDS)
    with steer_hooks.ActivationSteerer(p, 1, np.ones(8), 0.0):
        assert torch.equal(p.m(input_ids=IDS), ref)


def test_edits_text_only_and_removes():
    p = Policy()
    ref = p.m(input_ids=IDS)
    with steer_hooks.ActivationSteerer(p, 1, np.ones(8), 2.0) as s:
        out = p.m(input_ids=IDS)
        assert s.n_calls == 1 and s.n_fires == 1
    assert torch.equal(out[0, :2], ref[0, :2]) and torch.all(out[0, 2:] == ref[0, 2:] + 2.0)
    assert torch.equal(p.m(input_ids=IDS), ref)
    assert not p.m._forward_pre_hooks and not any(b._forward_hooks for b in p.m.transformer.blocks)


def test_last_block_refused():
    with pytest.raises(ValueError):
        steer_hooks.ActivationSteerer(Policy(), 3, np.ones(8), 1.0)
