"""The reusable part of examples/teacher_student.py: a model whose per-layer
trainable mask lives in a buffer, so freezing rewinds with the weights, in
the forward-scaled 1/sqrt(fan_in) parametrization."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("tracklab")  # the example imports it at module level
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from teacher_student import MLP3, N_LAYERS  # noqa: E402

from rewind.state import TrainerState  # noqa: E402


def test_starts_fully_frozen_and_unfreezes_per_layer():
    m = MLP3(4, 8, 1)
    assert not m.has_trainable()
    assert all(not p.requires_grad for p in m.parameters())
    m.set_trainable(2, True)
    assert m.has_trainable()
    assert all(p.requires_grad for p in m.layers[1].parameters())
    assert all(not p.requires_grad for p in m.layers[0].parameters())
    assert all(not p.requires_grad for p in m.layers[2].parameters())
    m.set_trainable(2, False)
    assert not m.has_trainable()


def test_invalid_layer_is_rejected():
    m = MLP3(4, 8, 1)
    for bad in (0, N_LAYERS + 1):
        with pytest.raises(ValueError):
            m.set_trainable(bad, True)
        with pytest.raises(ValueError):
            m.perturb(bad, 0.1)


def test_perturb_touches_only_that_layer():
    torch.manual_seed(0)
    m = MLP3(4, 8, 1)
    before = [p.detach().clone() for p in m.parameters()]
    m.perturb(1, 0.5)
    after = list(m.parameters())
    n_l1 = len(list(m.layers[0].parameters()))
    assert all(not torch.equal(a, b) for a, b in zip(before[:n_l1], after[:n_l1]))
    assert all(torch.equal(a, b) for a, b in zip(before[n_l1:], after[n_l1:]))


def test_trainable_mask_rewinds_with_the_snapshot():
    """The mask is a buffer, so a TrainerState restore brings it back;
    sync_requires_grad() (called by train_step) re-applies it to the flags."""
    m = MLP3(4, 8, 1)
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    m.set_trainable(3, True)
    snap = TrainerState.capture(0, m, opt, 0.1)          # layer 3 trainable
    m.set_trainable(3, False)
    m.set_trainable(1, True)                              # now layer 1 instead
    assert m.trainable.tolist() == [True, False, False]

    snap.restore(m, opt)
    assert m.trainable.tolist() == [False, False, True]   # buffer restored...
    m.sync_requires_grad()                                # ...and flags follow
    assert all(p.requires_grad for p in m.layers[2].parameters())
    assert all(not p.requires_grad for p in m.layers[0].parameters())


def test_init_is_unit_normal_weights_and_zero_bias():
    torch.manual_seed(3)
    m = MLP3(200, 300, 1)
    for layer in m.layers:
        assert torch.all(layer.bias == 0)
    w = m.layers[0].weight
    assert abs(w.mean().item()) < 0.02 and abs(w.std().item() - 1) < 0.02


def test_preactivations_are_order_one_for_any_width():
    """Standard normal input -> every preactivation has unit variance at
    init, independent of d and hidden, because forward() divides by
    sqrt(fan_in). ReLU halves the variance between layers, so the second
    layer's preactivation is checked against 1/2."""
    torch.manual_seed(4)
    for d, h in ((8, 32), (512, 32), (16, 1024)):
        m = MLP3(d, h, 1)
        x = torch.randn(4096, d)
        with torch.no_grad():
            z1 = m.preactivation(x, 1)
            z2 = m.preactivation(torch.relu(z1), 2)
        assert abs(z1.var().item() - 1) < 0.1, (d, h, z1.var().item())
        assert abs(z2.var().item() - 0.5) < 0.1, (d, h, z2.var().item())


def test_gain_scales_every_preactivation():
    torch.manual_seed(5)
    a = MLP3(8, 16, 1, gain=1.0)
    b = MLP3(8, 16, 1, gain=3.0)
    b.load_state_dict(a.state_dict())
    x = torch.randn(64, 8)
    with torch.no_grad():
        assert torch.allclose(b.preactivation(x, 1), 3 * a.preactivation(x, 1))
        h = torch.relu(a.preactivation(x, 1))
        assert torch.allclose(b.preactivation(h, 2), 3 * a.preactivation(h, 2))
    assert math.isclose(b.gain, 3.0)


def test_activation_choice():
    m = MLP3(4, 8, 1, act="tanh")
    assert m.act is torch.tanh
    with pytest.raises(KeyError):
        MLP3(4, 8, 1, act="sigmoid")
