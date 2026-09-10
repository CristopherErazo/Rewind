from __future__ import annotations

import pytest
import torch

from rewind import RunMailbox, TrainerController
from tests.fakes import FakeRun


class Toy:
    """A one-parameter model trained on synthetic data drawn from the torch
    RNG, so the loss sequence is a deterministic function of (weights, RNG)
    and rewinds are exactly reproducible."""

    def __init__(self):
        torch.manual_seed(0)
        self.model = torch.nn.Linear(4, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.losses: list[float] = []

    def train_step(self) -> dict:
        x = torch.randn(8, 4)
        y = x.sum(dim=1, keepdim=True)
        loss = ((self.model(x) - y) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        return {"train_loss": loss.item()}

    def evaluate(self) -> dict:
        # Draws from the torch RNG on purpose, like a real eval batch would:
        # a rewind is only exact if the snapshot precedes this draw.
        with torch.no_grad():
            x = torch.randn(16, 4)
            eval_loss = ((self.model(x) - x.sum(dim=1, keepdim=True)) ** 2).mean().item()
            w = self.model.weight.abs().sum().item()
        return {"weight_l1": w, "eval_loss": eval_loss}


@pytest.fixture
def toy():
    return Toy()


@pytest.fixture
def fake_run(tmp_path):
    return FakeRun(tmp_path / "exp" / "run_001")


@pytest.fixture
def make_controller(toy, fake_run):
    def _make(total_steps=50, control=True, **kwargs):
        ctrl = TrainerController(
            toy.model, toy.optimizer, total_steps, toy.train_step, fake_run,
            eval_fn=toy.evaluate,
            control=RunMailbox(fake_run.run_dir) if control else None,
            **kwargs,
        )
        return ctrl
    return _make
