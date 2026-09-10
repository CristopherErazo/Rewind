"""examples/teacher_student.py -- layer-wise interventions on a student MLP.

A frozen 3-layer *teacher* MLP with fixed random weights labels Gaussian
inputs x in R^d. A *student* with the same architecture learns to imitate it.
Every student layer starts frozen, so nothing is learned until you unfreeze
a layer from the dashboard. Run it from a terminal, then open
examples/dashboard_attach.py and pick the `teacher_student` experiment:

    uv run examples/teacher_student.py --steps 5000
    uv run shiny run examples/dashboard_attach.py

Commands this run adds to the built-ins (pause, resume, stop, set_lr, rewind):

    unfreeze(layer)         let layer 1, 2 or 3 learn
    freeze(layer)           stop it again
    perturb(layer, scale)   add Gaussian noise to that layer's weights only

Metrics: `train_loss`, `eval_loss`, and `w_norm_<k>`, the Frobenius norm of
each layer's weight matrix, which makes it obvious which layers are moving,
which are frozen, and which one a perturbation hit.

Parametrization
---------------
Weights are drawn as N(0, 1) and biases start at zero; the width scaling
lives in the forward pass, where every preactivation is

    gain * (W x / sqrt(fan_in) + b)

with fan_in = d for the first layer and n_hidden for the others. Every
preactivation is therefore order 1 whatever d and n_hidden are, and `gain`
(default 1) sets how far into the nonlinearity the network operates.

What this example demonstrates beyond examples/toy_train.py
------------------------------------------------------------
* Commands with several arguments, and argument validation: an unknown layer
  raises ValueError, which the controller records as a `failed` event
  without stopping training.
* Intervention state that rewinds correctly. Which layers are trainable is
  stored in a model *buffer* (`MLP3.trainable`), so it is part of
  `state_dict()` and therefore of every snapshot. After a rewind the buffer
  comes back and `sync_requires_grad()` re-applies it at the start of the
  next step. Kept only as `requires_grad` flags, the freeze state would drift
  after a rewind exactly like the LR schedulers the README warns about.
* Handlers that change state call `controller.snapshot_now()` first, so
  "rewind to the step of the command" undoes the command.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn
from tracklab import ExperimentTracker

from rewind import ActionSpec, ArgSpec, RunMailbox, TrainerController

N_LAYERS = 3
ACTIVATIONS = {"relu": torch.relu, "tanh": torch.tanh}


class MLP3(nn.Module):
    """Linear -> act -> Linear -> act -> Linear in the 1/sqrt(fan_in)
    forward-scaled parametrization (see module docstring), with a per-layer
    trainable mask stored as a buffer so that it snapshots and rewinds with
    the weights."""

    def __init__(self, d_in: int, hidden: int, d_out: int, act: str = "relu", gain: float = 1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_in, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, d_out),
        ])
        self.act = ACTIVATIONS[act]
        self.gain = float(gain)
        self.register_buffer("trainable", torch.zeros(N_LAYERS, dtype=torch.bool))
        self.reset_parameters()
        self.sync_requires_grad()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """N(0, 1) weights, zero biases. The 1/sqrt(fan_in) is applied in
        forward(), not baked into the weights."""
        for layer in self.layers:
            layer.weight.normal_(0.0, 1.0)
            layer.bias.zero_()

    def preactivation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """gain * (W x / sqrt(fan_in) + b) for layer k (1-based)."""
        layer = self.layer(k)
        return self.gain * (x @ layer.weight.T / math.sqrt(layer.in_features) + layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for k in range(1, N_LAYERS + 1):
            x = self.preactivation(x, k)
            if k < N_LAYERS:
                x = self.act(x)
        return x

    # ---- layer-wise interventions ----

    def layer(self, k: int) -> nn.Linear:
        if not 1 <= k <= N_LAYERS:
            raise ValueError(f"layer must be 1..{N_LAYERS}, got {k}")
        return self.layers[k - 1]

    def set_trainable(self, k: int, flag: bool) -> None:
        self.layer(k)  # validates k
        self.trainable[k - 1] = flag
        self.sync_requires_grad()

    def sync_requires_grad(self) -> None:
        """Make requires_grad follow the buffer. Called after every restore
        (via train_step) because load_state_dict sets the buffer, not the flags."""
        for i, layer in enumerate(self.layers):
            for p in layer.parameters():
                p.requires_grad_(bool(self.trainable[i]))

    def perturb(self, k: int, scale: float) -> None:
        with torch.no_grad():
            for p in self.layer(k).parameters():
                p.add_(scale * torch.randn_like(p))

    def has_trainable(self) -> bool:
        return bool(self.trainable.any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--exp", default="teacher_student")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--d", type=int, default=128, help="input dimension; keep <= hidden so a frozen "
                    "first layer does not discard input directions")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--act", choices=sorted(ACTIVATIONS), default="tanh",
                    help="relu is genuinely nonlinear; tanh at gain 1 is close to linear")
    ap.add_argument("--gain", type=float, default=2.0,
                    help="multiplies every preactivation (teacher and student)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--teacher-seed", type=int, default=1234, help="fixed: same teacher every run")
    ap.add_argument("--seed", type=int, default=0, help="student init and data stream")
    ap.add_argument("--sleep", type=float, default=0.005, help="seconds per step, so there is time to click")
    args = ap.parse_args()

    # The teacher is a function of --teacher-seed only; nothing else touches
    # the RNG before it is built, so every run sees the same teacher.
    torch.manual_seed(args.teacher_seed)
    teacher = MLP3(args.d, args.hidden, 1, act=args.act, gain=args.gain)
    for p in teacher.parameters():
        p.requires_grad_(False)

    torch.manual_seed(args.seed)
    student = MLP3(args.d, args.hidden, 1, act=args.act, gain=args.gain)  # all frozen at start
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    def batch(n: int):
        x = torch.randn(n, args.d)
        with torch.no_grad():
            return x, teacher(x)

    def train_step() -> dict:
        student.sync_requires_grad()  # after a rewind the buffer is restored; flags must follow
        x, y = batch(args.batch)
        loss = nn.functional.mse_loss(student(x), y)
        if student.has_trainable():  # backward() would fail with no trainable parameter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if args.sleep:
            time.sleep(args.sleep)
        return {"train_loss": loss.item()}

    def evaluate() -> dict:
        with torch.no_grad():
            x, y = batch(1024)
            out = {"eval_loss": nn.functional.mse_loss(student(x), y).item()}
        for k in range(1, N_LAYERS + 1):
            out[f"w_norm_{k}"] = student.layer(k).weight.norm().item()
        return out

    # ---- handlers: fn(controller, cmd); snapshot first so the change can be rewound ----

    def unfreeze(controller: TrainerController, cmd: dict) -> None:
        controller.snapshot_now()
        student.set_trainable(cmd["layer"], True)

    def freeze(controller: TrainerController, cmd: dict) -> None:
        controller.snapshot_now()
        student.set_trainable(cmd["layer"], False)

    def perturb(controller: TrainerController, cmd: dict) -> None:
        controller.snapshot_now()
        student.perturb(cmd["layer"], cmd["scale"])

    layer_arg = ArgSpec("int", default=1, description=f"1..{N_LAYERS}, input side first")

    run = ExperimentTracker(args.exp, base_dir=args.base_dir).start_run(vars(args), artifacts=True)
    controller = TrainerController(
        student, optimizer, args.steps, train_step, run,
        eval_fn=evaluate,
        control=RunMailbox(run.run_dir),
        enable_rewind=True,
        train_log_every=10,
        status_every=10,
    )
    controller.eval_schedule = controller.every(10)
    controller.register_handler("unfreeze", unfreeze, spec=ActionSpec(
        "unfreeze", "Unfreeze layer", args={"layer": layer_arg},
        description="Let this layer's weights and bias be trained.",
    ))
    controller.register_handler("freeze", freeze, spec=ActionSpec(
        "freeze", "Freeze layer", args={"layer": layer_arg},
        description="Stop training this layer; its weights stay as they are.",
    ))
    controller.register_handler("perturb", perturb, spec=ActionSpec(
        "perturb", "Perturb layer",
        args={"layer": layer_arg,
              "scale": ArgSpec("float", default=0.5, description="noise std; weights are O(1)")},
        description="Add Gaussian noise to this layer's weights and bias only (snapshots first).",
    ))
    print(f"run {run.run_id} -> {run.run_dir}")
    print("all student layers start frozen; unfreeze one from the dashboard to start learning")
    controller.run_loop()


if __name__ == "__main__":
    main()
