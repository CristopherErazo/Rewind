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


ACTIVATIONS = {"relu": torch.relu, "tanh": torch.tanh}


class MLP(nn.Module):
    """
    Standard Multi-layer perceptron with L hidden layers, each followed by an activation function. 
    """
    def __init__(self, d_in: int, hidden: int, d_out: int, act: str = "relu", gain: float = 1.0, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = ACTIVATIONS[act]
        self.gain = float(gain)
        self.n_layers = n_layers

        # Input layer
        self.layers.append(nn.Linear(d_in, hidden, bias=False))

        # Hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(nn.Linear(hidden, hidden, bias=False))

        # Output layer
        self.layers.append(nn.Linear(hidden, d_out, bias=False))

        self.register_buffer("trainable", torch.zeros(n_layers, dtype=torch.bool))
        self.sync_requires_grad()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for k in range(self.n_layers):
            x = self.layers[k](x)
            x = self.act(self.gain * x)
        return x


    # ---- layer-wise interventions ----

    def layer(self, k: int) -> nn.Linear:
        if not 1 <= k <= self.n_layers:
            raise ValueError(f"layer must be 1..{self.n_layers}, got {k}")
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
    ap.add_argument("--d", type=int, default=128, help="input dimension")
    ap.add_argument("--hidden", type=int, default=32, help="hidden layers width")
    ap.add_argument("--act", choices=sorted(ACTIVATIONS), default="tanh")
    ap.add_argument("--gain", type=float, default=2.0, help="multiplies every preactivation (teacher and student)")
    ap.add_argument("--n-layers", type=int, default=3, help="teacher and student have same depth")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.5e-3)
    ap.add_argument("--teacher-seed", type=int, default=1234, help="fixed: same teacher every run")
    ap.add_argument("--seed", type=int, default=0, help="student init and data stream")
    ap.add_argument("--sleep", type=float, default=0.01, help="seconds per step, so there is time to click")
    args = ap.parse_args()

    # The teacher is a function of --teacher-seed only; nothing else touches
    # the RNG before it is built, so every run sees the same teacher.
    torch.manual_seed(args.teacher_seed)

    teacher = MLP(args.d, args.hidden, 1, act=args.act, gain=args.gain, n_layers=args.n_layers)
    for p in teacher.parameters():
        p.requires_grad_(False)

    torch.manual_seed(args.seed)
    student = MLP(args.d, args.hidden, 1, act=args.act, gain=args.gain, n_layers=args.n_layers)  # all frozen at start
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
        return out

    # ---- handlers: fn(controller, cmd); snapshot first so the change can be rewound ----

    def unfreeze(controller: TrainerController, cmd: dict) -> None:
        controller.snapshot_now()
        student.set_trainable(cmd["layer"], True)

    def freeze(controller: TrainerController, cmd: dict) -> None:
        controller.snapshot_now()
        student.set_trainable(cmd["layer"], False)

    layer_arg = ArgSpec("int", default=1, description=f"1..{args.n_layers}, input side first")

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
    print(f"run {run.run_id} -> {run.run_dir}")
    print("all student layers start frozen; unfreeze one from the dashboard to start learning")
    controller.run_loop()


if __name__ == "__main__":
    main()
