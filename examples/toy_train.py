"""examples/toy_train.py -- the smallest controllable training script.

A two-layer MLP regresses a noisy sum of its inputs. Batches are drawn from
the torch RNG, so a rewind replays exactly the same losses on the new
branch. Run it from a terminal, then open examples/dashboard_attach.py and
pick the run in the sidebar:

    PY=../.venv/Scripts/python.exe          # shared venv, see CLAUDE.md
    $PY examples/toy_train.py --steps 5000 --exp toy
    ../.venv/Scripts/shiny.exe run --reload examples/dashboard_attach.py

The script also shows the one custom intervention every project ends up
wanting: a `perturb` command that adds noise to the weights, registered with
an ActionSpec so it appears in the dashboard's control panel automatically.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tracklab import ExperimentTracker

from rewind import ActionSpec, ArgSpec, RunMailbox, TrainerController


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--exp", default="toy")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--sleep", type=float, default=0.01, help="seconds per step, so there is time to click")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, args.hidden), torch.nn.Tanh(), torch.nn.Linear(args.hidden, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def batch(n: int):
        x = torch.randn(n, 8)
        y = x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n, 1)
        return x, y

    def train_step() -> dict:
        x, y = batch(64)
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.sleep:
            time.sleep(args.sleep)
        return {"train_loss": loss.item()}

    def evaluate() -> dict:
        with torch.no_grad():
            x, y = batch(512)
            return {"eval_loss": torch.nn.functional.mse_loss(model(x), y).item()}

    def perturb(controller: TrainerController, cmd: dict) -> None:
        controller.snapshot_now()  # so the perturbation can be rewound
        with torch.no_grad():
            for p in controller.model.parameters():
                p.add_(cmd["scale"] * torch.randn_like(p))

    run = ExperimentTracker(args.exp, base_dir=args.base_dir).start_run(vars(args), artifacts=True)
    controller = TrainerController(
        model, optimizer, args.steps, train_step, run,
        eval_fn=evaluate,
        control=RunMailbox(run.run_dir),
        enable_rewind=True,
        train_log_every=10,
        status_every=10,
    )
    controller.eval_schedule = controller.every(50)
    controller.register_handler("perturb", perturb, spec=ActionSpec(
        "perturb", "Perturb weights",
        args={"scale": ArgSpec("float", default=0.05, description="noise std")},
        description="Add Gaussian noise to every parameter (snapshots first).",
    ))
    print(f"run {run.run_id} -> {run.run_dir}")
    controller.run_loop()


if __name__ == "__main__":
    main()
