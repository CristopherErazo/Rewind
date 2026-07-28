"""
TrainerController: owns the training loop and dispatches commands found in
run.poll_commands() to registered handlers.

Written entirely against RunHandle -- it has no idea tracklab, or any
specific project's model/data, exist. Built-in commands (pause, resume,
set_lr, rewind, stop) are handled internally; anything project-specific
(e.g. perturbing a layer) is added via register_handler() from the project
side, with no changes needed here.
"""

import time

from rewind.snapshot import SnapshotManager
from rewind.run_handle import RunHandle


def _handle_set_lr(controller, cmd):
    for g in controller.optimizer.param_groups:
        g["lr"] = cmd["lr"]


def _handle_rewind(controller, cmd):
    snap = controller.snapshots.nearest_before(cmd["step"])
    snap.restore(controller.model, controller.optimizer)
    controller.step = snap.step

    # lineage belongs to the snapshot being restored, not to whatever
    # branch the controller happened to be on when the command arrived
    parent_branch_id = snap.branch_id
    controller._branch_counter += 1
    controller.branch_id = f"b{controller._branch_counter}@t{snap.step}"

    controller.run.track_metric(
        controller.step,
        tags={"branch_id": controller.branch_id},
        parent_branch_id=parent_branch_id,
        fork_step=snap.step,
    )


class TrainerController:
    def __init__(self, model, optimizer, train_step_fn, eval_fn, total_steps,
                 run: RunHandle, logger_kwargs: dict | None = None,
                 log_metrics: list[str] | None = None):
        self.model = model
        self.optimizer = optimizer
        self.train_step_fn = train_step_fn   # () -> None, does one full train step
        self.eval_fn = eval_fn               # () -> dict of metrics
        self.total_steps = total_steps
        self.run = run
        self.logger = run.get_logger(**(logger_kwargs or {}))
        self.log_metrics = log_metrics       # None -> log every metric eval_fn returns

        self.snapshots = SnapshotManager(run)
        self.step = 0
        self.paused = False
        self.branch_id = "root"
        self._branch_counter = 0
        self.eval_schedule: set[int] = set()
        self._stop = False

        self._handlers: dict[str, callable] = {}
        self._register_builtin_handlers()

    # ---------------- extension point for project-specific commands ----------------
    def register_handler(self, cmd_type: str, handler):
        """handler(controller, cmd) -> None. Anything not built into rewind
        gets registered here by the project -- rewind never needs to know
        what it does, only that it exists."""
        self._handlers[cmd_type] = handler

    def _register_builtin_handlers(self):
        self._handlers.update({
            "pause":  lambda c, cmd: setattr(c, "paused", True),
            "resume": lambda c, cmd: setattr(c, "paused", False),
            "set_lr": _handle_set_lr,
            "rewind": _handle_rewind,
            "stop":   lambda c, cmd: setattr(c, "_stop", True),
        })

    # ---------------- on-demand snapshot, for handlers doing something risky ----------------
    def snapshot_now(self):
        return self.snapshots.force_snapshot(
            self.step, self.model, self.optimizer,
            self.optimizer.param_groups[0]["lr"], self.branch_id,
        )

    def _apply(self, cmd: dict):
        handler = self._handlers.get(cmd["type"])
        if handler is None:
            self.logger.warning(f"unknown command type: {cmd['type']!r}, ignoring")
            return
        handler(self, cmd)
        self.logger.info(f"applied {cmd['type']} at step {self.step} (branch={self.branch_id})")
        self.run.track_metric(self.step, tags={"branch_id": self.branch_id},
                               note=f"applied {cmd['type']}", event=cmd)

    def _log_eval(self, metrics: dict):
        keys = self.log_metrics if self.log_metrics is not None else metrics.keys()
        parts = [f"{k}={metrics[k]:.4f}" for k in keys if k in metrics]
        self.logger.info(f"step {self.step}/{self.total_steps} | branch={self.branch_id} | "
                          + " | ".join(parts))

    def run_loop(self):
        self.logger.info(f"starting training | run_id={self.run.run_id} | "
                          f"total_steps={self.total_steps}")
        self.run.set_status(step=0, branch_id=self.branch_id, running=True, paused=False)

        while self.step < self.total_steps and not self._stop:
            for cmd in self.run.poll_commands():
                self._apply(cmd)

            if self.paused:
                self.run.set_status(step=self.step, branch_id=self.branch_id,
                                     running=True, paused=True)
                time.sleep(0.1)
                continue

            if self.step in self.eval_schedule:
                metrics = self.eval_fn()
                self.run.track_metric(self.step, tags={"branch_id": self.branch_id}, **metrics)
                self._log_eval(metrics)
                self.run.set_status(
                    step=self.step, branch_id=self.branch_id, running=True, paused=False,
                    lr=self.optimizer.param_groups[0]["lr"], **metrics,
                )

            self.snapshots.maybe_snapshot(
                self.step, self.model, self.optimizer,
                self.optimizer.param_groups[0]["lr"], self.branch_id,
            )

            self.train_step_fn()
            self.step += 1

        self.logger.info(f"training finished at step {self.step} (branch={self.branch_id})")
        self.run.set_status(step=self.step, branch_id=self.branch_id,
                             running=False, paused=False, done=True)
        self.run.finalize()