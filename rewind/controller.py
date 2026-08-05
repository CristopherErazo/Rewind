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
from rewind.protocols import TrackingHandle, ControlHandle
from rewind.snapshot import SnapshotManager
from rewind.control import _NullControl


EVAL_ARTIFACTS_GROUP = "eval_artifacts"


def _handle_set_lr(controller, cmd):
    """Set the learning rate of all param groups to cmd['lr']."""
    for g in controller.optimizer.param_groups:
        g["lr"] = cmd["lr"]


def _handle_rewind(controller, cmd):
    """Rewind the model/optimizer state to the nearest snapshot at or before cmd['step'].
    Create a new branch_id for the rewinded state, and log the fork in metrics.csv.
    """
    # Find the nearest snapshot at or before the requested step. This will be a TrainerState object.
    # The nearest_before method will walk the lineage of branches to find the correct snapshot.
    if controller.enable_rewind is False:
        controller.logger.error("rewind is disabled; construct with enable_rewind=True")
        # raise exception
        raise RuntimeError("rewind is disabled; construct with enable_rewind=True")
    snap = controller.snapshots.nearest_before(cmd["step"], controller.branch_id)
    snap.restore(controller.model, controller.optimizer)
    controller.step = snap.step

    # Create a new branch_id for the rewinded state, and register it with the SnapshotManager.
    parent_branch_id = snap.branch_id
    controller._branch_counter += 1
    controller.branch_id = f"b{controller._branch_counter}@t{snap.step}"
    controller.snapshots.register_branch(controller.branch_id, parent_branch_id, snap.step)

    # Log the fork in metrics.csv, so users can see the lineage of branches.
    controller.run.track_metric(
        controller.step, tags=controller._branch_tags(),
        parent_branch_id=parent_branch_id, fork_step=snap.step,
    )


class TrainerController:
    """
    TrainerController: owns the training loop and dispatches commands found in
    run.poll_commands() to registered handlers.

    Args:
        model: The model to train and evaluate.
        optimizer: The optimizer to use for training.
        total_steps: The total number of training steps to run.
        train_step_fn: A callable that performs a single training step.
        eval_fn: A callable that evaluates the model and returns a dict of metrics.       
        run: A TrackingHandle for logging metrics and artifacts.
        eval_art_fun: A callable that returns a dict of evaluation artifacts (optional).
        control: A ControlHandle for receiving commands (optional).
        enable_rewind: Whether to enable rewind functionality (optional).
        logger_kwargs: Additional keyword arguments for the logger (optional).
        log_metrics: A list of metric names to log during evaluation (optional).
    
    Provides methods to register custom command handlers, take snapshots, and run the training loop.
    The TrainerController is designed to be agnostic of the specific model or data being used, 
    and can be extended with project-specific command handlers.
    """
    def __init__(self, model, optimizer, total_steps, train_step_fn, eval_fn, 
                 run: TrackingHandle,
                 eval_art_fun = None,
                 control: ControlHandle | None = None,
                 enable_rewind: bool  = False,     
                 logger_kwargs: dict | None = None,
                 log_metrics: list[str] | None = None):
        
        self.model = model
        self.optimizer = optimizer
        self.train_step_fn = train_step_fn
        self.eval_fn = eval_fn
        self.eval_art_fn = eval_art_fun
        self.total_steps = total_steps
        self.run = run
        self.control = control or _NullControl()
        self.logger = run.get_logger(**(logger_kwargs or {}))
        self.log_metrics = log_metrics

        self.enable_control = control is not None
        self.enable_rewind = enable_rewind # even if control is given, user can disable rewind if they want
        self.snapshots = SnapshotManager(run) if self.enable_rewind else None

        self.step = 0
        self.paused = False
        self.branch_id = "root"
        self._branch_counter = 0
        self.eval_schedule: set[int] = set()
        self.eval_artifacts_schedule: set[int] = set()
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
        handlers = {
            "pause":  lambda c, cmd: setattr(c, "paused", True),
            "resume": lambda c, cmd: setattr(c, "paused", False),
            "stop":   lambda c, cmd: setattr(c, "_stop", True),
            "set_lr": _handle_set_lr,
        }
        if self.snapshots is not None:
            handlers["rewind"] = _handle_rewind
        self._handlers.update(handlers)

    # ---------------- on-demand snapshot, for handlers doing something risky ----------------
    def snapshot_now(self):
        """Force a snapshot of the current model/optimizer state, even if it's not on the eval schedule."""
        if self.snapshots is None:
            # Log and Raise an error
            self.logger.error("snapshotting is disabled; construct with enable_rewind=True")
            raise RuntimeError("snapshotting is disabled; construct with enable_rewind=True")
        return self.snapshots.force_snapshot(
            self.step, self.model, self.optimizer,
            self.optimizer.param_groups[0]["lr"], self.branch_id,
        )
    
    # ---------------- shared branch-scoping helpers ----------------
    def _branch_tags(self) -> dict:
        """No tag at all if rewind is disabled — keeps metrics.csv free of a
        constant 'root' column for users who never branch."""
        return {"branch_id": self.branch_id} if self.enable_rewind else {}
    
    def _branch_scoped_name(self, base_name: str) -> str:
        """If rewind is enabled, prefix the artifact name with the branch_id,
        so that artifacts from different branches don't collide. If rewind is
        disabled, return the base name unchanged."""
        return f"{base_name}__{self.branch_id}" if self.enable_rewind else base_name

    def _apply(self, cmd: dict):
        """Dispatch a command to its registered handler, or log a warning if no handler is registered."""
        cmd_type = cmd.get("type")
        handler = self._handlers.get(cmd_type)
        if handler is None:
            self.logger.warning(f"unknown command type: {cmd_type!r}, ignoring")
            return
        try:
            handler(self, cmd)
        except Exception:
            self.logger.exception(f"command {cmd_type!r} failed at step {self.step}, ignoring")
            self.run.track_metric(self.step, tags=self._branch_tags(),
                                   note=f"command failed: {cmd_type}", event=cmd)
            return
        self.logger.info(f"applied {cmd_type} at step {self.step} (branch={self.branch_id})")
        self.run.track_metric(self.step, tags=self._branch_tags(),
                               note=f"applied {cmd_type}", event=cmd)

    def _log_eval(self, metrics: dict):
        keys = self.log_metrics if self.log_metrics is not None else metrics.keys()
        parts = [f"{k}={metrics[k]:.4f}" for k in keys if k in metrics]

        msg = f"step {self.step}/{self.total_steps} | " 
        msg += f"branch={self.branch_id} | " if self.enable_rewind else ""
        msg += " | ".join(parts)
        self.logger.info(msg)


    def _log_eval_artifacts(self, artifacts: dict):
        """
        Save evaluation artifacts to the run, under the group specified and
        with names prefixed by the current branch_id in case of rewinding. 
        Log how many artifacts were saved.

        Parameters:
        artifacts (dict): A dictionary {key:value} where keys are either name strings or tuples of (name, group)
        and values are either data or tuples of (data, type). The group defaults to None if not provided, 
        and the type defaults to 'tensor' if not provided. 
        """
        for key, value in artifacts.items():
            data, atype = value if isinstance(value, tuple) else (value, "tensor")
            name, group = key if isinstance(key, tuple) else (key, None)
            self.run.track_artifact(
                data, step=self.step, group=group,
                name=self._branch_scoped_name(name), type=atype,
            )
        msg = f"saved {len(artifacts)} eval artifact(s) at step {self.step}"
        msg += f" (branch={self.branch_id})" if self.enable_rewind else ""
        self.logger.info(msg)


    def run_loop(self):
        self.logger.info(f"starting training | run_id={self.run.run_id} | "
                          f"total_steps={self.total_steps}")
        self.run.set_status(step=0, branch_id=self.branch_id, running=True, paused=False)

        while self.step < self.total_steps and not self._stop:
            for cmd in self.control.poll_commands():
                self._apply(cmd)

            if self.paused:
                self.run.set_status(step=self.step, branch_id=self.branch_id,
                                         running=True, paused=True)
                time.sleep(0.1)
                continue

            if self.step in self.eval_schedule:
                metrics = self.eval_fn()
                self.run.track_metric(self.step, tags=self._branch_tags(), **metrics)
                self._log_eval(metrics)
                self.run.set_status(step=self.step, branch_id=self.branch_id, 
                                        running=True, paused=False)

            if self.eval_art_fn is not None and self.step in self.eval_artifacts_schedule:
                self._log_eval_artifacts(self.eval_art_fn())

            if self.snapshots is not None:
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