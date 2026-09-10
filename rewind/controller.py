"""rewind/controller.py

TrainerController owns the training loop and dispatches commands found on
its ControlHandle to registered handlers.

It is written against TrackingHandle / ControlHandle only: no tracker, no
project model or data. Built-in commands (pause, resume, stop, set_lr, and
rewind when enabled) live here as module-level handlers; anything
project-specific is added with `register_handler()`.

Lifecycle, as seen in control/status.json's `state` field:

    running -> paused -> running ... -> done | stopped
                                     -> crashed      (exception, re-raised)
                                     -> interrupted  (Ctrl-C, re-raised)

`run.finalize()` is always called, whatever the exit path.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from .control import RunState, RunStatus, _NullControl
from .events import append_event
from .protocols import ControlHandle, TrackingHandle
from .registry import ActionSpec, BUILTIN_ACTIONS, REWIND_ACTION, coerce_args, write_actions
from .snapshot import SnapshotManager

EVAL_ARTIFACTS_GROUP = "eval_artifacts"


# ---------------------------------------------------------------------------
# built-in handlers: handler(controller, cmd) -> None
# ---------------------------------------------------------------------------

def _handle_pause(controller: "TrainerController", cmd: dict) -> None:
    controller.paused = True
    controller._set_state(RunState.PAUSED)


def _handle_resume(controller: "TrainerController", cmd: dict) -> None:
    controller.paused = False
    controller._set_state(RunState.RUNNING)


def _handle_stop(controller: "TrainerController", cmd: dict) -> None:
    controller._stop = True
    controller.paused = False  # a paused run must fall through to the exit path


def _handle_set_lr(controller: "TrainerController", cmd: dict) -> None:
    for g in controller.optimizer.param_groups:
        g["lr"] = cmd["lr"]
    controller.status.update(lr=cmd["lr"])


def _handle_rewind(controller: "TrainerController", cmd: dict) -> None:
    """Restore the nearest snapshot at or before cmd['step'] and continue on
    a fresh branch. The fork is recorded in events.jsonl; metric rows carry
    the new branch_id tag from here on."""
    if controller.snapshots is None:
        raise RuntimeError("rewind is disabled; construct with enable_rewind=True")
    snap = controller.snapshots.nearest_before(cmd["step"], controller.branch_id)
    snap.restore(controller.model, controller.optimizer)

    parent_branch_id = controller.branch_id
    from_step = controller.step
    controller.step = snap.step
    controller._branch_counter += 1
    controller.branch_id = f"b{controller._branch_counter}@t{snap.step}"
    controller.snapshots.register_branch(controller.branch_id, snap.branch_id, snap.step)

    controller.status.update(step=controller.step, branch_id=controller.branch_id,
                             lr=controller._current_lr())
    append_event(controller.run.run_dir, "fork", controller.step, controller.branch_id,
                 parent_branch_id=parent_branch_id, snapshot_branch_id=snap.branch_id,
                 fork_step=snap.step, from_step=from_step)


# ---------------------------------------------------------------------------

class TrainerController:
    """Run a step-based training loop that can be steered from outside.

    Parameters
    ----------
    model, optimizer
        The objects to train, snapshot and restore. `optimizer.param_groups`
        is used by `set_lr` and for the `lr` field in status.
    total_steps
        Loop exit condition: `step` counts train_step_fn calls on the
        current branch, so a rewind makes the run longer in wall-clock.
    train_step_fn
        `() -> dict | None`. One optimizer step. If it returns a dict of
        floats and `train_log_every > 0`, they are logged as metrics every
        `train_log_every` steps.
    run
        A TrackingHandle. Its `run_dir` hosts the control/ folder.
    eval_fn
        Optional `() -> dict`. Called at every step in `eval_schedule`; the
        dict is logged as metrics.
    eval_artifacts_fn
        Optional `() -> dict`. Called at every step in
        `eval_artifacts_schedule`. Keys are `name` or `(name, group)`,
        values are `data` or `(data, type)`.
    control
        A ControlHandle (usually `RunMailbox(run.run_dir)`). None gives a
        plain loop that never reads commands.
    enable_rewind
        Build a SnapshotManager and register the `rewind` command.
    status_every
        Write status.json every N steps (state transitions always write).
    poll_every
        Poll the mailbox every N steps (a paused loop polls continuously).
    train_log_every
        Log train_step_fn's returned dict every N steps. 0 disables.
    logger_kwargs, log_metrics
        Passed to `run.get_logger`; which eval metrics to echo to the log.

    Determinism contract for exact rewinds: every source of randomness in
    train_step_fn must derive from the torch / numpy / python RNGs, which
    are captured and restored. Anything else with state (an LR scheduler, a
    data iterator, a GradScaler) is not restored yet and will drift after a
    rewind.
    """

    def __init__(self, model, optimizer, total_steps: int, train_step_fn: Callable,
                 run: TrackingHandle, *,
                 eval_fn: Callable[[], dict] | None = None,
                 eval_artifacts_fn: Callable[[], dict] | None = None,
                 control: ControlHandle | None = None,
                 enable_rewind: bool = False,
                 status_every: int = 10,
                 poll_every: int = 1,
                 train_log_every: int = 0,
                 logger_kwargs: dict | None = None,
                 log_metrics: list[str] | None = None):
        if status_every < 1 or poll_every < 1 or train_log_every < 0:
            raise ValueError("status_every and poll_every must be >= 1, train_log_every >= 0")

        self.model = model
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.train_step_fn = train_step_fn
        self.eval_fn = eval_fn
        self.eval_artifacts_fn = eval_artifacts_fn
        self.run = run
        self.control = control or _NullControl()
        self.enable_rewind = enable_rewind
        self.status_every = status_every
        self.poll_every = poll_every
        self.train_log_every = train_log_every
        self.logger = run.get_logger(**(logger_kwargs or {}))
        self.log_metrics = log_metrics

        self.snapshots = SnapshotManager(run) if enable_rewind else None
        self.status = RunStatus(run.run_dir)

        self.step = 0
        self.paused = False
        self.branch_id = "root"
        self.eval_schedule: set[int] = set()
        self.eval_artifacts_schedule: set[int] = set()

        self._branch_counter = 0
        self._stop = False
        self._handlers: dict[str, Callable] = {}
        self._custom_specs: list[ActionSpec] = []
        self._spec_by_name: dict[str, ActionSpec] = {}
        self._register_builtin_handlers()

    # ---------------- extension points ----------------

    def register_handler(self, cmd_type: str, handler: Callable, spec: ActionSpec | None = None) -> None:
        """Register `handler(controller, cmd)` for commands of type
        `cmd_type`. Passing `spec` makes the command appear in the dashboard
        and enables argument coercion. Registering an existing type
        (including a built-in) replaces it."""
        self._handlers[cmd_type] = handler
        if spec is not None:
            if spec.name != cmd_type:
                raise ValueError(f"spec.name {spec.name!r} must equal cmd_type {cmd_type!r}")
            self._custom_specs = [s for s in self._custom_specs if s.name != cmd_type]
            self._custom_specs.append(spec)

    def every(self, n: int, start: int = 0, end: int | None = None) -> set[int]:
        """Convenience for schedules: `controller.eval_schedule = controller.every(50)`."""
        end = self.total_steps if end is None else end
        return set(range(start, end + 1, n))

    def snapshot_now(self):
        """Force a snapshot of the current state; handlers that mutate
        weights should call this first so the change can be rewound."""
        if self.snapshots is None:
            raise RuntimeError("snapshotting is disabled; construct with enable_rewind=True")
        return self.snapshots.force_snapshot(
            self.step, self.model, self.optimizer, self._current_lr(), self.branch_id,
        )

    # ---------------- internals ----------------

    def _register_builtin_handlers(self) -> None:
        self._handlers.update({
            "pause": _handle_pause,
            "resume": _handle_resume,
            "stop": _handle_stop,
            "set_lr": _handle_set_lr,
        })
        if self.snapshots is not None:
            self._handlers["rewind"] = _handle_rewind

    def _build_specs(self) -> list[ActionSpec]:
        """Specs for every handler that has one, built at run_loop time so
        the list can never disagree with the handler table."""
        builtins = list(BUILTIN_ACTIONS) + ([REWIND_ACTION] if self.snapshots is not None else [])
        custom_names = {s.name for s in self._custom_specs}
        specs = [s for s in builtins if s.name in self._handlers and s.name not in custom_names]
        specs += [s for s in self._custom_specs if s.name in self._handlers]
        return specs

    def _current_lr(self) -> float | None:
        groups = getattr(self.optimizer, "param_groups", None)
        return groups[0].get("lr") if groups else None

    def _set_state(self, state: RunState, **extra) -> None:
        self.status.update(step=self.step, branch_id=self.branch_id, state=state,
                           lr=self._current_lr(), **extra)
        append_event(self.run.run_dir, "state", self.step, self.branch_id, state=state.value, **extra)

    def _branch_tags(self) -> dict:
        """No tag at all if rewind is disabled, so non-branching users get a
        metrics table without a constant 'root' column."""
        return {"branch_id": self.branch_id} if self.enable_rewind else {}

    def _branch_scoped_name(self, base_name: str) -> str:
        return f"{base_name}__{self.branch_id}" if self.enable_rewind else base_name

    def _poll_and_apply(self) -> None:
        for cmd in self.control.poll_commands():
            self._apply(cmd)

    def _apply(self, cmd: dict) -> None:
        cmd_type = cmd.get("type") if isinstance(cmd, dict) else None
        handler = self._handlers.get(cmd_type)
        if handler is None:
            self.logger.warning(f"unknown command type {cmd_type!r}, ignoring")
            append_event(self.run.run_dir, "failed", self.step, self.branch_id,
                         command=cmd, error=f"unknown command type {cmd_type!r}")
            return
        try:
            spec = self._spec_by_name.get(cmd_type)
            if spec is not None:
                cmd = coerce_args(spec, cmd)
            handler(self, cmd)
        except Exception as e:
            self.logger.exception(f"command {cmd_type!r} failed at step {self.step}, ignoring")
            append_event(self.run.run_dir, "failed", self.step, self.branch_id,
                         command=cmd, error=f"{type(e).__name__}: {e}")
            return
        self.logger.info(f"applied {cmd_type} at step {self.step} (branch={self.branch_id})")
        append_event(self.run.run_dir, "applied", self.step, self.branch_id, command=cmd)

    def _log_eval(self, metrics: dict) -> None:
        keys = self.log_metrics if self.log_metrics is not None else list(metrics.keys())
        parts = []
        for k in keys:
            if k in metrics:
                v = metrics[k]
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        msg = f"step {self.step}/{self.total_steps} | "
        msg += f"branch={self.branch_id} | " if self.enable_rewind else ""
        self.logger.info(msg + " | ".join(parts))

    def _log_eval_artifacts(self, artifacts: dict) -> None:
        for key, value in artifacts.items():
            data, atype = value if isinstance(value, tuple) else (value, "tensor")
            name, group = key if isinstance(key, tuple) else (key, None)
            self.run.track_artifact(data, step=self.step, group=group,
                                    name=self._branch_scoped_name(name), type=atype)
        msg = f"saved {len(artifacts)} eval artifact(s) at step {self.step}"
        msg += f" (branch={self.branch_id})" if self.enable_rewind else ""
        self.logger.info(msg)

    # ---------------- the loop ----------------

    def run_loop(self) -> None:
        self.logger.info(f"starting training | run_id={self.run.run_id} | total_steps={self.total_steps}")
        specs = self._build_specs()
        self._spec_by_name = {s.name: s for s in specs}
        write_actions(self.run.run_dir, specs)
        self.status.update(total_steps=self.total_steps, pid=os.getpid())
        self._set_state(RunState.RUNNING)

        try:
            self._loop()
        except KeyboardInterrupt:
            self.logger.warning(f"interrupted at step {self.step}")
            self._set_state(RunState.INTERRUPTED)
            raise
        except Exception as e:
            self.logger.exception(f"training crashed at step {self.step}")
            self._set_state(RunState.CRASHED, error=f"{type(e).__name__}: {e}")
            raise
        else:
            final = RunState.STOPPED if self._stop else RunState.DONE
            self.logger.info(f"training {final.value} at step {self.step} (branch={self.branch_id})")
            self._set_state(final)
        finally:
            self.run.finalize()

    def _loop(self) -> None:
        while self.step < self.total_steps and not self._stop:
            if self.paused or self.step % self.poll_every == 0:
                self._poll_and_apply()

            if self.paused:
                time.sleep(0.1)
                continue
            if self._stop:
                break

            if self.eval_fn is not None and self.step in self.eval_schedule:
                metrics = self.eval_fn()
                self.run.track_metric(self.step, tags=self._branch_tags(), **metrics)
                self._log_eval(metrics)

            if self.eval_artifacts_fn is not None and self.step in self.eval_artifacts_schedule:
                self._log_eval_artifacts(self.eval_artifacts_fn())

            if self.snapshots is not None:
                self.snapshots.maybe_snapshot(
                    self.step, self.model, self.optimizer, self._current_lr(), self.branch_id,
                )

            out = self.train_step_fn()
            if (self.train_log_every and isinstance(out, dict) and out
                    and self.step % self.train_log_every == 0):
                self.run.track_metric(self.step, tags=self._branch_tags(), **out)

            self.step += 1
            if self.step % self.status_every == 0:
                self.status.update(step=self.step, lr=self._current_lr())

        # A completed run evaluates its final weights if the schedule asks
        # for it; the loop body only evaluates *before* each train step.
        if not self._stop and self.eval_fn is not None and self.step in self.eval_schedule:
            metrics = self.eval_fn()
            self.run.track_metric(self.step, tags=self._branch_tags(), **metrics)
            self._log_eval(metrics)
        if not self._stop and self.eval_artifacts_fn is not None and self.step in self.eval_artifacts_schedule:
            self._log_eval_artifacts(self.eval_artifacts_fn())
