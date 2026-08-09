"""rewind/dashboard/launch.py

RunLauncher owns the lifecycle of exactly one subprocess-backed training run:
spawning it, learning its run_id via a handshake (never predicting it),
tracking its PID on disk, and exposing liveness/kill controls.

Deliberately has no Shiny dependency, so it can be unit-tested and used
standalone (a notebook, a script, a future non-Shiny UI) without spinning up
an app. The Shiny control_panel/experiment_picker modules are thin wrappers
around this class.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

from ._fsutil import atomic_write  # rewind's own helper -- not a tracklab import
from ._layout import handshake_path as _handshake_path
from ._layout import pending_dir, process_path

HANDSHAKE_TIMEOUT_S = 30.0   # max time to wait for the subprocess to claim a run_id
HANDSHAKE_POLL_S = 0.05
TERMINATE_GRACE_S = 5.0      # SIGTERM -> SIGKILL escalation window


class LaunchError(RuntimeError):
    """Raised when a subprocess exits, or times out, before claiming a run_id."""


@dataclass
class ProcessRecord:
    """Persisted to `<run_dir>/control/process.json` (see rewind._layout).
    Small and boring on purpose —
    this is what lets a *new* dashboard session reattach to a run that's
    still going after the dashboard itself was restarted."""

    pid: int
    run_id: str
    cmd: list[str]
    started_at: float
    launch_token: str

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> ProcessRecord:
        return cls(**d)


class RunLauncher:
    """
    Usage
    -----
        launcher = RunLauncher(exp_dir, entrypoint="my_project.launcher")
        record = launcher.launch({"model_args.vocab_size": "128"})
        ...
        launcher.is_alive()
        launcher.terminate()

    One instance == one owned process. Don't reuse an instance across runs;
    construct a new one (or use `RunLauncher.attach` for reattaching to a
    run started by a previous dashboard session).
    """

    def __init__(self, exp_name: str, base_dir: Path, entrypoint: str, python: str | None = None):
        self.exp_dir = base_dir / exp_name
        self.exp_name = exp_name
        self.base_dir = base_dir
        self.entrypoint = entrypoint
        self.python = python or sys.executable
        self._proc: subprocess.Popen | None = None
        self._record: ProcessRecord | None = None

    # ------------------------------------------------------------------ #
    # Launching
    # ------------------------------------------------------------------ #

    def launch(self, overrides: dict[str, str]) -> ProcessRecord:
        """Spawn the training subprocess and block briefly until it tells us
        (not: until we guess) which run_id it claimed."""
        if self._proc is not None and self.is_alive():
            raise LaunchError(
                "This RunLauncher already owns a live process; construct a new "
                "instance per launch rather than reusing one."
            )

        token = uuid.uuid4().hex
        pending_dir(self.exp_dir).mkdir(parents=True, exist_ok=True)
        handshake_path = _handshake_path(self.exp_dir, token)

        cli_overrides = [f"{k}={v}" for k, v in overrides.items()]
        
        for k , v in {'launch_token':token,
                      'experiment_name':self.exp_name,
                      'base_dir':self.base_dir}.items():
            cli_overrides.append(f"extra_args.{k}={v}")

        cmd = [self.python, "-m", self.entrypoint, *cli_overrides]

        # start_new_session=True puts the child in its own process group, so
        # terminate() can be extended later to signal the whole group if a
        # project's train loop ever spawns its own workers.
        proc = subprocess.Popen(cmd, start_new_session=True)

        try:
            run_id = self._await_handshake(handshake_path, proc)
        except LaunchError:
            handshake_path.unlink(missing_ok=True)
            raise

        record = ProcessRecord(
            pid=proc.pid,
            run_id=run_id,
            cmd=cmd,
            started_at=time.time(),
            launch_token=token,
        )
        run_dir = self.exp_dir / run_id
        atomic_write(process_path(run_dir), json.dumps(record.to_json(), indent=2))
        handshake_path.unlink(missing_ok=True)

        self._proc = proc
        self._record = record
        return record

    def _await_handshake(self, handshake_path: Path, proc: subprocess.Popen) -> str:
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
        while time.monotonic() < deadline:
            if handshake_path.exists():
                data = json.loads(handshake_path.read_text())
                return data["run_id"]
            exit_code = proc.poll()
            if exit_code is not None:
                raise LaunchError(
                    f"Subprocess exited with code {exit_code} before claiming a "
                    f"run_id. cmd={proc.args}"
                )
            time.sleep(HANDSHAKE_POLL_S)

        proc.kill()
        raise LaunchError(
            "Timed out waiting for the subprocess to claim a run_id "
            f"(waited {HANDSHAKE_TIMEOUT_S}s). It may be hanging during import/setup."
        )

    # ------------------------------------------------------------------ #
    # Reattaching (dashboard restarted, subprocess is still running)
    # ------------------------------------------------------------------ #

    @classmethod
    def attach(cls, run_dir: Path) -> RunLauncher | None:
        """Rebuild a lifecycle handle from a persisted process.json. Returns
        None if there's no record (nothing was ever launched for this run,
        e.g. it was created directly via the CLI)."""
        record_path = process_path(run_dir)
        if not record_path.exists():
            return None
        record = ProcessRecord.from_json(json.loads(record_path.read_text()))
        launcher = cls.__new__(cls)  # skip __init__: we don't need entrypoint/exp_dir to attach
        launcher.exp_dir = run_dir.parent
        launcher.entrypoint = None
        launcher.python = None
        launcher._proc = None
        launcher._record = record
        return launcher

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def record(self) -> ProcessRecord | None:
        return self._record

    def is_alive(self) -> bool:
        if self._record is None:
            return False
        return _pid_alive(self._record.pid)

    def terminate(self, force: bool = False) -> None:
        """SIGTERM, wait up to TERMINATE_GRACE_S, escalate to SIGKILL. Pass
        force=True to skip straight to SIGKILL (e.g. a user hitting "kill"
        a second time after a graceful terminate didn't take)."""
        if self._record is None or not self.is_alive():
            return

        pid = self._record.pid
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return

        if force:
            return

        deadline = time.monotonic() + TERMINATE_GRACE_S
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                return
            time.sleep(0.1)

        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def write_handshake(exp_dir: Path, token: str, run_id: str) -> None:
    """Called from the training-process side (see icl.launcher.build_controller)
    once a run_id has been claimed, to tell the RunLauncher waiting in the
    parent process which run_id was actually chosen. This is the write half
    of the handshake RunLauncher._await_handshake() polls for -- kept here,
    next to that method, so the `.pending/<token>.json` path convention is
    defined in exactly one place (_layout.py) rather than reconstructed by
    hand on both the launching and the launched side."""
    atomic_write(_handshake_path(exp_dir, token), json.dumps({"run_id": run_id}))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just isn't ours
    return True