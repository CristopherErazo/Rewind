"""rewind/launch.py

RunLauncher owns the lifecycle of exactly one subprocess-backed training run:
spawning it, learning its run_id via a handshake (never predicting it),
persisting its PID, and exposing liveness / kill controls that work on both
POSIX and Windows.

No Shiny dependency, so it can be used from a script or tested directly.

Windows note: `os.kill(pid, 0)` is NOT a liveness probe on Windows. Any
signal other than the CTRL events makes CPython call TerminateProcess with
that value as the exit code, so the old `_pid_alive` would have killed the
training run. Liveness now goes through psutil when available, then a
ctypes OpenProcess check on Windows, and `os.kill(pid, 0)` only on POSIX.
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

from ._fsutil import atomic_write
from ._layout import handshake_path as _handshake_path
from ._layout import pending_dir, process_path

HANDSHAKE_TIMEOUT_S = 30.0
HANDSHAKE_POLL_S = 0.05
TERMINATE_GRACE_S = 5.0

_IS_WINDOWS = sys.platform.startswith("win")

try:  # optional, gives the most reliable liveness/kill on every platform
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None


class LaunchError(RuntimeError):
    """Raised when a subprocess exits, or times out, before claiming a run_id."""


@dataclass
class ProcessRecord:
    """Persisted to `control/process.json` so a later dashboard session can
    reattach to a run it did not start."""

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
        launcher = RunLauncher("my-exp", Path("./data"), entrypoint="my_project.launcher")
        record = launcher.launch({"model_args.vocab_size": "128"})
        launcher.is_alive()
        launcher.terminate()

    One instance == one owned process. Use `RunLauncher.attach(run_dir)` to
    get a handle on a run started by someone else (an earlier dashboard
    session, or a terminal); that handle can probe and kill but not relaunch.
    """

    def __init__(self, exp_name: str, base_dir: Path, entrypoint: str, python: str | None = None):
        self.base_dir = Path(base_dir)
        self.exp_name = exp_name
        self.exp_dir = self.base_dir / exp_name
        self.entrypoint = entrypoint
        self.python = python or sys.executable
        self._proc: subprocess.Popen | None = None
        self._record: ProcessRecord | None = None

    # ------------------------------------------------------------------ #
    # Launching
    # ------------------------------------------------------------------ #

    def launch(self, overrides: dict[str, object]) -> ProcessRecord:
        """Spawn `python -m <entrypoint> key=value ...` and block until the
        child writes its handshake file with the run_id it claimed."""
        if self._proc is not None and self.is_alive():
            raise LaunchError("This RunLauncher already owns a live process; "
                              "construct a new instance per launch.")
        if not self.entrypoint:
            raise LaunchError("RunLauncher has no entrypoint; attached handles cannot launch.")

        token = uuid.uuid4().hex
        pending_dir(self.exp_dir).mkdir(parents=True, exist_ok=True)
        handshake_path = _handshake_path(self.exp_dir, token)

        cli_overrides = [f"{k}={v}" for k, v in overrides.items()]
        injected = {"launch_token": token, "experiment_name": self.exp_name, "base_dir": str(self.base_dir)}
        cli_overrides += [f"extra_args.{k}={v}" for k, v in injected.items()]
        cmd = [self.python, "-m", self.entrypoint, *cli_overrides]

        popen_kwargs: dict = {}
        if _IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)

        try:
            run_id = self._await_handshake(handshake_path, proc)
        except LaunchError:
            handshake_path.unlink(missing_ok=True)
            raise

        record = ProcessRecord(pid=proc.pid, run_id=run_id, cmd=cmd,
                               started_at=time.time(), launch_token=token)
        atomic_write(process_path(self.exp_dir / run_id), json.dumps(record.to_json(), indent=2))
        handshake_path.unlink(missing_ok=True)

        self._proc = proc
        self._record = record
        return record

    def _await_handshake(self, handshake_path: Path, proc: subprocess.Popen) -> str:
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
        while time.monotonic() < deadline:
            if handshake_path.exists():
                try:
                    return json.loads(handshake_path.read_text(encoding="utf-8"))["run_id"]
                except (json.JSONDecodeError, KeyError, OSError):
                    pass  # half-written; try again next tick
            exit_code = proc.poll()
            if exit_code is not None:
                raise LaunchError(f"Subprocess exited with code {exit_code} before claiming a "
                                  f"run_id. cmd={proc.args}")
            time.sleep(HANDSHAKE_POLL_S)
        proc.kill()
        raise LaunchError(f"Timed out waiting for the subprocess to claim a run_id "
                          f"(waited {HANDSHAKE_TIMEOUT_S}s). It may be hanging during import/setup.")

    # ------------------------------------------------------------------ #
    # Reattaching
    # ------------------------------------------------------------------ #

    @classmethod
    def attach(cls, run_dir: Path) -> RunLauncher | None:
        """Rebuild a lifecycle handle from a persisted process.json. None if
        nothing was ever launched for this run (e.g. started from a CLI
        without a launcher)."""
        run_dir = Path(run_dir)
        record_path = process_path(run_dir)
        try:
            record = ProcessRecord.from_json(json.loads(record_path.read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError, TypeError, OSError):
            return None
        launcher = cls(exp_name=run_dir.parent.name, base_dir=run_dir.parent.parent,
                       entrypoint="", python=None)
        launcher._record = record
        return launcher

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def record(self) -> ProcessRecord | None:
        return self._record

    def is_alive(self) -> bool:
        if self._proc is not None:
            return self._proc.poll() is None
        if self._record is None:
            return False
        return pid_alive(self._record.pid)

    def terminate(self, force: bool = False) -> None:
        """Ask the process to exit, wait up to TERMINATE_GRACE_S, then kill.
        `force=True` kills immediately."""
        if not self.is_alive():
            return
        pid = self._record.pid if self._record else (self._proc.pid if self._proc else None)
        if pid is None:
            return

        if self._proc is not None:
            (self._proc.kill if force else self._proc.terminate)()
            try:
                self._proc.wait(timeout=TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=TERMINATE_GRACE_S)
            return

        _signal_pid(pid, force=force)
        if force:
            return
        deadline = time.monotonic() + TERMINATE_GRACE_S
        while time.monotonic() < deadline:
            if not pid_alive(pid):
                return
            time.sleep(0.1)
        _signal_pid(pid, force=True)


def write_handshake(exp_dir: Path, token: str, run_id: str) -> None:
    """Training-process side of the launch handshake: call this right after
    the tracker has claimed a run_id."""
    atomic_write(_handshake_path(Path(exp_dir), token), json.dumps({"run_id": run_id}))


# ---------------------------------------------------------------------------
# process helpers (module-level so the dashboard can use them on attached runs)
# ---------------------------------------------------------------------------

def pid_alive(pid: int) -> bool:
    """Liveness probe that is safe on Windows (never sends a signal there)."""
    if pid is None or pid <= 0:
        return False
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False
    if _IS_WINDOWS:
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_pid_alive(pid: int) -> bool:  # pragma: no cover (Windows only)
    import ctypes
    from ctypes import wintypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _signal_pid(pid: int, force: bool) -> None:
    """Terminate a process we do not own a Popen for."""
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            (p.kill if force else p.terminate)()
        except psutil.Error:
            pass
        return
    if _IS_WINDOWS:
        # There is no graceful signal for a console process we don't own;
        # taskkill /F is the honest option, /T takes the process tree along.
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    try:
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass
