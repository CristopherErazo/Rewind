"""rewind/control.py

The control plane's two file-backed halves plus the run state vocabulary.

RunStatus   trainer -> dashboard, one writer, many readers. `control/status.json`.
RunMailbox  dashboard -> trainer, many writers, one reader. `control/commands/`.
RunState    the closed set of values `status["state"]` can take.
"""

from __future__ import annotations

import json
import os
import time
from enum import Enum
from pathlib import Path

from ._fsutil import atomic_write
from ._layout import commands_dir, status_path


class RunState(str, Enum):
    """Lifecycle of a controlled run. Exactly one of these is in
    status.json's `state` field at any time.

    running      training steps are being executed
    paused       loop is idling, still polling commands
    stopped      a `stop` command ended the run before total_steps
    done         reached total_steps
    crashed      train/eval raised; `error` in status carries the message
    interrupted  KeyboardInterrupt (Ctrl-C) in the training process
    """
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    DONE = "done"
    CRASHED = "crashed"
    INTERRUPTED = "interrupted"

    @property
    def live(self) -> bool:
        """True while the process is still in its loop and accepts commands."""
        return self in (RunState.RUNNING, RunState.PAUSED)


LIVE_STATES = frozenset(s.value for s in RunState if s.live)


class RunStatus:
    """Holds the current status in memory (safe: only the training process
    writes this file), merges partial updates, and rewrites the whole file
    atomically. Every write stamps `updated_at` so readers can tell a stale
    file from a live one."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._state: dict = {}

    def update(self, **fields) -> None:
        for k, v in fields.items():
            if isinstance(v, Enum):
                fields[k] = v.value
        self._state.update(fields)
        self._state["updated_at"] = time.time()
        atomic_write(status_path(self.run_dir), json.dumps(self._state, indent=2))

    @property
    def current(self) -> dict:
        return dict(self._state)


def read_status(run_dir: Path) -> dict | None:
    """Dashboard-side reader. None if the file is missing or half-written."""
    path = status_path(run_dir)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


class RunMailbox:
    """Command mailbox for one run directory. Usable standalone (dashboard,
    given only a path) or handed to TrainerController as its `control`."""

    _seq = 0  # process-wide, see send_command

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.commands_dir = commands_dir(self.run_dir)
        self.commands_dir.mkdir(parents=True, exist_ok=True)

    def send_command(self, cmd: dict) -> None:
        """Write a command file. The trainer picks it up on its next poll.

        Filenames are `<time_ns>-<pid>-<seq>.json`. The clock alone is not
        enough: on Windows two back-to-back calls can see the same
        time_ns(), and the second file would silently overwrite the first.
        The per-process sequence number breaks ties within one sender, the
        pid breaks ties between senders."""
        RunMailbox._seq += 1
        name = f"{time.time_ns():020d}-{os.getpid()}-{RunMailbox._seq:06d}.json"
        atomic_write(self.commands_dir / name, json.dumps(cmd))

    def poll_commands(self) -> list[dict]:
        """Read all pending commands in send order, deleting them. Cheap when
        the directory is empty, which is the common case on every step."""
        try:
            with os.scandir(self.commands_dir) as it:
                names = [e.name for e in it if e.name.endswith(".json")]
        except FileNotFoundError:
            return []
        if not names:
            return []

        def _key(name: str) -> tuple:
            # numeric order on every dash-separated part; legacy plain
            # `<time_ns>.json` names still sort correctly
            parts = name[:-5].split("-")
            try:
                return tuple(int(x) for x in parts)
            except ValueError:
                return (0,)

        cmds = []
        for name in sorted(names, key=_key):
            path = self.commands_dir / name
            try:
                cmds.append(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
            finally:
                path.unlink(missing_ok=True)
        return cmds


class _NullControl:
    """Inert control channel: no commands directory, no filesystem reads.
    Used when the user just wants tracking."""

    def send_command(self, cmd: dict) -> None:
        raise RuntimeError("this run has no control channel; construct TrainerController with control=...")

    def poll_commands(self) -> list[dict]:
        return []
