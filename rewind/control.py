"""
Module for controlling a training run via a mailbox of files in a directory. 
The RunMailbox class provides methods to set and get the status of the run, 
send commands to the run, and poll for commands sent to the run. 
The NullControl class is an inert control channel that does not allow any commands to be sent or received.
"""

import json
import os
import time
from pathlib import Path




def _atomic_write(path: Path, data: str):
    """Write to a temp file then rename into place, so a reader (the
    dashboard, polling once a second) never sees a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


class RunMailbox:
    """Status board + command mailbox for one run directory. Usable
    standalone (dashboard, given only a path) or composed inside
    ControllableRun (training process, given a live Run)."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.status_path = self.run_dir / "status.json"
        self.commands_dir = self.run_dir / "commands"
        self.commands_dir.mkdir(parents=True, exist_ok=True)



    def send_command(self, cmd: dict):
        """Write a command to the mailbox. The training process will pick it up on its next poll and act on it."""
        path = self.commands_dir / f"{time.time_ns()}.json"
        _atomic_write(path, json.dumps(cmd))

    def poll_commands(self) -> list[dict]:
        """Read all commands from the mailbox, then delete them. Returns a list of command dicts."""
        cmds = []
        # Use key=lambda p: int(p.stem) to sort numerically by the filename
        sorted_paths = sorted(self.commands_dir.glob("*.json"), key=lambda p: int(p.stem))
        
        for path in sorted_paths:
            try:
                cmds.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                continue
            finally:
                path.unlink(missing_ok=True)
        return cmds

class _NullControl:
    """Inert control channel: no commands directory, no status.json,
    no filesystem writes at all. Used when the user just wants tracking."""

    def send_command(self, cmd: dict):
        raise RuntimeError("this run has no control channel; construct TrainerController with control=...")
    def poll_commands(self) -> list[dict]: return []
