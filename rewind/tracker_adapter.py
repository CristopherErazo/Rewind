"""
The only file in rewind allowed to import tracklab.

Wraps a tracklab.Run so it satisfies the RunHandle protocol that
TrainerController depends on. Status board and command mailbox don't exist
on tracklab.Run -- they're added here as plain files inside run.run_dir,
without needing any change to tracklab itself. Everything else is a direct
pass-through to the real Run methods (using tracklab's actual signatures).
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


class ControllableRun:
    def __init__(self, run):
        self._run = run
        self.status_path = Path(run.run_dir) / "status.json"
        self.commands_dir = Path(run.run_dir) / "commands"
        self.commands_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- pass-through: tracklab already does these ----------------
    def track_metric(self, step, note=None, **metrics):
        self._run.track_metric(step, note=note, **metrics)

    def track_artifact(self, data, step=None, name='', type='pickle'):
        self._run.track_artifact(data, step=step, name=name, type=type)

    def load_artifact(self, name='', step=None, type='pickle'):
        # requires the load_artifact method added to tracklab.Run
        return self._run.load_artifact(name=name, step=step, type=type)

    def get_logger(self, **kwargs):
        return self._run.get_logger(**kwargs)

    def finalize(self):
        self._run.finalize()

    @property
    def run_id(self):
        return self._run.run_id

    @property
    def run_dir(self):
        return self._run.run_dir

    # ---------------- new: status board ----------------
    def set_status(self, **kwargs):
        _atomic_write(self.status_path, json.dumps(kwargs))

    def get_status(self) -> dict:
        if not self.status_path.exists():
            return {}
        try:
            return json.loads(self.status_path.read_text())
        except json.JSONDecodeError:
            return {}  # caught the file mid-write; try again next poll

    # ---------------- new: command mailbox ----------------
    def send_command(self, cmd: dict):
        path = self.commands_dir / f"{time.time_ns()}.json"
        _atomic_write(path, json.dumps(cmd))

    def poll_commands(self) -> list[dict]:
        """Return all commands currently in the mailbox, deleting them from disk."""
        cmds = []
        for path in sorted(self.commands_dir.glob("*.json")):
            try:
                cmds.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                continue
            finally:
                path.unlink(missing_ok=True)
        return cmds