"""rewind/events.py

The audit journal: `control/events.jsonl`, one JSON object per line.

Everything the controller does in response to the outside world is recorded
here rather than in the tracker's metrics table, so metrics.jsonl stays a
clean (step, metric, value) log and the dashboard can show command history
and branch lineage from one file. Only the training process writes it.

Event kinds written by the controller:
  applied   a command ran successfully           payload: command dict
  failed    a command raised                     payload: command dict, error
  fork      a rewind created a new branch        payload: parent_branch_id, fork_step
  state     lifecycle transition                 payload: state (and error if crashed)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ._fsutil import append_line
from ._layout import events_path


def append_event(run_dir: Path, kind: str, step: int, branch_id: str, **payload) -> None:
    row = {"time": time.time(), "kind": kind, "step": step, "branch_id": branch_id, **payload}
    append_line(events_path(run_dir), json.dumps(row, default=str))


def read_events(run_dir: Path) -> list[dict]:
    """Dashboard-side reader; skips a trailing half-written line."""
    path = events_path(run_dir)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
