"""rewind/_layout.py

The canonical location of everything Rewind owns inside a TrackLab run_dir.

TrackLab owns run_dir's root (config.json, metrics.jsonl, artifacts/, logs/)
and knows nothing about Rewind. Everything Rewind adds -- the command
mailbox, the action registry, run status, the audit journal, and the
launcher's process record -- lives under one `control/` subfolder instead of
as loose files at run_dir root. This folder only exists for runs that
actually have the control plane enabled; a bare TrackLab Run never creates
it.

Every module that needs one of these paths should import it from here
rather than hardcoding "control" or "commands" as a string literal -- that's
the whole point: the layout changes in exactly one place if it ever needs to
move again.
"""

from __future__ import annotations

from pathlib import Path


def control_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "control"


def commands_dir(run_dir: Path) -> Path:
    return control_dir(run_dir) / "commands"


def status_path(run_dir: Path) -> Path:
    return control_dir(run_dir) / "status.json"


def actions_path(run_dir: Path) -> Path:
    return control_dir(run_dir) / "actions.json"


def events_path(run_dir: Path) -> Path:
    """Audit journal (rewind/events.py): applied/failed commands, forks and
    lifecycle transitions, kept out of the tracker's metrics table."""
    return control_dir(run_dir) / "events.jsonl"


def process_path(run_dir: Path) -> Path:
    return control_dir(run_dir) / "process.json"


def pending_dir(exp_dir: Path) -> Path:
    """Handshake staging area, one level ABOVE run_dir -- a run_id doesn't
    exist yet when a launch token is generated, so there's nowhere under a
    run_dir to put this. Lives at the experiment level instead. Filenames
    here (`.pending`) never match `next_run_id`'s `run_*` glob, so it can't
    interfere with run-id scanning."""
    return Path(exp_dir) / ".pending"


def handshake_path(exp_dir: Path, token: str) -> Path:
    return pending_dir(exp_dir) / f"{token}.json"