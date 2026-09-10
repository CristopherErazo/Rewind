"""In-memory TrackingHandle used by the tests.

This is also the reference for what the protocol truly requires: if the
controller needs something FakeRun does not provide, the protocol (and this
file) must grow, not the tests' mocking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


class FakeRun:
    def __init__(self, run_dir: Path, run_id: str = "run_001"):
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list[dict] = []          # one row per track_metric call
        self.artifacts: dict[tuple, Any] = {}  # (group, name, step) -> data
        self.finalized = 0

    def track_metric(self, step, note=None, tags=None, **metrics):
        self.metrics.append({"step": step, "note": note, "tags": dict(tags or {}), **metrics})

    def track_artifact(self, data, step=None, group=None, name="", type="pickle"):
        self.artifacts[(group, name, step)] = data

    def load_artifact(self, group=None, name="", step=None, type="pickle"):
        return self.artifacts[(group, name, step)]

    def get_logger(self, **kwargs):
        logger = logging.getLogger(f"fake.{self.run_dir}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    def finalize(self):
        self.finalized += 1
