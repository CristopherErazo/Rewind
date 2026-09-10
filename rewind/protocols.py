"""rewind/protocols.py

The two seams between Rewind and the outside world.

TrackingHandle is what the controller needs from an experiment tracker:
metrics, artifacts, a logger, and a run directory it can put its control
plane into. A TrackLab `Run` satisfies it unchanged. `tests/fakes.py` has an
in-memory implementation that doubles as the reference for what the
protocol really requires.

ControlHandle is the command channel. `RunMailbox` (control.py) implements
it with files; a `_NullControl` turns the controller into a plain loop.
"""

from __future__ import annotations

from typing import Any, Protocol


class TrackingHandle(Protocol):
    run_id: str
    run_dir: Any  # anything Path() accepts; the control/ folder lives under it

    def track_metric(self, step: int, note: str | None = None,
                     tags: dict | None = None, **metrics) -> None: ...

    def track_artifact(self, data: Any, step: int | None = None,
                       group: str | None = None, name: str = "",
                       type: str = "pickle") -> None: ...

    def load_artifact(self, group: str | None = None, name: str = "",
                      step: int | None = None, type: str = "pickle") -> Any: ...

    def get_logger(self, **kwargs) -> Any: ...

    def finalize(self) -> None: ...


class ControlHandle(Protocol):
    def send_command(self, cmd: dict) -> None: ...

    def poll_commands(self) -> list[dict]: ...
