"""rewind/dashboard/config.py

DashboardConfig bundles everything the dashboard needs to know about a
project. The minimum is a base directory:

    app = build_dashboard(DashboardConfig(base_dir="./data"))

That gives an attach-first dashboard: pick an experiment and a run that is
already training (started from a terminal, a notebook, anywhere) and steer
it. Launching new runs from the UI is optional and needs both `config_cls`
(to build the form) and `entrypoint` (to spawn the process).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .schema import FieldSpec


class DashboardExtension(Protocol):
    """A project-supplied tab. Just another Shiny module mounted as one
    extra nav panel. Ids inside an extension must be unique across the whole
    app (extensions are not namespaced yet)."""

    id: str
    label: str

    def ui(self) -> Any: ...

    def server(self, input, output, session, ctx: "RunContext") -> None: ...


@dataclass
class RunContext:
    """What an extension's server() receives. Every field is a zero-arg
    reactive callable, so reading it inside a render/calc/effect stays
    reactive and re-fires when the active run changes."""

    reader: Callable[[], Any]              # () -> tracklab.ExperimentReader for the selected experiment
    run_dir: Callable[[], Path | None]     # () -> active run_dir, or None
    mailbox: Callable[[], Any | None]      # () -> rewind.RunMailbox bound to run_dir, or None
    status: Callable[[], dict | None]      # () -> parsed control/status.json, or None
    metrics: Callable[[], list[dict]]      # () -> metric rows (long format) for the active run
    events: Callable[[], list[dict]]       # () -> control/events.jsonl rows


@dataclass
class DashboardConfig:
    base_dir: Path
    config_cls: type | None = None         # project config dataclass; enables the launch form
    entrypoint: str | None = None          # "python -m <entrypoint>" target; enables launching
    field_overrides: dict[str, Callable[[FieldSpec], Any]] = field(default_factory=dict)
    extensions: list[DashboardExtension] = field(default_factory=list)
    poll_interval_s: float = 1.0           # live files of the active run
    listing_interval_s: float = 2.0        # experiment / run directory listings
    title: str = "Rewind"

    @property
    def launchable(self) -> bool:
        return self.config_cls is not None and bool(self.entrypoint)
