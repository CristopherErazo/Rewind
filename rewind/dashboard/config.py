"""rewind/dashboard/config.py

DashboardConfig bundles everything the engine needs to know about a
project. A new project's entire dashboard is:

    from rewind.dashboard import build_dashboard, DashboardConfig
    app = build_dashboard(DashboardConfig(
        base_dir="./data",
        config_cls=TrainerArgs,
        entrypoint="my_project.launcher",
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .schema import FieldSpec


class DashboardExtension(Protocol):
    """A project-supplied panel that doesn't fit the generic form/plot mold
    (design doc, "Metrics panel: sensible default + extension slots"). It's
    just another Shiny module, mounted as one extra tab -- no plugin
    machinery beyond "put it in this list"."""

    id: str
    label: str

    def ui(self) -> Any: ...

    def server(self, input, output, session, ctx: RunContext) -> None: ...


@dataclass
class RunContext:
    """Handles an extension typically needs -- the same reader machinery the
    built-in panels use, nothing extension-specific added on top.

    All three fields are zero-arg callables (in practice, the app's own
    reactive.Calc objects), the same convention every built-in module
    already uses (see e.g. control_panel_server's `run_dir: reactive.Calc`
    parameter). Reading ctx.run_dir() inside a @render/@reactive.calc/
    @reactive.effect stays reactive -- it re-fires when the selected run
    changes. A RunContext holding already-resolved values instead would
    freeze at whatever was active the moment the extension's server()
    function ran (once, at session start), never updating again.

    Known limitation: extensions are mounted directly, not wrapped in their
    own Shiny module/namespace, so any Shiny input/output id an extension
    defines (e.g. via ui.output_text_verbatim("foo")) must be unique across
    the whole app, not just within the extension. Fine for one or two
    extensions; worth revisiting with proper per-extension namespacing if
    that ever becomes a real collision instead of a documented constraint.
    """

    reader: Callable[[], Any]  # () -> tracklab.ExperimentReader, for the selected experiment
    run_dir: Callable[[], Path | None]  # () -> active run_dir, or None if nothing's selected
    mailbox: Callable[[], Any | None]  # () -> rewind.control.RunMailbox bound to run_dir, or None


@dataclass
class DashboardConfig:
    base_dir: Path
    config_cls: type  # e.g. TrainerArgs -- introspected by schema.form_fields
    entrypoint: str  # "python -m <entrypoint> ..." target for RunLauncher
    field_overrides: dict[str, Callable[[FieldSpec], Any]] = field(default_factory=dict)
    extensions: list[DashboardExtension] = field(default_factory=list)
    poll_interval_s: float = 1.0