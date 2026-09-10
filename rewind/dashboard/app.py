"""rewind/dashboard/app.py

Assembles the App from the panel modules. This is deliberately the only
file that knows about every panel at once.

Reactivity design
-----------------
* One `reactive.Value` holds the active run_dir. It is set by the run picker
  in the sidebar (attach) or by a successful launch.
* Directory listings (experiments, runs) are `reactive.poll`s whose check
  function is a cheap listing stamp, so new folders appear without a reload.
* One shared `reactive.poll` per active run reads status.json, events.jsonl
  and the metrics stream. Its check function only stats the files; the body
  does the actual reading once per change. Panels never read files.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from shiny import App, reactive, ui

from tracklab import ExperimentReader
from tracklab.live import MetricsStream

from .config import DashboardConfig, RunContext
from .schema import form_fields
from ..control import RunMailbox, read_status
from ..events import read_events
from .._fsutil import file_stamp
from .._layout import events_path, status_path
from ..launch import RunLauncher
from .modules import (control_panel, events_panel, experiment_picker, launcher_form,
                      metrics_panel, run_picker, runs_table, status_bar)

# A MetricsStream tracks a byte offset, so it is reused across polls. Keyed
# by run_dir; a module-level cache is fine for a local, single-user tool.
_stream_cache: dict[Path, MetricsStream] = {}


def _get_stream(run_dir: Path) -> MetricsStream:
    if run_dir not in _stream_cache:
        _stream_cache[run_dir] = MetricsStream(run_dir)
    return _stream_cache[run_dir]


def _live_stamp(run_dir: Path | None) -> tuple:
    """Cheap change detector for the active run's live files."""
    if run_dir is None:
        return (None,)
    return (str(run_dir), file_stamp(run_dir / "metrics.jsonl"),
            file_stamp(status_path(run_dir)), file_stamp(events_path(run_dir)))


def _read_live(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {"metrics": [], "status": None, "events": []}
    df = _get_stream(run_dir).poll()
    return {"metrics": df.to_dict("records"), "status": read_status(run_dir),
            "events": read_events(run_dir)}


def _dir_stamp(path: Path | None) -> tuple:
    """Names of subdirectories plus whether each has started writing files
    the dashboard cares about. Cheap enough to run every couple of seconds."""
    if path is None or not path.is_dir():
        return ()
    out = []
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_dir() and not e.name.startswith("."):
                    out.append((e.name, os.path.exists(os.path.join(e.path, "metrics.jsonl")),
                                os.path.exists(os.path.join(e.path, "control", "status.json"))))
    except OSError:
        return ()
    return tuple(sorted(out))


def _list_runs(exp_dir: Path | None) -> list[str]:
    """Every run folder, newest last, whether or not it has metrics yet.
    (ExperimentReader.list_runs hides runs until metrics.jsonl exists,
    which is exactly when a freshly launched run needs to be selectable.)"""
    if exp_dir is None or not exp_dir.is_dir():
        return []
    names = [p.name for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]

    def key(n: str):
        digits = "".join(ch for ch in n if ch.isdigit())
        return (int(digits) if digits else -1, n)

    return sorted(names, key=key)


def build_dashboard(cfg: DashboardConfig) -> App:
    """Build a ready-to-run Shiny App from a DashboardConfig.

    With only `base_dir` set you get the attach-first dashboard: experiment
    picker, run picker, status bar, live metrics, control panel and events.
    Set `config_cls` and `entrypoint` as well to add the launch form.
    Assign the result to a module-level `app` and run with
    `shiny run --reload path/to/script.py`.
    """
    cfg.base_dir = Path(cfg.base_dir)
    cfg.base_dir.mkdir(parents=True, exist_ok=True)

    if cfg.config_cls is not None and not dataclasses.is_dataclass(cfg.config_cls):
        raise TypeError(f"DashboardConfig.config_cls must be a dataclass, got {cfg.config_cls!r}.")
    if (cfg.config_cls is None) != (not cfg.entrypoint):
        raise ValueError("Set both config_cls and entrypoint to enable launching, or neither.")
    if cfg.field_overrides:
        if cfg.config_cls is None:
            raise ValueError("field_overrides needs config_cls.")
        valid = {f.path for f in form_fields(cfg.config_cls)}
        unknown = set(cfg.field_overrides) - valid
        if unknown:
            raise ValueError(f"field_overrides references unknown field(s): {sorted(unknown)}. "
                             f"Known fields: {sorted(valid)}")

    sidebar_items = [
        experiment_picker.experiment_picker_ui("exp"),
        run_picker.run_picker_ui("run"),
    ]
    if cfg.launchable:
        sidebar_items.append(launcher_form.launcher_form_ui("launch", cfg))

    app_ui = ui.page_fluid(
        # Outputs here refresh every second by design. Per-output spinners
        # would flash constantly and their min-height nudges the layout; the
        # fade dims a recalculating output's children to 30% instantly and
        # snaps back when the value lands, a visible flicker on the status
        # bar. The top-of-page pulse remains as the busy cue.
        ui.busy_indicators.use(spinners=False, fade=False),
        ui.h2(cfg.title),
        ui.layout_sidebar(
            ui.sidebar(*sidebar_items, width=340, open="desktop"),
            status_bar.status_bar_ui("status"),
            ui.navset_tab(
                # Controls are one thin toolbar row above the plot, so an
                # intervention never hides the metrics.
                ui.nav_panel("Live", control_panel.control_panel_ui("control"),
                             metrics_panel.metrics_panel_ui("metrics")),
                ui.nav_panel("Events", events_panel.events_panel_ui("events")),
                ui.nav_panel("Runs", runs_table.runs_table_ui("runs")),
                *[ui.nav_panel(ext.label, ext.ui()) for ext in cfg.extensions],
            ),
        ),
    )

    def server(input, output, session):
        # ---- experiment / run selection -------------------------------
        experiments = reactive.poll(lambda: _dir_stamp(cfg.base_dir), cfg.listing_interval_s)(
            lambda: sorted(p.name for p in cfg.base_dir.iterdir() if p.is_dir() and not p.name.startswith(".")))
        selected_experiment = experiment_picker.experiment_picker_server("exp", experiments)

        @reactive.calc
        def exp_dir() -> Path | None:
            name = selected_experiment()
            return cfg.base_dir / name if name else None

        runs = reactive.poll(lambda: _dir_stamp(exp_dir()), cfg.listing_interval_s)(
            lambda: _list_runs(exp_dir()))

        # Set by a successful launch so the run picker jumps to the new run.
        launched_run_id: reactive.Value[str | None] = reactive.value(None)
        selected_run = run_picker.run_picker_server("run", runs, launched_run_id)

        @reactive.calc
        def active_run_dir() -> Path | None:
            d, r = exp_dir(), selected_run()
            return d / r if (d and r) else None

        if cfg.launchable:
            def _on_launched(launcher: RunLauncher) -> None:
                launched_run_id.set(launcher.record.run_id)
            launcher_form.launcher_form_server("launch", cfg, selected_experiment, _on_launched)

        # ---- one shared poll for the active run -------------------------
        live = reactive.poll(lambda: _live_stamp(active_run_dir()), cfg.poll_interval_s)(
            lambda: _read_live(active_run_dir()))

        status = lambda: live()["status"]
        metrics = lambda: live()["metrics"]
        events = lambda: live()["events"]

        @reactive.calc
        def reader() -> ExperimentReader:
            return ExperimentReader(selected_experiment() or "", base_dir=cfg.base_dir)

        @reactive.calc
        def active_mailbox() -> RunMailbox | None:
            rd = active_run_dir()
            return RunMailbox(rd) if rd else None

        @reactive.calc
        def active_launcher() -> RunLauncher | None:
            rd = active_run_dir()
            return RunLauncher.attach(rd) if rd else None

        # ---- panels ------------------------------------------------------
        status_bar.status_bar_server("status", active_run_dir, status, active_launcher)
        clicked_step = metrics_panel.metrics_panel_server("metrics", metrics)
        control_panel.control_panel_server("control", active_run_dir, status, active_mailbox, active_launcher,
                                           step_hint=clicked_step)
        events_panel.events_panel_server("events", events)
        runs_table.runs_table_server("runs", reader, runs)

        ctx = RunContext(reader=reader, run_dir=active_run_dir, mailbox=active_mailbox,
                         status=status, metrics=metrics, events=events)
        for ext in cfg.extensions:
            ext.server(input, output, session, ctx=ctx)

    return App(app_ui, server)
