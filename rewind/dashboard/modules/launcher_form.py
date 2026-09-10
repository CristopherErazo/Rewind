"""rewind/dashboard/modules/launcher_form.py

Renders the launch form from DashboardConfig.config_cls via
schema.form_fields() and owns the RunLauncher.launch() call. Only mounted
when DashboardConfig.launchable is True.
"""

from __future__ import annotations

from typing import Callable

from shiny import module, reactive, render, ui

from ..config import DashboardConfig
from ...launch import LaunchError, RunLauncher
from ..schema import FieldSpec, form_fields, input_id


def _default_widget(f: FieldSpec):
    widget_id = input_id(f.path)
    if f.widget == "switch":
        return ui.input_switch(widget_id, f.label, value=bool(f.default))
    if f.widget == "select":
        return ui.input_select(widget_id, f.label, choices=f.choices or [], selected=f.default)
    if f.widget == "numeric":
        return ui.input_numeric(widget_id, f.label, value=f.default)
    return ui.input_text(widget_id, f.label, value=str(f.default) if f.default is not None else "")


@module.ui
def launcher_form_ui(cfg: DashboardConfig):
    fields = form_fields(cfg.config_cls)

    sections: dict[str, list] = {}
    for f in fields:
        section = f.path.split(".")[0] if "." in f.path else "general"
        override = cfg.field_overrides.get(f.path)
        widget = override(f) if override is not None else _default_widget(f)
        sections.setdefault(section, []).append(widget)

    return ui.card(
        ui.card_header("Launch a new run"),
        ui.accordion(
            *[ui.accordion_panel(name.replace("_", " ").title(), *widgets)
              for name, widgets in sections.items()],
            open=False,
        ),
        ui.input_action_button("launch_btn", "Launch run", class_="btn-primary"),
        ui.output_text("launch_status"),
    )


@module.server
def launcher_form_server(input, output, session, cfg: DashboardConfig,
                         experiment_name: Callable[[], str | None],
                         on_launched: Callable[[RunLauncher], None]):
    fields = form_fields(cfg.config_cls)
    status = reactive.value("")

    @reactive.effect
    @reactive.event(input.launch_btn, ignore_init=True)
    def _launch():
        exp_name = experiment_name()
        if not exp_name:
            status.set("Pick or name an experiment first.")
            return
        overrides = {f.path: input[input_id(f.path)]() for f in fields}
        launcher = RunLauncher(exp_name, cfg.base_dir, entrypoint=cfg.entrypoint)
        status.set("Launching...")
        try:
            # Blocks the session while waiting for the child's run_id
            # handshake (usually well under a second; 30s worst case).
            record = launcher.launch(overrides)
        except LaunchError as e:
            status.set(f"Launch failed: {e}")
            return
        status.set(f"Launched {record.run_id} (pid {record.pid})")
        on_launched(launcher)

    @render.text
    def launch_status():
        return status.get()
