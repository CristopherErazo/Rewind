from pathlib import Path
from typing import Callable
from shiny import module, ui, reactive, render
from ..launch import RunLauncher
from ..registry import write_action

@module.ui
def control_panel_ui():
    return ui.card(
        ui.card_header("Run Control Plane"),
        ui.layout_columns(
            ui.input_action_button("pause_btn", "Pause", class_="btn-warning"),
            ui.input_action_button("resume_btn", "Resume", class_="btn-success"),
            ui.input_action_button("stop_btn", "Graceful Stop", class_="btn-secondary"),
            ui.input_action_button("kill_btn", "Force Kill", class_="btn-danger")
        ),
        ui.output_text("run_status_display")
    )

@module.server
def control_panel_server(input, output, session, selected_run_id: Callable[[], str], base_dir: Path, launcher: RunLauncher):
    
    def _get_run_dir() -> Path | None:
        run_id = selected_run_id()
        return (base_dir / run_id) if run_id else None

    @reactive.effect
    @reactive.event(input.pause_btn)
    def _pause():
        r_dir = _get_run_dir()
        if r_dir:
            write_action(r_dir, action="pause")
            ui.notification_show("Pause signal dispatched", type="warning")

    @reactive.effect
    @reactive.event(input.resume_btn)
    def _resume():
        r_dir = _get_run_dir()
        if r_dir:
            write_action(r_dir, action="resume")
            ui.notification_show("Resume signal dispatched", type="message")

    @reactive.effect
    @reactive.event(input.stop_btn)
    def _stop():
        r_dir = _get_run_dir()
        if r_dir:
            write_action(r_dir, action="stop")
            ui.notification_show("Stop signal dispatched", type="warning")

    @reactive.effect
    @reactive.event(input.kill_btn)
    def _kill():
        run_id = selected_run_id()
        if run_id and launcher.kill(run_id):
            ui.notification_show(f"Process {run_id} killed.", type="error")

    @output
    @render.text
    def run_status_display():
        run_id = selected_run_id()
        if not run_id:
            return "No run selected."
        status = launcher.get_status(run_id)
        return f"Selected Run: {run_id} | Launcher Process State: {status.upper()}"