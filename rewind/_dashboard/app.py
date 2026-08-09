from shiny import App, ui
from .config import DashboardConfig
from .launch import RunLauncher
from .modules import (
    experiment_picker,
    launcher_form,
    control_panel,
    runs_table,
    metrics_panel
)

def build_app(config: DashboardConfig) -> App:
    launcher = RunLauncher(tracklab_dir=config.base_dir)

    app_ui = ui.page_fluid(
        ui.h2("Rewind MLOps Control Plane"),
        ui.hr(),
        ui.layout_sidebar(
            ui.sidebar(
                experiment_picker.experiment_picker_ui("picker"),
                ui.hr(),
                launcher_form.launcher_form_ui("launcher_form")
            ),
            ui.layout_columns(
                control_panel.control_panel_ui("controls"),
                metrics_panel.metrics_panel_ui("metrics"),
                col_widths=[12, 12]
            ),
            ui.hr(),
            runs_table.runs_table_ui("runs_table")
        )
    )

    def server(input, output, session):
        # 1. Picker returns a reactive value containing the selected run ID
        selected_run_id = experiment_picker.experiment_picker_server(
            "picker", base_dir=config.base_dir
        )

        # 2. Form launches runs and triggers picker refresh
        launcher_form.launcher_form_server(
            "launcher_form",
            launcher=launcher,
            script_path=config.script_path,
            on_launch_cb=lambda: None # Optional refresh signal
        )

        # 3. Controls and Metrics receive the reactive selection reader
        control_panel.control_panel_server(
            "controls",
            selected_run_id=selected_run_id,
            base_dir=config.base_dir,
            launcher=launcher
        )

        metrics_panel.metrics_panel_server(
            "metrics",
            selected_run_id=selected_run_id,
            base_dir=config.base_dir
        )

        runs_table.runs_table_server(
            "runs_table",
            base_dir=config.base_dir,
            launcher=launcher
        )

    return App(app_ui, server)