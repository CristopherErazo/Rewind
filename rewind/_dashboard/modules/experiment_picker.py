from pathlib import Path
from typing import Optional
from shiny import module, ui, reactive

@module.ui
def experiment_picker_ui():
    return ui.card(
        ui.card_header("Experiment Selector"),
        ui.input_select("selected_run", "Select Run", choices=[]),
        ui.input_action_button("refresh_btn", "Refresh Runs", class_="btn-sm btn-outline-secondary")
    )

@module.server
def experiment_picker_server(input, output, session, base_dir: Path):
    # Reactive value to store the currently selected run ID
    selected_run_id = reactive.value(None)

    @reactive.effect
    @reactive.event(input.refresh_btn, ignore_none=False)
    def update_run_list():
        if not base_dir.exists():
            return
            
        # Discover directories starting with run_
        run_dirs = sorted(
            [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            reverse=True
        )
        
        current_selection = input.selected_run()
        ui.update_select(
            "selected_run",
            choices=run_dirs,
            selected=current_selection if current_selection in run_dirs else (run_dirs[0] if run_dirs else None)
        )

    @reactive.effect
    def _sync_selection():
        selected_run_id.set(input.selected_run())

    # Return the reactive reader so other modules can react to run selection changes
    return selected_run_id