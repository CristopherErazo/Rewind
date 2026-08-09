from pathlib import Path
from typing import Callable
from shiny import module, ui, render, reactive
import pandas as pd
import json

@module.ui
def metrics_panel_ui():
    return ui.card(
        ui.card_header("Metrics Visualizer"),
        ui.output_table("metrics_summary_table")
    )

@module.server
def metrics_panel_server(input, output, session, selected_run_id: Callable[[], str], base_dir: Path):
    @output
    @render.table
    def metrics_summary_table():
        reactive.invalidate_later(2) # Refresh metrics periodically
        
        run_id = selected_run_id()
        if not run_id:
            return pd.DataFrame({"Info": ["Select a run to view metrics"]})
            
        metrics_file = base_dir / run_id / "metrics.json" # Assumes TrackLab writes here
        
        if not metrics_file.exists():
            return pd.DataFrame({"Info": [f"No metrics.json found for {run_id}"]})

        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
            return pd.DataFrame([data])
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})