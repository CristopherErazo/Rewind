from pathlib import Path
from shiny import module, ui, render, reactive
import pandas as pd
from ..launch import RunLauncher

@module.ui
def runs_table_ui():
    return ui.card(
        ui.card_header("All Experiments Overview"),
        ui.output_data_frame("runs_dataframe")
    )

@module.server
def runs_table_server(input, output, session, base_dir: Path, launcher: RunLauncher):
    @output
    @render.data_frame
    def runs_dataframe():
        # Reactive timer auto-refreshes every 3 seconds
        reactive.invalidate_later(3)
        
        if not base_dir.exists():
            return render.DataGrid(pd.DataFrame())

        data = []
        for d in sorted(base_dir.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("run_"):
                status = launcher.get_status(d.name)
                data.append({"Run ID": d.name, "Status": status, "Path": str(d)})

        df = pd.DataFrame(data) if data else pd.DataFrame(columns=["Run ID", "Status", "Path"])
        return render.DataGrid(df)