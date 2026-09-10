"""rewind/dashboard/modules/runs_table.py

One row per run in the selected experiment, columns are the config fields
that vary across runs (ExperimentReader.summarize_runs). Re-renders when the
run listing changes.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from shiny import module, render, ui


@module.ui
def runs_table_ui():
    return ui.card(ui.card_header("Runs"), ui.output_data_frame("table"))


@module.server
def runs_table_server(input, output, session, reader: Callable, runs: Callable[[], list[str]]):

    @render.data_frame
    def table():
        runs()  # dependency: refresh when the run listing changes
        try:
            df = reader().summarize_runs()
        except Exception:
            df = pd.DataFrame()
        return render.DataGrid(df, filters=True, width="100%")
