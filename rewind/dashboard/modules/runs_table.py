"""rewind/dashboard/modules/runs_table.py

Nothing project-specific: ExperimentReader.summarize_runs() already handles
flattening whatever config schema a project uses.
"""

from __future__ import annotations

from shiny import module, reactive, render, ui


@module.ui
def runs_table_ui():
    return ui.card(ui.card_header("Runs"), ui.output_data_frame("table"))


@module.server
def runs_table_server(input, output, session, reader: reactive.Calc):
    @render.data_frame
    def table():
        return render.DataGrid(reader().summarize_runs(), filters=True)