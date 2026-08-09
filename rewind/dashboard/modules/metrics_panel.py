"""rewind/dashboard/modules/metrics_panel.py

Default live view: one line chart per distinct metric name, faceted by
branch_id automatically when Rewind branching produces that column. Reads
whatever is in metrics.jsonl -- no project awareness required. Projects that
want a bespoke plot add a DashboardExtension instead of touching this file.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from shiny import module, reactive
from shinywidgets import output_widget, render_widget


@module.ui
def metrics_panel_ui():
    return output_widget("chart")


@module.server
def metrics_panel_server(input, output, session, metrics_df: reactive.Calc):
    """metrics_df: zero-arg callable returning the current metrics DataFrame
    (or None), e.g. the "metrics" slice of the app-level shared poll."""

    @render_widget
    def chart():
        df: pd.DataFrame | None = metrics_df()
        if df is None or df.empty:
            return px.line(title="Waiting for metrics...")
        color = "branch_id" if "branch_id" in df.columns else None
        fig = px.line(df, x="step", y="value", color=color, facet_row="metric")
        fig.update_yaxes(matches=None)  # each metric keeps its own y-scale
        fig.update_layout(margin=dict(t=30, b=10), height=180 * df["metric"].nunique())
        return fig