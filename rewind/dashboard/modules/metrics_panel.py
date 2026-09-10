"""rewind/dashboard/modules/metrics_panel.py

Default live view: one line chart per metric, colored by branch_id when the
run has branched. Reads the long-format rows the app's shared poll provides.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
from shiny import module
from shinywidgets import output_widget, render_widget

MAX_POINTS_PER_TRACE = 4000


@module.ui
def metrics_panel_ui():
    return output_widget("chart")


@module.server
def metrics_panel_server(input, output, session, metrics: Callable[[], list[dict]]):

    @render_widget
    def chart():
        rows = metrics()
        if not rows:
            return px.line(title="Waiting for metrics...")
        df = pd.DataFrame(rows)
        if not {"step", "metric", "value"} <= set(df.columns):
            return px.line(title="metrics.jsonl is not in (step, metric, value) format")
        df = df.dropna(subset=["value"])
        color = "branch_id" if "branch_id" in df.columns else None

        # Stride-downsample each trace so long runs stay responsive.
        keys = ["metric"] + ([color] if color else [])
        parts = []
        for _, g in df.groupby(keys, sort=False):
            if len(g) > MAX_POINTS_PER_TRACE:
                g = g.iloc[:: max(1, len(g) // MAX_POINTS_PER_TRACE)]
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)

        fig = px.line(df, x="step", y="value", color=color, facet_row="metric")
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(margin=dict(t=30, b=10), height=max(220, 200 * df["metric"].nunique()))
        return fig
