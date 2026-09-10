"""rewind/dashboard/modules/metrics_panel.py

Default live view: one subplot per metric, one line per branch_id when the
run has branched. Reads the long-format rows the app's shared poll provides.

Design notes
------------
* The Plotly widget is built once per *structure*, meaning the ordered set
  of metrics and the (metric, branch) pairs that have traces. Every other
  poll pushes the new x/y arrays into the existing traces inside a
  `batch_update()`, so the browser restyles the lines in place. The previous
  version returned a fresh `px.line` figure from the renderer on every poll;
  shinywidgets turned each one into a brand-new widget, and the browser tore
  down and re-created the whole plot every second, which is what made the
  live tab lag, flicker and burn CPU.
* `structure` is a reactive.value that is only `set()` when the key
  changes, so the renderer does not re-run for ordinary new points.
* Each trace is stride-downsampled to MAX_POINTS_PER_TRACE so long runs stay
  cheap to serialize and draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from shiny import module, reactive
from shinywidgets import output_widget, render_widget

MAX_POINTS_PER_TRACE = 4000
ROW_HEIGHT_PX = 200
MIN_HEIGHT_PX = 220
REQUIRED_COLUMNS = {"step", "metric", "value"}


@dataclass(frozen=True)
class Trace:
    metric: str
    branch: str | None      # None when the run has no branch_id column
    x: np.ndarray
    y: np.ndarray

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.metric, self.branch)


def parse_rows(rows: list[dict]) -> tuple[list[str], list[Trace]] | None:
    """Long-format rows -> (ordered metric names, one Trace per metric and
    branch). Returns None when the rows are not (step, metric, value)."""
    if not rows:
        return [], []
    df = pd.DataFrame(rows)
    if not REQUIRED_COLUMNS <= set(df.columns):
        return None
    df = df.dropna(subset=["value"])
    if df.empty:
        return [], []
    metrics = [str(m) for m in dict.fromkeys(df["metric"])]
    has_branch = "branch_id" in df.columns
    if has_branch:
        df = df.assign(branch_id=df["branch_id"].fillna("root").astype(str))
        keys = ["metric", "branch_id"]
    else:
        keys = ["metric"]
    traces = []
    for key, g in df.groupby(keys, sort=False):
        metric, branch = (key[0], key[1]) if has_branch else (key[0], None)
        if len(g) > MAX_POINTS_PER_TRACE:
            g = g.iloc[:: max(1, len(g) // MAX_POINTS_PER_TRACE)]
        traces.append(Trace(str(metric), branch, g["step"].to_numpy(), g["value"].to_numpy(dtype=float)))
    return metrics, traces


def structure_key(parsed: tuple[list[str], list[Trace]] | None) -> tuple | None:
    """What has to change for the figure to be rebuilt rather than updated."""
    if parsed is None:
        return ("invalid",)
    metrics, traces = parsed
    return (tuple(metrics), tuple(t.key for t in traces))


def build_figure(parsed: tuple[list[str], list[Trace]] | None) -> go.Figure:
    """A go.Figure (shinywidgets wraps it in a FigureWidget) whose trace
    order matches `parsed[1]`, so `push_points` can update by position."""
    if parsed is None:
        return go.Figure(layout=dict(title="metrics.jsonl is not in (step, metric, value) format",
                                     height=MIN_HEIGHT_PX))
    metrics, traces = parsed
    if not metrics:
        return go.Figure(layout=dict(title="Waiting for metrics...", height=MIN_HEIGHT_PX))

    fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True,
                        subplot_titles=metrics, vertical_spacing=min(0.15, 0.3 / len(metrics)))
    branches = list(dict.fromkeys(t.branch for t in traces if t.branch is not None))
    palette = qualitative.Plotly
    color_of = {b: palette[i % len(palette)] for i, b in enumerate(branches)}
    shown: set[str] = set()
    for t in traces:
        row = metrics.index(t.metric) + 1
        if t.branch is None:
            scatter = go.Scatter(x=t.x, y=t.y, mode="lines", name=t.metric, showlegend=False)
        else:
            scatter = go.Scatter(x=t.x, y=t.y, mode="lines", name=t.branch, legendgroup=t.branch,
                                 showlegend=t.branch not in shown, line=dict(color=color_of[t.branch]))
            shown.add(t.branch)
        fig.add_trace(scatter, row=row, col=1)
    fig.update_xaxes(title_text="step", row=len(metrics), col=1)
    fig.update_layout(margin=dict(t=30, b=10), height=max(MIN_HEIGHT_PX, ROW_HEIGHT_PX * len(metrics)),
                      legend_title_text="branch" if branches else None, uirevision="keep")
    return fig


def push_points(widget: go.FigureWidget, traces: list[Trace]) -> None:
    """Replace every trace's data in place; one message to the browser."""
    with widget.batch_update():
        for target, t in zip(widget.data, traces):
            target.x = t.x
            target.y = t.y


@module.ui
def metrics_panel_ui():
    return output_widget("chart")


@module.server
def metrics_panel_server(input, output, session, metrics: Callable[[], list[dict]]):

    @reactive.calc
    def parsed():
        return parse_rows(metrics())

    structure: reactive.Value[tuple | None] = reactive.value(None)
    rendered_key: list = [None]  # structure the current widget was built for

    @reactive.effect
    def _track_structure():
        key = structure_key(parsed())
        with reactive.isolate():
            if structure.get() != key:
                structure.set(key)

    @render_widget
    def chart():
        key = structure()
        with reactive.isolate():
            current = parsed()
        rendered_key[0] = key
        return build_figure(current)

    @reactive.effect
    def _push_points():
        current = parsed()
        widget = chart.widget  # re-runs after every re-render as well
        if widget is None or current is None:
            return
        if rendered_key[0] != structure_key(current):
            return  # a rebuild is pending; it will carry these points
        _, traces = current
        if len(widget.data) == len(traces):
            push_points(widget, traces)
