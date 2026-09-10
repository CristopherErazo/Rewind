"""The pure half of the live metrics panel: parsing rows into traces, the
structure key that decides rebuild-vs-update, figure construction and the
in-place point push. No Shiny session needed."""

from __future__ import annotations

import pytest

pytest.importorskip("shiny")
pytest.importorskip("shinywidgets")
pytest.importorskip("plotly")

import numpy as np  # noqa: E402

from rewind.dashboard.modules.metrics_panel import (  # noqa: E402
    MAX_POINTS_PER_TRACE, build_figure, parse_rows, push_points, structure_key,
)


def _rows(n_steps: int, branches=("root",), eval_every=50):
    rows = []
    for b in branches:
        for s in range(0, n_steps, 10):
            rows.append({"step": s, "metric": "train_loss", "value": 1.0 / (s + 1), "branch_id": b})
            if s % eval_every == 0:
                rows.append({"step": s, "metric": "eval_loss", "value": 2.0 / (s + 1), "branch_id": b})
    return rows


def test_parse_rows_edge_cases():
    assert parse_rows([]) == ([], [])
    assert parse_rows([{"step": 0, "loss": 1.0}]) is None            # wrong shape
    assert parse_rows([{"step": 0, "metric": "a", "value": None}]) == ([], [])


def test_parse_rows_orders_metrics_and_traces():
    metrics, traces = parse_rows(_rows(200, branches=("root", "b1@t100")))
    assert metrics == ["train_loss", "eval_loss"]  # first-seen order
    assert [t.key for t in traces] == [("train_loss", "root"), ("eval_loss", "root"),
                                       ("train_loss", "b1@t100"), ("eval_loss", "b1@t100")]
    tl = traces[0]
    assert tl.x.tolist() == list(range(0, 200, 10)) and tl.y.dtype == float


def test_parse_rows_without_branch_column():
    rows = [{"step": s, "metric": "loss", "value": float(s)} for s in range(5)]
    metrics, traces = parse_rows(rows)
    assert metrics == ["loss"] and [t.key for t in traces] == [("loss", None)]


def test_parse_rows_downsamples_long_traces():
    rows = [{"step": s, "metric": "loss", "value": 0.5, "branch_id": "root"}
            for s in range(3 * MAX_POINTS_PER_TRACE)]
    _, traces = parse_rows(rows)
    assert len(traces) == 1
    assert MAX_POINTS_PER_TRACE <= len(traces[0].x) <= MAX_POINTS_PER_TRACE + 1
    assert traces[0].x[1] - traces[0].x[0] == 3  # even stride


def test_structure_key_stable_for_new_points_and_changes_for_new_branch():
    k1 = structure_key(parse_rows(_rows(200)))
    k2 = structure_key(parse_rows(_rows(400)))          # same metrics/branches, more points
    k3 = structure_key(parse_rows(_rows(400, branches=("root", "b1@t100"))))
    assert k1 == k2 and k2 != k3
    assert structure_key(None) == ("invalid",)
    assert structure_key(([], [])) == ((), ())


def test_build_figure_matches_trace_order_and_legend():
    parsed = parse_rows(_rows(200, branches=("root", "b1@t100")))
    fig = build_figure(parsed)
    metrics, traces = parsed
    assert len(fig.data) == len(traces)
    assert [d.name for d in fig.data] == ["root", "root", "b1@t100", "b1@t100"]
    assert [d.showlegend for d in fig.data] == [True, False, True, False]  # one legend entry per branch
    assert fig.layout.height == 200 * len(metrics)
    # no automargin: the plot area must not slide when tick labels change width
    assert fig.layout.yaxis.automargin is False and fig.layout.yaxis2.automargin is False
    assert fig.layout.margin.l == 64
    # traces for the second metric land on the second subplot
    assert fig.data[0].yaxis == "y" and fig.data[1].yaxis == "y2"


def test_build_figure_placeholders():
    assert "Waiting" in build_figure(([], [])).layout.title.text
    assert "format" in build_figure(None).layout.title.text


def test_push_points_updates_in_place():
    parsed = parse_rows(_rows(200))
    fig = build_figure(parsed)
    _, traces = parse_rows(_rows(400))
    push_points(fig, traces)
    assert len(fig.data) == len(traces)
    assert np.array_equal(np.asarray(fig.data[0].x), traces[0].x)
    assert np.array_equal(np.asarray(fig.data[1].y), traces[1].y)


def test_clicked_step_of_reads_first_point():
    from types import SimpleNamespace
    from rewind.dashboard.modules.metrics_panel import clicked_step_of
    assert clicked_step_of(SimpleNamespace(xs=[1000.0], ys=[0.5])) == 1000
    assert clicked_step_of(SimpleNamespace(xs=[999.6])) == 1000
    assert clicked_step_of(SimpleNamespace(xs=[])) is None
    assert clicked_step_of(SimpleNamespace()) is None
