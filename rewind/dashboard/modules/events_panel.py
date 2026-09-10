"""rewind/dashboard/modules/events_panel.py

Table view of `control/events.jsonl`: lifecycle transitions, applied and
failed commands, and forks. Newest first.
"""

from __future__ import annotations

import json
import time
from typing import Callable

import pandas as pd
from shiny import module, render, ui


@module.ui
def events_panel_ui():
    return ui.card(ui.card_header("Events"), ui.output_data_frame("table"))


def _payload(row: dict) -> str:
    skip = {"time", "kind", "step", "branch_id"}
    rest = {k: v for k, v in row.items() if k not in skip}
    return json.dumps(rest, default=str) if rest else ""


@module.server
def events_panel_server(input, output, session, events: Callable[[], list[dict]]):

    @render.data_frame
    def table():
        rows = events()
        if not rows:
            return render.DataGrid(pd.DataFrame(columns=["time", "kind", "step", "branch", "details"]))
        df = pd.DataFrame([{
            "time": time.strftime("%H:%M:%S", time.localtime(r.get("time", 0))),
            "kind": r.get("kind"),
            "step": r.get("step"),
            "branch": r.get("branch_id"),
            "details": _payload(r),
        } for r in reversed(rows)])
        return render.DataGrid(df, filters=True, width="100%")
