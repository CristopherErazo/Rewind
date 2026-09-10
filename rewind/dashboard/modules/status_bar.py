"""rewind/dashboard/modules/status_bar.py

One line above the tabs: run id, state badge, step progress, branch, lr,
age of the last status write, and whether the process is alive when a
process.json exists. This is the feedback loop for every command sent from
the control panel.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from shiny import module, reactive, render, ui

from ...control import LIVE_STATES
from ...launch import RunLauncher

_BADGE = {
    "running": "success", "paused": "warning", "done": "primary",
    "stopped": "secondary", "crashed": "danger", "interrupted": "danger",
}


@module.ui
def status_bar_ui():
    return ui.card(ui.output_ui("bar"), class_="mb-2")


@module.server
def status_bar_server(input, output, session, run_dir: Callable[[], Path | None],
                      status: Callable[[], dict | None],
                      launcher: Callable[[], RunLauncher | None]):

    @render.ui
    def bar():
        reactive.invalidate_later(1.0)  # keep the "updated Ns ago" counter moving
        rd = run_dir()
        if rd is None:
            return ui.span("No run selected.", class_="text-muted")
        st = status()
        if not st:
            return ui.span(f"{rd.name}: no control/status.json yet (not a rewind-controlled run, "
                           "or it has not started).", class_="text-muted")

        state = st.get("state", "unknown")
        step, total = st.get("step", 0), st.get("total_steps")
        pct = (100 * step / total) if total else None
        age = time.time() - st.get("updated_at", 0)
        lr = st.get("lr")

        items = [
            ui.strong(rd.name),
            ui.span(state, class_=f"badge bg-{_BADGE.get(state, 'secondary')} ms-2 me-2"),
            ui.span(f"step {step}" + (f" / {total} ({pct:.0f}%)" if total else "")),
        ]
        if st.get("branch_id") and st["branch_id"] != "root":
            items.append(ui.span(f"branch {st['branch_id']}", class_="ms-3"))
        if lr is not None:
            items.append(ui.span(f"lr {lr:.3g}", class_="ms-3"))
        items.append(ui.span(f"updated {age:.0f}s ago", class_="ms-3 text-muted"))
        if state in LIVE_STATES and age > 30:
            items.append(ui.span("stale: no status write for >30s", class_="ms-2 text-danger"))
        if state == "crashed" and st.get("error"):
            items.append(ui.div(ui.code(st["error"]), class_="mt-1 text-danger small"))

        lc = launcher()
        if lc is not None:
            alive = lc.is_alive()
            items.append(ui.span(f"pid {lc.record.pid} " + ("alive" if alive else "dead"),
                                 class_="ms-3 " + ("text-success" if alive else "text-muted")))
        return ui.div(*items, class_="d-flex flex-wrap align-items-center small")
