"""rewind/dashboard/modules/run_picker.py

Selects the run the whole dashboard is looking at. This is the attach path:
any run folder in the experiment can be picked, whether it was started from
this dashboard, another one, or a terminal. A successful launch pushes its
run_id in through `launched_run_id` so the picker follows it.
"""

from __future__ import annotations

from typing import Callable

from shiny import module, reactive, ui


@module.ui
def run_picker_ui():
    return ui.card(
        ui.card_header("Run"),
        ui.input_select("run", "Active run", choices=[]),
        ui.input_switch("follow_latest", "Follow newest run", value=True),
    )


@module.server
def run_picker_server(input, output, session, runs: Callable[[], list[str]],
                      launched_run_id: reactive.Value):

    @reactive.effect
    def _refresh_choices():
        names = runs()
        with reactive.isolate():
            current = input.run()
            follow = input.follow_latest()
        if follow or current not in names:
            selected = names[-1] if names else None
        else:
            selected = current
        ui.update_select("run", choices=names, selected=selected)

    @reactive.effect
    @reactive.event(launched_run_id)
    def _jump_to_launched():
        rid = launched_run_id.get()
        if rid:
            names = runs()
            ui.update_select("run", choices=sorted(set(names) | {rid}), selected=rid)

    @reactive.calc
    def selected_run() -> str | None:
        return input.run() or None

    return selected_run
