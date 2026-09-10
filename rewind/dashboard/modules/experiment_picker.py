"""rewind/dashboard/modules/experiment_picker.py

Pick an existing experiment folder under base_dir, or type a new name (used
by the launch form). The list of names comes from the app as a reactive
callable so it refreshes when folders appear.
"""

from __future__ import annotations

from typing import Callable

from shiny import module, reactive, render, ui


@module.ui
def experiment_picker_ui():
    return ui.card(
        ui.card_header("Experiment"),
        ui.input_select("existing", "Existing experiments", choices=[]),
        ui.input_text("new_name", "...or name a new one", placeholder="my-experiment"),
        ui.output_text("selected_label"),
    )


@module.server
def experiment_picker_server(input, output, session, experiment_names: Callable[[], list[str]]):

    @reactive.effect
    def _refresh_choices():
        names = experiment_names()
        with reactive.isolate():
            current = input.existing()
        selected = current if current in names else (names[-1] if names else None)
        ui.update_select("existing", choices=names, selected=selected)

    @reactive.calc
    def selected_experiment() -> str | None:
        new_name = (input.new_name() or "").strip()
        if new_name:
            return new_name
        return input.existing() or None

    @render.text
    def selected_label():
        sel = selected_experiment()
        return f"Selected: {sel}" if sel else "No experiment selected"

    return selected_experiment
