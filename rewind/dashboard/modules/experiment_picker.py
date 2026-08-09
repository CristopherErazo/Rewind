"""rewind/dashboard/modules/experiment_picker.py

Lists experiments under base_dir, lets the user pick one or name a new one.
Deliberately dumb -- this is the one panel nothing should ever need to
extend, so it stays a directory listing plus a text input.
"""

from __future__ import annotations

from pathlib import Path

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
def experiment_picker_server(input, output, session, base_dir: Path):

    @reactive.calc
    def experiment_names() -> list[str]:
        if not base_dir.exists():
            return []
        return sorted(p.name for p in base_dir.iterdir() if p.is_dir())

    @reactive.effect
    def _refresh_choices():
        ui.update_select("existing", choices=experiment_names())

    @reactive.calc
    def selected_experiment() -> str | None:
        new_name = input.new_name().strip()
        if new_name:
            return new_name
        return input.existing() or None

    @render.text
    def selected_label():
        sel = selected_experiment()
        return f"Selected: {sel}" if sel else "No experiment selected"

    # Module servers can return reactive values for the parent app to
    # compose with -- this is how build_dashboard() learns what's selected
    # without experiment_picker needing to know about anything downstream.
    return selected_experiment