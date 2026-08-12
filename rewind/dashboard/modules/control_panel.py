"""rewind/dashboard/modules/control_panel.py

Fully generic: renders whatever ActionSpec list it finds in
`<run_dir>/actions.json`, and sends commands through the run's mailbox. This
never imports project code -- a custom handler (e.g. "perturb layer")
appears here automatically the moment it's registered with a spec on the
TrainerController side. That's the entire payoff of the registry design.
"""

from __future__ import annotations

from pathlib import Path

from shiny import module, reactive, render, ui

from rewind.control import RunMailbox

from ...registry import ActionSpec, read_actions


def _action_ui(spec: ActionSpec):
    if spec.kind == "button":
        return ui.input_action_button(
            spec.html_id, spec.label, title=spec.description, class_="btn-sm mb-2"
        )

    arg_inputs = [
        ui.input_numeric(spec.arg_html_id(arg_name), arg_name, value=arg.default)
        if arg.kind in ("int", "float")
        else ui.input_text(spec.arg_html_id(arg_name), arg_name, value=str(arg.default or ""))
        for arg_name, arg in spec.args.items()
    ]
    return ui.div(
        ui.strong(spec.label),
        ui.p(spec.description, class_="text-muted small mb-1"),
        *arg_inputs,
        ui.input_action_button(spec.html_id, "Send", class_="btn-sm mb-3"),
    )


@module.ui
def control_panel_ui():
    # Populated dynamically server-side, since the action list isn't known
    # until a run is selected and it has written its actions.json.
    return ui.card(ui.card_header("Controls"), 
                   ui.output_text("counter"),
                   ui.output_ui("actions_container"))


@module.server
def control_panel_server(input, output, session, run_dir: reactive.Calc):
    # @reactive.calc
    # def specs() -> list[ActionSpec]:
    #     rd = run_dir()
    #     return read_actions(Path(rd)) if rd else []

    @reactive.calc
    def specs() -> list[ActionSpec]:
        reactive.invalidate_later(1.0)   # keep re-reading actions.json until it exists, then keep it fresh
        rd = run_dir()
        result =  read_actions(Path(rd)) if rd else []
        # print(f"[specs] run_dir={rd}, n_specs={len(result)}")   # <-- add this
        return result

    @render.ui
    def actions_container():
        current = specs()
        if not current:
            return ui.p("No active run selected.", class_="text-muted")
        return [_action_ui(s) for s in current]

    @reactive.calc
    def mailbox() -> RunMailbox | None:
        rd = run_dir()
        return RunMailbox(Path(rd)) if rd else None

    # Action buttons increment a Shiny counter on each click. We track the
    # last-seen count per action (keyed by html_id, the same id the button
    # was rendered with -- write_actions() already guarantees these are
    # unique) so a click is dispatched exactly once, even though this
    # effect re-runs whenever *any* tracked button changes.
    _last_counts = reactive.value({})

    @reactive.effect
    def _dispatch():
        # print(f"[_dispatch] fired, run_dir={run_dir()}")   # <-- add this
        mb = mailbox()
        if mb is None:
            print("[_dispatch] mailbox is None, returning")   # <-- add this
            return
        with reactive.isolate():
            last = _last_counts.get()
        # last = _last_counts.get()
        updated = dict(last)
        for spec in specs():
            if not hasattr(input, spec.html_id):
                print(f"[_dispatch] no input attr for {spec.html_id}")   # <-- add this
                continue
            count = getattr(input, spec.html_id)()
            # print(f"[_dispatch] {spec.html_id}: count={count}, last={last.get(spec.html_id, 0)}")   # <-- add this
            if count != last.get(spec.html_id, 0):
                print(f"[_dispatch] {spec.html_id}: {last.get(spec.html_id, 0)} -> {count}")
        
            if count > last.get(spec.html_id, 0):
                with reactive.isolate():
                    args = {
                        arg_name: getattr(input, spec.arg_html_id(arg_name))()
                        for arg_name in spec.args
                    }
                print(f"[_dispatch] SENDING: {spec.name}")
                # print(f"[_dispatch] sending command: {spec.name}")   # <-- add this
                mb.send_command({"type": spec.name, **args})
            updated[spec.html_id] = count
        _last_counts.set(updated)

    @render.text
    def counter():
        return f'N counts = {_last_counts.get()}'