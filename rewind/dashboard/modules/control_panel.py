"""rewind/dashboard/modules/control_panel.py

Fully generic: renders whatever ActionSpec list the active run wrote to
`control/actions.json`, and sends commands through the run's mailbox. A
custom handler registered with a spec on the trainer side appears here with
no dashboard changes.

Design notes
------------
* `specs` is a reactive.poll keyed on the actions file's stamp, so the
  action widgets are rendered once per run (and again only if the file
  changes). The previous version re-rendered every second, which reset the
  buttons' click counters and dropped clicks.
* One effect per action is created whenever the spec list changes; the
  previous set is destroyed first. Each effect tracks the last click count
  it dispatched (`_last_counts`, keyed by button id, kept across re-binds):
  a count of 0 is a freshly rendered button and resets the tracker, a count
  not above the tracker is stale (e.g. the same button id on a previously
  selected run) and is ignored, anything else is a real click. This is
  deliberately not `reactive.event(ignore_init=True)`: when the effect is
  created before the client has reported the button, ignore_init swallows
  the first real click instead of the initial zero.
* Argument values are coerced with the spec before sending, and the trainer
  coerces again on receipt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from shiny import module, reactive, render, ui

from ...control import LIVE_STATES, RunMailbox
from ...launch import RunLauncher
from ..._fsutil import file_stamp
from ..._layout import actions_path
from ...registry import ActionSpec, coerce_args, read_actions


def _arg_input(spec: ActionSpec, arg_name: str, arg):
    aid = spec.arg_html_id(arg_name)
    label = arg_name if not arg.description else f"{arg_name} ({arg.description})"
    if arg.kind == "bool":
        return ui.input_switch(aid, label, value=bool(arg.default))
    if arg.kind in ("int", "float"):
        return ui.input_numeric(aid, label, value=arg.default,
                                step=1 if arg.kind == "int" else None)
    return ui.input_text(aid, label, value="" if arg.default is None else str(arg.default))


def _action_ui(spec: ActionSpec, disabled: bool):
    if spec.kind == "button":
        return ui.input_action_button(spec.html_id, spec.label, title=spec.description,
                                      class_="btn-sm me-2 mb-2", disabled=disabled)
    return ui.div(
        ui.strong(spec.label),
        ui.p(spec.description, class_="text-muted small mb-1"),
        *[_arg_input(spec, n, a) for n, a in spec.args.items()],
        ui.input_action_button(spec.html_id, "Send", class_="btn-sm btn-outline-primary mb-3",
                               disabled=disabled),
    )


@module.ui
def control_panel_ui():
    return ui.card(
        ui.card_header("Controls"),
        ui.output_ui("actions_container"),
        ui.output_text("last_sent"),
        ui.hr(),
        ui.output_ui("process_controls"),
    )


@module.server
def control_panel_server(input, output, session, run_dir: Callable[[], Path | None],
                         status: Callable[[], dict | None],
                         mailbox: Callable[[], RunMailbox | None],
                         launcher: Callable[[], RunLauncher | None]):

    specs = reactive.poll(
        lambda: (str(run_dir()), file_stamp(actions_path(run_dir())) if run_dir() else None), 1.0,
    )(lambda: read_actions(run_dir()) if run_dir() else [])

    @reactive.calc
    def controls_enabled() -> bool:
        st = status()
        return bool(st) and st.get("state") in LIVE_STATES

    # Rendered once per spec list. Enabled/disabled is applied through
    # update_action_button so a state change does not re-render the widgets.
    @render.ui
    def actions_container():
        current = specs()
        if run_dir() is None:
            return ui.p("Select a run in the sidebar.", class_="text-muted")
        if not current:
            return ui.p("This run has not published any actions (control/actions.json missing).",
                        class_="text-muted")
        with reactive.isolate():
            disabled = not controls_enabled()
        buttons = [s for s in current if s.kind == "button"]
        forms = [s for s in current if s.kind == "form"]
        return ui.div(ui.div(*[_action_ui(s, disabled) for s in buttons]),
                      *[_action_ui(s, disabled) for s in forms])

    @reactive.effect
    def _toggle_enabled():
        enabled = controls_enabled()
        for spec in specs():
            ui.update_action_button(spec.html_id, disabled=not enabled)

    last_sent_msg = reactive.value("")
    _effects: list = []
    _last_counts: dict[str, int] = {}

    @reactive.effect
    def _bind_actions():
        current = specs()
        for eff in _effects:
            eff.destroy()
        _effects.clear()
        for spec in current:
            _effects.append(_make_dispatcher(spec))

    def _make_dispatcher(spec: ActionSpec):
        @reactive.effect
        def _dispatch():
            count = int(input[spec.html_id]())  # SilentException until the client reports it
            last = _last_counts.get(spec.html_id, 0)
            if count == 0:
                _last_counts[spec.html_id] = 0  # freshly rendered button
                return
            if count <= last:
                return  # stale count from an earlier render / run
            _last_counts[spec.html_id] = count
            with reactive.isolate():
                mb = mailbox()
                if mb is None:
                    return
                raw = {n: input[spec.arg_html_id(n)]() for n in spec.args}
            try:
                cmd = coerce_args(spec, {"type": spec.name, **raw})
            except ValueError as e:
                last_sent_msg.set(f"Not sent: {e}")
                return
            mb.send_command(cmd)
            args = {k: v for k, v in cmd.items() if k != "type"}
            last_sent_msg.set(f"Sent {spec.name}" + (f" {args}" if args else ""))
        return _dispatch

    @render.text
    def last_sent():
        return last_sent_msg.get()

    # ---- process-level controls (only for runs with a process.json) -------
    @render.ui
    def process_controls():
        lc = launcher()
        if lc is None:
            return ui.p("No process record for this run; kill is unavailable "
                        "(the run was not started through a RunLauncher).", class_="text-muted small")
        return ui.div(
            ui.input_action_button("kill", "Kill process", class_="btn-sm btn-danger"),
            ui.span(" Terminates the training process immediately; the run ends without a final "
                    "status write.", class_="text-muted small"),
        )

    @reactive.effect
    @reactive.event(input.kill, ignore_init=True)
    def _kill():
        lc = launcher()
        if lc is not None:
            lc.terminate(force=True)
            last_sent_msg.set(f"Killed pid {lc.record.pid}")
