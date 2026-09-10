"""rewind/dashboard/modules/control_panel.py

Fully generic: renders whatever ActionSpec list the active run wrote to
`control/actions.json`, and sends commands through the run's mailbox. A
custom handler registered with a spec on the trainer side appears here with
no dashboard changes.

Layout
------
One thin toolbar row directly above the live plot, so interventions never
take the metrics out of sight:

* button actions (pause, resume, stop, ...) are always-visible small
  buttons, with player-style icons for the three built-ins; the action's
  description is a hover tooltip.
* form actions (set_lr, rewind, perturb, ...) are inline input groups:
  `[label | input(s) | send]`. Type a value and press Enter (or click the
  return-arrow button) to send. No menu to open or close. The label's hover
  tooltip carries the description, each argument's type, default and
  meaning, and, for an integer `step` argument, the hint that clicking a
  point on the plot fills it in (`step_hint`, wired by app.py).
* the "Sent ..." confirmation, a not-live note and the kill button (when a
  process record exists) sit at the right end of the same row.

Design notes
------------
* Argument inputs are bare `<input>` tags (no shiny.ui wrapper) so they can
  be direct children of a Bootstrap input-group. Shiny's bindings key on
  `input[type=number|text|checkbox]`, so they bind, and `ui.update_numeric`
  finds them by id like any other input. Raw tags do not apply the module
  namespace, so their ids go through `resolve_id` explicitly.
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
from shiny.module import resolve_id

from ...control import LIVE_STATES, RunMailbox
from ...launch import RunLauncher
from ..._fsutil import file_stamp
from ..._layout import actions_path
from ...registry import ActionSpec, ArgSpec, coerce_args, read_actions

STEP_ARG = "step"  # integer arguments with this name receive clicks on the plot

_CSS = """
.rw-controls { font-size: .875rem; }
.rw-controls .rw-note { font-size: .8rem; }
.rw-controls bslib-tooltip { display: contents; }
.rw-controls .btn svg { vertical-align: -0.125em; }
.rw-controls .btn .action-icon + .action-label { margin-left: .3rem; }
.rw-group .input-group-text { border-top-right-radius: 0; border-bottom-right-radius: 0; }
.rw-group .form-control { width: 6.5em; flex: 0 0 auto; }
.rw-group .form-control[type=text] { width: 8em; }
.rw-group .form-control::placeholder { color: var(--bs-secondary-color); opacity: .6; }
"""

# Enter inside an input group sends it. blur() first so the input's change
# event (which carries the new value) reaches the server before the click.
_JS = """
document.addEventListener('keydown', function (e) {
  if (e.key !== 'Enter') return;
  var group = e.target.closest('.rw-group');
  if (!group) return;
  var send = group.querySelector('button.rw-send');
  if (!send || send.disabled) return;
  e.preventDefault();
  e.target.blur();
  setTimeout(function () { send.click(); }, 0);
});
"""

# Small player-style glyphs, drawn inline so no icon package is needed.
_GLYPHS = {
    "pause": '<rect x="3.5" y="2.5" width="3.2" height="11" rx="1"/><rect x="9.3" y="2.5" width="3.2" height="11" rx="1"/>',
    "resume": '<polygon points="4,2.5 13,8 4,13.5"/>',
    "stop": '<rect x="3" y="3" width="10" height="10" rx="1.5"/>',
    # return arrow: distinct from the play triangle, and hints "Enter sends"
    "send": ('<path d="M13 3v4.5a1.5 1.5 0 0 1-1.5 1.5H4.6" fill="none" stroke="currentColor" '
             'stroke-width="1.6" stroke-linecap="round"/><polyline points="7.2,6.2 4.2,9 7.2,11.8" '
             'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'),
}


def _glyph(name: str):
    body = _GLYPHS.get(name)
    if body is None:
        return None
    return ui.HTML(f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 16 16" '
                   f'fill="currentColor" aria-hidden="true">{body}</svg>')


def _arg_meta(arg: ArgSpec) -> str:
    meta = arg.kind
    if arg.default is not None:
        meta += f", default {arg.default}"
    if not arg.required:
        meta += ", optional"
    return meta


def _arg_input(spec: ActionSpec, arg_name: str, arg: ArgSpec):
    """Bare input tag, a direct child of the input-group (see design notes)."""
    aid = resolve_id(spec.arg_html_id(arg_name))  # raw tags do not namespace themselves
    attrs = {"id": aid, "class_": "form-control", "placeholder": arg_name,
             "aria-label": f"{spec.label}: {arg_name}", "data-update-on": "change"}
    if arg.kind == "bool":
        box = ui.tags.input(type="checkbox", id=aid, class_="form-check-input mt-0",
                            checked="checked" if arg.default else None, **{"aria-label": arg_name})
        return ui.div(box, ui.span(arg_name, class_="ms-1"), class_="input-group-text")
    if arg.kind in ("int", "float"):
        attrs["class_"] += " shiny-input-number"
        return ui.tags.input(type="number", value=None if arg.default is None else arg.default,
                             step=1 if arg.kind == "int" else "any", **attrs)
    return ui.tags.input(type="text", value="" if arg.default is None else str(arg.default), **attrs)


def _takes_step_hint(spec: ActionSpec) -> bool:
    arg = spec.args.get(STEP_ARG)
    return arg is not None and arg.kind == "int"


def _form_tooltip(spec: ActionSpec):
    lines = []
    if spec.description:
        lines.append(ui.div(spec.description))
    for n, a in spec.args.items():
        lines.append(ui.div(ui.code(n), f" ({_arg_meta(a)})", f": {a.description}" if a.description else ""))
    if _takes_step_hint(spec):
        lines.append(ui.div("Click a point on the plot to fill in the step.", class_="fst-italic"))
    lines.append(ui.div("Enter sends.", class_="text-muted"))
    return lines


def _button_ui(spec: ActionSpec, disabled: bool):
    btn = ui.input_action_button(spec.html_id, spec.label, icon=_glyph(spec.name),
                                 class_="btn-sm btn-outline-secondary", disabled=disabled)
    return ui.tooltip(btn, spec.description, placement="bottom") if spec.description else btn


def _form_ui(spec: ActionSpec, disabled: bool):
    label = ui.tooltip(ui.span(spec.label, class_="input-group-text"), *_form_tooltip(spec),
                       placement="bottom")
    send = ui.input_action_button(spec.html_id, "", icon=_glyph("send"),
                                  class_="btn-sm btn-outline-primary rw-send", disabled=disabled,
                                  title=f"Send {spec.name}")
    return ui.div(label, *[_arg_input(spec, n, a) for n, a in spec.args.items()], send,
                  class_="input-group input-group-sm flex-nowrap w-auto rw-group")


@module.ui
def control_panel_ui():
    return ui.div(
        ui.tags.style(_CSS),
        ui.tags.script(_JS),
        ui.output_ui("actions_container", class_="d-flex flex-wrap align-items-center gap-1"),
        ui.div(
            ui.output_ui("state_note", inline=True),
            ui.output_text("last_sent", inline=True),
            ui.output_ui("process_controls", inline=True),
            class_="d-flex flex-wrap align-items-center gap-2 ms-auto rw-note text-muted",
        ),
        class_="rw-controls d-flex flex-wrap align-items-center gap-2 mb-2",
    )


@module.server
def control_panel_server(input, output, session, run_dir: Callable[[], Path | None],
                         status: Callable[[], dict | None],
                         mailbox: Callable[[], RunMailbox | None],
                         launcher: Callable[[], RunLauncher | None],
                         step_hint: Callable[[], int | None] | None = None):
    """`step_hint` is a reactive callable (e.g. the metrics panel's clicked
    step); each new value is written into every integer `step` argument."""

    specs = reactive.poll(
        lambda: (str(run_dir()), file_stamp(actions_path(run_dir())) if run_dir() else None), 1.0,
    )(lambda: read_actions(run_dir()) if run_dir() else [])

    # status() changes on every poll (updated_at, step). Anything rendered
    # from it must depend on a reactive.value that is only set() when the
    # displayed fact changes, otherwise the output recalculates every second
    # and Shiny's busy indicator (min-height 32px on a recalculating output)
    # makes even an empty note jump the layout around it.
    controls_enabled: reactive.Value[bool] = reactive.value(False)
    note_text: reactive.Value[str] = reactive.value("")

    @reactive.effect
    def _track_status():
        st = status()
        enabled = bool(st) and st.get("state") in LIVE_STATES
        note = "" if (run_dir() is None or not st or enabled) else \
            f"run is {st.get('state', 'not live')}, controls disabled"
        controls_enabled.set(enabled)  # no-op when unchanged
        note_text.set(note)

    @render.ui
    def state_note():
        return ui.span(note_text()) if note_text() else None

    # Rendered once per spec list. Enabled/disabled is applied through
    # update_action_button so a state change does not re-render the widgets.
    @render.ui
    def actions_container():
        current = specs()
        if run_dir() is None:
            return ui.span("Select a run in the sidebar.", class_="text-muted rw-note")
        if not current:
            return ui.span("This run has not published any actions (control/actions.json missing).",
                           class_="text-muted rw-note")
        with reactive.isolate():
            disabled = not controls_enabled()
        buttons = [s for s in current if s.kind == "button"]
        forms = [s for s in current if s.kind == "form"]
        return ui.TagList(
            *[_button_ui(s, disabled) for s in buttons],
            ui.div(class_="vr mx-1") if buttons and forms else None,
            *[_form_ui(s, disabled) for s in forms],
        )

    @reactive.effect
    def _toggle_enabled():
        enabled = controls_enabled()
        for spec in specs():
            ui.update_action_button(spec.html_id, disabled=not enabled)

    if step_hint is not None:
        @reactive.effect
        def _fill_step():
            step = step_hint()
            if step is None:
                return
            for spec in specs():
                if _takes_step_hint(spec):
                    ui.update_numeric(spec.arg_html_id(STEP_ARG), value=int(step))

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
            return ui.tooltip(ui.span("kill is unavailable", class_="text-muted"),
                              "No process record for this run (it was not started through a "
                              "RunLauncher), so the dashboard cannot terminate its process.",
                              placement="bottom")
        return ui.tooltip(
            ui.input_action_button("kill", "Kill process", class_="btn-sm btn-outline-danger"),
            "Terminates the training process immediately; the run ends without a final status write.",
            placement="bottom",
        )

    @reactive.effect
    @reactive.event(input.kill, ignore_init=True)
    def _kill():
        lc = launcher()
        if lc is not None:
            lc.terminate(force=True)
            last_sent_msg.set(f"Killed pid {lc.record.pid}")
