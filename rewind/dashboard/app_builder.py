"""
build_dashboard(): assembles a full Shiny app from a config dataclass and a
service module path. This is the only thing a project needs to call --
everything else in rewind.dashboard is internal plumbing.
"""

import sys
import subprocess

from shiny import App, reactive, render, ui

from rewind.dashboard.launch_form import build_launch_form, collect_overrides
from rewind.dashboard.components import status_panel, metrics_plot_panel, control_panel
from rewind.tracker_adapter import RunMailbox


def build_dashboard(config_cls, service_module: str, experiment_name: str,
                     runs_base_dir: str = "./data", extra_controls=None):
    """
    config_cls       the project's top-level config dataclass (e.g. TrainerArgs)
    service_module    dotted path to the entry point, e.g. "icl.runtime.service"
    experiment_name    passed through to both the launched runs and the reader
    runs_base_dir        where tracklab stores experiments (matches ExperimentReader)
    extra_controls         optional list of extra UI elements + their own
                            @reactive.effect handlers for project-specific
                            commands (e.g. a "perturb layer" button) -- see
                            the docstring at the bottom of this file
    """
    from tracklab import ExperimentReader
    reader = ExperimentReader(experiment_name, base_dir=runs_base_dir)

    extra_ui = [c["ui"] for c in (extra_controls or [])]

    app_ui = ui.page_fluid(
        ui.h2(f"{experiment_name} — training control panel"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Launch a new run"),
                *build_launch_form(config_cls),
                ui.input_action_button("launch_btn", "Launch run", class_="btn-primary"),
                ui.hr(),
                ui.h4("Pick a run to watch"),
                ui.input_select("selected_run", "Run", choices=[]),
                ui.input_action_button("refresh_runs_btn", "Refresh run list"),
                ui.hr(),
                *control_panel(),
                *([ui.hr()] + extra_ui if extra_ui else []),
            ),
            status_panel(),
            metrics_plot_panel(),
        ),
    )

    def server(input, output, session):
        def mailbox_for_selected():
            run_id = input.selected_run()
            if not run_id:
                return None
            return RunMailbox(reader.exp_dir / run_id)

        # ---- launch ----
        @reactive.effect
        @reactive.event(input.launch_btn)
        def _launch():
            overrides = collect_overrides(input, config_cls)
            cmd = [sys.executable, "-m", service_module,
                   f"extra_args.experiment_name={experiment_name}", *overrides]
            subprocess.Popen(cmd)

        @reactive.effect
        @reactive.event(input.refresh_runs_btn)
        def _refresh_runs():
            ui.update_select("selected_run", choices=reader.list_runs())

        # ---- built-in controls ----
        @reactive.effect
        @reactive.event(input.pause_btn)
        def _pause():
            mb = mailbox_for_selected()
            if mb:
                mb.send_command({"type": "pause"})

        @reactive.effect
        @reactive.event(input.resume_btn)
        def _resume():
            mb = mailbox_for_selected()
            if mb:
                mb.send_command({"type": "resume"})

        @reactive.effect
        @reactive.event(input.set_lr_btn)
        def _set_lr():
            mb = mailbox_for_selected()
            if mb:
                mb.send_command({"type": "set_lr", "lr": float(input.new_lr())})

        @reactive.effect
        @reactive.event(input.rewind_btn)
        def _rewind():
            mb = mailbox_for_selected()
            if mb:
                mb.send_command({"type": "rewind", "step": int(input.rewind_step())})

        # ---- extra, project-specific controls ----
        for extra in (extra_controls or []):
            extra["register"](input, mailbox_for_selected)

        # ---- live monitor ----
        @reactive.calc
        def _heartbeat():
            reactive.invalidate_later(1.0)
            return None

        @render.text
        def status_text():
            _heartbeat()
            mb = mailbox_for_selected()
            return str(mb.get_status()) if mb else "No run selected."

        @render.plot
        def loss_plot():
            _heartbeat()
            run_id = input.selected_run()
            if not run_id:
                return
            import matplotlib.pyplot as plt
            df = reader.load_metrics(run_id)
            if df.empty:
                return
            loss_rows = df[df["metric"] == "loss"]
            fig, ax = plt.subplots()
            if "branch_id" in loss_rows.columns:
                for branch, sub in loss_rows.groupby("branch_id"):
                    ax.plot(sub["step"], sub["value"], label=str(branch))
                ax.legend()
            else:
                ax.plot(loss_rows["step"], loss_rows["value"])
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            return fig

    return App(app_ui, server)


# extra_controls entries look like:
#
#   {
#       "ui": ui.TagList(
#           ui.input_text("layer_name", "Layer"),
#           ui.input_numeric("perturb_scale", "Scale", value=0.01),
#           ui.input_action_button("perturb_btn", "Perturb"),
#       ),
#       "register": lambda input, mailbox_for_selected: (
#           reactive.effect(reactive.event(input.perturb_btn)(lambda: (
#               mailbox_for_selected() and mailbox_for_selected().send_command(
#                   {"type": "perturb_layer", "layer": input.layer_name(),
#                    "scale": input.perturb_scale()}
#               )
#           )))
#       ),
#   }