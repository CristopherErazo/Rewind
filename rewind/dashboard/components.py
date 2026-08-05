"""
Generic, reusable pieces of the dashboard UI: the status readout, the
metrics plot, and the built-in control buttons. None of these know
anything about a specific model or project.
"""

from shiny import ui


def status_panel():
    return ui.output_text_verbatim("status_text")


def metrics_plot_panel():
    return ui.output_plot("loss_plot")


def control_panel():
    return [
        ui.h4("Controls"),
        ui.input_action_button("pause_btn", "Pause"),
        ui.input_action_button("resume_btn", "Resume"),
        ui.input_numeric("new_lr", "New learning rate", value=0.001),
        ui.input_action_button("set_lr_btn", "Apply LR"),
        ui.input_numeric("rewind_step", "Rewind to step", value=0),
        ui.input_action_button("rewind_btn", "Rewind"),
    ]