"""rewind.dashboard -- a Shiny UI for attaching to, watching and steering a
run through Rewind's control plane. Requires the `dashboard` extra
(tracklab, shiny, shinywidgets, plotly, pandas).

Minimal, attach-only:

    from rewind.dashboard import build_dashboard, DashboardConfig
    app = build_dashboard(DashboardConfig(base_dir="./data"))

With launching enabled (form generated from a config dataclass):

    app = build_dashboard(DashboardConfig(
        base_dir="./data", config_cls=TrainerArgs, entrypoint="my_project.launcher"))

Run with `shiny run --reload path/to/script.py`.
"""

from .app import build_dashboard
from .config import DashboardConfig, DashboardExtension, RunContext
from .schema import FieldSpec, form_fields, input_id
from ..launch import LaunchError, ProcessRecord, RunLauncher, write_handshake
from ..registry import ActionSpec, ArgSpec, read_actions, write_actions

__all__ = [
    "build_dashboard", "DashboardConfig", "DashboardExtension", "RunContext",
    "RunLauncher", "ProcessRecord", "LaunchError", "write_handshake",
    "ActionSpec", "ArgSpec", "read_actions", "write_actions",
    "FieldSpec", "form_fields", "input_id",
]
