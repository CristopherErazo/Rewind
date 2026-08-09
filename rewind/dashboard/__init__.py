"""rewind.dashboard -- a project-agnostic Shiny UI for launching, watching,
and steering a run through Rewind's control plane.

The only function most projects ever need:

    from rewind.dashboard import build_dashboard, DashboardConfig

Everything project-specific comes in through DashboardConfig; everything
else (experiment picker, launch form, runs table, live metrics, control
panel) is generic and works unmodified across projects. See
build_dashboard()'s docstring in app.py for the full contract, or
icl/scripts/dashboard_basic.py for the minimal working example.

Public API
----------
build_dashboard(cfg) -> shiny.App   The one entrypoint. Assign the result to
                                     a module-level `app` and run with
                                     `shiny run --reload <script>`.
DashboardConfig                     The contract a project fills in.
DashboardExtension                  Protocol for a custom tab (see config.py).
RunContext                          What a DashboardExtension's server() receives.
RunLauncher, ProcessRecord, LaunchError
                                     Subprocess lifecycle -- usable directly,
                                     without Shiny, e.g. from a script or the
                                     `rewind` CLI.
ActionSpec, ArgSpec, read_actions, write_actions
                                     The self-describing control registry.
FieldSpec, form_fields, input_id
                                     Config-dataclass introspection behind
                                     the auto-generated launch form.
"""

from .app import build_dashboard
from .config import DashboardConfig, DashboardExtension, RunContext
from ..launch import LaunchError, ProcessRecord, RunLauncher, write_handshake
from ..registry import ActionSpec, ArgSpec, read_actions, write_actions
from .schema import FieldSpec, form_fields, input_id

__all__ = [
    "build_dashboard",
    "DashboardConfig",
    "DashboardExtension",
    "RunContext",
    "RunLauncher",
    "ProcessRecord",
    "LaunchError",
    "write_handshake",
    "ActionSpec",
    "ArgSpec",
    "read_actions",
    "write_actions",
    "FieldSpec",
    "form_fields",
    "input_id",
]