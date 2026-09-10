"""examples/dashboard_attach.py -- attach-only dashboard for examples/toy_train.py.

    ../.venv/Scripts/shiny.exe run --reload examples/dashboard_attach.py

No launcher: start runs from a terminal, pick them in the sidebar, steer them
from the Control tab.
"""

from pathlib import Path

from rewind.dashboard import DashboardConfig, build_dashboard

app = build_dashboard(DashboardConfig(base_dir=Path(__file__).resolve().parent / "data", title="Rewind (toy)"))
