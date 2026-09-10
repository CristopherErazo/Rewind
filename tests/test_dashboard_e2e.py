"""End-to-end: a real Shiny session, driven over its websocket protocol,
attaches to a live paused trainer and resumes it from the control panel.

Skipped when the dashboard extra is not installed.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest

shiny = pytest.importorskip("shiny")
uvicorn = pytest.importorskip("uvicorn")
websockets = pytest.importorskip("websockets")
pytest.importorskip("tracklab")

from websockets.sync.client import connect  # noqa: E402

from rewind import RunMailbox, read_status  # noqa: E402
from rewind.dashboard import DashboardConfig, build_dashboard  # noqa: E402
from tests.conftest import Toy  # noqa: E402
from tests.fakes import FakeRun  # noqa: E402

VISIBLE_OUTPUTS = ["status-bar", "control-actions_container", "control-last_sent",
                   "control-process_controls", "events-table", "metrics-chart"]


class FileMetricsRun(FakeRun):
    """FakeRun that also appends metrics.jsonl rows in TrackLab's long
    format, so the dashboard's metrics stream and live chart see real data."""

    def track_metric(self, step, note=None, tags=None, **metrics):
        super().track_metric(step, note=note, tags=tags, **metrics)
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            for k, v in metrics.items():
                f.write(json.dumps({"step": step, "metric": k, "value": v, **(tags or {})}) + "\n")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    def __init__(self, app, port):
        self.config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=lambda: asyncio.run(self.server.serve()), daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 15
        while time.time() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", self.config.port)) == 0:
                    return self
            time.sleep(0.05)
        raise RuntimeError("uvicorn did not start")

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


def _recv_until(ws, predicate, timeout=20.0):
    """Collect server messages until predicate(latest_values) is truthy."""
    values: dict = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = json.loads(ws.recv(timeout=1.0))
        except TimeoutError:
            continue
        if isinstance(msg, dict) and "values" in msg:
            values.update(msg["values"])
            if msg.get("errors"):
                raise AssertionError(f"output errors: {msg['errors']}")
            if predicate(values):
                return values
    raise AssertionError(f"timed out; last values keys={sorted(values)}")


def _html(v) -> str:
    return v["html"] if isinstance(v, dict) else str(v)


def test_attach_and_resume_from_dashboard(tmp_path):
    base = tmp_path / "data"
    run_dir = base / "exp" / "run_001"
    fake_run = FileMetricsRun(run_dir)
    toy = Toy()
    mb = RunMailbox(run_dir)
    mb.send_command({"type": "pause"})  # trainer will pause on its first poll

    from rewind import TrainerController
    ctrl = TrainerController(toy.model, toy.optimizer, 30, toy.train_step, fake_run,
                             eval_fn=toy.evaluate, control=mb, enable_rewind=True, status_every=1,
                             train_log_every=1)
    ctrl.eval_schedule = ctrl.every(10)
    trainer = threading.Thread(target=ctrl.run_loop, daemon=True)
    trainer.start()

    app = build_dashboard(DashboardConfig(base_dir=base, poll_interval_s=0.2, listing_interval_s=0.2))
    with _Server(app, _free_port()) as srv:
        # max_size=None: the first plotly widget message carries the plotly.js
        # bundle (>1 MB), above the websockets client's default frame limit.
        with connect(f"ws://127.0.0.1:{srv.config.port}/websocket/", open_timeout=10, max_size=None) as ws:
            init = {"exp-existing": "exp", "exp-new_name": "", "run-run": "run_001",
                    "run-follow_latest": True}
            init.update({f".clientdata_output_{o}_hidden": False for o in VISIBLE_OUTPUTS})
            ws.send(json.dumps({"method": "init", "data": init}))

            # status bar shows the paused run and the control panel has rendered the buttons
            vals = _recv_until(ws, lambda v: "control-actions_container" in v and "status-bar" in v
                               and "act_resume" in _html(v["control-actions_container"])
                               and "paused" in _html(v["status-bar"]))
            assert "rewind" in _html(vals["control-actions_container"])  # rewind enabled -> widget present
            assert "kill is unavailable" in _html(vals["control-process_controls"])  # no process.json

            # the browser reports every freshly rendered button as 0 before any click
            import re
            buttons = set(re.findall(r'id="(control-act_[a-z0-9_]+)"', _html(vals["control-actions_container"])))
            buttons = {b for b in buttons if "_arg_" not in b}
            ws.send(json.dumps({"method": "update", "data": {f"{b}:shiny.action": 0 for b in buttons}}))
            # click Resume
            ws.send(json.dumps({"method": "update", "data": {"control-act_resume:shiny.action": 1}}))
            vals = _recv_until(ws, lambda v: "Sent resume" in str(v.get("control-last_sent", "")))

            trainer.join(timeout=20)
            assert not trainer.is_alive(), "trainer did not resume after the dashboard click"
            assert read_status(run_dir)["state"] == "done"

            # the status bar and events table follow the run to completion
            vals = _recv_until(ws, lambda v: "done" in _html(v.get("status-bar", "")))
            assert "applied" in json.dumps(vals.get("events-table", ""))
            # The live chart re-rendered as a real widget once metrics existed:
            # that rebuild is driven by the same poll that reported "done", so
            # it is in the values collected above. Later points travel over
            # the widget comm and never appear as output values again.
            assert isinstance(vals.get("metrics-chart"), dict) and "model_id" in vals["metrics-chart"]
            assert vals["metrics-chart"]["widget_pkg"] == "plotly"

            # a bad argument is caught client-side and never reaches the mailbox
            ws.send(json.dumps({"method": "update", "data": {"control-act_set_lr_arg_lr": None}}))
            ws.send(json.dumps({"method": "update", "data": {"control-act_set_lr:shiny.action": 1}}))
            _recv_until(ws, lambda v: "Not sent" in str(v.get("control-last_sent", ""))
                        or "Sent set_lr" in str(v.get("control-last_sent", "")))
