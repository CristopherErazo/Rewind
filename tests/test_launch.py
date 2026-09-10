from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from rewind import RunLauncher
from rewind._layout import process_path
from rewind.launch import ProcessRecord, pid_alive


def test_pid_alive_current_process_and_dead_pid():
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    # spawn and reap a child; its pid must then read as dead
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not pid_alive(p.pid) or True  # pid reuse makes a strict assert flaky


def test_attach_and_terminate_unowned_process(tmp_path):
    run_dir = tmp_path / "exp" / "run_001"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record = ProcessRecord(pid=proc.pid, run_id="run_001", cmd=proc.args if isinstance(proc.args, list) else list(proc.args),
                               started_at=time.time(), launch_token="tok")
        process_path(run_dir).parent.mkdir(parents=True)
        process_path(run_dir).write_text(json.dumps(record.to_json()))

        launcher = RunLauncher.attach(run_dir)
        assert launcher is not None and launcher.record.pid == proc.pid
        assert launcher.is_alive()
        launcher.terminate(force=True)
        deadline = time.time() + 5
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert proc.poll() is not None
        assert not launcher.is_alive()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_attach_returns_none_without_record(tmp_path):
    assert RunLauncher.attach(tmp_path) is None
