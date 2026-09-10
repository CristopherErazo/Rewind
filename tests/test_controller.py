from __future__ import annotations

import json
import threading
import time

import pytest

from rewind import ActionSpec, ArgSpec, RunMailbox, RunState, read_actions, read_events, read_status
from rewind._layout import status_path


def _kinds(run_dir):
    return [e["kind"] for e in read_events(run_dir)]


def test_plain_loop_runs_to_done_and_finalizes(make_controller, fake_run):
    ctrl = make_controller(total_steps=20, control=False)
    ctrl.eval_schedule = ctrl.every(10)
    ctrl.run_loop()
    st = read_status(fake_run.run_dir)
    assert st["state"] == RunState.DONE.value
    assert st["step"] == 20 and st["total_steps"] == 20 and "pid" in st and "updated_at" in st
    assert fake_run.finalized == 1
    assert [m["step"] for m in fake_run.metrics] == [0, 10, 20]
    assert all(m["tags"] == {} for m in fake_run.metrics)  # no branch tag without rewind
    assert [s.name for s in read_actions(fake_run.run_dir)] == ["pause", "resume", "stop", "set_lr"]


def test_train_metrics_logged_every_n(make_controller, fake_run):
    ctrl = make_controller(total_steps=10, control=False, train_log_every=5)
    ctrl.run_loop()
    assert [(m["step"], "train_loss" in m) for m in fake_run.metrics] == [(0, True), (5, True)]


def test_status_write_throttle(make_controller, fake_run, monkeypatch):
    writes = []
    import rewind.control as control_mod
    orig = control_mod.atomic_write

    def counting(path, content):
        if path == status_path(fake_run.run_dir):
            writes.append(json.loads(content).get("step"))
        orig(path, content)

    monkeypatch.setattr(control_mod, "atomic_write", counting)
    ctrl = make_controller(total_steps=30, control=False, status_every=10)
    ctrl.run_loop()
    # initial (total_steps/pid) + running + steps 10, 20, 30 + done
    assert writes == [None, 0, 10, 20, 30, 30]


def test_crash_writes_crashed_state_and_finalizes(make_controller, fake_run, toy):
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("nan loss")
        return toy.train_step()

    ctrl = make_controller(total_steps=10, control=False)
    ctrl.train_step_fn = boom
    with pytest.raises(RuntimeError, match="nan loss"):
        ctrl.run_loop()
    st = read_status(fake_run.run_dir)
    assert st["state"] == RunState.CRASHED.value and "nan loss" in st["error"] and st["step"] == 2
    assert fake_run.finalized == 1
    assert _kinds(fake_run.run_dir)[-1] == "state"


def test_keyboard_interrupt_marks_interrupted(make_controller, fake_run):
    ctrl = make_controller(total_steps=10, control=False)

    def interrupt():
        raise KeyboardInterrupt

    ctrl.train_step_fn = interrupt
    with pytest.raises(KeyboardInterrupt):
        ctrl.run_loop()
    assert read_status(fake_run.run_dir)["state"] == RunState.INTERRUPTED.value
    assert fake_run.finalized == 1


def test_stop_command_ends_with_stopped(make_controller, fake_run):
    ctrl = make_controller(total_steps=1000)
    mb = RunMailbox(fake_run.run_dir)
    mb.send_command({"type": "stop"})
    ctrl.run_loop()
    assert ctrl.step == 0
    assert read_status(fake_run.run_dir)["state"] == RunState.STOPPED.value
    assert "applied" in _kinds(fake_run.run_dir)


def test_pause_then_resume_from_another_thread(make_controller, fake_run):
    ctrl = make_controller(total_steps=40, status_every=1)
    mb = RunMailbox(fake_run.run_dir)
    mb.send_command({"type": "pause"})

    def resume_later():
        deadline = time.time() + 5
        while time.time() < deadline:
            st = read_status(fake_run.run_dir)
            if st and st["state"] == "paused":
                mb.send_command({"type": "set_lr", "lr": 0.01})
                mb.send_command({"type": "resume"})
                return
            time.sleep(0.02)
        raise AssertionError("never paused")

    t = threading.Thread(target=resume_later)
    t.start()
    ctrl.run_loop()
    t.join()
    assert read_status(fake_run.run_dir)["state"] == "done"
    assert ctrl.optimizer.param_groups[0]["lr"] == 0.01
    states = [e["state"] for e in read_events(fake_run.run_dir) if e["kind"] == "state"]
    assert states == ["running", "paused", "running", "done"]


def test_bad_argument_is_rejected_and_loop_continues(make_controller, fake_run):
    ctrl = make_controller(total_steps=5)
    RunMailbox(fake_run.run_dir).send_command({"type": "set_lr", "lr": "fast"})
    RunMailbox(fake_run.run_dir).send_command({"type": "nonsense"})
    ctrl.run_loop()
    events = read_events(fake_run.run_dir)
    failed = [e for e in events if e["kind"] == "failed"]
    assert len(failed) == 2
    assert "expected float" in failed[0]["error"]
    assert "unknown command" in failed[1]["error"]
    assert read_status(fake_run.run_dir)["state"] == "done"


def test_custom_handler_with_spec_and_coercion(make_controller, fake_run):
    seen = {}

    def perturb(controller, cmd):
        seen.update(cmd)

    ctrl = make_controller(total_steps=3)
    ctrl.register_handler("perturb", perturb, spec=ActionSpec(
        "perturb", "Perturb", args={"scale": ArgSpec("float", default=0.1), "layer": ArgSpec("int")}))
    RunMailbox(fake_run.run_dir).send_command({"type": "perturb", "layer": "2"})
    ctrl.run_loop()
    assert seen == {"type": "perturb", "layer": 2, "scale": 0.1}
    assert [s.name for s in read_actions(fake_run.run_dir)][-1] == "perturb"


def test_register_handler_spec_name_mismatch(make_controller):
    ctrl = make_controller()
    with pytest.raises(ValueError):
        ctrl.register_handler("a", lambda c, cmd: None, spec=ActionSpec("b", "B"))


def test_rewind_spec_only_when_enabled(make_controller, fake_run):
    ctrl = make_controller(total_steps=1, enable_rewind=True)
    ctrl.run_loop()
    assert "rewind" in [s.name for s in read_actions(fake_run.run_dir)]


def test_rewind_is_exact_and_forks_a_branch(make_controller, fake_run, toy):
    """Train 30 steps, rewind to 20 on the way to 40, and check the rewound
    branch replays exactly the losses the root branch saw at steps 20..29."""
    ctrl = make_controller(total_steps=40, enable_rewind=True, status_every=1)
    ctrl.eval_schedule = ctrl.every(10)
    ctrl.snapshots.ring_every = 10
    mb = RunMailbox(fake_run.run_dir)
    sent = {"done": False}
    orig_step = toy.train_step

    def step_and_maybe_rewind():
        out = orig_step()
        if ctrl.step == 29 and not sent["done"]:   # about to become 30
            sent["done"] = True
            mb.send_command({"type": "rewind", "step": 25})
        return out

    ctrl.train_step_fn = step_and_maybe_rewind
    ctrl.run_loop()

    root_losses = toy.losses[:30]
    replay = toy.losses[30:40]
    assert replay == root_losses[20:30]
    assert ctrl.branch_id == "b1@t20"
    forks = [e for e in read_events(fake_run.run_dir) if e["kind"] == "fork"]
    assert len(forks) == 1 and forks[0]["fork_step"] == 20 and forks[0]["parent_branch_id"] == "root"
    st = read_status(fake_run.run_dir)
    assert st["branch_id"] == "b1@t20" and st["state"] == "done"
    # metric rows are tagged by branch, and none carry command noise
    tags = {m["tags"]["branch_id"] for m in fake_run.metrics}
    assert tags == {"root", "b1@t20"}
    assert all(m["note"] is None for m in fake_run.metrics)
