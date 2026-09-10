from __future__ import annotations

import json

import pytest

from rewind import ActionSpec, ArgSpec, RunMailbox, read_actions
from rewind._fsutil import atomic_write, file_stamp
from rewind._layout import actions_path, commands_dir
from rewind.events import append_event, read_events
from rewind.registry import coerce_args, write_actions


def test_atomic_write_creates_parents_and_replaces(tmp_path):
    p = tmp_path / "a" / "b" / "f.json"
    atomic_write(p, "1")
    atomic_write(p, "2")
    assert p.read_text() == "2"
    assert [x.name for x in p.parent.iterdir()] == ["f.json"]  # no temp leftovers


def test_file_stamp_none_for_missing(tmp_path):
    assert file_stamp(tmp_path / "nope") is None
    p = tmp_path / "x"
    p.write_text("abc")
    size, _ = file_stamp(p)
    assert size == 3


def test_mailbox_roundtrip_in_send_order(tmp_path):
    mb = RunMailbox(tmp_path)
    for i in range(5):
        mb.send_command({"type": "set_lr", "lr": i})
    assert [c["lr"] for c in mb.poll_commands()] == [0, 1, 2, 3, 4]
    assert mb.poll_commands() == []  # consumed


def test_mailbox_skips_bad_json_and_deletes_it(tmp_path):
    mb = RunMailbox(tmp_path)
    (commands_dir(tmp_path) / "1.json").write_text("{not json")
    mb.send_command({"type": "pause"})
    assert [c["type"] for c in mb.poll_commands()] == ["pause"]
    assert list(commands_dir(tmp_path).iterdir()) == []


def test_action_spec_json_roundtrip_and_ids():
    spec = ActionSpec("perturb layer", "Perturb", args={"scale 2": ArgSpec("float", 0.1)})
    back = ActionSpec.from_json(json.loads(json.dumps(spec.to_json())))
    assert back == spec
    assert spec.html_id == "act_perturb_layer"
    assert spec.arg_html_id("scale 2") == "act_perturb_layer_arg_scale_2"
    assert spec.kind == "form" and ActionSpec("x", "X").kind == "button"


def test_write_actions_rejects_id_collision(tmp_path):
    with pytest.raises(ValueError):
        write_actions(tmp_path, [ActionSpec("a b", "A"), ActionSpec("a-b", "B")])


def test_read_actions_missing_or_corrupt(tmp_path):
    assert read_actions(tmp_path) == []
    actions_path(tmp_path).parent.mkdir(parents=True)
    actions_path(tmp_path).write_text("[garbage")
    assert read_actions(tmp_path) == []


@pytest.mark.parametrize("kind,raw,expected", [
    ("int", "3", 3), ("int", 3.0, 3), ("float", "1e-3", 1e-3),
    ("bool", "yes", True), ("bool", "0", False), ("bool", 1, True), ("str", 5, "5"),
])
def test_argspec_coerce(kind, raw, expected):
    assert ArgSpec(kind).coerce(raw) == expected


@pytest.mark.parametrize("kind,raw", [("int", "x"), ("int", 2.5), ("float", "abc"), ("bool", "maybe")])
def test_argspec_coerce_rejects(kind, raw):
    with pytest.raises(ValueError):
        ArgSpec(kind).coerce(raw)


def test_coerce_args_fills_defaults_drops_unknown_reports_missing():
    spec = ActionSpec("f", "F", args={"a": ArgSpec("int"), "b": ArgSpec("float", default=0.5)})
    assert coerce_args(spec, {"type": "f", "a": "2", "junk": 1}) == {"type": "f", "a": 2, "b": 0.5}
    with pytest.raises(ValueError, match="a: missing"):
        coerce_args(spec, {"type": "f"})


def test_events_roundtrip_skips_partial_line(tmp_path):
    append_event(tmp_path, "applied", 3, "root", command={"type": "pause"})
    append_event(tmp_path, "fork", 2, "b1@t2", parent_branch_id="root", fork_step=2)
    from rewind._layout import events_path
    with open(events_path(tmp_path), "a") as f:
        f.write('{"kind": "half"')
    rows = read_events(tmp_path)
    assert [r["kind"] for r in rows] == ["applied", "fork"]
    assert rows[0]["command"] == {"type": "pause"}
