from __future__ import annotations

import pytest
import torch

from rewind.snapshot import SNAPSHOT_GROUP, SnapshotManager
from tests.fakes import FakeRun


def _mgr(tmp_path, **kw):
    return SnapshotManager(FakeRun(tmp_path), **kw)


def _model_opt():
    m = torch.nn.Linear(2, 1)
    return m, torch.optim.SGD(m.parameters(), lr=0.1)


def test_ring_and_disk_cadence(tmp_path):
    mgr = _mgr(tmp_path, ring_size=3, ring_every=2, disk_every=4)
    m, o = _model_opt()
    for step in range(9):
        mgr.maybe_snapshot(step, m, o, 0.1, "root")
    assert sorted(k[1] for k in mgr._ring) == [4, 6, 8]  # ring holds the last 3
    assert [k[1] for k in mgr._disk_index] == [0, 4, 8]
    assert set(mgr.run.artifacts) == {(SNAPSHOT_GROUP, "trainer_state__root", s) for s in (0, 4, 8)}


def test_nearest_before_prefers_ring_then_disk(tmp_path):
    mgr = _mgr(tmp_path, ring_size=2, ring_every=2, disk_every=4)
    m, o = _model_opt()
    for step in range(9):
        mgr.maybe_snapshot(step, m, o, 0.1, "root")
    assert mgr.nearest_before(7, "root").step == 6   # in ring
    assert mgr.nearest_before(3, "root").step == 0   # evicted from ring, found on disk
    with pytest.raises(KeyError):
        mgr.nearest_before(-1, "root")


def test_nearest_before_walks_lineage_and_caps_at_fork(tmp_path):
    mgr = _mgr(tmp_path, ring_size=100, ring_every=1, disk_every=10**9)
    m, o = _model_opt()
    for step in range(0, 10):
        mgr.maybe_snapshot(step, m, o, 0.1, "root")
    mgr.register_branch("b1@t5", "root", 5)
    for step in range(5, 8):
        mgr.maybe_snapshot(step, m, o, 0.1, "b1@t5")
    mgr.register_branch("b2@t6", "b1@t5", 6)
    # b2 has no snapshots of its own: fall to b1 (<=6), then root (<=5)
    assert mgr.nearest_before(9, "b2@t6").branch_id == "b1@t5"
    assert mgr.nearest_before(9, "b2@t6").step == 6
    assert mgr.nearest_before(3, "b2@t6").branch_id == "root"
    assert mgr.nearest_before(3, "b2@t6").step == 3
    # a root step above the fork is never reachable from the child
    snap = mgr.nearest_before(8, "b1@t5")
    assert (snap.branch_id, snap.step) == ("b1@t5", 7)


def test_restore_reproduces_weights_and_rng(tmp_path):
    mgr = _mgr(tmp_path, ring_size=5, ring_every=1)
    m, o = _model_opt()
    torch.manual_seed(1)
    snap = mgr.force_snapshot(0, m, o, 0.1, "root")
    a = torch.randn(3)
    with torch.no_grad():
        m.weight.add_(1.0)
    snap.restore(m, o)
    b = torch.randn(3)
    assert torch.equal(a, b)
    assert torch.equal(m.weight, snap.model_state["weight"])
