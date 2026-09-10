from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from rewind.snapshot import SNAPSHOT_GROUP, SnapshotManager
from rewind.state import TrainerState
from tests.fakes import FakeRun


class TorchDiskRun(FakeRun):
    """FakeRun whose type='torch' artifacts really go through torch.save and
    a default-argument torch.load, exactly like TrackLab's do. With PyTorch
    >= 2.6 that load is weights_only=True and rejects pickled classes."""

    def track_artifact(self, data, step=None, group=None, name="", type="pickle"):
        path = self.run_dir / f"{group}__{name}__{step}.pt"
        torch.save(data, path)
        self.artifacts[(group, name, step)] = path

    def load_artifact(self, group=None, name="", step=None, type="pickle"):
        return torch.load(self.artifacts[(group, name, step)], map_location="cpu")


def _draws() -> tuple:
    """One draw from each RNG the snapshot restores."""
    return (torch.randn(3).tolist(), random.random(), float(np.random.rand()))


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


def test_state_dict_roundtrip_through_default_torch_load(tmp_path):
    """to_dict() must survive torch.save -> torch.load() with default
    arguments (weights_only=True) and restore weights, optimizer state and
    all three RNG streams exactly."""
    torch.manual_seed(7); random.seed(7); np.random.seed(7)
    m = torch.nn.Linear(2, 1)
    o = torch.optim.Adam(m.parameters(), lr=0.1)
    m(torch.randn(4, 2)).sum().backward(); o.step()  # Adam now has tensor state

    state = TrainerState.capture(5, m, o, 0.1, "b1@t5")
    expected = _draws()

    torch.save(state.to_dict(), tmp_path / "s.pt")
    back = TrainerState.from_dict(torch.load(tmp_path / "s.pt", map_location="cpu"))

    assert (back.step, back.lr, back.branch_id) == (5, 0.1, "b1@t5")
    with torch.no_grad():
        m.weight.add_(1.0)
    torch.randn(10); random.random(); np.random.rand()  # advance every RNG
    back.restore(m, o)
    assert torch.equal(m.weight, state.model_state["weight"])
    assert torch.equal(o.state[m.weight]["exp_avg"], state.optim_state["state"][0]["exp_avg"])
    assert _draws() == expected


def test_disk_snapshot_evicted_from_ring_loads_under_weights_only(tmp_path):
    """The bug behind run_002's failed rewind: a snapshot only on disk came
    back through torch.load, which refused the pickled dataclass."""
    mgr = SnapshotManager(TorchDiskRun(tmp_path), ring_size=2, ring_every=2, disk_every=4)
    m = torch.nn.Linear(2, 1)
    o = torch.optim.Adam(m.parameters(), lr=0.1)
    torch.manual_seed(3)
    reference = None
    for step in range(9):
        if step == 0:
            reference = TrainerState.capture(0, m, o, 0.1, "root")
        mgr.maybe_snapshot(step, m, o, 0.1, "root")
        m(torch.randn(4, 2)).sum().backward(); o.step(); o.zero_grad()
    assert ("root", 0) not in mgr._ring  # evicted: only the disk copy is left

    snap = mgr.nearest_before(3, "root")
    assert isinstance(snap, TrainerState)
    assert (snap.step, snap.branch_id) == (0, "root")
    assert torch.equal(snap.model_state["weight"], reference.model_state["weight"])
    reference.restore(m, o); a = _draws()
    snap.restore(m, o); b = _draws()
    assert a == b


def test_forced_snapshot_is_not_overwritten_by_scheduled_one(tmp_path):
    """Rewind + perturb in one poll: the fork step is always a disk step, so
    the scheduled snapshot used to overwrite the handler's pre-perturb
    snapshot under the same (branch, step) key, and only the perturbed state
    reached disk. First write wins: the pre-intervention state is kept in
    the ring and is what goes to disk."""
    mgr = SnapshotManager(TorchDiskRun(tmp_path), ring_size=4, ring_every=2, disk_every=2)
    m = torch.nn.Linear(2, 1)
    o = torch.optim.SGD(m.parameters(), lr=0.1)
    before = m.weight.detach().clone()

    first = mgr.force_snapshot(4, m, o, 0.1, "b1@t4")      # handler: about to perturb
    with torch.no_grad():
        m.weight.add_(1.0)                                  # the perturbation
    assert mgr.force_snapshot(4, m, o, 0.1, "b1@t4") is first  # second call keeps the earliest
    mgr.maybe_snapshot(4, m, o, 0.1, "b1@t4")               # scheduled ring + disk snapshot, same key

    assert torch.equal(mgr._ring[("b1@t4", 4)].model_state["weight"], before)
    on_disk = mgr.nearest_before(4, "b1@t4")
    assert on_disk.step == 4 and torch.equal(on_disk.model_state["weight"], before)
    # ...and the ring only ever holds one entry for that key
    assert [k for k in mgr._ring] == [("b1@t4", 4)]
    # evict the ring copy; the disk copy is still the pre-perturb state
    for step in (6, 8, 10, 12, 14):
        mgr.maybe_snapshot(step, m, o, 0.1, "b1@t4")
    assert ("b1@t4", 4) not in mgr._ring
    assert torch.equal(mgr.nearest_before(4, "b1@t4").model_state["weight"], before)
