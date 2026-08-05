"""
Two-tier snapshotting, speaking RunHandle instead of any specific tracker:
  - a small in-memory ring buffer for recent steps (cheap, kept dense)
  - run.track_artifact(..., type='pickle') for older steps, saved less often

Disk snapshot *discovery* isn't something RunHandle exposes generically, so
this manager tracks which steps it saved to disk itself, in memory -- fine,
since rewinding only ever happens live, within the same process that did
the saving.
---
Two-tier snapshotting, keyed by (branch_id, step) rather than step alone so
two branches passing through the same step number never collide -- neither
in the in-memory ring nor in the artifact filenames on disk. Snapshots live
under their own tracklab artifact group ("trainer_state") so they never
share an index.csv with eval-time artifacts.
"""

from collections import OrderedDict

from rewind.state import TrainerState
from rewind.protocols import TrackingHandle


SNAPSHOT_GROUP = "trainer_state"
SNAPSHOT_ARTIFACT_NAME = "trainer_state"


class SnapshotManager:
    def __init__(self, run: TrackingHandle, ring_size: int = 20,
                 ring_every: int = 10, disk_every: int = 200):
        self.run = run
        self.ring_size = ring_size
        self.ring_every = ring_every
        self.disk_every = disk_every
        self._ring: "OrderedDict[tuple[str, int], TrainerState]" = OrderedDict()
        self._disk_index: list[tuple[str, int]] = []              # (branch_id, step)
        self._lineage: dict[str, tuple[str, int]] = {}             # branch_id -> (parent, fork_step)


    def register_branch(self, branch_id: str, parent_branch_id: str, fork_step: int):
        """Called once when a rewind creates a new branch, so nearest_before
        can walk fork lineage instead of guessing across branches."""
        self._lineage[branch_id] = (parent_branch_id, fork_step)

    def maybe_snapshot(self, step, model, optimizer, lr, branch_id="root"):
        take_ring = step % self.ring_every == 0
        take_disk = step % self.disk_every == 0
        if not (take_ring or take_disk):
            return
        state = TrainerState.capture(step, model, optimizer, lr, branch_id)
        self._store(state, ring=take_ring, disk=take_disk)

    def force_snapshot(self, step, model, optimizer, lr, branch_id="root"):
        """Snapshot immediately, regardless of schedule -- for handlers about
        to do something risky (e.g. perturbing weights) that want a
        guaranteed rewind point right before the change."""
        state = TrainerState.capture(step, model, optimizer, lr, branch_id)
        self._store(state, ring=True, disk=False)
        return state

    def _store(self, state: TrainerState, ring: bool, disk: bool):
        key = (state.branch_id, state.step)
        if ring:
            self._ring[key] = state
            if len(self._ring) > self.ring_size:
                self._ring.popitem(last=False)
        if disk:
            name = self._scoped_name(state.branch_id)
            self.run.track_artifact(state, step=state.step, group=SNAPSHOT_GROUP,
                                     name=name, type="torch")
            self._disk_index.append(key)

    def _scoped_name(self, branch_id: str) -> str:
        return f"{SNAPSHOT_ARTIFACT_NAME}__{branch_id}"
    

    def nearest_before(self, step: int, branch_id: str) -> TrainerState:
        """Return the most recent snapshot at or before the given step, on the
        given branch. If none exists on that branch, walk the parent lineage
        until a snapshot is found or the root is reached."""
        b, upper = branch_id, step
        while True:
            ring_candidates = [k for k in self._ring if k[0] == b and k[1] <= upper]
            if ring_candidates:
                return self._ring[max(ring_candidates, key=lambda k: k[1])]

            disk_candidates = [k for k in self._disk_index if k[0] == b and k[1] <= upper]
            if disk_candidates:
                nearest_step = max(k[1] for k in disk_candidates)
                return self.run.load_artifact(group=SNAPSHOT_GROUP, name=self._scoped_name(b),
                                               step=nearest_step, type="torch")

            if b not in self._lineage:
                raise KeyError(f"No snapshot at or before step {step} reachable from branch {branch_id!r}")
            parent, fork_step = self._lineage[b]
            b, upper = parent, min(upper, fork_step)