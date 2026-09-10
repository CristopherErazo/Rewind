"""rewind/snapshot.py

Two-tier snapshotting, speaking TrackingHandle instead of any specific
tracker:
  - a small in-memory ring buffer of TrainerState objects for recent steps
    (cheap, kept dense)
  - run.track_artifact(..., type="torch") for older steps, saved less often

Snapshots are keyed by (branch_id, step) rather than step alone so two
branches passing through the same step number never collide, neither in the
ring nor in the artifact filenames on disk. Disk snapshots live under their
own artifact group ("trainer_state") so they never share an index with
eval-time artifacts.

What goes to disk is `TrainerState.to_dict()`, a plain dict of tensors and
scalars, never the dataclass itself: `torch.load` defaults to
`weights_only=True` since PyTorch 2.6 and refuses arbitrary classes. Disk
snapshot *discovery* is not something TrackingHandle exposes, so the manager
remembers which (branch, step) pairs it saved, in memory: rewinding only
ever happens live, in the process that did the saving.
"""

from collections import OrderedDict

from rewind.state import TrainerState
from rewind.protocols import TrackingHandle


SNAPSHOT_GROUP = "trainer_state"
SNAPSHOT_ARTIFACT_NAME = "trainer_state"


class SnapshotManager:
    """First write wins: at most one state is kept per (branch_id, step),
    and it is the earliest one captured at that step. The loop applies
    commands before the scheduled snapshot, so a handler's force_snapshot
    (taken right before it mutates weights) is that earliest state, and a
    scheduled snapshot at the same step reuses it rather than overwriting it
    with the post-intervention state. "Rewind to S" therefore always means
    the state before anything that happened at step S."""

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
        key = (branch_id, step)
        if key in self._ring:
            # A handler already snapshotted this step before mutating the
            # weights (force_snapshot). That is the state *before* the
            # intervention: keep it, and let it reach disk on a disk step
            # instead of capturing the post-intervention state under the
            # same key. See "first write wins" in the class docstring.
            state = self._ring[key]
        else:
            state = TrainerState.capture(step, model, optimizer, lr, branch_id)
        self._store(state, ring=take_ring, disk=take_disk)

    def force_snapshot(self, step, model, optimizer, lr, branch_id="root"):
        """Snapshot immediately, regardless of schedule -- for handlers about
        to do something risky (e.g. perturbing weights) that want a
        guaranteed rewind point right before the change. A second call at
        the same (branch, step) returns the existing snapshot rather than
        re-capturing, so the earliest state of the step is the one kept."""
        key = (branch_id, step)
        if key in self._ring:
            return self._ring[key]
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
            self.run.track_artifact(state.to_dict(), step=state.step, group=SNAPSHOT_GROUP,
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
                payload = self.run.load_artifact(group=SNAPSHOT_GROUP, name=self._scoped_name(b),
                                                 step=nearest_step, type="torch")
                return TrainerState.from_dict(payload)

            if b not in self._lineage:
                raise KeyError(f"No snapshot at or before step {step} reachable from branch {branch_id!r}")
            parent, fork_step = self._lineage[b]
            b, upper = parent, min(upper, fork_step)