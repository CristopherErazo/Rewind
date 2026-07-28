"""
Two-tier snapshotting, speaking RunHandle instead of any specific tracker:
  - a small in-memory ring buffer for recent steps (cheap, kept dense)
  - run.track_artifact(..., type='pickle') for older steps, saved less often

Disk snapshot *discovery* isn't something RunHandle exposes generically, so
this manager tracks which steps it saved to disk itself, in memory -- fine,
since rewinding only ever happens live, within the same process that did
the saving.
"""

from collections import OrderedDict

from rewind.state import TrainerState
from rewind.run_handle import RunHandle

SNAPSHOT_ARTIFACT_NAME = "trainer_state"


class SnapshotManager:
    def __init__(self, run: RunHandle, ring_size: int = 20,
                 ring_every: int = 10, disk_every: int = 200):
        self.run = run
        self.ring_size = ring_size
        self.ring_every = ring_every
        self.disk_every = disk_every
        self._ring: "OrderedDict[int, TrainerState]" = OrderedDict()
        self._disk_steps: list[int] = []

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
        if ring:
            self._ring[state.step] = state
            if len(self._ring) > self.ring_size:
                self._ring.popitem(last=False)  # evict oldest
        if disk:
            self.run.track_artifact(state, step=state.step, name=SNAPSHOT_ARTIFACT_NAME, type='pickle')
            self._disk_steps.append(state.step)

    def nearest_before(self, step: int) -> TrainerState:
        ring_candidates = [s for s in self._ring if s <= step]
        if ring_candidates:
            return self._ring[max(ring_candidates)]

        disk_candidates = [s for s in self._disk_steps if s <= step]
        if not disk_candidates:
            raise KeyError(f"No snapshot at or before step {step}")
        nearest = max(disk_candidates)
        return self.run.load_artifact(name=SNAPSHOT_ARTIFACT_NAME, step=nearest, type='pickle')