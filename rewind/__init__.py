"""rewind: steer a PyTorch training loop while it runs.

    from rewind import TrainerController, RunMailbox, ActionSpec, ArgSpec

    controller = TrainerController(model, optimizer, total_steps, train_step, run,
                                   eval_fn=evaluate,
                                   control=RunMailbox(run.run_dir),
                                   enable_rewind=True)
    controller.eval_schedule = controller.every(50)
    controller.run_loop()

The dashboard lives in `rewind.dashboard` and needs the `dashboard` extra.
"""

from .controller import TrainerController
from .control import RunMailbox, RunState, read_status
from .events import read_events
from .launch import LaunchError, ProcessRecord, RunLauncher, write_handshake
from .protocols import ControlHandle, TrackingHandle
from .registry import ActionSpec, ArgSpec, read_actions
from .snapshot import SnapshotManager
from .state import TrainerState

__all__ = [
    "TrainerController",
    "RunMailbox", "RunState", "read_status", "read_events",
    "ActionSpec", "ArgSpec", "read_actions",
    "SnapshotManager", "TrainerState",
    "RunLauncher", "ProcessRecord", "LaunchError", "write_handshake",
    "TrackingHandle", "ControlHandle",
]
