# rewind

A small library for turning a training loop into something you can steer while it runs: pause, resume, change hyperparameters live, and rewind to an earlier step — via deterministic snapshot + replay, not dense checkpointing.

`rewind` doesn't know what your model is, what your data looks like, or which experiment tracker you use. You give it a plain training loop expressed as functions; it gives you back a controllable one.

## What it does

- **Pause / resume** a run without killing the process
- **Change hyperparameters live** (e.g. learning rate) mid-training
- **Rewind to an earlier step**, exactly — model, optimizer, and RNG state restored, not approximated
- **No dense checkpointing** — a small in-memory ring covers recent steps; older steps fall back to disk via your tracker's artifact storage. Rewinding replays deterministically from the nearest snapshot
- **Branches, not overwrites** — diverging after a rewind tags a new branch id instead of erasing the original run
- **Extensible to custom interventions** (e.g. perturbing a layer's weights) via a handler registry, with no changes to `rewind` itself

## What it doesn't do

Track metrics/configs/artifacts durably (that's your experiment tracker's job — `rewind` only needs one exposing a small interface, see below), or require a dashboard. Also: if you don't need live control at all, skip `rewind` and write your plain training loop directly — nothing else depends on it existing.

## Quick start

```python
from rewind.controller import TrainerController
from rewind.tracklab_adapter import ControllableRun   # or your own adapter

def train_step_fn():
    batch = get_batch()
    loss = compute_loss(model, batch)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

def eval_fn():
    return {"loss": compute_eval_loss(model)}

run = ControllableRun(my_tracklab_run)
controller = TrainerController(model, optimizer, train_step_fn, eval_fn,
                                total_steps=10_000, run=run)
controller.eval_schedule = set(range(0, 10_000, 50))
controller.run_loop()
```

From another process (e.g. a dashboard), send commands through the same `run` object:

```python
run.send_command({"type": "pause"})
run.send_command({"type": "set_lr", "lr": 1e-4})
run.send_command({"type": "rewind", "step": 4000})
status = run.get_status()
```

## Architecture

```
rewind/
    state.py             capture/restore model + optimizer + RNG state
    snapshot.py           ring buffer + disk fallback, "nearest snapshot ≤ step X"
    controller.py            the training loop, dispatches commands to handlers
    run_handle.py               RunHandle Protocol -- what controller depends on
    tracklab_adapter.py            example adapter implementing RunHandle
    dashboard/                        generic Shiny UI: status, plot, controls
```

### Integrating a tracker

`TrainerController` is written against `RunHandle`, a `Protocol` — not any specific tracker:

```python
class RunHandle(Protocol):
    def track_metric(self, step, **metrics): ...
    def track_artifact(self, data, step=None, name='', type='pickle'): ...
    def load_artifact(self, name='', step=None, type='pickle'): ...
    def set_status(self, **kwargs): ...
    def get_status(self) -> dict: ...
    def send_command(self, cmd: dict): ...
    def poll_commands(self) -> list[dict]: ...
```

Write one adapter implementing this shape for your tracker of choice — nothing else in `rewind` changes. `set_status`/`get_status`/`send_command`/`poll_commands` are usually the only methods missing; the included `tracklab_adapter.py` implements them as plain files inside the run's directory, no tracker changes required.

### Custom manipulations

```python
def perturb_layer(controller, cmd):
    controller.snapshot_now()  # rewind point right before the change
    param = dict(controller.model.named_parameters())[cmd["layer"]]
    with torch.no_grad():
        param.add_(torch.randn_like(param) * cmd.get("scale", 0.01))

controller.register_handler("perturb_layer", perturb_layer)
```

Sent the same way as any built-in command. `rewind` never needs to know what a handler does — only that a command type maps to a callable.

### Dashboard

```python
from rewind.dashboard import build_dashboard
from myproject.config import TrainerArgs

app = build_dashboard(config_cls=TrainerArgs, service_module="myproject.runtime.service")
```

Project-specific controls can be added via `extra_controls` without modifying `rewind.dashboard`.

## Design principles

- One-way dependencies: your project may depend on `rewind`; `rewind` never depends on it, or on a specific tracker
- Policy (when/what to snapshot) lives here; storage lives in your tracker
- Determinism over density: exact state capture + replay, not saving every step
- Branch, don't overwrite
- Extend by registering, never by editing `rewind`'s source

## Status

Early / personal research infrastructure. API may change.