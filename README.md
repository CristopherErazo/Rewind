# rewind

A small library for turning a PyTorch training loop into something you can steer while it runs: pause, resume, change hyperparameters live, run your own interventions, and rewind to an earlier step exactly, on a new branch instead of over the old one.

`rewind` does not know what your model is, what your data looks like, or how your project is configured. You give it a plain training step as a function; it gives you back a loop that reads commands from a mailbox while it trains, plus a generic dashboard that talks to that mailbox.

## What it does

- **Pause / resume / stop** a run without killing the process
- **Change the learning rate live**, or register any command of your own (perturb weights, swap a loss term, dump a probe) with two lines
- **Rewind to an earlier step, exactly**: model, optimizer and RNG state restored from a snapshot, not approximated
- **Branches, not overwrites**: continuing after a rewind tags a new branch id; metrics from both branches stay in the log
- **Self-describing controls**: every command a run accepts is declared in a file, and the dashboard renders a widget for each one, including yours
- **Attach from anywhere**: the dashboard reads and writes plain files inside the run directory, so it can steer a run started from a terminal, a notebook, or a job script
- **Honest status**: a crashed or interrupted run is marked as such; a run never looks alive when it is not

## What it does not do

Track metrics or configs durably: that is your experiment tracker's job. The core needs a tracker exposing a small interface (see below); the sibling project [TrackLab](https://github.com/CristopherErazo/TrackLab) satisfies it and is what the dashboard reads. If you do not need live control, skip `rewind` and write your loop directly.

## Install

`rewind` is a [uv](https://docs.astral.sh/uv/) project. The core depends only on `numpy` and `torch`; the dashboard is an extra that pulls in TrackLab (from GitHub), Shiny, Plotly and pandas.

```bash
git clone https://github.com/CristopherErazo/Rewind
cd Rewind
uv sync --extra dashboard     # creates .venv with everything, including pytest
uv run pytest -q              # 51 tests, about 5 seconds
```

Without `--extra dashboard` you get the core only; the dashboard end-to-end test skips itself.

## Try the example

Two files in `examples/` show the whole workflow. Run them from the repo root in two terminals.

Terminal 1, start a controllable training run:

```bash
uv run examples/toy_train.py --steps 5000
```

A two-layer MLP regresses a noisy sum of its inputs, slowly on purpose (10 ms per step, `--sleep 0` to go fast). It writes to `examples/data/toy/run_001/` and prints the run directory. Batches come from the torch RNG, so rewinds replay exactly.

Terminal 2, open the dashboard and attach to it:

```bash
uv run shiny run --reload examples/dashboard_attach.py
```

Open the printed URL (usually `http://127.0.0.1:8000`). In the sidebar pick experiment `toy` and the run (the "Follow newest run" switch is on by default). Then:

- **Live** tab: `train_loss` and `eval_loss` update as the run goes.
- **Control** tab: click **Pause**, watch the status bar turn to `paused`, click **Resume**. Set a new learning rate and **Send**. Send **Perturb weights** with a noise scale and watch the loss jump. Type an earlier step into **Rewind to step** and **Send**: the status bar shows a new branch such as `b1@t2400`, the Live plot grows a second colored line, and the loss curve replays. Snapshots are dense (every 10 steps) only over the last 200 steps and sparse (every 200 steps) before that, so a rewind to step 150 sent at step 400 lands on `b1@t0`, the nearest snapshot at or before the request.
- **Events** tab: every command, its arguments, each branch fork and each state change, newest first.
- **Runs** tab: one row per run with the config fields that differ between them.

Stop the run with **Stop**, or Ctrl-C the terminal; either way the status bar shows the final state. The dashboard has no launch form in this example because no entrypoint is configured; see "Launching from the dashboard" below.

## Quick start in your own script

```python
import torch
from tracklab import ExperimentTracker
from rewind import TrainerController, RunMailbox, ActionSpec, ArgSpec

model, optimizer = build_model_and_optimizer()

def train_step() -> dict:            # one optimizer step; returned dict is logged
    x, y = next_batch()
    loss = loss_fn(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"train_loss": loss.item()}

def evaluate() -> dict:
    return {"eval_loss": compute_eval_loss(model)}

run = ExperimentTracker("my-exp", base_dir="./data").start_run(config, artifacts=True)

controller = TrainerController(
    model, optimizer, 10_000, train_step, run,
    eval_fn=evaluate,
    control=RunMailbox(run.run_dir),   # omit for a plain, uncontrollable loop
    enable_rewind=True,                # snapshots + the `rewind` command
    train_log_every=10,
)
controller.eval_schedule = controller.every(100)
controller.run_loop()
```

Everything after `run` is keyword-only. Useful knobs: `status_every` (how often `status.json` is rewritten, default 10 steps), `poll_every` (how often the mailbox is checked, default every step), `eval_artifacts_fn` with `eval_artifacts_schedule` for saving tensors through the tracker.

### Sending commands without the dashboard

Commands are dicts written to the run's mailbox from any process:

```python
from rewind import RunMailbox, read_status
mb = RunMailbox("./data/my-exp/run_003")
mb.send_command({"type": "pause"})
mb.send_command({"type": "set_lr", "lr": 1e-4})
mb.send_command({"type": "rewind", "step": 4000})
print(read_status("./data/my-exp/run_003")["state"])
```

Built-ins: `pause`, `resume`, `stop`, `set_lr(lr)`, and `rewind(step)` when rewind is enabled.

### Registering your own intervention

```python
def perturb(controller, cmd):
    controller.snapshot_now()            # rewind point right before the change
    with torch.no_grad():
        for p in controller.model.parameters():
            p.add_(cmd["scale"] * torch.randn_like(p))

controller.register_handler("perturb", perturb, spec=ActionSpec(
    "perturb", "Perturb weights",
    args={"scale": ArgSpec("float", default=0.05, description="noise std")},
    description="Add Gaussian noise to every parameter.",
))
```

A handler is `fn(controller, cmd)`. The `ActionSpec` is optional, but with it the dashboard renders a widget for the command and both sides coerce `scale` to a float. A handler that raises is logged as a `failed` event and the loop continues.

## How it works

### Two protocols, one controller

`TrainerController` depends on two `Protocol`s in `rewind/protocols.py`:

- `TrackingHandle`: `track_metric`, `track_artifact`, `load_artifact`, `get_logger`, `finalize`, plus `run_id` and a filesystem `run_dir`. A TrackLab `Run` satisfies it unchanged; `tests/fakes.py` has a 30-line in-memory version.
- `ControlHandle`: `send_command`, `poll_commands`. `RunMailbox` implements it with files.

### The run directory is the IPC channel

Everything `rewind` writes lives under `control/` inside the tracker's run directory:

| File | Written by | Purpose |
| --- | --- | --- |
| `control/commands/*.json` | dashboard or script | Mailbox; the trainer reads and deletes in send order |
| `control/status.json` | trainer | `state` (running, paused, stopped, done, crashed, interrupted), `step`, `total_steps`, `branch_id`, `lr`, `pid`, `updated_at` |
| `control/actions.json` | trainer | The commands this run accepts, with argument types; drives the control panel |
| `control/events.jsonl` | trainer | Audit journal: applied and failed commands, forks, state changes |
| `control/process.json` | launcher | PID record so a dashboard can attach to and kill a run it did not start |

All whole-file writes are atomic, and the dashboard only polls file sizes and mtimes until something changes.

### Snapshots and branches

Snapshots are CPU copies of model and optimizer state plus torch, python, numpy and CUDA RNG state, kept in a small in-memory ring (every 10 steps, last 20) and on disk through the tracker's artifacts every 200 steps. On disk a snapshot is a plain dict of tensors and scalars (`TrainerState.to_dict()`), so it loads under `torch.load`'s default `weights_only=True`. `rewind` restores the nearest snapshot at or before the requested step, walking back through parent branches if needed, and continues under a new branch id `b<n>@t<step>`. A snapshot at step `S` is always the state *before* anything that happened at `S`: when a handler calls `snapshot_now()` and a scheduled snapshot falls on the same step, the earlier one is kept, so rewinding to the step of an intervention undoes it. Metric rows carry a `branch_id` tag, so the dashboard plots branches as separate lines.

**Determinism contract.** A rewind is exact when every source of randomness in your training step comes from the torch, numpy or python RNGs, which are restored. Anything else with state is not restored yet: an LR scheduler, a `DataLoader` iterator, a `GradScaler`. Those drift after a rewind. State hooks for them are the next milestone.

### Launching from the dashboard

The dashboard is attach-first, but it can also start runs. Give `DashboardConfig` a config dataclass (fields marked `field(metadata={"ui": True})` become form widgets) and a `python -m` entrypoint:

```python
from rewind.dashboard import build_dashboard, DashboardConfig
app = build_dashboard(DashboardConfig(
    base_dir="./data", config_cls=TrainerArgs, entrypoint="my_project.train"))
```

The entrypoint receives `key=value` overrides using the dataclass's dotted field paths, plus `extra_args.launch_token`, `extra_args.experiment_name` and `extra_args.base_dir`. It must claim a run through the tracker and then call `rewind.write_handshake(exp_dir, token, run_id)`. A helper that parses these arguments for you is planned; until then see `RunLauncher` in `rewind/launch.py`.

Custom tabs implement `rewind.dashboard.DashboardExtension` and receive a `RunContext` of reactive callables (`run_dir`, `status`, `metrics`, `events`, `mailbox`, `reader`).

## Design principles

- One-way dependencies: your project may depend on `rewind`; `rewind` never depends on it, or on a specific tracker
- Policy (when and what to snapshot) lives here; storage lives in your tracker
- Determinism over density: exact state capture and replay, not a checkpoint every step
- Branch, don't overwrite
- Extend by registering, never by editing `rewind`'s source

## Status and limitations

Personal research infrastructure that works end to end on Windows and POSIX; the API may still change.

- Rewind works within the process that took the snapshots; there is no resume-after-restart yet
- Snapshot cadence is not configurable from `TrainerController` yet, and each ring snapshot is a full CPU copy of model plus optimizer, so watch memory with large models
- The dashboard is TrackLab-specific and single-user
- Dashboard extensions share one Shiny namespace, so their ids must be unique across the app
