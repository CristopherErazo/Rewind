# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`rewind` turns a plain PyTorch training loop into one that can be steered while it runs: pause/resume, live hyperparameter changes, and exact rewind to an earlier step via snapshot + deterministic replay, with branches instead of overwrites. The core is tracker-agnostic and project-agnostic: the controller is written against two small `Protocol`s and everything project-specific is injected (handlers, config dataclass, entrypoint). The Shiny dashboard is TrackLab-specific. The sibling repos `../TrackLab` (the file-based tracker whose `Run` satisfies `TrackingHandle`) and `../ICL` (the research project that consumes both) live next to this one. The roadmap lives outside the repo at `../Rewind_PLAN.md`; Milestone 1 (working personal tool) is done, Milestones 2 and 3 are pending.

## Environment and commands

There is no per-repo virtualenv. All three sibling packages are installed editable into the shared venv one level up; the system Python has none of the dependencies.

```bash
# from this repo, in Git Bash
PY=../.venv/Scripts/python.exe
$PY -m pytest -q -p no:warnings              # 40 tests, ~5s; includes a real Shiny session test
$PY examples/toy_train.py --steps 5000        # controllable toy run, writes to examples/data/
../.venv/Scripts/shiny.exe run --reload examples/dashboard_attach.py   # attach-only dashboard
```

`pyproject.toml` is a uv project (`uv_build` backend, flat layout like TrackLab): `numpy`/`torch` are core deps, the `dashboard` extra adds tracklab, shiny, shinywidgets, plotly and pandas, and `[tool.uv.sources]` pulls tracklab from `https://github.com/CristopherErazo/TrackLab` (a commented `path = "../TrackLab"` line switches to the sibling checkout). `uv.lock` is committed. For a standalone environment use `uv sync --extra dashboard`; the shared `../.venv` above is not managed by uv, so do not run `uv sync` against it (it removes packages absent from the lock). `requirements.txt` is gone. Tests use `tests/fakes.py::FakeRun`, an in-memory `TrackingHandle`; if the controller needs something FakeRun lacks, the protocol must grow, not the mocking. `tests/test_dashboard_e2e.py` runs uvicorn in a thread and drives the Shiny websocket protocol directly (init message with `.clientdata_output_<id>_hidden: False` to make outputs render, `update` messages with `<id>:shiny.action` for button clicks).

`.gitignore` ignores `_scratch/`; underscore-prefixed modules are tracked normally.

## Architecture

### Two protocols, one controller

`rewind/protocols.py`: `TrackingHandle` (metrics with `tags`, artifacts, logger, `run_id`, `run_dir`) and `ControlHandle` (`send_command`/`poll_commands`). `TrainerController(model, optimizer, total_steps, train_step_fn, run, *, eval_fn=None, eval_artifacts_fn=None, control=None, enable_rewind=False, status_every=10, poll_every=1, train_log_every=0, ...)` in `controller.py` is keyword-only after `run`. No `control` gives a plain loop; `enable_rewind` builds the `SnapshotManager` and registers `rewind`. Branch tag `branch_id` and artifact suffix `__<branch>` are emitted only when rewind is enabled. `train_step_fn` may return a dict of metrics, logged every `train_log_every` steps. `controller.every(n)` builds schedule sets that include `total_steps`; a completed run evaluates its final weights.

### The filesystem is the IPC channel

All Rewind files live under `control/` inside the run dir; paths come exclusively from `rewind/_layout.py`. Every whole-file write goes through `_fsutil.atomic_write`; appends through `_fsutil.append_line`.

| Path | Writer -> Reader | Purpose |
| --- | --- | --- |
| `control/commands/<time_ns>-<pid>-<seq>.json` | dashboard -> trainer | Mailbox. The pid/seq suffix exists because `time_ns()` repeats on Windows and two commands used to overwrite each other. |
| `control/status.json` | trainer -> dashboard | `RunStatus`: `{state, step, total_steps, branch_id, lr, pid, updated_at, error?}`. `state` is a `RunState` value: running, paused, stopped, done, crashed, interrupted. Written on every transition and every `status_every` steps. |
| `control/events.jsonl` | trainer -> dashboard | Audit journal (`rewind/events.py`): kinds `state`, `applied`, `failed`, `fork`. Command history never goes into metrics.jsonl. |
| `control/actions.json` | trainer -> dashboard | `ActionSpec` list written at `run_loop()` start; drives the control panel and trainer-side argument coercion. |
| `control/process.json` | launcher -> dashboard | `ProcessRecord` so `RunLauncher.attach()` can probe/kill a run it did not start. |
| `<exp_dir>/.pending/<token>.json` | trainer -> launcher | Launch handshake. |

### Run loop and command dispatch

`run_loop()` writes actions and `state=running`, then loops: poll commands (every `poll_every` steps, continuously while paused) -> if paused sleep 0.1s -> eval -> eval artifacts -> `maybe_snapshot` -> `train_step_fn()` -> `step += 1` -> throttled status write. Exceptions write `state=crashed` with the error string and re-raise; `KeyboardInterrupt` writes `interrupted`; `run.finalize()` always runs. Commands are `{"type": name, **args}`; args are coerced against the registered `ActionSpec` (`registry.coerce_args`) before the handler runs, and a failing or unknown command is recorded as a `failed` event while the loop continues. `register_handler(name, fn, spec=ActionSpec(...))` adds a project command; `spec.name` must equal `name`. Handlers that mutate weights should call `controller.snapshot_now()` first.

### Snapshots and branching

`SnapshotManager` (`snapshot.py`) keeps `TrainerState` objects (CPU model/optimizer state plus torch/python/numpy/cuda RNG) keyed by `(branch_id, step)`: an in-memory ring (`ring_every=10`, `ring_size=20`) and disk via `track_artifact(group="trainer_state")` every `disk_every=200`. Cadence is not yet configurable from `TrainerController` (Milestone 2). `rewind` restores `nearest_before(step, branch)`, walking parent lineage capped at fork steps, creates branch `b<n>@t<step>`, and writes a `fork` event. Only model, optimizer and RNG are restored: LR schedulers, data iterators and scalers drift after a rewind unless batches derive from the torch RNG (Milestone 2 adds state hooks).

### Launching and process control

`RunLauncher.launch(overrides)` spawns `python -m <entrypoint> key=value ...` plus `extra_args.launch_token/experiment_name/base_dir`, waits for the child's `write_handshake`, and writes `process.json`. Liveness uses `pid_alive` (psutil, else ctypes `OpenProcess` on Windows, else `os.kill(pid, 0)` on POSIX). Never reintroduce `os.kill(pid, 0)` on Windows: it terminates the process. Kill goes through the owned `Popen`, psutil, `taskkill /T /F`, or signals, in that order of availability.

### Dashboard (`rewind/dashboard/`)

Attach-first: `build_dashboard(DashboardConfig(base_dir=...))` is a complete dashboard; `config_cls` + `entrypoint` (both or neither) add the launch form. `app.py` owns all reactivity: directory listings are `reactive.poll`s keyed on a listing stamp; the active run is chosen by `modules/run_picker.py` (or a successful launch); one `reactive.poll` per active run has a cheap check (`file_stamp` of metrics/status/events) and a body that reads all three once. Modules receive zero-arg callables and never read files. Layout: sidebar (experiment picker, run picker, optional launch form), `status_bar` above the tabs, tabs Live / Control / Events / Runs / extensions.

`control_panel.py` renders `actions.json` once per change (poll on the file stamp) and binds one effect per action, tracking the last dispatched click count per button id. Do not switch it back to `reactive.event(ignore_init=True)`: when the effect is created before the client reports the button, `ignore_init` swallows the first real click. `RunContext` for extensions exposes `reader, run_dir, mailbox, status, metrics, events`; extensions are not namespaced.

## Current state to be aware of

- `README.md` predates the protocol split and describes modules that no longer exist. The authoritative public surface is `rewind/__init__.py` and `rewind/dashboard/__init__.py`; rewriting the README is Milestone 3.
- `../ICL` currently has no live Python files importing `rewind` (only stale `.pyc`), so API changes here break nothing downstream yet. ICL's entrypoint will need the `rewind.cli` override parser planned for Milestone 3.
- `examples/data/` is created by the toy script and is ignored by the `data/` rule.
