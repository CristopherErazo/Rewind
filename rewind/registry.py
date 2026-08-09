"""rewind/dashboard/registry.py

The self-describing control surface. A running TrainerController declares
what commands it accepts -- built-ins (pause/resume/stop/set_lr) plus
anything a project registers with `register_handler(..., spec=...)` -- as a
list of ActionSpec. That list is written once to `<run_dir>/actions.json` at
run start. The dashboard's control_panel module reads that file and renders
generic controls for whatever it finds. It never imports project code: this
is what makes a project-specific command (e.g. "perturb_layer") show up in
the UI the moment it's registered, with zero dashboard changes.

Integration touch points (outside this file):
  - rewind.controller.TrainerController.register_handler gains an optional
    `spec: ActionSpec` kwarg; the controller keeps its own running list
    starting from BUILTIN_ACTIONS and appends each spec it's given.
  - TrainerController.run_loop() calls write_actions(run.run_dir, self._specs)
    once, before entering the poll loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

from ._fsutil import atomic_write
from ._layout import actions_path

ArgKind = Literal["int", "float", "str", "bool"]

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _slugify(text: str) -> str:
    """Turn an arbitrary display string into a safe HTML/Shiny id fragment:
    non-alphanumerics collapsed to underscores, lowercased, and guaranteed
    not to start with a digit (invalid for an HTML id)."""
    slug = _SLUG_RE.sub("_", text.strip()).strip("_").lower()
    if not slug:
        slug = "action"
    if slug[0].isdigit():
        slug = f"a_{slug}"
    return slug


@dataclass
class ArgSpec:
    kind: ArgKind
    default: Any = None
    description: str = ""


@dataclass
class ActionSpec:
    name: str  # command "type", e.g. "set_lr", "perturb layer" -- matches the
    # `cmd_type` a handler is registered under in TrainerController. Free-form
    # on purpose: it only needs to be a valid dict key, not a valid HTML id.
    label: str  # button / section text shown in the UI
    args: dict[str, ArgSpec] = field(default_factory=dict)
    description: str = ""

    @property
    def kind(self) -> Literal["button", "form"]:
        """Derived, not stored. "button" if there are no args, "form"
        otherwise -- kept as a property so it can never disagree with
        `args`, which was possible when this was its own field."""
        return "button" if not self.args else "form"

    @property
    def html_id(self) -> str:
        """Safe Shiny input id derived from `name`, e.g. "perturb layer" ->
        "act_perturb_layer". control_panel.py should always go through this
        (and arg_html_id below) rather than building ids by hand."""
        return f"act_{_slugify(self.name)}"

    def arg_html_id(self, arg_name: str) -> str:
        return f"{self.html_id}_arg_{_slugify(arg_name)}"

    def to_json(self) -> dict:
        # `kind` and the *_html_id properties are derived, not persisted --
        # from_json reconstructs an ActionSpec that recomputes them the
        # same way, so there's nothing to keep in sync on disk.
        return {"name": self.name, "label": self.label,
                "args": {k: asdict(v) for k, v in self.args.items()},
                "description": self.description}

    @classmethod
    def from_json(cls, d: dict) -> ActionSpec:
        args = {k: ArgSpec(**v) for k, v in d.get("args", {}).items()}
        return cls(
            name=d["name"],
            label=d["label"],
            args=args,
            description=d.get("description", ""),
        )


# Specs for the handlers TrainerController always registers itself, so a
# bare project (nothing custom registered) still gets a working panel.
BUILTIN_ACTIONS: list[ActionSpec] = [
    ActionSpec("pause", "Pause", description="Pause after the current step."),
    ActionSpec("resume", "Resume", description="Resume a paused run."),
    ActionSpec("stop", "Stop", description="Stop the run and finalize the tracker."),
    ActionSpec(
        "set_lr",
        "Set learning rate",
        args={"lr": ArgSpec("float", default=1e-3, description="New learning rate")},
        description="Update the optimizer's learning rate live.",
    ),
]

def write_actions(run_dir: Path, specs: list[ActionSpec]) -> None:
    seen: dict[str, str] = {}
    for spec in specs:
        if spec.html_id in seen:
            raise ValueError(
                f"Action names {seen[spec.html_id]!r} and {spec.name!r} both "
                f"produce the UI id {spec.html_id!r} -- rename one of them."
            )
        seen[spec.html_id] = spec.name

    payload = [s.to_json() for s in specs]
    atomic_write(actions_path(run_dir), json.dumps(payload, indent=2))


def read_actions(run_dir: Path) -> list[ActionSpec]:
    path = actions_path(run_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [ActionSpec.from_json(d) for d in payload]