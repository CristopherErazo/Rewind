"""rewind/registry.py

The self-describing control surface. A running TrainerController declares
what commands it accepts -- built-ins plus anything a project registers with
`register_handler(..., spec=...)` -- as a list of ActionSpec, written once to
`control/actions.json` at run start. The dashboard's control_panel renders
generic controls for whatever it finds there and never imports project code.

The same specs are used on the trainer side to coerce incoming command
arguments (`coerce_args`), so the trainer never trusts the UI's types.
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
_TRUE = {"1", "true", "yes", "on", "t", "y"}
_FALSE = {"0", "false", "no", "off", "f", "n", ""}


def _slugify(text: str) -> str:
    """Arbitrary display string -> safe HTML/Shiny id fragment."""
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
    required: bool = True

    def coerce(self, value: Any) -> Any:
        """Convert a raw value (from JSON, a form, or a CLI) to `kind`.
        Raises ValueError with a readable message on failure."""
        try:
            if self.kind == "bool":
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                s = str(value).strip().lower()
                if s in _TRUE:
                    return True
                if s in _FALSE:
                    return False
                raise ValueError(value)
            if self.kind == "int":
                if isinstance(value, bool):
                    return int(value)
                if isinstance(value, float):
                    if not value.is_integer():
                        raise ValueError(value)
                    return int(value)
                return int(str(value).strip())
            if self.kind == "float":
                return float(value)
            if self.kind == "str":
                return str(value)
        except (TypeError, ValueError):
            pass
        raise ValueError(f"expected {self.kind}, got {value!r}")


@dataclass
class ActionSpec:
    name: str  # command "type"; must match the handler's registered cmd_type
    label: str  # button / section text shown in the UI
    args: dict[str, ArgSpec] = field(default_factory=dict)
    description: str = ""

    @property
    def kind(self) -> Literal["button", "form"]:
        return "button" if not self.args else "form"

    @property
    def html_id(self) -> str:
        return f"act_{_slugify(self.name)}"

    def arg_html_id(self, arg_name: str) -> str:
        return f"{self.html_id}_arg_{_slugify(arg_name)}"

    def to_json(self) -> dict:
        return {"name": self.name, "label": self.label,
                "args": {k: asdict(v) for k, v in self.args.items()},
                "description": self.description}

    @classmethod
    def from_json(cls, d: dict) -> ActionSpec:
        args = {k: ArgSpec(**v) for k, v in d.get("args", {}).items()}
        return cls(name=d["name"], label=d["label"], args=args,
                   description=d.get("description", ""))


def coerce_args(spec: ActionSpec, cmd: dict) -> dict:
    """Return a copy of `cmd` with every declared argument coerced to its
    kind, defaults filled in, and undeclared keys (other than "type")
    dropped. Raises ValueError listing every problem at once."""
    out = {"type": cmd.get("type", spec.name)}
    problems = []
    for arg_name, arg in spec.args.items():
        if arg_name in cmd and cmd[arg_name] is not None:
            try:
                out[arg_name] = arg.coerce(cmd[arg_name])
            except ValueError as e:
                problems.append(f"{arg_name}: {e}")
        elif arg.default is not None:
            out[arg_name] = arg.coerce(arg.default)
        elif arg.required:
            problems.append(f"{arg_name}: missing")
    if problems:
        raise ValueError(f"bad arguments for {spec.name!r}: " + "; ".join(problems))
    return out


# Specs for the handlers TrainerController always registers itself.
BUILTIN_ACTIONS: list[ActionSpec] = [
    ActionSpec("pause", "Pause", description="Pause after the current step."),
    ActionSpec("resume", "Resume", description="Resume a paused run."),
    ActionSpec("stop", "Stop", description="Stop the run and finalize the tracker."),
    ActionSpec(
        "set_lr", "Set learning rate",
        args={"lr": ArgSpec("float", default=1e-3, description="New learning rate")},
        description="Update the optimizer's learning rate live.",
    ),
]

# Added only when the controller was constructed with enable_rewind=True.
REWIND_ACTION = ActionSpec(
    "rewind", "Rewind to step",
    args={"step": ArgSpec("int", description="Restore the nearest snapshot at or before this step")},
    description="Restore model/optimizer/RNG from a snapshot and continue on a new branch.",
)


def write_actions(run_dir: Path, specs: list[ActionSpec]) -> None:
    seen: dict[str, str] = {}
    for spec in specs:
        if spec.html_id in seen:
            raise ValueError(
                f"Action names {seen[spec.html_id]!r} and {spec.name!r} both "
                f"produce the UI id {spec.html_id!r} -- rename one of them."
            )
        seen[spec.html_id] = spec.name
    atomic_write(actions_path(run_dir), json.dumps([s.to_json() for s in specs], indent=2))


def read_actions(run_dir: Path) -> list[ActionSpec]:
    path = actions_path(run_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    return [ActionSpec.from_json(d) for d in payload]
