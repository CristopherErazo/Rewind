"""rewind/dashboard/schema.py

Turns a (possibly nested) config dataclass into a flat list of FieldSpec the
launcher_form module renders generically. This replaces a hand-maintained
per-project widget list and an accompanying "which CLI override does this
widget map to" dict -- both instead derived from the dataclass itself.

Only handles the common case (leaf field -> widget). A field can still be
rendered with something bespoke via DashboardConfig.field_overrides; this
module doesn't need to know about that escape hatch at all.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, get_type_hints

WidgetKind = Literal["numeric", "switch", "select", "text"]


def input_id(path: str) -> str:
    """Shiny input id for a given dotted field path. Shared between the
    default widget builder and any custom widget passed via
    DashboardConfig.field_overrides -- an override MUST use this so
    launcher_form.py can read the submitted value back under the id it
    expects."""
    return f"field_{path.replace('.', '__')}"

@dataclass
class FieldSpec:
    path: str  # dotted path, e.g. "model_args.vocab_size" -- also the CLI override key
    label: str  # human label, e.g. "Vocab size"
    widget: WidgetKind
    default: Any
    choices: list[str] | None = None


def form_fields(cfg_cls: type, prefix: str = "") -> list[FieldSpec]:
    """Recursively walk a dataclass (nested dataclass fields become dotted
    prefixes), returning one FieldSpec per leaf field."""
    fields: list[FieldSpec] = []
    hints = get_type_hints(cfg_cls)

    for f in dataclasses.fields(cfg_cls):
        path = f"{prefix}{f.name}"
        ftype = hints.get(f.name, f.type)

        if dataclasses.is_dataclass(ftype):
            fields.extend(form_fields(ftype, prefix=f"{path}."))
            continue
        if not f.metadata.get('ui',False):
            continue
                
        fields.append(
            FieldSpec(
                path=path,
                label=_label(f.name),
                widget=_widget_for(ftype),
                default=_default_of(f),
                choices=[c.name for c in ftype] if _is_enum(ftype) else None,
            )
        )
    return fields


def _default_of(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return f.default_factory()  # type: ignore[misc]
        except Exception:
            return None
    return None


def _is_enum(ftype: Any) -> bool:
    return isinstance(ftype, type) and issubclass(ftype, Enum)


def _widget_for(ftype: Any) -> WidgetKind:
    # bool check first: bool is a subclass of int in Python, so it must be
    # tested before the int/float check below or every checkbox becomes a
    # number input.
    if isinstance(ftype, type) and issubclass(ftype, bool):
        return "switch"
    if _is_enum(ftype):
        return "select"
    if isinstance(ftype, type) and issubclass(ftype, (int, float)):
        return "numeric"
    return "text"


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


