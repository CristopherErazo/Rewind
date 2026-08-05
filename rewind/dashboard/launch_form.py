"""
Turns any (possibly nested) config dataclass into Shiny input widgets, and
reads them back as OmegaConf-style CLI overrides -- so a new project never
needs to hand-write launch-form UI code.
"""

import dataclasses
from shiny import ui


def _leaf_fields(cls, prefix=""):
    """Yields (dotted_path, type) for every leaf field, recursing into
    nested dataclasses, skipping computed fields (init=False)."""
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        path = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(f.type):
            yield from _leaf_fields(f.type, prefix=f"{path}.")
        else:
            yield path, f.type


def _get_nested(obj, dotted_path):
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def _input_id(dotted_path):
    return dotted_path.replace(".", "__")  # Shiny input ids can't contain dots


def build_launch_form(config_cls):
    defaults = config_cls()
    inputs = []
    for path, _ in _leaf_fields(config_cls):
        value = _get_nested(defaults, path)
        input_id = _input_id(path)
        if isinstance(value, bool):
            inputs.append(ui.input_checkbox(input_id, path, value=value))
        elif isinstance(value, (int, float)):
            inputs.append(ui.input_numeric(input_id, path, value=value))
        else:
            inputs.append(ui.input_text(input_id, path, value=str(value)))
    return inputs


def collect_overrides(input, config_cls):
    """Reads current form values and returns them as
    'model_args.d_model=64' style strings, ready for the CLI."""
    overrides = []
    for path, _ in _leaf_fields(config_cls):
        value = input[_input_id(path)]()
        overrides.append(f"{path}={value}")
    return overrides