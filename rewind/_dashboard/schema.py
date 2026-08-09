import dataclasses
from typing import List, Tuple, Any
from shiny import ui

def _leaf_fields(cls, prefix: str = ""):
    """
    Yields (dotted_path, field_obj) for every leaf field.
    Recurses into nested dataclasses and skips:
      1. Computed fields (init=False)
      2. Fields with metadata {"ui": False}
    """
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        
        path = f"{prefix}{f.name}"
        
        if dataclasses.is_dataclass(f.type):
            yield from _leaf_fields(f.type, prefix=f"{path}.")
        else:
            if not f.metadata.get("ui",False):
                continue
            yield path, f

def _get_nested(obj: Any, dotted_path: str) -> Any:
    """Navigates a nested dataclass instance using a dotted path."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def _input_id(dotted_path: str) -> str:
    """Sanitizes dotted paths for Shiny input IDs (dots to double underscores)."""
    return dotted_path.replace(".", "__")


def generate_ui_inputs(config_cls) -> List[Any]:
    """
    Generates dynamic Shiny UI inputs for a given dataclass.
    """
    defaults = config_cls()
    inputs = []
    current_section = None
    
    for path, f in _leaf_fields(config_cls):
        value = _get_nested(defaults, path)
        input_id = _input_id(path)
        
        # Use custom label from metadata, or fall back to humanizing the path
        label = f.metadata.get("label", path.split(".")[-1].replace("_", " ").title())
        
        # Optionally add section headers when switching top-level groups
        top_level = path.split(".")[0]
        if top_level != current_section and "." in path:
            current_section = top_level
            header_text = current_section.replace("_", " ").title()
            print("Header: ", header_text)
            inputs.append(ui.h5(header_text, class_="mt-3 mb-2 text-primary border-bottom"))

        print(label)
        # Input component generation
        if isinstance(value, bool):
            inputs.append(ui.input_checkbox(input_id, label, value=value))
        elif isinstance(value, int):
            inputs.append(ui.input_numeric(input_id, label, value=value))
        elif isinstance(value, float):
            step = f.metadata.get("step", 0.01)
            min = f.metadata.get("min",0.0)
            inputs.append(ui.input_numeric(input_id, label, value=value, step=step,min=min))
        else:
            inputs.append(ui.input_text(input_id, label, value=str(value)))
            
    return inputs

def collect_overrides(input_dict, config_cls) -> list[str]:
    """
    Formats form values specifically for OmegaConf.from_cli().
    Outputs: ['model_args.d_model=64', 'extra_args.seed=42', ...]
    """
    overrides = []
    
    for path, f in _leaf_fields(config_cls):
        input_key = _input_id(path)
        val = input_dict[input_key]()
        
        # Format booleans as lowercase true/false for OmegaConf
        if isinstance(val, bool):
            overrides.append(f"{path}={str(val).lower()}")
        else:
            overrides.append(f"{path}={val}")
            
    return overrides

if __name__ == '__main__':
    from icl import TrainerArgs
    # trainer_args = TrainerArgs()

    inputs = generate_ui_inputs(TrainerArgs)
        
