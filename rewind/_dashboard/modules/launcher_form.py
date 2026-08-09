import dataclasses
from shiny import module, ui, reactive
from ..launch import RunLauncher
from ..schema import generate_ui_inputs

@module.ui
def launcher_form_ui(config_cls):
    return ui.card(
        ui.card_header("New Experiment"),
        ui.p("Configure and launch a new steerable run.", class_="text-muted small"),
        
        # 1. Dynamically unpack the UI inputs generated from the dataclass
        *generate_ui_inputs(config_cls),
        
        ui.hr(),
        ui.input_action_button("launch_btn", "Launch Run", class_="btn-primary w-100")
    )

@module.server
def launcher_form_server(input, output, session, launcher: RunLauncher, script_path: str, config_cls, on_launch_cb=None):
    @reactive.effect
    @reactive.event(input.launch_btn)
    def handle_launch():
        # 2. Claim ID atomically
        run_id = launcher.reserve_run_id()
        
        # 3. Dynamically build arguments based on the dataclass fields
        extra_args = []
        for field in dataclasses.fields(config_cls):
            # Fetch the reactive input value dynamically by field name
            # Equivalent to calling input.learning_rate()
            val = getattr(input, field.name)()
            
            # Format argument flags correctly based on type
            if field.type == bool:
                if val:  # Only append boolean flags if they are checked/True
                    extra_args.append(f"--{field.name}")
            else:
                extra_args.extend([f"--{field.name}", str(val)])
        
        # 4. Launch the process
        pid = launcher.launch(script_path=script_path, run_id=run_id, extra_args=extra_args)
        ui.notification_show(f"Launched {run_id} (PID: {pid})", type="message")
        
        if callable(on_launch_cb):
            on_launch_cb()