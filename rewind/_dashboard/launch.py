import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

class RunLauncher:
    def __init__(self, tracklab_dir: Path | str):
        self.tracklab_dir = Path(tracklab_dir)
        self.tracklab_dir.mkdir(parents=True, exist_ok=True)
        # In-memory registry mapping run_id -> Popen object
        self._active_processes: Dict[str, subprocess.Popen] = {}

    def reserve_run_id(self, prefix: str = "run_") -> str:
        """
        Fixes the Run-ID Race Condition.
        Calculates the next ID and immediately creates the directory to lock it.
        """
        existing_runs = [
            d.name for d in self.tracklab_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]
        
        run_numbers = []
        for run in existing_runs:
            try:
                run_numbers.append(int(run.replace(prefix, "")))
            except ValueError:
                continue
                
        next_num = max(run_numbers) + 1 if run_numbers else 1
        next_run_id = f"{prefix}{next_num:03d}"
        
        # CLAIM THE ID: Create the directory so no other process can take it
        run_dir = self.tracklab_dir / next_run_id
        run_dir.mkdir(exist_ok=False) 
        
        return next_run_id

    def launch(self, script_path: str, run_id: str, extra_args: Optional[List[str]] = None) -> int:
        """
        Fixes Process Management. 
        Launches the training script, passes the run_id, and tracks the PID.
        """
        if extra_args is None:
            extra_args = []
            
        # Explicitly tell the script which run_id to use
        command = [sys.executable, script_path, "--run_id", run_id] + extra_args
        
        # Launch non-blocking and capture the object
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL, # Or route to a log file if you prefer
            stderr=subprocess.DEVNULL, 
            env=os.environ.copy()
        )
        
        self._active_processes[run_id] = process
        return process.pid
        
    def get_status(self, run_id: str) -> str:
        """
        Returns the current state of the process.
        """
        process = self._active_processes.get(run_id)
        if not process:
            return "unknown"
            
        returncode = process.poll()
        if returncode is None:
            return "running"
        elif returncode == 0:
            return "completed"
        else:
            return "failed"
            
    def kill(self, run_id: str) -> bool:
        """
        Safely terminates a frozen or unwanted run.
        """
        process = self._active_processes.get(run_id)
        if process and process.poll() is None:
            process.terminate() # Graceful SIGTERM
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill() # Brutal SIGKILL if it ignores the terminate
            return True
        return False