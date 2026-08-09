"""rewind/_fsutil.py

A tiny, generic filesystem primitive Rewind needs in a few places (command
files, actions.json, process.json): write to a temp file, then os.replace
into the final path, so a concurrent reader never sees a torn write.

This existed before, inline, inside control.py's RunMailbox.send_command.
It's pulled out here so registry.py and launch.py can reuse it too, instead
of importing tracklab.utils.atomic_write for something this small and
generic. rewind/control.py should be updated to import from here as well,
so there's exactly one copy of this logic instead of two.
"""

from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path


# def atomic_write(path: Path, content: str) -> None:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
#     try:
#         with os.fdopen(fd, "w") as f:
#             f.write(content)
#         os.replace(tmp_path, path)
#     except Exception:
#         Path(tmp_path).unlink(missing_ok=True)
#         raise

def atomic_write(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        for attempt in range(10):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise