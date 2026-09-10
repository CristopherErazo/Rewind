"""rewind/_fsutil.py

Small filesystem primitives shared by control.py, registry.py, events.py
and launch.py.

`atomic_write` writes to a temp file in the same directory and then
`os.replace`s it into place, so a concurrent reader never sees a torn file.
On Windows `os.replace` can fail with PermissionError while another process
has the target open for reading, so it retries with a short backoff.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def atomic_write(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
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


def append_line(path: Path, line: str) -> None:
    """Append one complete line to a text file, flushed immediately, so a
    concurrent reader only ever sees whole lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()


def file_stamp(path: Path) -> tuple | None:
    """Cheap change detector for polling: (size, mtime_ns) or None if the
    file does not exist."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)
