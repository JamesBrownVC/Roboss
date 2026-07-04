"""Simple GPU slot semaphore (file lock, cross-process safe on Linux)."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path


class GPUSemaphore:
    def __init__(self, slots: int = 1, lock_dir: Path | None = None):
        self.slots = max(1, slots)
        root = lock_dir or Path(os.environ.get("V2R_GPU_LOCK_DIR", "/tmp/v2r_gpu_locks"))
        root.mkdir(parents=True, exist_ok=True)
        self.lock_files = [root / f"slot_{i}.lock" for i in range(self.slots)]

    @contextmanager
    def acquire(self, timeout_s: float = 3600.0, poll_s: float = 0.5):
        deadline = time.time() + timeout_s
        held: Path | None = None
        try:
            while time.time() < deadline:
                for lf in self.lock_files:
                    try:
                        fd = os.open(str(lf), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.write(fd, str(os.getpid()).encode())
                        os.close(fd)
                        held = lf
                        break
                    except FileExistsError:
                        continue
                if held is not None:
                    yield
                    return
                time.sleep(poll_s)
            raise TimeoutError(f"GPU semaphore: no slot free within {timeout_s}s")
        finally:
            if held is not None and held.exists():
                held.unlink(missing_ok=True)
