"""Per-episode file locks.

{workspace}/.lock is created with O_CREAT|O_EXCL and holds a small JSON
payload {pid, host, acquired_at}. Context manager. Stale-lock takeover when
the owning pid is dead (same host) or the lock file is older than one hour.
The lock is removed on exit.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Optional

STALE_AGE_S = 3600.0  # 1 hour


class LockHeldError(RuntimeError):
    """The episode workspace is locked by another live process."""


def pid_alive(pid: Any) -> bool:
    """Best-effort liveness check for a pid on this host."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # access denied == process exists but we cannot open it
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            code = ctypes.c_ulong(0)
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True  # cannot tell; be conservative
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


class EpisodeLock:
    """Exclusive per-episode lock at {workspace}/.lock.

    Usage:
        with EpisodeLock(ws.root):
            ... run stages ...

    Raises LockHeldError if the lock is held by a live process and the
    optional timeout budget is exhausted.
    """

    def __init__(
        self,
        workspace_root: Path | str,
        timeout_s: float = 0.0,
        poll_s: float = 0.25,
        stale_age_s: float = STALE_AGE_S,
    ):
        self.path = Path(workspace_root) / ".lock"
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.stale_age_s = float(stale_age_s)
        self._held = False

    # -- helpers ---------------------------------------------------------
    def _read_holder(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _describe_holder(self) -> str:
        h = self._read_holder()
        return f"pid={h.get('pid')} host={h.get('host')}"

    def _try_takeover(self) -> bool:
        """Remove the lock file if it is stale. True == caller should retry."""
        holder = self._read_holder()
        pid: Optional[Any] = holder.get("pid")
        same_host = holder.get("host") in (None, socket.gethostname())
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return True  # vanished between checks; retry acquire
        stale = (
            age > self.stale_age_s
            or (pid is not None and same_host and not pid_alive(pid))
            # unreadable payload: give the writer a grace period, then treat
            # as stale (crashed between O_CREAT and write)
            or (pid is None and age > 30.0)
        )
        if not stale:
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return False
        return True

    # -- api ---------------------------------------------------------------
    def acquire(self) -> "EpisodeLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._try_takeover():
                    continue
                if time.monotonic() >= deadline:
                    raise LockHeldError(
                        f"episode lock held ({self._describe_holder()}): {self.path}"
                    )
                time.sleep(self.poll_s)
                continue
            payload = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "acquired_at": time.time(),
            }
            try:
                os.write(fd, json.dumps(payload).encode("utf-8"))
            finally:
                os.close(fd)
            self._held = True
            return self

    def release(self) -> None:
        if not self._held:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._held = False

    def __enter__(self) -> "EpisodeLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False
