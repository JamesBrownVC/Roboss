"""Central log store with ring buffer and SSE subscriber fan-out."""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Any

MAX_ENTRIES = 2000

_STEP_AGENT = [
    (re.compile(r"Parsing intent", re.I), "intent"),
    (re.compile(r"Building world contract", re.I), "contract"),
    (re.compile(r"Planning .* scenario", re.I), "scenarios"),
    (re.compile(r"repair|violate|dropping .* scenario", re.I), "validator"),
    (re.compile(r"Compiling prompts", re.I), "compiler"),
    (re.compile(r"Generating canvas|start frame|Visual anchors", re.I), "canvas"),
]


def _agent_from_message(message: str, default: str = "pipeline") -> str:
    for pattern, agent in _STEP_AGENT:
        if pattern.search(message):
            return agent
    return default


class LogStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._next_id = 1
        self._subscribers: list[asyncio.Queue] = []

    def append(
        self,
        message: str,
        *,
        level: str = "info",
        agent: str | None = None,
        batch_id: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        agent_name = agent or _agent_from_message(message)
        entry = {
            "id": 0,
            "ts": time.time(),
            "level": level,
            "agent": agent_name,
            "message": message.strip(),
            "batch_id": batch_id,
            "job_id": job_id,
        }
        with self._lock:
            entry["id"] = self._next_id
            self._next_id += 1
            self._entries.append(entry)
            if len(self._entries) > MAX_ENTRIES:
                self._entries = self._entries[-MAX_ENTRIES:]
            subscribers = list(self._subscribers)
        for queue in subscribers:
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                pass
        return entry

    def get_since(self, since_id: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if since_id is None:
                return list(self._entries)
            return [entry for entry in self._entries if entry["id"] > since_id]

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)


LOG_STORE = LogStore()


def batch_logger(batch_id: str, job_id: str | None = None):
    """Return a callable compatible with agents.run_pipeline(progress=...)."""

    def _log(
        message: str,
        level: str | None = None,
        agent: str | None = None,
    ) -> None:
        resolved = level
        if resolved is None:
            resolved = "error" if message.lower().startswith("error") else "info"
            if "failed" in message.lower() or "skipped" in message.lower():
                resolved = "warn"
        LOG_STORE.append(
            message,
            level=resolved,
            agent=agent,
            batch_id=batch_id,
            job_id=job_id,
        )

    return _log
