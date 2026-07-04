"""Tiny .env loader shared by agents and verifier.

It intentionally avoids an extra dependency. Existing environment variables
win over values from .env, so shell-provided secrets are never overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path


def _candidate_paths() -> list[Path]:
    roots = [Path.cwd(), Path(__file__).resolve().parent]
    paths: list[Path] = []
    for root in roots:
        for parent in (root, *root.parents):
            path = parent / ".env"
            if path not in paths:
                paths.append(path)
    return paths


def _clean(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def load_dotenv() -> Path | None:
    """Load KEY=VALUE pairs from the nearest .env file if present."""
    for path in _candidate_paths():
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = _clean(value)
        return path
    return None
