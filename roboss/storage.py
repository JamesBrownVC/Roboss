"""Local file storage with frontend-friendly asset URLs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import get_settings


@dataclass(frozen=True)
class StoredFile:
    path: str
    url: str
    size_bytes: int


class LocalStorageService:
    """Stores artifacts under runs/ and exposes stable /assets URLs."""

    def __init__(self, root: Path | None = None, base_url: str = "/assets"):
        settings = get_settings()
        self.root = (root or settings.runs_dir).resolve()
        self.base_url = base_url.rstrip("/")
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str | Path) -> Path:
        path = (self.root / relative_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"path escapes storage root: {relative_path}")
        return path

    def relative(self, path: str | Path) -> str:
        p = Path(path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path escapes storage root: {path}")
        return p.relative_to(self.root).as_posix()

    def url_for(self, path: str | Path) -> str:
        rel = self.relative(path)
        return f"{self.base_url}/{rel}"

    def describe(self, path: str | Path) -> StoredFile:
        p = Path(path).resolve()
        return StoredFile(
            path=str(p),
            url=self.url_for(p),
            size_bytes=p.stat().st_size,
        )

    def save_bytes(self, relative_path: str | Path, data: bytes) -> StoredFile:
        path = self.resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.describe(path)

    def save_json(self, relative_path: str | Path, data: Any) -> StoredFile:
        raw = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        return self.save_bytes(relative_path, raw)

    def collect_files(self, directory: str | Path) -> dict[str, dict[str, Any]]:
        base = Path(directory).resolve()
        if base != self.root and self.root not in base.parents:
            raise ValueError(f"path escapes storage root: {directory}")
        files: dict[str, dict[str, Any]] = {}
        if not base.exists():
            return files
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel_to_dir = path.relative_to(base).as_posix()
            item = self.describe(path)
            files[rel_to_dir] = {
                "path": item.path,
                "url": item.url,
                "size_bytes": item.size_bytes,
            }
        return files

    def write_manifest(self, run_dir: str | Path,
                       extra: dict[str, Any] | None = None) -> StoredFile:
        run_path = Path(run_dir).resolve()
        manifest = {
            "run_id": self.relative(run_path),
            "root": str(run_path),
            "files": self.collect_files(run_path),
        }
        if extra:
            manifest.update(extra)
        return self.save_json(Path(self.relative(run_path)) / "manifest.json",
                              manifest)


def get_storage() -> LocalStorageService:
    return LocalStorageService()
