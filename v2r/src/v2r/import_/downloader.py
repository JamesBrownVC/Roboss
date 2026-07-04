"""Download videos from the dataset catalog into data/raw/{source_id}/."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from .catalog import (
    DatasetCatalog,
    DatasetSource,
    DirectUrlSource,
    HuggingFaceSource,
    YouTubeSource,
    load_catalog,
)


@dataclass
class ImportResult:
    source_id: str
    subject: str
    videos: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _log(msg: str, log: Callable[[str], None]) -> None:
    log(msg)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def download_url(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.pexels.com/",
        "Accept": "video/mp4,*/*",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest.write_bytes(data)
    return dest


def _normalize_video(src: Path, dest: Path, target_fps: float = 30.0) -> Path:
    """Re-encode to h264 mp4 at target fps if needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if src.suffix.lower() == ".mp4" and abs(fps - target_fps) < 1.0 and src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
        return dest

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dest), fourcc, target_fps, (w, h))
    cap = cv2.VideoCapture(str(src))
    step = fps / target_fps if fps > 0 else 1.0
    idx = 0.0
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i >= int(idx):
            out.write(frame)
            idx += step
        frame_i += 1
    cap.release()
    out.release()
    return dest


def import_direct_url(source: DirectUrlSource, out_dir: Path, log: Callable[[str], None]) -> ImportResult:
    result = ImportResult(source_id=source.id, subject=source.subject)
    for i, url in enumerate(source.urls):
        dest = out_dir / f"{source.id}_{i:03d}.mp4"
        if dest.exists() and dest.stat().st_size > 10_000:
            result.videos.append(dest)
            continue
        try:
            _log(f"  downloading {url[:80]}...", log)
            tmp = out_dir / f".tmp_{source.id}_{i:03d}.mp4"
            download_url(url, tmp)
            _normalize_video(tmp, dest)
            tmp.unlink(missing_ok=True)
            result.videos.append(dest)
        except Exception as e:
            result.errors.append(f"{url}: {e}")
    return result


def import_youtube(source: YouTubeSource, out_dir: Path, log: Callable[[str], None]) -> ImportResult:
    result = ImportResult(source_id=source.id, subject=source.subject)
    for i, url in enumerate(source.urls):
        dest = out_dir / f"{source.id}_{i:03d}.mp4"
        if dest.exists():
            result.videos.append(dest)
            continue
        try:
            _log(f"  yt-dlp {url}", log)
            cmd = [
                "yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best",
                "-o", str(out_dir / f".tmp_{source.id}_{i:03d}.%(ext)s"),
                url,
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            tmp_files = list(out_dir.glob(f".tmp_{source.id}_{i:03d}.*"))
            if not tmp_files:
                raise FileNotFoundError("yt-dlp produced no file")
            _normalize_video(tmp_files[0], dest)
            for t in tmp_files:
                t.unlink(missing_ok=True)
            result.videos.append(dest)
        except Exception as e:
            result.errors.append(f"{url}: {e}")
    return result


def import_huggingface(source: HuggingFaceSource, out_dir: Path, log: Callable[[str], None]) -> ImportResult:
    result = ImportResult(source_id=source.id, subject=source.subject)
    try:
        from datasets import load_dataset
    except ImportError:
        result.errors.append("datasets package not installed (pip install datasets)")
        return result

    try:
        _log(f"  HF load_dataset({source.dataset}, split={source.split})", log)
        ds = load_dataset(source.dataset, split=source.split, streaming=True)
    except Exception as e:
        result.errors.append(f"load_dataset failed: {e}")
        return result

    col = source.video_column
    count = 0
    for row in ds:
        if count >= source.max_episodes:
            break
        dest = out_dir / f"{source.id}_{count:03d}.mp4"
        if dest.exists() and dest.stat().st_size > 10_000:
            result.videos.append(dest)
            count += 1
            continue
        try:
            item = row.get(col)
            if item is None:
                result.errors.append(f"row {count}: column {col!r} missing")
                count += 1
                continue

            # HF Video / PIL / path / bytes
            if isinstance(item, str):
                p = Path(item)
                if p.is_file():
                    _normalize_video(p, dest)
                    result.videos.append(dest)
                else:
                    result.errors.append(f"row {count}: path not found {item}")
            elif hasattr(item, "save"):
                # PIL Image frame sequence - rare
                result.errors.append(f"row {count}: unsupported image column type")
            elif isinstance(item, dict) and "path" in item:
                _normalize_video(Path(item["path"]), dest)
                result.videos.append(dest)
            else:
                # datasets Video feature: decode frames and write mp4
                import numpy as np

                frames = item.get("frames") if isinstance(item, dict) else None
                if frames is None and hasattr(item, "__getitem__"):
                    try:
                        arr = np.array(item)
                        if arr.ndim == 4:
                            frames = arr
                    except Exception:
                        pass
                if frames is not None and len(frames) > 0:
                    h, w = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(str(dest), fourcc, 30.0, (w, h))
                    for fr in frames:
                        if fr.shape[2] == 3:
                            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                        else:
                            vw.write(fr)
                    vw.release()
                    result.videos.append(dest)
                else:
                    result.errors.append(f"row {count}: cannot decode video column")
        except Exception as e:
            result.errors.append(f"row {count}: {e}")
        count += 1
    return result


def import_source(source: DatasetSource, data_root: Path, log: Callable[[str], None]) -> ImportResult:
    out_dir = data_root / source.id
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(source, DirectUrlSource):
        return import_direct_url(source, out_dir, log)
    if isinstance(source, HuggingFaceSource):
        return import_huggingface(source, out_dir, log)
    if isinstance(source, YouTubeSource):
        return import_youtube(source, out_dir, log)
    raise ValueError(f"unknown source type: {source}")


def import_all(
    root: Path,
    subject: Optional[str] = None,
    source_ids: Optional[list[str]] = None,
    log: Callable[[str], None] = print,
) -> tuple[list[ImportResult], Path]:
    catalog = load_catalog(root)
    data_root = Path(catalog.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root

    sources = catalog.by_subject(subject)
    if source_ids:
        ids = set(source_ids)
        sources = [s for s in sources if s.id in ids]

    results: list[ImportResult] = []
    for source in sources:
        _log(f"[import] {source.id} ({source.subject})", log)
        results.append(import_source(source, data_root, log))

    manifest = {
        "sources": [
            {
                "source_id": r.source_id,
                "subject": r.subject,
                "n_videos": len(r.videos),
                "videos": [str(v) for v in r.videos],
                "errors": r.errors,
            }
            for r in results
        ],
        "total_videos": sum(len(r.videos) for r in results),
    }
    manifest_path = data_root / "import_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"Wrote {manifest_path} ({manifest['total_videos']} videos)", log)
    return results, data_root
