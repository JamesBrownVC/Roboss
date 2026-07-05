"""Pluggable video generation backends.

`VideoGenBackend` is the interface; two implementations ship:

  MockBackend   local cv2 renderer producing moving human-like silhouettes
                (deterministic per variant_id). Always available; used for
                demos, CI, and whenever the API is missing/quota-limited.

  OmniBackend   Gemini Omni Flash via the Interactions API (default when a
                key is present). Blocking POST /v1beta/interactions, then
                file-state polling + download. Requires GEMINI_API_KEY.

  VeoBackend    Google Veo via the Gemini API (models/veo-*:predictLongRunning
                + operation polling + file download). Requires GEMINI_API_KEY.

To plug in a different generator, subclass VideoGenBackend and register it in
`get_backend`.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import gemini
from .gemini import GeminiError
from .spec import JobDirs, JobSpec, VariantSpec


@dataclass
class GenResult:
    variant_id: str
    ok: bool
    path: Optional[Path] = None
    backend: str = ""
    error: str = ""
    meta: dict = field(default_factory=dict)


class VideoGenBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, variant: VariantSpec, dest: Path,
                 log: Callable[[str], None] = print) -> GenResult:
        """Generate one video for `variant` and write it to `dest`."""

    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Mock backend: cv2-rendered walking/waving silhouette
# ---------------------------------------------------------------------------


class MockBackend(VideoGenBackend):
    """Renders a deterministic articulated human silhouette moving in a simple
    room. Camera azimuth/height from the variant's prompt hash shift the
    projection slightly so different cams look like different viewpoints."""

    name = "mock"

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0):
        self.width, self.height, self.fps = width, height, fps

    def generate(self, variant: VariantSpec, dest: Path,
                 log: Callable[[str], None] = print) -> GenResult:
        try:
            n_frames = int(variant.duration_s * self.fps)
            seed = int.from_bytes(
                hashlib.blake2b(variant.variant_id.encode(), digest_size=4).digest(), "little")
            rng = np.random.default_rng(seed)
            # event seed controls the motion; cam seed controls viewpoint shift
            ev_seed = int.from_bytes(
                hashlib.blake2b(variant.event_id.encode(), digest_size=4).digest(), "little")
            ev_rng = np.random.default_rng(ev_seed)
            speed = float(ev_rng.uniform(0.8, 1.4))
            wave = "wave" in variant.prompt.lower() or "hello" in variant.prompt.lower()
            cam_shift = float(rng.uniform(-0.15, 0.15))
            cam_scale = float(rng.uniform(0.85, 1.1))

            dest.parent.mkdir(parents=True, exist_ok=True)
            vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (self.width, self.height))
            if not vw.isOpened():
                return GenResult(variant.variant_id, False, backend=self.name,
                                 error=f"cv2.VideoWriter failed for {dest}")
            bg_tone = int(rng.integers(30, 70))
            for f in range(n_frames):
                t = f / self.fps
                frame = self._render_frame(t, speed, wave, cam_shift, cam_scale, bg_tone)
                vw.write(frame)
            vw.release()
            return GenResult(variant.variant_id, True, path=dest, backend=self.name,
                             meta={"n_frames": n_frames, "fps": self.fps,
                                   "width": self.width, "height": self.height})
        except Exception as e:
            return GenResult(variant.variant_id, False, backend=self.name,
                             error=f"{type(e).__name__}: {e}")

    # -- drawing ------------------------------------------------------------
    def _render_frame(self, t: float, speed: float, wave: bool,
                      cam_shift: float, cam_scale: float, bg_tone: int) -> np.ndarray:
        w, h = self.width, self.height
        frame = np.full((h, w, 3), bg_tone, dtype=np.uint8)
        # floor gradient
        cv2.rectangle(frame, (0, int(h * 0.72)), (w, h), (bg_tone + 25,) * 3, -1)

        phase = 2 * math.pi * 1.2 * speed * t
        cx = (0.5 + cam_shift + 0.10 * math.sin(phase * 0.35)) * w
        scale = cam_scale * h * 0.55
        bob = 0.015 * math.sin(2 * phase) * scale
        hip = np.array([cx, h * 0.55 + bob])
        neck = hip + [0, -0.42 * scale]
        head = neck + [0, -0.10 * scale]
        col = (225, 225, 225)

        def seg(a, b, thickness):
            cv2.line(frame, tuple(np.int32(a)), tuple(np.int32(b)), col, thickness)

        cv2.circle(frame, tuple(np.int32(head)), int(0.06 * scale), col, -1)
        seg(neck, hip, int(0.09 * scale))
        # legs
        for s in (-1, 1):
            swing = 0.14 * math.sin(phase + (0 if s > 0 else math.pi)) * scale
            knee = hip + [s * 0.03 * scale + swing * 0.5, 0.22 * scale]
            ankle = hip + [s * 0.03 * scale + swing, 0.45 * scale]
            seg(hip, knee, int(0.05 * scale))
            seg(knee, ankle, int(0.045 * scale))
        # arms
        for s in (-1, 1):
            sho = neck + [s * 0.10 * scale, 0.02 * scale]
            if wave and s > 0:
                ang = -1.9 + 0.5 * math.sin(2 * math.pi * 2.0 * t)
                elb = sho + [0.14 * scale * math.cos(ang * 0.5), 0.14 * scale * math.sin(ang * 0.5)]
                wri = elb + [0.15 * scale * math.cos(ang), 0.15 * scale * math.sin(ang)]
            else:
                arm = 0.12 * math.sin(phase + (math.pi if s > 0 else 0)) * scale
                elb = sho + [s * 0.02 * scale + arm * 0.4, 0.15 * scale]
                wri = sho + [s * 0.02 * scale + arm, 0.30 * scale]
            seg(sho, elb, int(0.04 * scale))
            seg(elb, wri, int(0.035 * scale))
        return frame


# ---------------------------------------------------------------------------
# Omni backend (Interactions API) — default when a key is available
# ---------------------------------------------------------------------------


class OmniBackend(VideoGenBackend):
    """Gemini Omni Flash: conversational video model on the Interactions API.

    Notes vs Veo: the create call is blocking (~40s per clip), duration is
    model-controlled (VariantSpec.duration_s is not honored), and prompts
    mentioning people can trip Google's content filter (HTTP 400 with a
    prohibited-content message) — generate_all's mock fallback catches that.
    """

    name = "omni"

    def __init__(self, model: str = gemini.DEFAULT_OMNI_MODEL,
                 timeout_s: float = 600.0):
        self.model = model
        self.timeout_s = timeout_s

    def available(self) -> bool:
        return gemini.have_api_key()

    def generate(self, variant: VariantSpec, dest: Path,
                 log: Callable[[str], None] = print) -> GenResult:
        try:
            log(f"[omni] {variant.variant_id}: starting generation")
            uri = gemini.omni_generate_video(
                variant.prompt, model=self.model,
                aspect_ratio=variant.aspect_ratio,
                timeout=self.timeout_s,
            )
            gemini.poll_file_active(uri, timeout_s=self.timeout_s)
            gemini.download_file(uri, dest)
            log(f"[omni] {variant.variant_id}: downloaded {dest.name}")
            return GenResult(variant.variant_id, True, path=dest, backend=self.name,
                             meta={"model": self.model, "uri": uri})
        except GeminiError as e:
            return GenResult(variant.variant_id, False, backend=self.name, error=str(e))
        except Exception as e:
            return GenResult(variant.variant_id, False, backend=self.name,
                             error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Veo backend
# ---------------------------------------------------------------------------


class VeoBackend(VideoGenBackend):
    name = "veo"

    def __init__(self, model: str = gemini.DEFAULT_VEO_MODEL,
                 timeout_s: float = 600.0):
        self.model = model
        self.timeout_s = timeout_s

    def available(self) -> bool:
        return gemini.have_api_key()

    def generate(self, variant: VariantSpec, dest: Path,
                 log: Callable[[str], None] = print) -> GenResult:
        try:
            log(f"[veo] {variant.variant_id}: starting generation")
            op = gemini.start_video_generation(
                variant.prompt, model=self.model,
                aspect_ratio=variant.aspect_ratio,
                duration_seconds=variant.duration_s,
            )
            uri = gemini.poll_video_operation(op, timeout_s=self.timeout_s)
            gemini.download_file(uri, dest)
            log(f"[veo] {variant.variant_id}: downloaded {dest.name}")
            return GenResult(variant.variant_id, True, path=dest, backend=self.name,
                             meta={"model": self.model, "operation": op})
        except GeminiError as e:
            return GenResult(variant.variant_id, False, backend=self.name, error=str(e))
        except Exception as e:
            return GenResult(variant.variant_id, False, backend=self.name,
                             error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Selection + parallel fan-out
# ---------------------------------------------------------------------------


def get_backend(name: str) -> VideoGenBackend:
    """`auto` prefers Omni when a key is present, otherwise mock.

    PROJECT STANDARD: video generation uses OMNI. Veo stays available only
    for explicit --backend veo requests - do not make it a default anywhere.
    """
    if name == "auto":
        omni = OmniBackend()
        return omni if omni.available() else MockBackend()
    if name == "omni":
        return OmniBackend()
    if name == "veo":
        return VeoBackend()
    if name == "mock":
        return MockBackend()
    raise ValueError(f"unknown video backend {name!r} (use mock | omni | veo | auto)")


def generate_all(
    spec: JobSpec,
    dirs: JobDirs,
    backend: VideoGenBackend,
    max_workers: int = 4,
    log: Callable[[str], None] = print,
) -> list[GenResult]:
    """Step 3: fan out generation in parallel; write per-variant sidecar JSON.

    If the primary backend fails for a variant (quota, network), that variant
    falls back to the mock backend so the pipeline stays demoable end-to-end.
    """
    import json

    mock = MockBackend() if backend.name != "mock" else backend
    results: list[GenResult] = []

    def _one(variant: VariantSpec) -> GenResult:
        dest = dirs.video_mp4(variant.variant_id)
        res = backend.generate(variant, dest, log=log)
        if not res.ok and backend.name != "mock":
            log(f"[gen] {variant.variant_id}: {backend.name} failed ({res.error}); mock fallback")
            res = mock.generate(variant, dest, log=log)
            res.meta["fallback_from"] = backend.name
        return res

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, v): v for v in spec.variants}
        for fut in as_completed(futures):
            variant = futures[fut]
            res = fut.result()
            results.append(res)
            sidecar = {
                "variant_id": variant.variant_id,
                "event_id": variant.event_id,
                "cam_id": variant.cam_id,
                "prompt": variant.prompt,
                "duration_s": variant.duration_s,
                "backend": res.backend,
                "ok": res.ok,
                "error": res.error,
                "meta": res.meta,
            }
            dirs.video_sidecar(variant.variant_id).write_text(
                json.dumps(sidecar, indent=2), encoding="utf-8")
            log(f"[gen] {variant.variant_id}: {'ok' if res.ok else 'FAILED'} ({res.backend})")

    results.sort(key=lambda r: r.variant_id)
    return results
