"""Job spec models + "director" prompt expansion (Gemini or deterministic mock).

A syngen job is a matrix of `events x cameras`:
  - each *event* is one independent subject-motion variation of the user prompt
    (different motion style / lighting / time of day);
  - each *camera* is a viewpoint that renders every event, so same-event
    multi-camera videos can feed the V2R multi-view session tier. Camera
    parameters (height, distance, azimuth, FOV) are saved into spec.json and
    later seed calibration priors for `v2r session`.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from .gemini import GeminiError, extract_json, generate_content

DIRECTOR_TEMPERATURE = 0.9

TIMES_OF_DAY = ["morning", "midday", "golden hour", "dusk", "night (interior lighting)"]
MOTION_STYLES = ["natural pace", "slow and deliberate", "brisk and energetic",
                 "hesitant with pauses", "smooth and continuous"]


class CameraSpec(BaseModel):
    cam_id: str
    description: str = ""
    height_m: float = 1.5
    distance_m: float = 3.0
    azimuth_deg: float = 0.0
    fov_deg: float = 60.0


class EventSpec(BaseModel):
    event_id: str
    subject: str = "person"
    motion_style: str = "natural pace"
    time_of_day: str = "midday"
    lighting: str = "natural light"
    scene: str = ""
    base_prompt: str


class VariantSpec(BaseModel):
    """One video to generate = one (event, camera) cell."""

    variant_id: str
    event_id: str
    cam_id: str
    prompt: str
    duration_s: int = 4
    aspect_ratio: str = "16:9"


class JobSpec(BaseModel):
    job_id: str
    user_prompt: str
    created_at: str = ""
    n_events: int = 1
    n_cameras: int = 1
    backend: str = "mock"
    director: str = "mock"           # "gemini" | "mock"
    world_notes: str = ""
    cameras: list[CameraSpec] = Field(default_factory=list)
    events: list[EventSpec] = Field(default_factory=list)
    variants: list[VariantSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Director schemas / prompts
# ---------------------------------------------------------------------------

DIRECTOR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "world_notes": {
            "type": "STRING",
            "description": "Shared world-consistency notes: environment, subject "
                           "appearance, layout that every camera and event must respect.",
        },
        "events": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "subject": {"type": "STRING"},
                    "motion_style": {"type": "STRING"},
                    "time_of_day": {"type": "STRING"},
                    "lighting": {"type": "STRING"},
                    "scene": {"type": "STRING"},
                    "base_prompt": {
                        "type": "STRING",
                        "description": "Complete video-generation prompt for this "
                                       "motion variation, without camera placement.",
                    },
                },
                "required": ["base_prompt"],
            },
        },
        "cameras": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "description": {
                        "type": "STRING",
                        "description": "Camera placement phrase to append to prompts, "
                                       "e.g. 'shot from a high corner camera, wide angle'.",
                    },
                    "height_m": {"type": "NUMBER"},
                    "distance_m": {"type": "NUMBER"},
                    "azimuth_deg": {"type": "NUMBER"},
                    "fov_deg": {"type": "NUMBER"},
                },
                "required": ["description"],
            },
        },
    },
    "required": ["events", "cameras"],
}


def _director_prompt(user_prompt: str, n_events: int, n_cameras: int,
                     extra_params: Optional[dict] = None) -> str:
    extras = ""
    if extra_params:
        extras = "\nAdditional user constraints: " + json.dumps(extra_params)
    return f"""You are a synthetic-data director for a robotics training pipeline.
The user wants training videos of: "{user_prompt}"{extras}

Design a generation plan:
- exactly {n_events} EVENT variations: same task, but vary subject motion style,
  time of day, and lighting. Each base_prompt must be a self-contained,
  physically-plausible video-generation prompt (single continuous shot, no cuts,
  realistic human motion, subject fully visible). Do NOT include camera placement
  in base_prompt.
- exactly {n_cameras} CAMERA viewpoints observing the SAME scene from different
  positions (vary height_m 0.8-3.0, distance_m 2-6, azimuth_deg spread around the
  subject, fov_deg 45-90). description is a short phrase appendable to a prompt.
- world_notes: consistency requirements so all cameras/events depict one coherent
  world (room layout, subject clothing, object positions).

Videos are later processed by pose trackers, so the subject must stay in frame,
motion must obey physics (no teleporting, no morphing limbs), static cameras only.

IMPORTANT content-safety constraint: refer to human subjects generically ("a
person", "an adult") — do NOT describe age, gender, hair, face, or other
demographic/physical traits (clothing color is fine). Detailed person
descriptions get prompts blocked by the video model's content filter."""


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def _slug(text: str, max_len: int = 24) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len] or "job"


def make_job_id(user_prompt: str) -> str:
    h = hashlib.blake2b(
        f"{user_prompt}:{datetime.now(timezone.utc).isoformat()}".encode(),
        digest_size=3,
    ).hexdigest()
    return f"{_slug(user_prompt)}_{h}"


def _compose_variants(cameras: list[CameraSpec], events: list[EventSpec],
                      duration_s: int) -> list[VariantSpec]:
    variants = []
    for ev in events:
        for cam in cameras:
            prompt = ev.base_prompt.rstrip(". ")
            if cam.description:
                prompt += f". {cam.description.rstrip('. ')}"
            prompt += f". {ev.lighting}, {ev.time_of_day}. Static camera, single continuous shot."
            variants.append(VariantSpec(
                variant_id=f"{ev.event_id}_{cam.cam_id}",
                event_id=ev.event_id,
                cam_id=cam.cam_id,
                prompt=prompt,
                duration_s=duration_s,
            ))
    return variants


def parse_director_response(raw: str, n_events: int, n_cameras: int) -> tuple[str, list[EventSpec], list[CameraSpec]]:
    """Parse+validate the director LLM JSON into typed specs (clamped to N)."""
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("director response is not a JSON object")
    ev_raw = data.get("events") or []
    cam_raw = data.get("cameras") or []
    if not ev_raw or not cam_raw:
        raise ValueError("director response missing events or cameras")

    events = []
    for i, e in enumerate(ev_raw[:n_events]):
        bp = str(e.get("base_prompt", "")).strip()
        if not bp:
            continue
        events.append(EventSpec(
            event_id=f"e{i:02d}",
            subject=str(e.get("subject", "person")),
            motion_style=str(e.get("motion_style", "natural pace")),
            time_of_day=str(e.get("time_of_day", "midday")),
            lighting=str(e.get("lighting", "natural light")),
            scene=str(e.get("scene", "")),
            base_prompt=bp,
        ))
    cameras = []
    for i, c in enumerate(cam_raw[:n_cameras]):
        cameras.append(CameraSpec(
            cam_id=f"cam{i}",
            description=str(c.get("description", "")),
            height_m=float(c.get("height_m", 1.5)),
            distance_m=float(c.get("distance_m", 3.0)),
            azimuth_deg=float(c.get("azimuth_deg", i * 360.0 / max(n_cameras, 1))),
            fov_deg=float(c.get("fov_deg", 60.0)),
        ))
    if not events or not cameras:
        raise ValueError("director response produced no usable events/cameras")
    world_notes = str(data.get("world_notes", ""))
    return world_notes, events, cameras


def mock_expand(user_prompt: str, n_events: int, n_cameras: int,
                seed: Optional[int] = None) -> tuple[str, list[EventSpec], list[CameraSpec]]:
    """Deterministic offline expansion (no API)."""
    if seed is None:
        seed = int.from_bytes(hashlib.blake2b(user_prompt.encode(), digest_size=4).digest(), "little")
    rng = np.random.default_rng(seed)
    events = []
    for i in range(n_events):
        tod = TIMES_OF_DAY[int(rng.integers(len(TIMES_OF_DAY)))]
        style = MOTION_STYLES[int(rng.integers(len(MOTION_STYLES)))]
        events.append(EventSpec(
            event_id=f"e{i:02d}",
            subject="person",
            motion_style=style,
            time_of_day=tod,
            lighting="soft natural light" if "night" not in tod else "warm interior lighting",
            scene="plain indoor room",
            base_prompt=f"A person: {user_prompt}. The motion is {style}, "
                        f"realistic human biomechanics, subject fully visible",
        ))
    cameras = []
    for i in range(n_cameras):
        az = i * 360.0 / max(n_cameras, 1)
        h = float(1.0 + 0.6 * (i % 3))
        d = float(2.5 + 0.8 * i)
        cameras.append(CameraSpec(
            cam_id=f"cam{i}",
            description=f"Shot from a static camera at {h:.1f}m height, "
                        f"{d:.1f}m away, {az:.0f} degrees around the subject",
            height_m=h, distance_m=d, azimuth_deg=az, fov_deg=60.0 + 5.0 * (i % 3),
        ))
    world_notes = ("Mock expansion (no LLM): one consistent indoor room, same subject "
                   "clothing across all events and cameras.")
    return world_notes, events, cameras


def expand_request(
    user_prompt: str,
    n_events: int,
    n_cameras: int,
    job_id: str,
    backend: str,
    duration_s: int = 4,
    extra_params: Optional[dict] = None,
    use_llm: bool = True,
    log=print,
) -> JobSpec:
    """Step 2: director expansion. Tries Gemini, falls back to the mock."""
    director = "mock"
    world_notes: str
    try:
        if not use_llm:
            raise GeminiError("LLM director disabled")
        raw = generate_content(
            [{"text": _director_prompt(user_prompt, n_events, n_cameras, extra_params)}],
            temperature=DIRECTOR_TEMPERATURE,
            response_schema=DIRECTOR_SCHEMA,
        )
        world_notes, events, cameras = parse_director_response(raw, n_events, n_cameras)
        # LLM may return fewer than requested; pad from the mock expansion
        if len(events) < n_events or len(cameras) < n_cameras:
            _, mev, mcam = mock_expand(user_prompt, n_events, n_cameras)
            events = (events + mev)[:n_events]
            for i, e in enumerate(events):
                e.event_id = f"e{i:02d}"
            cameras = (cameras + mcam)[:n_cameras]
            for i, c in enumerate(cameras):
                c.cam_id = f"cam{i}"
        director = "gemini"
        log(f"[director] Gemini expansion ok: {len(events)} events x {len(cameras)} cameras")
    except (GeminiError, ValueError, json.JSONDecodeError) as e:
        log(f"[director] Gemini unavailable ({e}); using deterministic mock expansion")
        world_notes, events, cameras = mock_expand(user_prompt, n_events, n_cameras)

    return JobSpec(
        job_id=job_id,
        user_prompt=user_prompt,
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        n_events=n_events,
        n_cameras=n_cameras,
        backend=backend,
        director=director,
        world_notes=world_notes,
        cameras=cameras,
        events=events,
        variants=_compose_variants(cameras, events, duration_s),
    )


# ---------------------------------------------------------------------------
# Job directory layout
# ---------------------------------------------------------------------------


class JobDirs:
    """data/syngen/{job_id}/ layout — single source of truth for syngen paths."""

    def __init__(self, root: Path, job_id: str):
        self.root = Path(root) / "data" / "syngen" / job_id
        self.job_id = job_id

    @property
    def spec_json(self) -> Path: return self.root / "spec.json"
    @property
    def status_json(self) -> Path: return self.root / "status.json"
    @property
    def videos_dir(self) -> Path: return self.root / "videos"
    def video_mp4(self, variant_id: str) -> Path:
        return self.videos_dir / f"{variant_id}.mp4"
    def video_sidecar(self, variant_id: str) -> Path:
        return self.videos_dir / f"{variant_id}.json"
    @property
    def verification_dir(self) -> Path: return self.root / "verification"
    def verification_json(self, variant_id: str) -> Path:
        return self.verification_dir / f"{variant_id}.json"
    @property
    def ingest_json(self) -> Path: return self.root / "ingest.json"
    @property
    def delivery_dir(self) -> Path: return self.root / "delivery"

    def create(self) -> "JobDirs":
        for d in (self.root, self.videos_dir, self.verification_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def save_spec(self, spec: JobSpec) -> None:
        self.spec_json.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    def load_spec(self) -> JobSpec:
        return JobSpec.model_validate_json(self.spec_json.read_text(encoding="utf-8"))
