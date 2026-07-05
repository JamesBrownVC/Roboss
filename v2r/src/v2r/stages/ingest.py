"""Ingest stage (master prompt 6.A): normalize input, protect PII, register
consent.

real mode (Ubuntu+CUDA production host) shells out to pinned tools:

    ffmpeg          system binary (pin the distro build in envs/ingest/)
                    transcode -> 30 fps, yuv420p, h264
    PySceneDetect   pip 'scenedetect[opencv]'  PIN: <PIN_ME_SCENEDETECT_VERSION>
                    shot segmentation -> probe.shots
    deface          pip 'deface'               PIN: <PIN_ME_DEFACE_VERSION>
                    face/plate blur (EgoBlur is an allowed swap-in, same contract)

Guard: if ffmpeg is missing in real mode the stage falls back to the cv2 path
used by synthetic mode and records tool='cv2-fallback' plus an explicit note in
the metrics (mp4v codec, no audio, scene/blur tools skipped).

synthetic mode (any host): cv2-only normalization of the operator-supplied
source video. Deterministic (pure nearest-neighbor resampling, no randomness).

Gate (config/qa.yaml `ingest`): min height, min fps, duration window, max
corrupt frames. Gate failure returns StageStatus.rejected (funnel outcome).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..schema import io as sio
from ..schema.models import ConsentRecord, StageStatus, VideoProbe
from ..schema.timeline import canonical_timestamps
from .base import (
    Stage,
    StageContext,
    StageResult,
    gate_from_thresholds,
    register_stage,
    run_tool,
)

TOOL_REAL = "ffmpeg+scenedetect+deface"
TOOL_FALLBACK = "cv2-fallback"
TOOL_SYNTH = "cv2"


def _dev_consent() -> ConsentRecord:
    return ConsentRecord(
        consent_id="dev-consent-0001",
        license="research-only-dev",
        subject_ids=["synthetic"],
        blur_applied=True,
        notes="dev harness default consent; replace with operator ledger in production",
    )


# ---------------------------------------------------------------------------
# probing / cv2 normalization helpers
# ---------------------------------------------------------------------------


def _fourcc_str(fcc: int) -> str:
    chars = [chr((fcc >> (8 * i)) & 0xFF) for i in range(4)]
    if fcc and all(32 <= ord(c) <= 126 for c in chars):
        return "".join(chars).strip()
    return ""


def _probe_source(src: Path) -> VideoProbe:
    """Decode-scan the source: width/height/fps, actual frame count, corrupt
    frames = container-declared frames that failed to decode."""
    cap = cv2.VideoCapture(str(src))
    width = height = 0
    fps = 0.0
    declared = 0
    codec = ""
    frames_read = 0
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = _fourcc_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frames_read += 1
    cap.release()
    corrupt = max(0, declared - frames_read) if declared > 0 else 0
    duration = frames_read / fps if fps > 0 else 0.0
    return VideoProbe(
        width=width,
        height=height,
        fps=fps,
        n_frames=frames_read,
        duration_s=duration,
        codec=codec,
        pix_fmt="",
        corrupt_frames=corrupt,
        original_path=str(src),
        shots=[(0.0, duration)],
    )


def _reencode_cv2(src: Path, dst: Path, probe: VideoProbe, hz: float) -> int:
    """Re-encode to `hz` fps (mp4v), nearest-neighbor frame resampling, same
    resolution. Output frame count matches canonical_timestamps(duration, hz)
    so downstream per-frame artifacts (poses, depth) align 1:1 with video."""
    t_out = canonical_timestamps(probe.duration_s, hz)
    src_idx = np.minimum(
        np.round(t_out * probe.fps).astype(np.int64), max(probe.n_frames - 1, 0)
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    vw = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), float(hz),
        (probe.width, probe.height),
    )
    if not vw.isOpened():
        cap.release()
        raise IOError(f"cv2.VideoWriter failed to open {dst}")
    cur = -1
    buf: Optional[np.ndarray] = None
    for target in src_idx:
        while cur < target:
            ret, frame = cap.read()
            cur += 1
            if ret:
                buf = frame
        if buf is None:  # every read failed so far: emit a black frame
            buf = np.zeros((probe.height, probe.width, 3), dtype=np.uint8)
        vw.write(buf)
    vw.release()
    cap.release()
    return len(t_out)


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


@register_stage
class Ingest(Stage):
    name = "ingest"

    def run(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        if ctx.source_video is None:
            raise ValueError("ingest requires ctx.source_video")
        src = Path(ctx.source_video)
        if not src.is_file():
            raise FileNotFoundError(f"source video not found: {src}")
        ws.raw_dir.mkdir(parents=True, exist_ok=True)

        notes: list[str] = []
        ctx.log(f"[ingest] probing {src.name}")
        probe = _probe_source(src)
        sio.write_json_model(ws.probe_path, probe)
        if not ws.consent_path.exists():
            sio.write_json_model(ws.consent_path, _dev_consent())

        metrics: dict = {
            "width": probe.width,
            "height": probe.height,
            "fps": probe.fps,
            "duration_s": probe.duration_s,
            "n_frames": probe.n_frames,
            "corrupt_frames": probe.corrupt_frames,
        }
        qa = ctx.cfg.qa.get("ingest", {})
        gate = gate_from_thresholds(metrics, [
            ("height", "ge", float(qa.get("min_height", 720)), True),
            ("fps", "ge", float(qa.get("min_fps", 15.0)), True),
            ("duration_s", "ge", float(qa.get("min_duration_s", 3.0)), True),
            ("duration_s", "le", float(qa.get("max_duration_s", 900.0)), True),
            ("corrupt_frames", "le", float(qa.get("max_corrupt_frames", 0)), True),
        ])
        outputs = [ws.rel(ws.probe_path), ws.rel(ws.consent_path)]
        if not gate.passed:
            return StageResult(
                status=StageStatus.rejected,
                metrics=metrics,
                failure_reason="; ".join(gate.reasons),
                outputs=outputs,
                gate=gate,
                tool=TOOL_REAL if ctx.mode == "real" else TOOL_SYNTH,
            )

        hz = ctx.cfg.pipeline.canonical_hz
        if ctx.mode == "real":
            tool, n_out, shots = self._normalize_real(ctx, src, probe, hz, notes)
            if shots is not None:
                probe.shots = shots
                sio.write_json_model(ws.probe_path, probe)
        elif abs(probe.fps - hz) <= 0.6 and src.suffix.lower() == ".mp4":
            # already at canonical rate: copy the container through UNCHANGED.
            # Critically this PRESERVES THE AUDIO TRACK (cv2 re-encode drops
            # it), which the transcription/utterance channel needs.
            import shutil

            ctx.log("[ingest] source already ~30 Hz mp4: copying through (audio preserved)")
            shutil.copy2(src, ws.video_path)
            n_out = probe.n_frames
            notes.append("copied through unchanged; audio track preserved")
            tool = TOOL_SYNTH
        else:
            ctx.log("[ingest] re-encoding to 30 Hz (cv2, synthetic mode)")
            n_out = _reencode_cv2(src, ws.video_path, probe, hz)
            notes.append("cv2 mp4v encoder (synthetic mode); h264 requires ffmpeg "
                         "(real mode); AUDIO DROPPED by cv2 - transcription needs "
                         "the agentic path or real-mode ffmpeg")
            tool = TOOL_SYNTH

        metrics["out_n_frames"] = n_out
        if notes:
            metrics["notes"] = "; ".join(notes)
        outputs.insert(0, ws.rel(ws.video_path))
        return StageResult(
            status=StageStatus.success,
            metrics=metrics,
            outputs=outputs,
            gate=gate,
            tool=tool,
        )

    # ------------------------------------------------------------------
    # real mode: ffmpeg + PySceneDetect + deface, each individually guarded
    # ------------------------------------------------------------------
    def _normalize_real(
        self,
        ctx: StageContext,
        src: Path,
        probe: VideoProbe,
        hz: float,
        notes: list[str],
    ) -> tuple[str, int, Optional[list[tuple[float, float]]]]:
        """Returns (tool, out_n_frames, shots-or-None)."""
        ws = ctx.ws
        env = ctx.cfg.stage(self.name).env  # null: ingest tools live host-side
        tmp = ws.raw_dir / "_ingest_tmp.mp4"

        # 1) ffmpeg transcode: 30 fps, yuv420p, h264. Audio dropped for parity
        #    with the cv2 fallback; multi-view sync reads audio from raw/cams/.
        ffmpeg_err = ""
        try:
            proc = run_tool(
                [
                    "ffmpeg", "-y", "-i", str(src),
                    "-vf", f"fps={hz:g}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an",
                    str(tmp),
                ],
                env_name=env,
            )
            ffmpeg_ok = proc.returncode == 0 and tmp.is_file()
            if not ffmpeg_ok:
                ffmpeg_err = (proc.stderr or proc.stdout or "")[-300:]
        except (FileNotFoundError, OSError) as e:
            ffmpeg_ok = False
            ffmpeg_err = str(e)

        if not ffmpeg_ok:
            ctx.log("[ingest] ffmpeg unavailable -> cv2 fallback")
            notes.append(
                "ffmpeg unavailable (" + (ffmpeg_err.strip() or "not found") + "); "
                "cv2-fallback used: mp4v codec, no audio, scene/blur tools skipped"
            )
            n_out = _reencode_cv2(src, ws.video_path, probe, hz)
            return TOOL_FALLBACK, n_out, None

        # 2) PySceneDetect shot segmentation (guarded; default = single shot).
        shots = self._detect_shots(tmp, env, probe.duration_s, notes)

        # 3) Blur (deface). On failure keep the unblurred transcode but say so.
        blurred = ws.raw_dir / "_ingest_blur.mp4"
        try:
            proc = run_tool(["deface", str(tmp), "-o", str(blurred)], env_name=env)
            if proc.returncode == 0 and blurred.is_file():
                blurred.replace(ws.video_path)
                tmp.unlink(missing_ok=True)
            else:
                raise RuntimeError((proc.stderr or "")[-300:])
        except (FileNotFoundError, OSError, RuntimeError) as e:
            notes.append(
                "deface blur failed (" + (str(e).strip() or "not found") + "); video NOT blurred"
            )
            tmp.replace(ws.video_path)

        cap = cv2.VideoCapture(str(ws.video_path))
        n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        return TOOL_REAL, n_out, shots

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_shots(
        video: Path, env: Optional[str], duration_s: float, notes: list[str]
    ) -> list[tuple[float, float]]:
        """PySceneDetect CLI -> [(start_s, end_s)]; single shot on any failure."""
        default = [(0.0, duration_s)]
        try:
            proc = run_tool(
                [
                    "scenedetect", "-i", str(video),
                    "detect-adaptive", "list-scenes", "-o", str(video.parent),
                ],
                env_name=env,
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or "")[-300:])
            csv_path = video.parent / (video.stem + "-Scenes.csv")
            if not csv_path.is_file():
                raise FileNotFoundError(str(csv_path))
            import pandas as pd

            df = pd.read_csv(csv_path, skiprows=1)  # row 0 is the timecode list
            shots = [
                (float(r["Start Time (seconds)"]), float(r["End Time (seconds)"]))
                for _, r in df.iterrows()
            ]
            return shots or default
        except Exception as e:  # optional tool: degrade, never fail the stage
            notes.append(
                "scenedetect unavailable (" + (str(e).strip() or "not found")
                + "); single shot assumed"
            )
            return default
