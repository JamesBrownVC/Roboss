"""Speech transcription for episode videos (Gemini native audio+video).

Why this channel exists: multimodal labels are not just about what the human
DOES — utterances aligned with action segments teach a robot WHAT TO SAY in a
given situation ("tiens, attrape !" while handing an object over). Gemini
processes the video's audio track natively, so no local ASR dependency is
needed; utterances come back timestamped and are aligned to the labeled
segments afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..schema.models import SegmentsFile, SourceTag, Utterance, UtterancesFile
from ..syngen import gemini

UTTERANCES_SCHEMA = {
    "type": "object",
    "properties": {
        "has_speech": {"type": "boolean"},
        "audio_notes": {"type": "string"},
        "utterances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t_start_s": {"type": "number"},
                    "t_end_s": {"type": "number"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                    "language": {"type": "string"},
                    "intent": {"type": "string",
                               "enum": ["greeting", "instruction", "comment",
                                        "response", "other"]},
                },
                "required": ["t_start_s", "t_end_s", "speaker", "text",
                             "language", "intent"],
            },
        },
    },
    "required": ["has_speech", "audio_notes", "utterances"],
}

_PROMPT = (
    "Listen to this video's AUDIO track and transcribe every spoken utterance "
    "with timestamps (seconds), speaker id (person_0, person_1, ... or "
    "'offscreen'), verbatim text in the original language, the language code, "
    "and the pragmatic intent (greeting/instruction/comment/response/other). "
    "If there is no speech, return has_speech=false and describe the audio in "
    "audio_notes (music, ambience, silence, no audio track). Never invent "
    "speech that is not clearly audible."
)


def transcribe_video(video: Path, api_key: Optional[str] = None,
                     model: Optional[str] = None) -> UtterancesFile:
    raw = gemini.analyze_video(
        Path(video), prompt=_PROMPT, response_schema=UTTERANCES_SCHEMA,
        model=model or gemini.DEFAULT_VISION_MODEL, api_key=api_key)
    data = gemini.extract_json(raw)
    utts = [Utterance(
        t_start_s=float(u["t_start_s"]), t_end_s=float(max(u["t_end_s"], u["t_start_s"])),
        speaker=str(u.get("speaker", "person_0"))[:40],
        text=str(u["text"])[:500],
        language=str(u.get("language", ""))[:16],
        intent=u.get("intent", "other"),
        conf=0.7,  # ASR via VLM: good but unverified against a reference
    ) for u in data.get("utterances", [])]
    return UtterancesFile(
        utterances=utts,
        has_speech=bool(data.get("has_speech", bool(utts))),
        audio_notes=str(data.get("audio_notes", ""))[:300],
        method="gemini native audio+video transcription",
        source=SourceTag.estimated,
    )


def align_with_segments(uf: UtterancesFile, segments_json: Path) -> UtterancesFile:
    """Attach the co-occurring action segment's skill to each utterance —
    the (situation, utterance) pair is the robot-speech training signal."""
    if not segments_json.is_file():
        return uf
    try:
        segs = SegmentsFile.model_validate_json(
            segments_json.read_text(encoding="utf-8")).segments
    except Exception:
        return uf
    for u in uf.utterances:
        mid = 0.5 * (u.t_start_s + u.t_end_s)
        for s in segs:
            if s.start_s <= mid <= s.end_s:
                u.aligned_segment = s.skill
                break
    return uf


def transcribe_to_workspace(ws, cfg, api_key: Optional[str] = None) -> dict:
    """Transcribe ws.video_path -> semantics/utterances.json. Returns a
    JSON-safe summary for stage metrics / agent observations."""
    import os

    if os.environ.get("V2R_DISABLE_TRANSCRIBE"):
        return {"available": False, "reason": "disabled via V2R_DISABLE_TRANSCRIBE"}
    # pointless when ingest re-encoded through cv2 (audio track dropped)
    ingest_manifest = ws.manifest_path("ingest")
    if ingest_manifest.is_file():
        try:
            notes = json.loads(ingest_manifest.read_text(encoding="utf-8"))\
                .get("metrics", {}).get("notes", "")
            if "AUDIO DROPPED" in notes:
                return {"available": False,
                        "reason": "ingest re-encode dropped the audio track "
                                  "(cv2); use real-mode ffmpeg or the agentic "
                                  "labeler path"}
        except Exception:  # noqa: BLE001
            pass
    key = api_key or gemini.get_api_key(cfg.root)
    if not key:
        return {"available": False, "reason": "GEMINI_API_KEY not set"}
    uf = transcribe_video(ws.video_path, api_key=key)
    uf = align_with_segments(uf, ws.segments_json)
    ws.utterances_json.parent.mkdir(parents=True, exist_ok=True)
    ws.utterances_json.write_text(uf.model_dump_json(indent=2), encoding="utf-8")
    return {
        "available": True,
        "artifact": ws.rel(ws.utterances_json),
        "has_speech": uf.has_speech,
        "n_utterances": len(uf.utterances),
        "languages": sorted({u.language for u in uf.utterances if u.language}),
        "audio_notes": uf.audio_notes,
        "sample": [{"t": round(u.t_start_s, 1), "speaker": u.speaker,
                    "text": u.text[:80], "intent": u.intent,
                    "during": u.aligned_segment}
                   for u in uf.utterances[:5]],
    }
