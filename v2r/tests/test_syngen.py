"""Tests for the syngen loop: director expansion parsing, verdict merging,
mock generation, and offline verification. All Gemini HTTP calls are mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from v2r.schema.models import FeasibilityRecommendation, FeasibilityReport
from v2r.syngen import gemini
from v2r.syngen.backends import MockBackend
from v2r.syngen.spec import (
    JobDirs,
    expand_request,
    make_job_id,
    mock_expand,
    parse_director_response,
)
from v2r.syngen.verify import TrackBReport, _keyword_skills, merge_verdict, physics_check


# ---------------------------------------------------------------------------
# Director expansion parsing
# ---------------------------------------------------------------------------

DIRECTOR_JSON = {
    "world_notes": "One kitchen, same actor in a blue shirt.",
    "events": [
        {"subject": "person", "motion_style": "slow", "time_of_day": "morning",
         "lighting": "soft window light", "scene": "kitchen",
         "base_prompt": "A person slowly picks up a mug from a counter"},
        {"subject": "person", "motion_style": "brisk", "time_of_day": "dusk",
         "lighting": "warm lamps", "scene": "kitchen",
         "base_prompt": "A person briskly picks up a mug from a counter"},
    ],
    "cameras": [
        {"description": "eye-level front camera", "height_m": 1.6,
         "distance_m": 3.0, "azimuth_deg": 0, "fov_deg": 60},
        {"description": "high corner camera, wide angle", "height_m": 2.8,
         "distance_m": 4.5, "azimuth_deg": 120, "fov_deg": 85},
    ],
}


def test_parse_director_response_plain_json():
    notes, events, cameras = parse_director_response(json.dumps(DIRECTOR_JSON), 2, 2)
    assert notes.startswith("One kitchen")
    assert [e.event_id for e in events] == ["e00", "e01"]
    assert [c.cam_id for c in cameras] == ["cam0", "cam1"]
    assert cameras[1].fov_deg == 85
    assert "mug" in events[0].base_prompt


def test_parse_director_response_markdown_fenced():
    raw = "```json\n" + json.dumps(DIRECTOR_JSON) + "\n```"
    _, events, cameras = parse_director_response(raw, 2, 2)
    assert len(events) == 2 and len(cameras) == 2


def test_parse_director_response_clamps_to_requested_counts():
    _, events, cameras = parse_director_response(json.dumps(DIRECTOR_JSON), 1, 1)
    assert len(events) == 1 and len(cameras) == 1


def test_parse_director_response_rejects_garbage():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_director_response("not json at all", 2, 2)
    with pytest.raises(ValueError):
        parse_director_response(json.dumps({"events": [], "cameras": []}), 2, 2)


def test_expand_request_uses_mocked_gemini(monkeypatch):
    calls = {}

    def fake_generate_content(parts, **kwargs):
        calls["temperature"] = kwargs.get("temperature")
        calls["schema"] = kwargs.get("response_schema")
        return json.dumps(DIRECTOR_JSON)

    monkeypatch.setattr("v2r.syngen.spec.generate_content", fake_generate_content)
    spec = expand_request("person picks up a mug", 2, 2, "job1", "mock",
                          duration_s=3, log=lambda m: None)
    assert spec.director == "gemini"
    assert calls["temperature"] == pytest.approx(0.9)
    assert calls["schema"] is not None
    assert len(spec.variants) == 4  # 2 events x 2 cams
    v = spec.variants[0]
    assert v.variant_id == "e00_cam0"
    assert "eye-level front camera" in v.prompt
    assert v.duration_s == 3


def test_expand_request_falls_back_to_mock_on_api_error(monkeypatch):
    def boom(parts, **kwargs):
        raise gemini.GeminiError("quota exceeded")

    monkeypatch.setattr("v2r.syngen.spec.generate_content", boom)
    spec = expand_request("person waves", 3, 2, "job2", "mock", log=lambda m: None)
    assert spec.director == "mock"
    assert len(spec.events) == 3 and len(spec.cameras) == 2
    assert len(spec.variants) == 6


def test_mock_expand_deterministic():
    a = mock_expand("person waves hello", 2, 2)
    b = mock_expand("person waves hello", 2, 2)
    assert a[1][0].base_prompt == b[1][0].base_prompt
    assert a[2][1].azimuth_deg == b[2][1].azimuth_deg


def test_make_job_id_slug():
    jid = make_job_id("Person picking up objects!")
    assert jid.startswith("person_picking_up_objec")


# ---------------------------------------------------------------------------
# Verdict merging
# ---------------------------------------------------------------------------


def _vlm(rec: str, artifacts=()) -> FeasibilityReport:
    return FeasibilityReport(
        physically_plausible=rec != "reject",
        tracking_likely_valid=rec == "proceed",
        ai_generated_artifacts=list(artifacts),
        confidence=0.8,
        recommendation=FeasibilityRecommendation(rec),
        judge_source="vlm",
    )


def test_merge_verdict_accept_when_both_pass():
    verdict, reasons = merge_verdict(_vlm("proceed"), TrackBReport(physics_ok=True))
    assert verdict == "accept"
    assert reasons == ["both tracks passed"]


def test_merge_verdict_reject_on_vlm_reject():
    verdict, reasons = merge_verdict(
        _vlm("reject", ["limb_morphing"]), TrackBReport(physics_ok=True))
    assert verdict == "reject"
    assert any("limb_morphing" in r for r in reasons)


def test_merge_verdict_reject_on_multiple_physics_violations():
    phys = TrackBReport(physics_ok=False,
                        reasons=["velocity_spike_ratio=0.3 > 0.15",
                                 "scale_jump_ratio=0.4 > 0.20"])
    verdict, _ = merge_verdict(_vlm("proceed"), phys)
    assert verdict == "reject"


def test_merge_verdict_review_on_single_borderline_track():
    phys = TrackBReport(physics_ok=False, reasons=["flow_consistency=0.2 < 0.25"])
    assert merge_verdict(_vlm("proceed"), phys)[0] == "review"
    assert merge_verdict(_vlm("human_review"), TrackBReport())[0] == "review"


def test_merge_verdict_reject_on_undecodable_video():
    phys = TrackBReport(physics_ok=False, reasons=["cannot open video"])
    assert merge_verdict(_vlm("proceed"), phys)[0] == "reject"


def test_keyword_skills_respects_vocab():
    verbs = ["grasp", "lift", "walk", "idle"]
    assert _keyword_skills("person picking up a box", verbs) == ["grasp"]
    assert _keyword_skills("someone walking around", verbs) == ["walk"]
    assert _keyword_skills("nothing matching here", verbs) == ["idle"]


# ---------------------------------------------------------------------------
# Mock backend + Track B on a real (rendered) video
# ---------------------------------------------------------------------------


def test_mock_backend_and_physics_check(tmp_path: Path):
    from v2r.syngen.spec import VariantSpec

    variant = VariantSpec(variant_id="e00_cam0", event_id="e00", cam_id="cam0",
                          prompt="person waves hello", duration_s=2)
    backend = MockBackend(width=320, height=240, fps=15.0)
    dest = tmp_path / "e00_cam0.mp4"
    res = backend.generate(variant, dest, log=lambda m: None)
    assert res.ok and dest.is_file() and dest.stat().st_size > 1000

    rep = physics_check(dest, log=lambda m: None)
    assert rep.n_frames > 10
    assert rep.flow_mean > 0.0  # the silhouette actually moves
    # a smooth deterministic render must not trip the hard physics gates
    assert rep.velocity_spike_ratio <= 0.15
    assert rep.scale_jump_ratio <= 0.5


def test_get_backend_names():
    from v2r.syngen.backends import MockBackend, OmniBackend, VeoBackend, get_backend

    assert isinstance(get_backend("mock"), MockBackend)
    assert isinstance(get_backend("omni"), OmniBackend)
    assert isinstance(get_backend("veo"), VeoBackend)
    with pytest.raises(ValueError):
        get_backend("nope")


def test_get_backend_auto_prefers_omni_with_key(monkeypatch):
    from v2r.syngen.backends import MockBackend, OmniBackend, get_backend

    monkeypatch.setattr("v2r.syngen.gemini.have_api_key", lambda root=None: True)
    assert isinstance(get_backend("auto"), OmniBackend)
    monkeypatch.setattr("v2r.syngen.gemini.have_api_key", lambda root=None: False)
    assert isinstance(get_backend("auto"), MockBackend)


def test_omni_generate_video_parses_interaction_response(monkeypatch):
    interaction = {
        "id": "v1_abc", "status": "completed", "object": "interaction",
        "steps": [
            {"type": "user_input", "content": [{"type": "text", "text": "..."}]},
            {"type": "thought", "content": [{"type": "thought", "text": "..."}]},
            {"type": "model_output", "content": [{
                "type": "video", "mime_type": "video/mp4",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/xyz:download?alt=media",
            }]},
        ],
    }
    captured = {}

    def fake_request(method, url, key, payload=None, timeout=0):
        captured["url"] = url
        captured["payload"] = payload
        return interaction

    monkeypatch.setattr(gemini, "_request", fake_request)
    monkeypatch.setattr(gemini, "get_api_key", lambda root=None: "k")
    uri = gemini.omni_generate_video("a sunset", aspect_ratio="16:9")
    assert uri.endswith("files/xyz:download?alt=media")
    assert captured["url"].endswith("/interactions")
    assert captured["payload"]["model"] == gemini.DEFAULT_OMNI_MODEL
    assert captured["payload"]["response_format"]["type"] == "video"
    assert captured["payload"]["response_format"]["delivery"] == "uri"


def test_omni_generate_video_raises_without_video(monkeypatch):
    monkeypatch.setattr(gemini, "_request",
                        lambda *a, **kw: {"status": "completed", "steps": []})
    monkeypatch.setattr(gemini, "get_api_key", lambda root=None: "k")
    with pytest.raises(gemini.GeminiError):
        gemini.omni_generate_video("a sunset")


def test_job_dirs_layout(tmp_path: Path):
    dirs = JobDirs(tmp_path, "jobx").create()
    assert dirs.videos_dir.is_dir() and dirs.verification_dir.is_dir()
    assert dirs.video_mp4("e00_cam0").name == "e00_cam0.mp4"
    assert dirs.verification_json("e00_cam0").parent.name == "verification"


# ---------------------------------------------------------------------------
# VLM judge fallback path (no API key)
# ---------------------------------------------------------------------------


def test_vlm_judge_offline_fallback(tmp_path: Path, monkeypatch):
    from v2r.syngen.spec import VariantSpec
    from v2r.syngen.verify import vlm_judge

    monkeypatch.setattr("v2r.syngen.verify.have_api_key", lambda: False)
    variant = VariantSpec(variant_id="e00_cam0", event_id="e00", cam_id="cam0",
                          prompt="person waves hello", duration_s=2)
    backend = MockBackend(width=320, height=240, fps=15.0)
    dest = tmp_path / "v.mp4"
    backend.generate(variant, dest, log=lambda m: None)

    report = vlm_judge(dest, variant, log=lambda m: None)
    assert report.judge_source == "rule_based"
    assert report.recommendation in (FeasibilityRecommendation.proceed,
                                     FeasibilityRecommendation.human_review)


def test_vlm_judge_mocked_gemini(tmp_path: Path, monkeypatch):
    from v2r.syngen.spec import VariantSpec
    from v2r.syngen.verify import vlm_judge

    monkeypatch.setattr("v2r.syngen.verify.have_api_key", lambda: True)
    monkeypatch.setattr(
        "v2r.syngen.verify.generate_content",
        lambda parts, **kw: json.dumps({
            "physically_plausible": True, "subject_visible": True,
            "camera_consistent": True, "artifacts": ["temporal_flicker"],
            "confidence": 0.77, "recommendation": "proceed", "notes": "ok",
        }))
    variant = VariantSpec(variant_id="e00_cam0", event_id="e00", cam_id="cam0",
                          prompt="person waves hello", duration_s=2)
    backend = MockBackend(width=320, height=240, fps=15.0)
    dest = tmp_path / "v.mp4"
    backend.generate(variant, dest, log=lambda m: None)

    report = vlm_judge(dest, variant, log=lambda m: None)
    assert report.judge_source == "vlm"
    assert report.confidence == pytest.approx(0.77)
    assert report.ai_generated_artifacts == ["temporal_flicker"]
    assert report.recommendation == FeasibilityRecommendation.proceed
