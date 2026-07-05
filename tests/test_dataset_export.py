"""Tests for the dataset ZIP export endpoint logic (server/app.py).

No HTTP, no models: build_dataset_archive is exercised directly against a
fabricated generated/ tree and an in-memory batch.
"""

import io
import json
import zipfile
from pathlib import Path

import server.app as app_module
from server.batches import BatchState, JobState, _batches

BID = "testbatch1234"


def _make_batch_tree(root: Path) -> None:
    bundle = root / BID / "bundle"
    (bundle / "frames").mkdir(parents=True)
    (bundle / "contract.json").write_text(
        json.dumps({"world_contract": {"world_id": "w1"}}), encoding="utf-8")
    (bundle / "canvas.png").write_bytes(b"png-bytes")
    (bundle / "frames" / "sc_1_start.png").write_bytes(b"frame-bytes")
    (bundle / "scenarios.json").write_text(json.dumps({"scenarios": [
        {"scenario_id": "sc_1", "title": "Slip",
         "expected_labels": ["human_slip"]},
    ]}), encoding="utf-8")
    (bundle / "intent.json").write_text(
        json.dumps({"raw_intention": "warehouse safety"}), encoding="utf-8")

    job_dir = root / BID / f"{BID}-job-1"
    job_dir.mkdir(parents=True)
    (job_dir / "sc_1.mp4").write_bytes(b"raw-video")
    (job_dir / "sc_1_labeled.mp4").write_bytes(b"labeled-video")
    (job_dir / "sc_1_report.json").write_text(json.dumps({
        "decision": "accept", "plausibility_score": 0.93,
        "plausible": True, "video_id": "sc_1",
    }), encoding="utf-8")
    (job_dir / "sc_1_labels.json").write_text(json.dumps({
        "labels": ["human_slip"], "frames": [],
    }), encoding="utf-8")


def _register_batch() -> None:
    job = JobState(
        id=f"{BID}-job-1", index=1, status="completed",
        videoUrl=f"/generated/{BID}/{BID}-job-1/sc_1.mp4",
        labeledVideoUrl=f"/generated/{BID}/{BID}-job-1/sc_1_labeled.mp4",
        reviewStatus="passed", labelStatus="completed",
        label={"labels": ["stale_in_memory_copy"]},
        scenario_id="sc_1",
    )
    batch = BatchState(id=BID, status="completed", count=1,
                       aspect_ratio="16:9", prompt="warehouse safety",
                       jobs=[job], completed=1)
    _batches[BID] = batch


def _names(archive_bytes: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        return set(z.namelist())


def _read_member(archive_bytes: bytes, name: str):
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        return z.read(name)


def test_dataset_archive_for_live_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "GENERATED_DIR", tmp_path)
    _make_batch_tree(tmp_path)
    _register_batch()
    try:
        content = app_module.build_dataset_archive(BID)
    finally:
        _batches.pop(BID, None)

    assert content is not None
    names = _names(content)
    assert {"dataset.json", "manifest.json", "world/contract.json",
            "world/canvas.png"} <= names
    assert {"samples/sc_1/video.mp4", "samples/sc_1/labeled_preview.mp4",
            "samples/sc_1/labels.json", "samples/sc_1/report.json",
            "samples/sc_1/scenario.json",
            "samples/sc_1/start_frame.png"} <= names

    index = json.loads(_read_member(content, "dataset.json"))
    assert index["batch_id"] == BID
    assert index["prompt"] == "warehouse safety"
    assert index["counts"] == {"samples": 1, "accepted": 1, "rejected": 0}
    sample = index["samples"][0]
    assert sample["scenario_id"] == "sc_1"
    assert sample["accepted"] is True
    assert sample["plausibility_score"] == 0.93

    # disk labels (fresh) must win over the stale in-memory copy
    labels = json.loads(_read_member(content, "samples/sc_1/labels.json"))
    assert labels["labels"] == ["human_slip"]


def test_dataset_archive_for_restored_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "GENERATED_DIR", tmp_path)
    _make_batch_tree(tmp_path)
    _batches.pop(BID, None)  # not in memory -> restored from disk

    content = app_module.build_dataset_archive(BID)
    assert content is not None
    names = _names(content)
    assert "samples/sc_1/video.mp4" in names
    assert "dataset.json" in names
    index = json.loads(_read_member(content, "dataset.json"))
    assert index["counts"]["samples"] == 1


def test_dataset_archive_missing_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "GENERATED_DIR", tmp_path)
    assert app_module.build_dataset_archive("nope") is None
