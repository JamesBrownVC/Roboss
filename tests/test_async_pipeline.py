"""Async orchestration tests for roboss.pipeline.

All model/network stages are stubbed via the module-level indirection
(_generate_video, _label_video, verify_video, compile_scenarios), so these
tests exercise only the concurrency plumbing: scenario chains must actually
overlap, and per-stage semaphores must cap concurrency.
"""

import json
import threading
import time
from pathlib import Path

import roboss.pipeline as rp

N = 4


def _fake_compile(n):
    def fake_compile(intention, outdir, count, start_frames, deterministic,
                     start_frame_workers, progress):
        bundle = Path(outdir)
        bundle.mkdir(parents=True, exist_ok=True)
        scenarios = [
            {"scenario_id": f"sc_{i}", "title": f"scenario {i}",
             "video_prompt": f"scenario {i} prompt", "verifier_packet": {
                 "scenario_prompt": f"scenario {i} prompt"}}
            for i in range(n)
        ]
        (bundle / "scenarios.json").write_text(
            json.dumps({"scenarios": scenarios}), encoding="utf-8")
        return {"outdir": str(bundle), "deterministic": True,
                "parallelism": {"start_frame_workers": 0}}
    return fake_compile


class _StageCounter:
    """Tracks how many stage calls run at the same moment."""

    def __init__(self):
        self.now = 0
        self.peak = 0
        self.total = 0
        self.lock = threading.Lock()

    def __enter__(self):
        with self.lock:
            self.now += 1
            self.total += 1
            self.peak = max(self.peak, self.now)

    def __exit__(self, *exc):
        with self.lock:
            self.now -= 1


def test_e2e_chains_run_concurrently_with_stage_caps(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOSS_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("ROBOSS_GEN_WORKERS", "2")
    monkeypatch.setenv("ROBOSS_VERIFY_WORKERS", "2")
    monkeypatch.setenv("ROBOSS_LABEL_WORKERS", "2")

    gen, ver, lab = _StageCounter(), _StageCounter(), _StageCounter()

    def fake_generate(prompt):
        with gen:
            time.sleep(0.15)
        return b"video-bytes"

    def fake_verify(video_path, scenario, report_path, *a, **k):
        with ver:
            time.sleep(0.1)
        return {"decision": "accept", "plausibility_score": 1.0}

    def fake_label(video_bytes):
        with lab:
            time.sleep(0.1)
        return {"labels": [], "summary": {}}

    monkeypatch.setattr(rp, "compile_scenarios", _fake_compile(N))
    monkeypatch.setattr(rp, "_generate_video", fake_generate)
    monkeypatch.setattr(rp, "verify_video", fake_verify)
    monkeypatch.setattr(rp, "_label_video", fake_label)

    started = time.monotonic()
    summary = rp.run_e2e_pipeline(
        intention="test", count=N, run_name="async_test",
        gate2=False, label=True, start_frames=False,
        progress=lambda *_: None)
    elapsed = time.monotonic() - started

    assert summary["batch_decision"] == "accept"
    assert [r["status"] for r in summary["results"]] == ["accept"] * N
    assert gen.total == ver.total == lab.total == N

    # chains genuinely overlapped, and no stage exceeded its semaphore
    assert gen.peak == 2, "generation should run 2-wide (cap ROBOSS_GEN_WORKERS)"
    assert ver.peak <= 2 and lab.peak <= 2
    serial_time = N * (0.15 + 0.1 + 0.1)
    assert elapsed < serial_time, "async pipeline must beat serial execution"

    # artifacts: per-scenario dirs + batch summary on disk
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "summary.json").is_file()
    for i in range(N):
        assert (run_dir / f"sc_{i}" / "generated.mp4").is_file()
        assert (run_dir / f"sc_{i}" / "labels.json").is_file()


def test_one_failing_chain_does_not_kill_the_batch(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOSS_RUNS_DIR", str(tmp_path))

    def flaky_generate(prompt):
        if "scenario 1" in prompt or prompt == "scenario 1":
            raise RuntimeError("generation exploded")
        return b"video-bytes"

    monkeypatch.setattr(rp, "compile_scenarios", _fake_compile(3))
    monkeypatch.setattr(rp, "_generate_video", flaky_generate)
    monkeypatch.setattr(
        rp, "verify_video",
        lambda *a, **k: {"decision": "accept", "plausibility_score": 1.0})
    monkeypatch.setattr(rp, "_label_video",
                        lambda b: {"labels": [], "summary": {}})

    summary = rp.run_e2e_pipeline(
        intention="test", count=3, run_name="flaky_test",
        gate2=False, label=True, start_frames=False,
        progress=lambda *_: None)

    statuses = {r["scenario_id"]: r["status"] for r in summary["results"]}
    assert statuses["sc_1"] == "error"
    assert statuses["sc_0"] == statuses["sc_2"] == "accept"
    assert summary["batch_decision"] == "reject"  # require_acceptance=True
    assert (Path(summary["run_dir"]) / "sc_1" / "error.json").is_file()
