from pathlib import Path

from roboss.pipeline import VideoPipelineResult, run_e2e_pipeline


class DummySettings:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.start_frame_workers = 2
        self.video_workers = 3


def test_e2e_parallel_results_keep_scenario_order(tmp_path, monkeypatch):
    scenarios = [
        {
            "scenario_id": "sc_01",
            "title": "one",
            "video_prompt": "prompt one",
            "verifier_packet": {"scenario_prompt": "one"},
        },
        {
            "scenario_id": "sc_02",
            "title": "two",
            "video_prompt": "prompt two",
            "verifier_packet": {"scenario_prompt": "two"},
        },
        {
            "scenario_id": "sc_03",
            "title": "three",
            "video_prompt": "prompt three",
            "verifier_packet": {"scenario_prompt": "three"},
        },
    ]

    def fake_compile_scenarios(*, outdir, **kwargs):
        out = Path(outdir)
        out.mkdir(parents=True)
        (out / "scenarios.json").write_text(
            '{"scenarios": ['
            '{"scenario_id": "sc_01", "title": "one", "video_prompt": "prompt one", "verifier_packet": {"scenario_prompt": "one"}},'
            '{"scenario_id": "sc_02", "title": "two", "video_prompt": "prompt two", "verifier_packet": {"scenario_prompt": "two"}},'
            '{"scenario_id": "sc_03", "title": "three", "video_prompt": "prompt three", "verifier_packet": {"scenario_prompt": "three"}}'
            ']}',
            encoding="utf-8",
        )
        return {
            "outdir": str(out),
            "scenarios": [s["scenario_id"] for s in scenarios],
            "parallelism": {"start_frame_workers": 2},
        }

    def fake_run_video_pipeline(*, outdir, scenario, **kwargs):
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        sid = scenario["scenario_id"]
        report = {
            "decision": "accept" if sid != "sc_02" else "reject",
            "plausibility_score": 1.0 if sid != "sc_02" else 0.4,
        }
        return VideoPipelineResult(
            outdir=out,
            video_path=out / "generated.mp4",
            report_path=out / "report.json",
            labels_path=None,
            report=report,
        )

    monkeypatch.setattr("roboss.pipeline.get_settings",
                        lambda: DummySettings(tmp_path))
    monkeypatch.setattr("roboss.storage.get_settings",
                        lambda: DummySettings(tmp_path))
    monkeypatch.setattr("roboss.pipeline.compile_scenarios",
                        fake_compile_scenarios)
    monkeypatch.setattr("roboss.pipeline.run_video_pipeline",
                        fake_run_video_pipeline)

    summary = run_e2e_pipeline(
        "make scenarios",
        count=3,
        run_name="parallel_test",
        video_workers=3,
        require_acceptance=True,
        label=False,
        gate2=False,
        progress=lambda *_: None,
    )

    assert [r["scenario_id"] for r in summary["results"]] == [
        "sc_01",
        "sc_02",
        "sc_03",
    ]
    assert summary["batch_decision"] == "reject"
    assert summary["parallelism"]["video_workers"] == 3
    assert [r["scenario_id"] for r in summary["accepted"]] == ["sc_01", "sc_03"]
    assert [r["scenario_id"] for r in summary["rejected"]] == ["sc_02"]
