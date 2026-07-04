"""Pipeline orchestrator: intention -> validated scenario bundle on disk.

    intent -> contract -> scenarios -> validate/repair loop -> compile
           -> (optional) canvas image -> (optional) start frames
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .canvas import generate_canvas, generate_start_frame
from .compiler import compile_scenario
from .config import AgentConfig
from .contract import build_contract
from .intent import parse_intent
from .llm import AgentError
from .scenarios import plan_scenarios, repair_scenarios
from .validator import validate_all


@dataclass
class PipelineResult:
    out_dir: str
    intent: dict
    contract: dict
    scenarios: list[dict]                  # compiled, valid only
    dropped: dict[str, list[str]] = field(default_factory=dict)
    canvas_path: str | None = None
    start_frame_paths: dict[str, str] = field(default_factory=dict)
    canvas_error: str | None = None


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def run_pipeline(intention: str,
                 out_dir: str,
                 cfg: AgentConfig,
                 count: int | None = None,
                 make_canvas: bool = False,
                 make_start_frames: bool = False,
                 start_frame_workers: int = 1,
                 progress=print) -> PipelineResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    progress("[1/5] Parsing intent ...")
    intent = parse_intent(intention, cfg, count_override=count)
    _write_json(out / "intent.json", intent)

    progress("[2/5] Building world contract ...")
    contract = build_contract(intent, cfg)
    _write_json(out / "contract.json", contract)
    n_entities = len(contract["world_contract"]["locked_entities"])
    progress(f"      world '{contract['world_contract']['world_id']}', "
             f"{n_entities} locked entities")

    progress(f"[3/5] Planning {intent['variation_count']} scenario "
             f"variations ...")
    scenarios = plan_scenarios(contract, intent["variation_count"], cfg)

    # deterministic validation with LLM repair loop
    for round_no in range(cfg.max_repair_rounds + 1):
        results = validate_all(contract, scenarios)
        invalid_ids = [sid for sid, errs in results.items() if errs]
        if not invalid_ids:
            break
        if round_no == cfg.max_repair_rounds:
            progress(f"      dropping {len(invalid_ids)} scenario(s) that "
                     f"still violate the contract: {invalid_ids}")
            break
        progress(f"      {len(invalid_ids)} scenario(s) violate the "
                 f"contract, repair round {round_no + 1} ...")
        invalid = [s for s in scenarios
                   if str(s.get("scenario_id")) in invalid_ids]
        errors = {sid: results[sid] for sid in invalid_ids}
        try:
            fixed = {str(s.get("scenario_id")): s
                     for s in repair_scenarios(contract, invalid, errors, cfg)}
        except AgentError as e:
            progress(f"      repair call failed ({e}); keeping originals")
            break
        scenarios = [fixed.get(str(s.get("scenario_id")), s)
                     for s in scenarios]

    results = validate_all(contract, scenarios)
    dropped = {sid: errs for sid, errs in results.items() if errs}
    valid = [s for s in scenarios
             if not results.get(str(s.get("scenario_id")), ["missing"])]

    progress(f"[4/5] Compiling prompts, keyframes and verifier packets for "
             f"{len(valid)} scenario(s) ...")
    compiled = [compile_scenario(contract, s) for s in valid]

    result = PipelineResult(out_dir=str(out), intent=intent,
                            contract=contract, scenarios=compiled,
                            dropped=dropped)

    if make_canvas or make_start_frames:
        progress(f"[5/5] Generating canvas anchor ({cfg.image_model}) ...")
        try:
            canvas = generate_canvas(contract, cfg)
        except AgentError as e:
            canvas, result.canvas_error = None, str(e)
        if canvas is None:
            result.canvas_error = result.canvas_error or "no image returned"
            progress(f"      canvas skipped: {result.canvas_error}")
        else:
            canvas_path = out / "canvas.png"
            canvas_path.write_bytes(canvas)
            result.canvas_path = str(canvas_path)
            if make_start_frames:
                frames_dir = out / "frames"
                frames_dir.mkdir(exist_ok=True)
                workers = max(1, int(start_frame_workers or 1))

                def _make_frame(sc: dict) -> tuple[str, bytes | None, str | None]:
                    sid = sc["scenario_id"]
                    try:
                        frame = generate_start_frame(canvas, sc, cfg)
                    except AgentError as e:
                        return sid, None, str(e)
                    return sid, frame, None

                frame_results: dict[str, bytes] = {}
                if workers == 1 or len(compiled) <= 1:
                    for sc in compiled:
                        sid, frame, err = _make_frame(sc)
                        if err:
                            progress(f"      start frame {sid} failed: {err}")
                        elif frame:
                            frame_results[sid] = frame
                else:
                    progress(f"      generating start frames with {workers} worker(s)")
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = [executor.submit(_make_frame, sc)
                                   for sc in compiled]
                        for future in as_completed(futures):
                            sid, frame, err = future.result()
                            if err:
                                progress(f"      start frame {sid} failed: {err}")
                            elif frame:
                                frame_results[sid] = frame

                for sc in compiled:
                    sid = sc["scenario_id"]
                    frame = frame_results.get(sid)
                    if frame:
                        p = frames_dir / f"{sid}_start.png"
                        p.write_bytes(frame)
                        result.start_frame_paths[sid] = str(p)
    else:
        progress("[5/5] Visual anchors skipped (use --canvas / "
                 "--start-frames)")

    # scenario files: one list + one verifier packet per scenario
    _write_json(out / "scenarios.json", {"scenarios": compiled})
    packets_dir = out / "verifier_packets"
    packets_dir.mkdir(exist_ok=True)
    for sc in compiled:
        _write_json(packets_dir / f"{sc['scenario_id']}.json",
                    sc["verifier_packet"])

    _write_json(out / "bundle.json", {
        "intention": intention,
        "world_id": contract["world_contract"]["world_id"],
        "scene_id": contract["scene_registry"]["scene_id"],
        "locked_entities": contract["world_contract"]["entity_ids"],
        "reference_assets_planned": len(contract["reference_assets"]),
        "identity_checks": [c["check_id"] for c in contract["identity_checks"]],
        "scenarios_valid": [s["scenario_id"] for s in compiled],
        "scenarios_dropped": dropped,
        "canvas": result.canvas_path,
        "canvas_error": result.canvas_error,
        "start_frames": result.start_frame_paths,
        "files": {
            "intent": "intent.json",
            "contract": "contract.json",
            "scenarios": "scenarios.json",
            "verifier_packets": "verifier_packets/",
        },
    })
    return result
