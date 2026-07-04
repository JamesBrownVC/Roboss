"""FastAPI application for the Roboss pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import compile_scenarios, run_e2e_pipeline, run_video_pipeline
from .schemas import (
    CompileRequest,
    E2ERequest,
    PipelineResponse,
    VideoPipelineRequest,
)
from .settings import get_settings
from .storage import get_storage

app = FastAPI(
    title="Roboss Synthetic Action Dataset Compiler",
    version="0.1.0",
)

_storage = get_storage()
app.mount("/assets", StaticFiles(directory=str(_storage.root)), name="assets")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "gemini_configured": bool(settings.gemini_api_key),
        "runs_dir": str(settings.runs_dir),
    }


@app.get("/runs", tags=["storage"])
def list_runs() -> dict:
    storage = get_storage()
    runs = []
    for path in sorted(p for p in storage.root.iterdir() if p.is_dir()):
        manifest_path = path / "manifest.json"
        item = {
            "run_id": path.relative_to(storage.root).as_posix(),
            "url": storage.url_for(path) if path.exists() else None,
            "manifest_url": storage.url_for(manifest_path)
            if manifest_path.exists() else None,
        }
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(
                    encoding="utf-8"))
                item["kind"] = manifest.get("kind")
                item["decision"] = manifest.get("decision")
                item["plausibility_score"] = manifest.get("plausibility_score")
            except json.JSONDecodeError:
                item["manifest_error"] = "invalid_json"
        runs.append(item)
    return {"runs": runs}


@app.get("/runs/{run_id:path}", tags=["storage"])
def get_run(run_id: str) -> dict:
    storage = get_storage()
    try:
        run_path = storage.resolve(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not run_path.exists() or not run_path.is_dir():
        raise HTTPException(status_code=404, detail="run not found")

    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500,
                                detail="manifest is invalid JSON") from exc
    else:
        manifest = {
            "run_id": storage.relative(run_path),
            "root": str(run_path),
            "files": storage.collect_files(run_path),
        }
    manifest["asset_base_url"] = f"/assets/{storage.relative(run_path)}"
    return manifest


@app.post("/scenario-bundles", response_model=PipelineResponse,
          tags=["scenario planning"])
def create_scenario_bundle(req: CompileRequest) -> PipelineResponse:
    try:
        data = compile_scenarios(
            intention=req.intention,
            outdir=req.outdir,
            count=req.count,
            start_frames=req.start_frames,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PipelineResponse(status="ok", outdir=data["outdir"], data=data)


@app.post("/verified-videos", response_model=PipelineResponse,
          tags=["video generation"])
def create_verified_video(req: VideoPipelineRequest) -> PipelineResponse:
    try:
        result = run_video_pipeline(
            prompt=req.prompt,
            outdir=req.outdir,
            scenario=req.scenario,
            scenario_path=req.scenario_path,
            gate2=req.gate2,
            label=req.label,
            device=req.device,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PipelineResponse(status="ok", outdir=str(result.outdir), data={
        "video": str(result.video_path),
        "report": str(result.report_path),
        "labels": str(result.labels_path) if result.labels_path else None,
        "decision": result.report["decision"],
        "score": result.report["plausibility_score"],
    })


@app.post("/verified-video-batches", response_model=PipelineResponse,
          tags=["video generation"])
def create_verified_video_batch(req: E2ERequest) -> PipelineResponse:
    try:
        data = run_e2e_pipeline(
            intention=req.intention,
            count=req.count,
            run_name=req.run_name,
            start_frames=req.start_frames,
            gate2=req.gate2,
            label=req.label,
            device=req.device,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PipelineResponse(status="ok", outdir=data["run_dir"], data=data)


# Backward-compatible aliases. Hidden from docs so the public API stays clean.
@app.post("/compile", response_model=PipelineResponse, include_in_schema=False)
def compile_endpoint(req: CompileRequest) -> PipelineResponse:
    return create_scenario_bundle(req)


@app.post("/pipeline", response_model=PipelineResponse, include_in_schema=False)
def pipeline_endpoint(req: VideoPipelineRequest) -> PipelineResponse:
    return create_verified_video(req)


@app.post("/e2e", response_model=PipelineResponse, include_in_schema=False)
def e2e_endpoint(req: E2ERequest) -> PipelineResponse:
    return create_verified_video_batch(req)
