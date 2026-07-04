# V2R Factory

Production pipeline: **raw human video → labeled, physics-checked, LeRobot v3 export**.

Source of truth: [`V2R_MASTER_PROMPT.md`](../V2R_MASTER_PROMPT.md) (repo root).

## Quick start (synthetic / CI — any host)

```bash
cd v2r
pip install -e ".[dev]"
python tests/data/make_sample.py   # creates tests/data/sample.mp4 if missing
v2r run --episode tests/data/sample.mp4 --stages all --robots g1
```

Synthetic mode produces **schema-valid** artifacts tagged `source=synthesized`. Use it for contract testing on Windows/macOS without CUDA tools.

## Research pipeline stages

| Stage | Tool (real mode) | Isolated env |
|-------|------------------|--------------|
| ingest | ffmpeg + PySceneDetect | orchestrator |
| feasibility_judge | physics heuristics + Qwen-VL judge | `envs/feasibility_judge/` |
| geometry | **ViPE** | `envs/vipe/` |
| human_body | **GVHMR** + Umeyama→ViPE | `envs/gvhmr/` |
| hands | **WiLoR** / HaMeR (MANO) | `envs/wilor/` |
| objects | Grounding DINO + SAM2 + **FoundationPose** | `envs/foundationpose/` |
| contact | geometric (orchestrator) | — |
| semantics | **Qwen2.5-VL** | `envs/semantics/` |
| retarget | **GMR** / mink / quadruped adapter | `envs/gmr/` |
| physics_validate | MuJoCo Tier-1 | orchestrator |
| qa | cross-checks + yield report | orchestrator |
| package | LeRobot v3 + EgoDex mirror | orchestrator |

## Switch synthetic → real (per stage)

Edit `config/pipeline.yaml`:

```yaml
default_mode: synthetic   # or real for all GPU stages
stages:
  geometry:     {enabled: true, mode: real, env: vipe}
  human_body:   {enabled: true, mode: real, env: gvhmr}
  hands:        {enabled: true, mode: real, env: wilor}
  objects:      {enabled: true, mode: real, env: foundationpose}
  semantics:    {enabled: true, mode: real, env: semantics}
  retarget:     {enabled: true, mode: real, env: gmr}
```

Or override once: `v2r run --episode ... --mode real` (requires CUDA Linux + envs installed).

Each env spec under `envs/{name}/README.md` lists **pinned commits** and `micromamba` install commands.

## Host requirements (real mode)

| Requirement | Notes |
|-------------|-------|
| **OS** | Ubuntu 22.04+ recommended |
| **GPU** | NVIDIA CUDA 12.x; 24 GB VRAM ideal (12 GB with reduced batch) |
| **SMPL-X / MANO** | Operator places registered models in `assets/body_models/` — never scraped |
| **Weights** | ViPE, GVHMR, WiLoR, FoundationPose, Qwen-VL checkpoints per env README |
| **Commercial** | See auto-generated `LICENSE_AUDIT.md`; set `licensing.permissive_only: true` for NC-free fallbacks |

Windows: use **synthetic mode** only unless you WSL2 + CUDA for individual stage envs.

## Feasibility judge (pre-analysis QA)

Runs **after ingest, before geometry** — rejects AI-generated or untrustworthy video early.

- **Physics checks**: temporal consistency, joint velocity/acceleration spikes, foot-slide heuristics, scale jumps, optical-flow vs pose disagreement
- **LLM judge** (structured JSON): `physically_plausible`, `tracking_likely_valid`, `ai_generated_artifacts`, `confidence`, `recommendation` (`proceed | reject | human_review`)
- **Outputs**: `qa/feasibility_report.json`, `qa/feasibility_mask.parquet` (per-frame valid/conf/source)
- **Modes**: synthetic (deterministic mock) | real (Qwen-VL in `envs/feasibility_judge/` or `V2R_JUDGE_API` OpenAI-compatible endpoint)
- **Gate**: reject when `recommendation=reject` or `physics_violation_frame_ratio > threshold` (`config/qa.yaml`)

## Multi-view GT tier (same event, multiple angles)

Session layout: `workspaces/sessions/{session_id}/`

```bash
v2r session create --id my_event --videos cam0:v0.mp4 cam1:v1.mp4 cam2:v2.mp4
v2r session sync --id my_event
v2r session calibrate --id my_event
v2r session triangulate --id my_event
v2r session fuse --id my_event --robots g1
# or full DAG:
v2r session run --id my_event --tier multiview
```

Multi-gen variants (same cam, multiple generations):

```bash
v2r session create --id my_event --videos cam0:best.mp4 cam1:v1.mp4 \
  --variants cam0:gen1.mp4 cam0:gen2.mp4
```

Strategy (`config/multiview.yaml`): align variants per camera, pick **best_frame** by cross-view reprojection confidence (or `mean_confidence` ensemble).

**Why multi-view GT tier** (5 reasons):

1. **Measured accuracy** — cross-view reprojection error is objective, not self-asserted monocular confidence
2. **True 3D joints** — RANSAC triangulation yields `source=triangulated` with error bars
3. **AI-video QA** — feasibility gate + multi-view cross-check catches physics violations single-view misses
4. **Internal benchmark** — monocular shadow pipeline runs on the same session; triangulation vs monocular error logged in `qa/cross_view_reproj.json`
5. **Export tier metadata** — `tier: monocular | multiview_gt` in LeRobot export (`monocular` = estimated kinematics; `multiview_gt` = triangulated with measured error)

Config: `config/multiview.yaml` (sync, calibration, triangulation thresholds).

## Dev harness (separate from production stages)

Lightweight import/timeseries path (MediaPipe + YOLO) — **not** production geometry/body:

```bash
v2r import-datasets
v2r extract-timeseries
v2r build-training-set
```

## Layout

```
v2r/
  config/           pipeline, robots, qa, licensing, verbs
  envs/             per-stage isolated env specs + tool_entry.py
  src/v2r/
    schema/         pydantic models + parquet/npz IO
    stages/         thin wrappers (synthetic + real subprocess)
    orchestrator/   CLI, DAG runner, manifests, GPU semaphore
    qa/             cross-checks, yield report, license audit
    export/         LeRobot / EgoDex writers
  tests/data/       sample.mp4
  workspaces/       episode outputs (gitignored)
```

## Outputs

Episode workspace: `workspaces/{episode_id}/` per master prompt §4.

After `package`: `export/lerobot/`, `export/egodex_mirror/`, `qa/yield_report.md`, manifests for every stage.

## Tests

```bash
pytest
```

## License audit

Regenerated on each `package` stage run → `LICENSE_AUDIT.md` at repo root.
