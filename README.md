# Roboss — Synthetic Action Dataset Compiler

An end-to-end pipeline that turns a plain-English idea into **physically
plausible, auto-labeled action videos** for training — and throws out the
ones that break physics. The heart of it is a **rejection gate** so that
impossible AI-generated footage never enters the dataset.

```
 idea / prompt
      │
      ▼
 1. SCENARIO COMPILER   (agents/)   idea → world contract → N validated scenarios
      │
      ▼
 2. VIDEO GENERATION    (Gemini)    prompt → generated.mp4
      │
      ▼
 3. VERIFICATION        (verifier/) the rejection gate — Gate 1 + Gate 2
      │
      ├─ REJECT ─▶ discard (report.json explains why)
      │
      └─ ACCEPT ─▶ 4. AUTO-LABELING (Gemini) → labels.json  ──▶ dataset
```

Two ways to drive it:

- **Single video** — one prompt straight through generate → verify → label
  (`run.sh pipeline` / `run_pipeline.py`).
- **Batch** — one idea fanned out into several scenario variations, each
  generated, verified and labeled (`e2e.sh`, which chains the scenario
  compiler into the pipeline).
- **API** — FastAPI endpoints over the same service layer
  (`uvicorn roboss.api:app`).

## Quick start

```bash
# one prompt → one video → verify → label
./run.sh pipeline "a warehouse rover drives down an aisle when barrels fall"

# one idea → N scenario variations, each generated + verified + labeled
./e2e.sh "industrial safety hazard in a warehouse" 5 my_run

# API server
./run.sh api

# verify an existing video only
./run.sh verify path/to/video.mp4

# run the tests (no models, no API)
./run.sh tests
```

`run.sh help` lists every command. All outputs land in `runs/<name>/`.

## The rejection gate (verifier)

```
generated video (+ optional scenario metadata)
        │
        ▼
 1. pose extraction        YOLO11-pose  → 17 COCO keypoints per person, tracked
 2. object det + tracking  YOLO11 + ByteTrack → object boxes with persistent IDs
        │
        ▼
 3. GATE 1: physics rule engine   10 deterministic checks (pure NumPy)
        │
        ▼
 4. GATE 2 (optional): semantic reviewer
    sampled frames + gate-1 findings → multimodal Gemini model
    → strict-JSON verdict on what rules cannot see
        │
        ▼
 5. report                 plausible? score, violations per gate, frames
```

## Gate 1 — formal checks (deterministic)

| # | check | flags |
|---|-------|-------|
| 1 | `trajectory_jump` | person/object teleports between consecutive frames |
| 2 | `body_deformation` | limb stretches beyond its own established length (foreshortening-safe) |
| 3 | `foot_skate` | grounded foot slides horizontally |
| 4 | `contact_incoherence` | object moves with no hand nearby and it is not free-falling |
| 5 | `object_disappearance` | track vanishes and reappears far away |
| 6 | `gravity_suspicion` | unsupported object hovers above the ground line |
| 7 | `object_materialization` | object/person appears or vanishes mid-frame, away from edges and people (tracker ID-switches are suppressed) |
| 8 | `levitation` | person hangs in the air too long and too statically for a jump |
| 9 | `telekinesis_suspicion` | object moves in lockstep with a hand gesture while the hand is too far to touch it |
| 10 | `object_deformation` | rigid object repeatedly snaps its shape (bbox aspect ratio) |

## Gate 2 — semantic reviewer (VLM, optional)

The rule engine cannot see extra limbs, morphing objects or magic glows.
Gate 2 samples ~10 frames (plus gate-1 suspicious frames), sends them with
the scenario prompt and gate-1 findings to a multimodal Gemini model
(`gemini-3.5-flash` by default), and receives a schema-constrained JSON
verdict (structured outputs — the model cannot free-form its answer).
Semantic types: `anatomical_anomaly`, `object_morphing`, `magic_effect`,
`impossible_gesture`, `scene_inconsistency`, `prompt_mismatch`.

Both gates feed one violation list (each entry is tagged `gate: formal|semantic`)
and one scoring formula — a video is rejected if either gate finds something
critical. Gate 2 is deliberately *not* the main judge: the deterministic
engine decides first, the VLM extends coverage.

Requires Gemini API credentials (`GEMINI_API_KEY`):

```bash
python -m verifier video.mp4 --gate2 --scenario scenario.example.json
```

All checks are camera-motion compensated (median displacement of all
tracks is treated as global pan and subtracted), and all thresholds are
normalized by the frame diagonal and fps, so they are resolution-independent.
Thresholds and score weights live in [verifier/config.py](verifier/config.py).

Scoring: `score = 1 − Σ weight(type) × worst_severity(type)`.
Accept if `score ≥ 0.72` and no single violation exceeds `0.85`.

## Setup

Requires **Python 3.13** (PyTorch does not support 3.14 yet). Using
[`uv`](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Or plain `venv`:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Put your key in a `.env` file at the project root (loaded automatically):

```
GEMINI_API_KEY=your_key_here
```

See [.env.example](.env.example) for optional knobs such as
`ROBOSS_VIDEO_MODEL`, `ROBOSS_LABEL_MODEL`, `ROBOSS_GATE2_ENABLED`,
`ROBOSS_LABEL_ON_ACCEPT`, and `ROBOSS_RUNS_DIR`.

YOLO11 weights (`yolo11n-pose.pt`, `yolo11n.pt`, ~12 MB total) download
automatically on first run. `run.sh` / `e2e.sh` create the venv for you if
it is missing.

## Runners

| command | what it does |
|---|---|
| `./run.sh tests` | physics + agent test suite (no models, no API) |
| `./run.sh verify <video.mp4>` | verify an existing video (Gate 1 + Gate 2) |
| `./run.sh pipeline "<prompt>"` | generate → verify → label one video |
| `./run.sh agents "<intention>"` | scenario compiler: idea → scenario bundle |
| `./run.sh api` | serve FastAPI on `127.0.0.1:8000` |
| `./run.sh all "<prompt>"` | tests, then the full pipeline |
| `./e2e.sh "<idea>" [count] [name]` | compile N scenarios, then generate + verify + label each |

`run.sh pipeline` flags: `--outdir DIR` (keep runs separate; default
`runs/latest` is overwritten), `--no-gate2`, `--device cpu|0`,
`--scenario FILE|none`. Extra flags pass through to the underlying tool.

Outputs per run land in `runs/<name>/`: `generated.mp4`, `report.json`,
and `labels.json` (only if accepted). Exit code `0` = accept, `2` = reject.

The batch path uses the compiled `video_prompt` from each scenario packet
when present. That prompt carries the object/scene identity anchors; the
short `scenario_prompt` is kept for reports and semantic review context.

## FastAPI

```bash
uvicorn roboss.api:app --host 127.0.0.1 --port 8000
```

Endpoints:

| endpoint | purpose |
|---|---|
| `GET /health` | env/runs sanity check |
| `POST /scenario-bundles` | intention → world contract + scenario bundle |
| `POST /verified-videos` | one prompt/scenario → video → verification → labels |
| `POST /verified-video-batches` | intention → scenarios → generate/verify/label each |
| `POST /robot-dataset-exports` | test endpoint: existing video → V2R robot data |
| `GET /runs` | list local runs with manifest URLs |
| `GET /runs/{run_id}` | run metadata + frontend file URLs |
| `GET /assets/...` | static access to local files under `runs/` |

Legacy aliases (`/compile`, `/pipeline`, `/e2e`) still work but are hidden
from the public docs.

### Local File Storage

The local MVP stores artifacts under `runs/` and exposes them to the
frontend through FastAPI:

```
Pipeline writes files
→ LocalStorageService saves/records them under runs/
→ manifest.json stores file metadata + /assets URLs
→ frontend calls /runs or /runs/{run_id}
→ frontend displays video/image/json from /assets/...
```

Example:

```bash
curl http://127.0.0.1:8000/runs
curl http://127.0.0.1:8000/runs/warehouse_v1
```

A video generated at `runs/my_run/sc_01/generated.mp4` is available as:

```
http://127.0.0.1:8000/assets/my_run/sc_01/generated.mp4
```

### Robot Data Export

James' `v2r/` pipeline is integrated after video verification. Enable it on
the full endpoints with:

```json
{
  "export_robot_data": true,
  "robots": ["g1"],
  "robot_data_mode": "synthetic",
  "robot_data_stages": "all"
}
```

Flow:

```
generated.mp4
→ Roboss verifier accepts
→ V2R ingest / kinematics / retarget / MuJoCo Tier-1 / package
→ runs/<run>/.../robot_data/manifest.json
→ frontend reads /assets URLs
```

To test only the video → data leg without generating a new video:

```bash
curl -X POST http://127.0.0.1:8000/robot-dataset-exports \
  -H "Content-Type: application/json" \
  -d '{"video_path":"runs/full_test_01/sc_01/generated.mp4","outdir":"runs/v2r_test/robot_data","robots":["g1"],"mode":"synthetic","stages":"all"}'
```

### Verifier CLI (existing video only)

```bash
python -m verifier path/to/generated_video.mp4 \
    --scenario scenario.example.json \
    --annotated annotated.mp4
```

Writes `<video>_report.json` and prints a summary:

```
  decision : REJECT
  score    : 0.57
  reason   : The generated video contains physical inconsistencies: ...
   - [0.88] trajectory_jump @ frames 41..42: The box center jumps 24% of the frame diagonal ...
   - [0.72] contact_incoherence @ frames 35..38: The box moves at 0.81 diag/s but no hand ...
```

Exit code: `0` = accept, `2` = reject (usable in a pipeline).

The `--scenario` metadata packet (see [scenario.example.json](scenario.example.json))
comes from the video-generation side; the report then also lists
`missing_expected_objects` so scenario mismatches are visible.

## Report schema

```json
{
  "video_id": "sample_001",
  "scenario": "Human slips near a humanoid robot carrying a box",
  "plausible": false,
  "plausibility_score": 0.57,
  "decision": "reject",
  "main_reason": "...",
  "violations": [
    {"type": "trajectory_jump", "severity": 0.88, "frames": [41, 42],
     "reason": "...", "entity": "box#3"}
  ],
  "suspicious_frames": [20, 21, 41, 42],
  "extracted_evidence": {
    "humans_detected": 1,
    "objects_detected": ["box"],
    "tracks": [{"id": 1, "label": "person", "frames_tracked": 86}],
    "detected_classes": ["person", "box"],
    "missing_expected_objects": []
  }
}
```

## Tests

The physics checks are pure NumPy over track data, so they run without
models or videos:

```bash
python -m pytest tests -q     # or: ./run.sh tests
```

## Project layout

```
run.sh              runner: tests / verify / pipeline / agents / api / all
e2e.sh              batch: idea → N scenarios → generate + verify + label each
run_pipeline.py     single video: generate → verify → label
gemini_service.py   Gemini video generation + auto-labeling
env_loader.py       tiny dependency-free .env loader
roboss/
  settings.py       centralized .env-backed settings
  storage.py        LocalStorageService + /assets-ready file metadata
  pipeline.py       reusable compile/generate/verify/label orchestration
  api.py            FastAPI app
agents/             scenario compiler (idea → world contract → scenarios)
verifier/
  config.py    thresholds + score weights (all tunable)
  tracks.py    Track / Evidence / Violation data structures
  extract.py   video → tracks (YOLO11-pose + YOLO11 detect/track)
  checks.py    the 10 gate-1 physics checks (model-free, unit-tested)
  gate2.py     gate-2 semantic reviewer (Gemini API, structured outputs)
  scoring.py   violations → score → accept/reject
  report.py    final JSON report (+ scenario match, per-gate status)
  viz.py       annotated demo video (skeletons, boxes, violation timeline)
  __main__.py  CLI
tests/
  test_checks.py   gate-1 physics checks
  test_agents.py   scenario compiler
```

## Models

| stage | model | where |
|---|---|---|
| scenario compiler (text) | `gemini-3.5-flash` | `agents/config.py` |
| scenario canvas (image) | `gemini-3.1-flash-image` | `agents/config.py` |
| video generation | `gemini-omni-flash-preview` | `gemini_service.py` |
| Gate 2 reviewer | `gemini-3.5-flash` | `verifier/config.py` |
| auto-labeling | `gemini-3.5-flash` | `gemini_service.py` |

## Known limitations

- 2D-only: no metric gravity check (`9.81 m/s²`) without camera calibration —
  the gravity check is a weak "unsupported hovering" heuristic.
- Object vocabulary is COCO-80; "humanoid robot" or "wet floor" need an
  open-vocabulary detector (Grounding DINO) — planned extension.
- Heavy camera motion beyond a global pan (zoom, rotation) can add noise;
  the median-displacement compensation only handles translation.
