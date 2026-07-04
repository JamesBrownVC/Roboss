# Roboss — Video Plausibility Verifier

The **rejection gate** of the Synthetic Action Dataset Compiler.

It receives an AI-generated action video, extracts human pose, object
tracks and contact evidence, then applies physics-aware checks to reject
videos with impossible motion. Output: a structured plausibility report
with a score, an accept/reject decision and frame-level reasons.

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

```powershell
python -m verifier video.mp4 --gate2 --scenario scenario.example.json
```

All checks are camera-motion compensated (median displacement of all
tracks is treated as global pan and subtracted), and all thresholds are
normalized by the frame diagonal and fps, so they are resolution-independent.
Thresholds and score weights live in [verifier/config.py](verifier/config.py).

Scoring: `score = 1 − Σ weight(type) × worst_severity(type)`.
Accept if `score ≥ 0.72` and no single violation exceeds `0.85`.

## Setup (Windows)

Requires Python 3.13 (PyTorch does not support 3.14 yet).

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

YOLO11 weights (`yolo11n-pose.pt`, `yolo11n.pt`, ~12 MB total) are
downloaded automatically on first run.

## Usage

```powershell
python -m verifier path\to\generated_video.mp4 `
    --scenario scenario.example.json `
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

```powershell
python -m pytest tests -q
```

## Project layout

```
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
  test_checks.py
```

## Known limitations

- 2D-only: no metric gravity check (`9.81 m/s²`) without camera calibration —
  the gravity check is a weak "unsupported hovering" heuristic.
- Object vocabulary is COCO-80; "humanoid robot" or "wet floor" need an
  open-vocabulary detector (Grounding DINO) — planned extension.
- Heavy camera motion beyond a global pan (zoom, rotation) can add noise;
  the median-displacement compensation only handles translation.
