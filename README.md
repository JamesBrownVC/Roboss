# Roboss Studio

React/Tailwind frontend for the Roboss video data pipeline: one-shot prompt to
generated, reviewed and annotated MP4 training videos.

## Frontend setup

```powershell
npm install
npm run dev
```

Open `http://127.0.0.1:5174`.

The dev server proxies `/api` and `/generated` to the backend at
`http://127.0.0.1:8010`. Start the FastAPI backend so the Studio page
can generate videos.

## Backend setup

Requires Python 3.13 and `GEMINI_API_KEY` (Gemini Omni Flash video generation).

Create a `.env` file in the project root (`C:\Users\adilo\Roboss\.env`):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn server.app:app --host 127.0.0.1 --port 8010 --reload
```

The backend orchestrates the agent pipeline (`python -m agents`), Gemini Omni Flash video generation, and the physics verifier. Generated MP4s are served from
`/generated`.

## Pages

- **Studio** - one prompt (plus optional image/video reference) in, an annotated video
  dataset out, with inline preview of the first samples, MP4 download, and live
  backend agent logs underneath the generation workspace.
- **Analytics** - dashboard with KPI cards and charts: videos generated, validation pass
  rate, datasets produced, success rate and latency per pipeline stage.
- **Monitor** - live robot view: connect the robot to watch its camera feed, real-time
  detection overlays, telemetry and action log.

## Video Plausibility Verifier

The verifier is the rejection gate of the Synthetic Action Dataset Compiler. It
receives an AI-generated action video, extracts human pose, object tracks and
contact evidence, then applies physics-aware checks to reject videos with
impossible motion. Output is a structured plausibility report with a score, an
accept/reject decision and frame-level reasons.

```text
generated video (+ optional scenario metadata)
        |
        v
 1. pose extraction        YOLO11-pose -> 17 COCO keypoints per person, tracked
 2. object det + tracking  YOLO11 + ByteTrack -> object boxes with persistent IDs
        |
        v
 3. Gate 1: physics rule engine   10 deterministic checks (pure NumPy)
        |
        v
 4. Gate 2 (optional): semantic reviewer
    sampled frames + gate-1 findings -> multimodal Gemini model
        |
        v
 5. report                 plausible? score, violations per gate, frames
```

### Gate 1 formal checks

| # | check | flags |
|---|-------|-------|
| 1 | `trajectory_jump` | person/object teleports between consecutive frames |
| 2 | `body_deformation` | limb stretches beyond its own established length |
| 3 | `foot_skate` | grounded foot slides horizontally |
| 4 | `contact_incoherence` | object moves with no hand nearby and it is not free-falling |
| 5 | `object_disappearance` | track vanishes and reappears far away |
| 6 | `gravity_suspicion` | unsupported object hovers above the ground line |
| 7 | `object_materialization` | object/person appears or vanishes mid-frame |
| 8 | `levitation` | person hangs in the air too long and too statically for a jump |
| 9 | `telekinesis_suspicion` | object moves with a far-away hand gesture |
| 10 | `object_deformation` | rigid object repeatedly snaps its bbox shape |

### Gate 2 semantic reviewer

Gate 2 samples frames, sends them with the scenario prompt and gate-1 findings to
a multimodal Gemini model (`gemini-3.5-flash` by default), and receives a
schema-constrained JSON verdict. It covers issues such as anatomical anomalies,
object morphing, magic effects, impossible gestures, scene inconsistencies and
prompt mismatch.

Requires Gemini API credentials (`GEMINI_API_KEY`):

```powershell
python -m verifier video.mp4 --gate2 --scenario scenario.example.json
```

All checks are camera-motion compensated and thresholds are normalized by the
frame diagonal and fps, so they are resolution-independent. Thresholds and score
weights live in [verifier/config.py](verifier/config.py).

Scoring: `score = 1 - sum(weight(type) * worst_severity(type))`. Accept if
`score >= 0.72` and no single violation exceeds `0.85`.

### Verifier setup

Requires Python 3.13 (PyTorch does not support 3.14 yet).

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

YOLO11 weights (`yolo11n-pose.pt`, `yolo11n.pt`, about 12 MB total) are
downloaded automatically on first run.

### Verifier usage

```powershell
python -m verifier path\to\generated_video.mp4 `
    --scenario scenario.example.json `
    --annotated annotated.mp4
```

Writes `<video>_report.json` and prints a summary. Exit code `0` means accept;
exit code `2` means reject.

The `--scenario` metadata packet, see
[scenario.example.json](scenario.example.json), comes from the video-generation
side. The report also lists `missing_expected_objects` so scenario mismatches are
visible.

### Tests

The physics checks are pure NumPy over track data, so they run without models or
videos:

```powershell
python -m pytest tests -q
```

### Project layout

```text
verifier/
  config.py    thresholds + score weights
  tracks.py    Track / Evidence / Violation data structures
  extract.py   video -> tracks (YOLO11-pose + YOLO11 detect/track)
  checks.py    the gate-1 physics checks
  gate2.py     gate-2 semantic reviewer
  scoring.py   violations -> score -> accept/reject
  report.py    final JSON report
  viz.py       annotated demo video
  __main__.py  CLI
tests/
  test_checks.py
```

## Notes

- The UI degrades gracefully when backend endpoints are missing: stats fall back
  to the browser's local run history.
- All UI text is in English; the theme is a dark green dashboard.
- The verifier is 2D-only and uses heuristic gravity checks without camera
  calibration.
