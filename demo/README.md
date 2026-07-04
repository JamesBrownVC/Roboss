# V2R Factory — Demo Frontend

Polished single-page demo for the V2R pipeline (video → robot training data).
FastAPI serves both the JSON API and the static frontend from one process.

## Launch

```bash
python demo/serve.py            # → http://localhost:8017
# optional custom port:
python demo/serve.py 9000
```

Requirements: `fastapi`, `uvicorn`, `pandas`, `pyarrow`, `opencv-python`,
`imageio-ffmpeg` (used once to transcode mp4v raw clips to browser-playable
h264, cached under `demo/.cache/`).

```bash
pip install fastapi uvicorn pandas pyarrow opencv-python imageio-ffmpeg
```

## What it reads

The API reads live pipeline artifacts from `../v2r` at request time — refresh
the page and it picks up whatever the backend agents have produced:

| Section | Real artifact | Fallback |
|---|---|---|
| Hero stats | `data/raw`, `data/timeseries`, `workspaces/` counts | zeros |
| Dataset explorer | `data/raw/import_manifest.json` + `data/raw/*/*.mp4` | mock cards |
| Timeseries viewer | `data/timeseries/{human,animal}/{stem}.parquet` | synthetic walking skeleton / dog track (marked "demo data") |
| Feasibility judge | `workspaces/*/qa/feasibility_report.json` | mock verdicts (always appended as contrast examples) |
| Multi-view tier | `workspaces/sessions/*/{session,sync,calibration}.json` + `qa/cross_view_reproj.json` | mock session |
| Yield funnel | `workspaces/*/manifests/*.manifest.json` + `qa/decision.json` | mock funnel |
| Export showcase | `workspaces/*/export/lerobot/{meta,features}.json` | mock card |

Anything mocked is flagged with an amber **demo data** badge in the UI.

## Skeleton overlay

If a human pose parquet exists for a selected raw video, the 33-joint
MediaPipe skeleton is drawn on a canvas synced to video playback. Animal
videos get YOLO track bounding boxes + velocity vectors. When no parquet
exists yet, a synthetic animated signal is shown instead (badged demo).

## Optional: refresh local extraction cache

`python demo/extract_cache.py` runs MediaPipe pose (and YOLO tracks, if
`ultralytics` is installed) over all imported clips plus two bundled
CC-licensed Wikimedia sample clips, writing parquet into `demo/.cache/ts/`.
The server uses these as a fallback whenever the pipeline's own
`data/timeseries/` parquet is missing or empty — this is what powers the
skeleton-overlay wow moment even before the main pipeline finishes.
Nothing under `v2r/` is ever written by the demo.

## API endpoints

- `GET /api/overview` — headline counts
- `GET /api/videos` — imported clips
- `GET /media/raw/{source_id}/{filename}` — video stream (h264-transcoded)
- `GET /api/timeseries/{subject}/{stem}` — per-frame pose/track JSON
- `GET /api/feasibility` — judge verdicts
- `GET /api/multiview` — sessions + cross-view reprojection
- `GET /api/yield` — funnel + per-episode stage grid
- `GET /api/exports` — LeRobot dataset cards
