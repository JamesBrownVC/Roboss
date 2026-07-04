# Roboss Studio

React/Tailwind frontend for the Roboss video data pipeline: one-shot prompt to
generated, reviewed and annotated MP4 training videos.

## Pages

- **Studio** - one prompt (plus optional image/video reference) in, an annotated video
  dataset out, with inline preview of the first samples and MP4 download.
- **Analytics** - dashboard with KPI cards and charts: videos generated, validation pass
  rate, datasets produced, success rate and latency per pipeline stage.
- **Monitor** - live robot view: connect the robot to watch its camera feed, real-time
  detection overlays, telemetry and action log.

## Setup

```powershell
npm install
npm run dev
```

Open `http://127.0.0.1:5174`.

The dev server proxies `/api` and `/generated` to the backend at `http://127.0.0.1:8010`.
Start the FastAPI backend separately so the Studio page can generate videos.

## Notes

- The UI degrades gracefully when backend endpoints are missing: stats fall back to
  the browser's local run history.
- All UI text is in English; the theme is a dark green dashboard.
