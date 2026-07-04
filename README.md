# Roboss Video Lab

Local React/Tailwind UI + Python FastAPI backend for generating MP4 videos with Gemini Omni Flash,
then annotating each MP4 with `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` through Crusoe.
The app launches a parallel batch of 6 camera-angle variants from the same scene prompt and shows each result with VLM-style zones in the browser.

## Setup

The backend reads the Gemini key from `API_KEY_GEMINI` or `GEMINI_API_KEY`.
It reads the Crusoe key from `API_KEY_CRUSOE` or `CRUSOE_API_KEY`.
It checks `.env` in this repo first, then the parent folder `.env`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
npm install
npm run dev
```

Open `http://127.0.0.1:5174`.
The API runs on `http://127.0.0.1:8010`.

## Notes

- The Gemini key is never sent to the browser.
- The Crusoe key is never sent to the browser.
- Batch jobs are created together and submitted in parallel as 6 distinct camera-angle generation + annotation pipelines.
- Generated videos, sampled frames, and annotation sidecars are saved in `generated/` and ignored by Git.
- Crusoe is called through an OpenAI-compatible chat completions URL. Use an Inference API key for `https://api.inference.crusoecloud.com/v1`.
- The annotator samples JPEG frames from each MP4 with `ffmpeg`, then sends each frame to Nemotron for object-detection zones.
- Annotation JSON uses `schema_version: "vlm-zones-v1"` with normalized `{x, y, width, height}` boxes per frame.
- Real generations can consume Gemini quota or credits.
