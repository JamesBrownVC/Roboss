"""One-off live probe of the Crusoe Managed Inference endpoint (text, image, fast)."""
import base64
import json
import ssl
import urllib.request
from pathlib import Path

import certifi
import cv2

ROOT = Path(__file__).resolve().parents[2]
key = [l.split("=", 1)[1].strip() for l in (ROOT / ".env").read_text().splitlines()
       if l.startswith("NVIDIA_API_KEY=")][0]
ctx = ssl.create_default_context(cafile=certifi.where())
BASE = "https://api.inference.crusoecloud.com/v1"
HDR = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
       "User-Agent": "v2r-labeler/0.1"}


def chat(payload):
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(payload).encode(), headers=HDR)
    with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
        return json.loads(r.read().decode())


out = chat({"model": "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B",
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 300})
print("OMNI TEXT:", repr(out["choices"][0]["message"]["content"])[:300])

cap = cv2.VideoCapture(str(ROOT / "v2r/data/syngen/veo1/videos/e00_cam0.mp4"))
ok, frame = cap.read()
cap.release()
frame = cv2.resize(frame, (384, 216))
_, buf = cv2.imencode(".jpg", frame)
b64 = base64.b64encode(buf.tobytes()).decode()
out = chat({"model": "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B", "max_tokens": 500,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "In one short sentence, what is in this image?"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]})
print("OMNI IMAGE:", repr(out["choices"][0]["message"]["content"])[:500])

out = chat({"model": "moonshotai/Kimi-K2.6", "max_tokens": 400, "temperature": 0.0,
            "messages": [{"role": "user",
                          "content": 'Reply with only JSON: {"ok": true}'}]})
print("KIMI RAW CHOICE:", json.dumps(out["choices"][0], default=str)[:1200])
