"""Calibrate aigen_forensics statistics on known real vs generated clips."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2r.agentic.tools import aigen_forensics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
clips = {
    "REAL pipette": ROOT / "data/raw/human_pexels_walk/human_pexels_walk_000.mp4",
    "REAL waves": ROOT / "data/raw/human_pexels_dance/human_pexels_dance_000.mp4",
    "REAL nighthut": ROOT / "data/raw/animal_pexels/animal_pexels_000.mp4",
    "REAL timelapse": ROOT / "data/raw/animal_pexels_wildlife/animal_pexels_wildlife_000.mp4",
    "REAL roboss C03": ROOT / "../roboss-film/selects/C03.mp4",
    "REAL roboss C01": ROOT / "../roboss-film/selects/C01.mp4",
    "GEN veo wave": ROOT / "data/syngen/veo1/videos/e00_cam0.mp4",
    "GEN cup e02": ROOT / "data/syngen/web_175617/videos/e02_cam0.mp4",
    "GEN e00": ROOT / "data/syngen/web_175617/videos/e00_cam0.mp4",
    "GEN e01": ROOT / "data/syngen/web_175617/videos/e01_cam0.mp4",
    "GEN omni": ROOT / "data/syngen/omni1/videos/e00_cam0.mp4",
}
for name, p in clips.items():
    out = aigen_forensics(p)
    if out.get("available"):
        print(f"{name:18} autocorr={out['noise_lag1_autocorr']:+.4f} "
              f"hf_cv={out['hf_energy_cv']:.4f}")
    else:
        print(name, out)
