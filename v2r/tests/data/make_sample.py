"""Create a minimal 3-second test clip for CI (720p, 30 fps)."""

from pathlib import Path

import cv2
import numpy as np


def main() -> Path:
    out = Path(__file__).resolve().parent / "sample.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h, fps, sec = 1280, 720, 30, 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    for i in range(fps * sec):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (40, 40, 40)
        cv2.rectangle(frame, (200, 200), (600, 500), (0, 180, 255), -1)
        cv2.putText(frame, f"frame {i}", (220, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return out


if __name__ == "__main__":
    main()
