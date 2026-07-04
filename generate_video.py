#!/usr/bin/env python3
"""Generate a video with Gemini Omni Flash (gemini-omni-flash-preview).

Usage:
    # Put GEMINI_API_KEY=your-api-key in a .env file next to this script
    python generate_video.py "A marble rolling on a chain reaction track"
    python generate_video.py "A cat playing with yarn" --image cat.png -o cat.mp4
    python generate_video.py "A futuristic city, cyberpunk style" --aspect-ratio 9:16
"""

import argparse
import base64
import mimetypes
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

MODEL = "gemini-omni-flash-preview"


def build_input(prompt: str, image_paths: list[str]):
    if not image_paths:
        return prompt
    parts = []
    for path in image_paths:
        data = base64.b64encode(Path(path).read_bytes()).decode()
        mime = mimetypes.guess_type(path)[0] or "image/png"
        parts.append({"type": "image", "data": data, "mime_type": mime})
    parts.append({"type": "text", "text": prompt})
    return parts


def download_via_uri(client: genai.Client, uri: str, output: Path) -> None:
    file_name = uri.split("/")[-1].split(":")[0]
    print("Waiting for video processing...")
    while True:
        f_info = client.files.get(name=f"files/{file_name}")
        if f_info.state.name == "ACTIVE":
            break
        if f_info.state.name == "FAILED":
            raise RuntimeError("Video generation failed.")
        time.sleep(5)
    output.write_bytes(client.files.download(file=uri))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a video with Gemini Omni Flash")
    parser.add_argument("prompt", help="Text prompt describing the video")
    parser.add_argument("--image", action="append", default=[], metavar="PATH",
                        help="Reference image (repeatable)")
    parser.add_argument("--aspect-ratio", choices=["16:9", "9:16"], default="16:9")
    parser.add_argument("--uri", action="store_true",
                        help="Use URI delivery (recommended for videos > 4 MB)")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output file (default: output.mp4)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found. Add it to the .env file.", file=sys.stderr)
        return 1

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    output = Path(args.output)

    response_format = {"type": "video", "aspect_ratio": args.aspect_ratio}
    if args.uri:
        response_format["delivery"] = "uri"

    print(f"Generating video with {MODEL}...")
    interaction = client.interactions.create(
        model=MODEL,
        input=build_input(args.prompt, args.image),
        response_format=response_format,
    )

    video = interaction.output_video
    if args.uri:
        download_via_uri(client, video.uri, output)
    else:
        output.write_bytes(base64.b64decode(video.data))

    print(f"Video saved to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
