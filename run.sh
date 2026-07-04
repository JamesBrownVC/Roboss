#!/usr/bin/env bash
#
# run.sh — one entry point for every part of the video-verification project.
#
# Usage:
#   ./run.sh <command> [args...]
#
# Commands:
#   tests                  Run the pure-NumPy physics test suite (no models/API).
#   verify   <video.mp4>   Verify an existing video (gate 1 + gate 2 semantic).
#   pipeline <prompt>      Full loop: generate -> verify -> label a video (Gemini).
#   agents   <intention>   Scenario contract compiler: intention -> scenarios.
#   api                    Start the FastAPI app on 127.0.0.1:8000.
#   all      <prompt>      Run tests, then the full pipeline end-to-end.
#   help                   Show this message.
#
# Any extra flags after the positional arg are forwarded to the underlying tool,
# e.g.  ./run.sh verify clip.mp4 --annotated out.mp4 --device 0
#        ./run.sh agents "a robot drops a box" --count 3 --canvas
#
set -euo pipefail

# Always operate from the project root (the dir this script lives in).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "error: $PY not found. Create the venv first:" >&2
    echo "  uv venv --python 3.13 .venv && uv pip install -r requirements.txt" >&2
    exit 1
fi

usage() {
    # Print the header comment block (skip the shebang), stopping at the
    # first non-comment line.
    awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
    tests)
        echo "=== Physics test suite ==="
        exec "$PY" -m pytest tests -q "$@"
        ;;

    verify)
        if [[ $# -lt 1 ]]; then echo "usage: ./run.sh verify <video.mp4> [flags]" >&2; exit 1; fi
        video="$1"; shift
        echo "=== Verify: $video ==="
        # --gate2 on by default; drop it from the forwarded flags to disable.
        exec "$PY" -m verifier "$video" --gate2 --scenario scenario.example.json "$@"
        ;;

    pipeline)
        if [[ $# -lt 1 ]]; then echo 'usage: ./run.sh pipeline "<prompt>" [flags]' >&2; exit 1; fi
        prompt="$1"; shift
        echo "=== Full pipeline (generate -> verify -> label) ==="
        exec "$PY" run_pipeline.py --prompt "$prompt" "$@"
        ;;

    agents)
        if [[ $# -lt 1 ]]; then echo 'usage: ./run.sh agents "<intention>" [flags]' >&2; exit 1; fi
        intention="$1"; shift
        echo "=== Scenario contract compiler ==="
        exec "$PY" -m agents "$intention" "$@"
        ;;

    api)
        echo "=== FastAPI server ==="
        exec "$PY" -m uvicorn roboss.api:app --host 127.0.0.1 --port 8000 "$@"
        ;;

    all)
        if [[ $# -lt 1 ]]; then echo 'usage: ./run.sh all "<prompt>" [flags]' >&2; exit 1; fi
        prompt="$1"; shift
        echo "########## STEP A: tests ##########"
        "$PY" -m pytest tests -q
        echo
        echo "########## STEP B: full pipeline ##########"
        "$PY" run_pipeline.py --prompt "$prompt" "$@"
        ;;

    help|-h|--help)
        usage
        ;;

    *)
        echo "error: unknown command '$cmd'" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac
