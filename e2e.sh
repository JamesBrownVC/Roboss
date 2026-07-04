#!/usr/bin/env bash
#
# e2e.sh — full end-to-end run of the Synthetic Action Dataset Compiler.
#
#   intention ──▶ scenario compiler (agents) ──▶ N scenario packets
#                                                       │
#                        for each packet: generate ──▶ verify ──▶ label
#
# Every scenario gets its own output folder with the generated .mp4, the
# verification report.json, and (if accepted) labels.json.
#
# Usage:
#   ./e2e.sh "<intention>" [count] [run_name]
#
# Examples:
#   ./e2e.sh "a human slips near a robot carrying a box"
#   ./e2e.sh "forklift nearly hits a worker" 3 forklift_test
#
set -euo pipefail

# ---- locate project + interpreter -----------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

# Bootstrap the venv if it is missing (needs `uv` + Python 3.13).
if [[ ! -x "$PY" ]]; then
    echo "[e2e] venv not found — creating it with uv (Python 3.13)..."
    uv venv --python 3.13 .venv
    uv pip install --python "$PY" -r requirements.txt
fi

# ---- arguments -------------------------------------------------------------
INTENTION="${1:-}"
COUNT="${2:-}"
RUN_NAME="${3:-run_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "$INTENTION" ]]; then
    echo 'usage: ./e2e.sh "<intention>" [count] [run_name]' >&2
    exit 1
fi

RUN_DIR="runs/$RUN_NAME"
BUNDLE_DIR="$RUN_DIR/scenarios"
PACKETS_DIR="$BUNDLE_DIR/verifier_packets"
mkdir -p "$RUN_DIR"

echo "############################################################"
echo "# END-TO-END RUN: $RUN_NAME"
echo "# intention: $INTENTION"
echo "# output   : $RUN_DIR"
echo "############################################################"

# ---- Step 0: sanity tests --------------------------------------------------
echo
echo "===== [0/3] Sanity tests ====="
"$PY" -m pytest tests -q

# ---- Step 1: compile scenarios --------------------------------------------
echo
echo "===== [1/3] Scenario compiler (intention -> scenarios) ====="
AGENT_ARGS=(--out "$BUNDLE_DIR")
[[ -n "$COUNT" ]] && AGENT_ARGS+=(--count "$COUNT")
"$PY" -m agents "$INTENTION" "${AGENT_ARGS[@]}"

if [[ ! -d "$PACKETS_DIR" ]]; then
    echo "[e2e] no verifier packets produced at $PACKETS_DIR — aborting." >&2
    exit 1
fi

mapfile -t PACKETS < <(find "$PACKETS_DIR" -name '*.json' | sort)
echo "[e2e] ${#PACKETS[@]} scenario packet(s) to process."

# ---- Step 2+3: per-scenario generate -> verify -> label --------------------
declare -a SUMMARY
i=0
for packet in "${PACKETS[@]}"; do
    i=$((i + 1))
    sid="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('scenario_id','scenario_'+str($i)))" "$packet")"
    prompt="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['scenario_prompt'])" "$packet")"
    out="$RUN_DIR/$sid"

    echo
    echo "===== [2/3] Scenario $i/${#PACKETS[@]}: $sid ====="
    echo "[e2e] prompt: $prompt"

    # The pipeline exits 2 on a rejected video — that is a valid outcome here,
    # so don't let `set -e` kill the whole run.
    set +e
    "$PY" run_pipeline.py --prompt "$prompt" --scenario "$packet" --outdir "$out"
    code=$?
    set -e

    case $code in
        0) SUMMARY+=("ACCEPT  $sid  -> $out") ;;
        2) SUMMARY+=("REJECT  $sid  -> $out") ;;
        *) SUMMARY+=("ERROR($code)  $sid  -> $out") ;;
    esac
done

# ---- summary ---------------------------------------------------------------
echo
echo "############################################################"
echo "# SUMMARY — $RUN_DIR"
echo "############################################################"
for line in "${SUMMARY[@]}"; do
    echo "  $line"
done
echo
echo "[e2e] done. Videos + reports are under $RUN_DIR/<scenario_id>/"
