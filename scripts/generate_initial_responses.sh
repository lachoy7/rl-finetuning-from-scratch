#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to the DPO checkpoint directory}"

python -m src.data.generate_initial_responses \
  --model-path "$MODEL_PATH" \
  --data-dir "${DATA_DIR:-.}" \
  --output-dir "${OUTPUT_DIR:-.}"
