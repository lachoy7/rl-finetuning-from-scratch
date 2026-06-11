#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for reward model evaluation}"

python -m src.eval.eval_sft \
  --model-path "${MODEL_PATH:-./nonlora_sft_smoltalk_9k_b8_lr1e-5_ga32}" \
  --reference "${REFERENCE:-Qwen/Qwen2.5-0.5B-Instruct}" \
  --num-samples "${NUM_SAMPLES:-100}" \
  --batch-size "${BATCH_SIZE:-8}"
