#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for reward model evaluation}"

python -m src.eval.eval_dpo \
  --sft-model "${SFT_MODEL:-./sft_smoltalk_model}" \
  --dpo-model "${DPO_MODEL:-./temp_dpo_model_epoch2}" \
  --num-samples "${NUM_SAMPLES:-100}" \
  --batch-size "${BATCH_SIZE:-8}" \
  ${SFT_COMPLETIONS_PICKLE:+--sft-completions-pickle "$SFT_COMPLETIONS_PICKLE"}
