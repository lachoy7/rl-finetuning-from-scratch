#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${HF_TOKEN:?Set HF_TOKEN for HuggingFace dataset access}"

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
fi
if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
  WANDB_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ "${NO_WANDB:-}" == "1" ]]; then
  WANDB_ARGS+=(--no-wandb)
fi

python -m src.train.train_dpo \
  --base-checkpoint "${BASE_CHECKPOINT:-./sft_smoltalk_e1_28k}" \
  --output-dir "${OUTPUT_DIR:-./final_dpo_modelv2}" \
  --train-size "${TRAIN_SIZE:-60000}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --epochs "${EPOCHS:-2}" \
  --beta "${BETA:-0.5}" \
  --lr "${LR:-1e-6}" \
  --accum-steps "${ACCUM_STEPS:-128}" \
  "${WANDB_ARGS[@]}"
