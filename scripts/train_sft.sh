#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

python -m src.train.train_sft \
  --resume-checkpoint "${RESUME_CHECKPOINT:-./sft_smoltalk_e1_18k}" \
  --output-dir "${OUTPUT_DIR:-./sft_smoltalk_e1_28k}" \
  --train-start "${TRAIN_START:-300000}" \
  --train-end "${TRAIN_END:-455000}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-1}" \
  --lr "${LR:-1e-7}" \
  --accum-steps "${ACCUM_STEPS:-16}" \
  "${WANDB_ARGS[@]}"
