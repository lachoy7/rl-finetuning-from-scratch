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

python -m src.train.train_score \
  --student-checkpoint "${STUDENT_CHECKPOINT:-./dpo_model}" \
  --stage "${STAGE:-all}" \
  --max-samples "${MAX_SAMPLES:-2000}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --lr "${LR:-1e-7}" \
  "${WANDB_ARGS[@]}"
