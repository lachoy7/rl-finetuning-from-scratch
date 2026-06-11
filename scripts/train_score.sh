#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m src.train.train_score \
  --student-checkpoint "${STUDENT_CHECKPOINT:-./dpo_model}" \
  --stage "${STAGE:-all}" \
  --max-samples "${MAX_SAMPLES:-2000}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --lr "${LR:-1e-7}"
