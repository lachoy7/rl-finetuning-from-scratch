#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m src.train.train_sft \
  --resume-checkpoint "${RESUME_CHECKPOINT:-./sft_smoltalk_e1_18k}" \
  --output-dir "${OUTPUT_DIR:-./sft_smoltalk_e1_28k}" \
  --train-start "${TRAIN_START:-300000}" \
  --train-end "${TRAIN_END:-455000}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-1}" \
  --lr "${LR:-1e-7}" \
  --accum-steps "${ACCUM_STEPS:-16}"
