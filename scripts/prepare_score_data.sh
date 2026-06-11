#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m src.data.prepare_score_data \
  --input "${INPUT:-leaderboard_subs.jsonl}" \
  --output "${OUTPUT:-leaderboard_raw.json}"
