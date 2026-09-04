#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: scripts/eval-prompt-variant.sh MODEL VARIANT [EPOCHS]}"
VARIANT="${2:?usage: scripts/eval-prompt-variant.sh MODEL VARIANT [EPOCHS]}"
EPOCHS="${3:-20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="runs/epochs/verify-${MODEL}-${VARIANT}-${EPOCHS}"

mkdir -p "$OUT_DIR"

for seed in $(seq 1 "$EPOCHS"); do
  echo "==> ${MODEL} [${VARIANT}] epoch ${seed}/${EPOCHS}"

  uv run pytest evals \
    --eval-model "$MODEL" \
    --eval-prompt-variant "$VARIANT" \
    --eval-seed "$seed" \
    --eval-json "${OUT_DIR}/epoch-$(printf '%02d' "$seed").json" || true
done

echo "==> summary"
uv run python -m evals.epoch_report "$OUT_DIR"