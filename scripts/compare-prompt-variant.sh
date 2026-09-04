#!/usr/bin/env bash
# Cross-model comparison of one PromptVariant against "default" — the same
# shape as epoch-benchmark.sh (every registered model, N epochs each, full
# suite so collection order/RNG-cascade stays controlled — see
# docs/journal/2026-09-04-trace-and-verdict-redesign.md), just crossed with
# a second variant axis and handed straight to epoch_report.py, which
# already groups by each file's own "model"/"prompt_variant" fields.
#
# Usage: scripts/compare-prompt-variant.sh VARIANT_NAME [EPOCHS]
# Example: scripts/compare-prompt-variant.sh add-amount-delta 5
set -euo pipefail

VARIANT="${1:?usage: scripts/compare-prompt-variant.sh VARIANT_NAME [EPOCHS]}"
EPOCHS="${2:-5}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run sumac models pull

OUT_DIRS=()
mapfile -t MODELS < <(uv run sumac models list --names-only)

for model in "${MODELS[@]}"; do
  for variant in default "$VARIANT"; do
    out_dir="runs/epochs/compare-${model}-${variant}"
    OUT_DIRS+=("$out_dir")
    mkdir -p "$out_dir"
    for seed in $(seq 1 "$EPOCHS"); do
      echo "==> ${model} [${variant}] epoch ${seed}/${EPOCHS}"
      uv run pytest evals --eval-model "$model" --eval-prompt-variant "$variant" \
        --eval-seed "$seed" \
        --eval-json "${out_dir}/epoch-$(printf '%02d' "$seed").json" || true
    done
  done
done

echo "==> summary (all models x variants)"
uv run python -m evals.epoch_report "${OUT_DIRS[@]}"
