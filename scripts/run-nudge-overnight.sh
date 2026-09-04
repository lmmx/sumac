#!/usr/bin/env bash
# The full default/nudge-v2/nudge-v3/nudge-v4 comparison across both
# registered models — docs/journal/2026-09-02-eval-suite.md's PromptVariant
# entries. Full suite (all 22 scenarios via `pytest evals`, not test_add.py
# alone), same seed range for every (model, variant) cell, output kept in
# separate directories per cell so nothing gets mixed together. Specific to
# this one overnight run, not general infra.
#
# Full-suite epochs are slow (~84s on 9B, ~40s on 4B) — 2 models x 4
# variants x EPOCHS is a couple of hours, not the ~15s/epoch the earlier
# fast (and, it turned out, RNG-position-corrupted — see the journal)
# narrowed run showed. That's expected; this is meant to run unattended.
#
# Deliberately new output directories (full-<model>-<variant>), not reused
# from the earlier reconnaissance runs (runs/epochs/qwen3.5-9b-default,
# runs/epochs/qwen3.5-9b-nudge-v2) — those were scoped to test_add.py only,
# a different scenario set, and mixing scopes into one directory would
# corrupt epoch_report.py's aggregation. The old ones are left alone.
set -euo pipefail

MODELS=(qwen3.5-4b qwen3.5-9b)
VARIANTS=(default nudge-v2 nudge-v3 nudge-v4)
EPOCHS=15

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run sumac models pull

out_dirs=()
for model in "${MODELS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    out_dir="runs/epochs/full-${model}-${variant}"
    out_dirs+=("$out_dir")
    mkdir -p "$out_dir"
    for seed in $(seq 1 "$EPOCHS"); do
      echo "==> ${model} [${variant}] epoch ${seed}/${EPOCHS}"
      uv run pytest evals --eval-model "$model" --eval-prompt-variant "$variant" \
        --eval-seed "$seed" \
        --eval-json "${out_dir}/epoch-$(printf '%02d' "$seed").json" || true
    done
  done
done

echo "==> summary"
uv run python -m evals.epoch_report "${out_dirs[@]}"
