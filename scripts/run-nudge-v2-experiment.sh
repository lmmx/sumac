#!/usr/bin/env bash
# The default-vs-nudge-v2 comparison from
# docs/journal/2026-09-02-eval-suite.md's PromptVariant entries — qwen3.5-9b
# only, 10 epochs each, same seed range for both conditions, output kept in
# separate directories so the two runs can't get mixed together. Specific to
# this one experiment, not general infra — scripts/epoch-benchmark.sh already
# covers "N epochs x every registered model"; this is deliberately narrower
# and not meant to grow into a generic variant-comparison tool.
#
# -k restricts each epoch to the two scenarios the nudge actually targets
# (via plain pytest test-selection, nothing new) — a full 22-scenario epoch
# takes ~1m24s on 9B, so 10 epochs x 2 conditions is ~28 minutes; these two
# alone run in a fraction of that, for fast iteration on the wording itself.
# Before trusting a result, rerun without -k (the full suite) to confirm
# nothing else regressed — narrowing the scenarios narrows what you learn.
set -euo pipefail

MODEL=qwen3.5-9b
EPOCHS=10
SCENARIOS="test_missing_item_discovers_new_product or test_multiple_products_with_omitted_amounts"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run sumac models pull "$MODEL"

for variant in default nudge-v2; do
  out_dir="runs/epochs/${MODEL}-${variant}"
  mkdir -p "$out_dir"
  for seed in $(seq 1 "$EPOCHS"); do
    echo "==> ${MODEL} [${variant}] epoch ${seed}/${EPOCHS}"
    uv run pytest evals -k "$SCENARIOS" --eval-model "$MODEL" --eval-prompt-variant "$variant" \
      --eval-seed "$seed" --eval-json "${out_dir}/epoch-$(printf '%02d' "$seed").json" || true
  done
done

echo "==> summary"
uv run python -m evals.epoch_report "runs/epochs/${MODEL}-default" "runs/epochs/${MODEL}-nudge-v2"
