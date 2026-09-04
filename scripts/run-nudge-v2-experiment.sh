#!/usr/bin/env bash
# The default-vs-nudge-v2 comparison from
# docs/journal/2026-09-02-eval-suite.md's PromptVariant entries — qwen3.5-9b
# only, 10 epochs each, same seed range for both conditions, output kept in
# separate directories so the two runs can't get mixed together. Specific to
# this one experiment, not general infra — scripts/epoch-benchmark.sh already
# covers "N epochs x every registered model"; this is deliberately narrower
# and not meant to grow into a generic variant-comparison tool.
#
# Runs evals/test_add.py only (10 scenarios), not the full 22 — but NOT via
# -k to just the two target scenarios either. `mistralrs.Runner(seed=...)`
# is seeded once per session and shared by every test in it (no per-request
# seed), so a test's effective sample depends on how many tokens every test
# that ran before it in the same session already generated. -k'ing straight
# to the two target tests skips their real predecessors and samples from a
# different point in the stream than the full-suite discovery run did —
# that's what produced the misleadingly-clean 10/10 result. Nothing in
# test_find.py/test_remove.py/test_reject.py/test_fixtures.py/
# test_termination.py runs before test_add.py either way (alphabetical
# collection order), so running the whole file reproduces the exact same
# RNG position for both target tests as the full suite would, while still
# skipping the other three-fifths of the scenario count.
set -euo pipefail

MODEL=qwen3.5-9b
EPOCHS=10

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run sumac models pull "$MODEL"

for variant in default nudge-v2; do
  out_dir="runs/epochs/${MODEL}-${variant}"
  mkdir -p "$out_dir"
  for seed in $(seq 1 "$EPOCHS"); do
    echo "==> ${MODEL} [${variant}] epoch ${seed}/${EPOCHS}"
    uv run pytest evals/test_add.py --eval-model "$MODEL" --eval-prompt-variant "$variant" \
      --eval-seed "$seed" --eval-json "${out_dir}/epoch-$(printf '%02d' "$seed").json" || true
  done
done

echo "==> summary"
uv run python -m evals.epoch_report "runs/epochs/${MODEL}-default" "runs/epochs/${MODEL}-nudge-v2"
