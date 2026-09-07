#!/usr/bin/env bash
# Idea 1's sampling comparison (docs/journal/2026-09-06-ask-latency-round-two.md).
# `llm.DEFAULT_TOP_K` is now 20 (measured: +28-30% tok/s over unset, 100%
# pass rate, 3 epochs — see the journal entry's "Current State" addendum),
# so "default" below is *not* the pre-idea-1 baseline any more; "no-top-k"
# reproduces that explicitly via `--eval-top-k 0`, mistral.rs's own spelling
# for "no limit" (`sampler.rs`: `top_k <= 0` full-sorts the vocabulary).
#
# Each config writes to its own runs/epochs/ directory and gets its own
# evals/epoch_report.py summary, printed separately rather than combined:
# epoch_report.py groups by a run's "model"/"prompt_variant"/"backend"
# fields only, none of which would tell two sampling configs apart, so
# feeding it multiple configs' directories at once would silently merge
# them into one row.
#
# Per the entry's measurement protocol, greedy is the one config that makes
# every later comparison exact — treat it as the reference point, not
# default/no-top-k against each other (both stochastic, not cleanly
# comparable to each other run-to-run).
#
# Usage: scripts/compare-sampling.sh MODEL [EPOCHS]
# Example: scripts/compare-sampling.sh qwen3.5-4b 20
set -euo pipefail

MODEL="${1:?usage: scripts/compare-sampling.sh MODEL [EPOCHS]}"
EPOCHS="${2:-20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run sumac models pull

CONFIGS=(greedy default no-top-k)
declare -A EXTRA_ARGS=(
  [greedy]="--eval-temperature 0.0"
  [default]=""
  [no-top-k]="--eval-top-k 0"
)

for config in "${CONFIGS[@]}"; do
  out_dir="runs/epochs/sampling-${MODEL}-${config}-${EPOCHS}"
  mkdir -p "$out_dir"
  for seed in $(seq 1 "$EPOCHS"); do
    echo "==> ${MODEL} [${config}] epoch ${seed}/${EPOCHS}"
    # shellcheck disable=SC2086
    uv run pytest evals --eval-model "$MODEL" --eval-seed "$seed" \
      ${EXTRA_ARGS[$config]} \
      --eval-json "${out_dir}/epoch-$(printf '%02d' "$seed").json" || true
  done
done

for config in "${CONFIGS[@]}"; do
  echo
  echo "==> summary: ${config}"
  uv run python -m evals.epoch_report "runs/epochs/sampling-${MODEL}-${config}-${EPOCHS}"
done
