#!/usr/bin/env bash
# Idea 1's sampling comparison (docs/journal/2026-09-06-ask-latency-round-two.md):
# baseline (current defaults, stochastic), top_k=20 (stochastic), and
# temperature=0.0 (greedy — no RNG draw, exactly reproducible run to run).
#
# Each config writes to its own runs/epochs/ directory and gets its own
# evals/epoch_report.py summary, printed separately rather than combined:
# epoch_report.py groups by a run's "model"/"prompt_variant"/"backend"
# fields only, none of which would tell two sampling configs apart, so
# feeding it multiple configs' directories at once would silently merge
# them into one row.
#
# Per the entry's measurement protocol, greedy is the one config that makes
# every later comparison exact — run it first and treat it as the
# reference point, not baseline/top_k=20 against each other (both are
# stochastic and not cleanly comparable to each other run-to-run).
#
# Usage: scripts/compare-sampling.sh MODEL [EPOCHS]
# Example: scripts/compare-sampling.sh qwen3.5-4b 20
set -euo pipefail

MODEL="${1:?usage: scripts/compare-sampling.sh MODEL [EPOCHS]}"
EPOCHS="${2:-20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run sumac models pull

CONFIGS=(greedy top_k20 baseline)
declare -A EXTRA_ARGS=(
  [greedy]="--eval-temperature 0.0"
  [top_k20]="--eval-top-k 20"
  [baseline]=""
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
