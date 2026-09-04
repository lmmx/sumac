#!/usr/bin/env bash
# Like verify-trace-redesign.sh's per-scenario trace+verdict join, but
# across every registered model (`sumac models list --names-only`) instead
# of qwen3.5-9b only — for checking whether a fix (or a regression) holds
# up the same way on both the 4b and 9b models, not just one of them.
#
# Cross-model *pass-rate* comparison already exists independently of this
# script: scripts/epoch-benchmark.sh runs every registered model and
# evals/epoch_report.py groups its output by each file's own "model" field.
# What that pair doesn't give you is one scenario's actual trace (did the
# nudge fire, what tools ran, in what order) side by side across models —
# that's what this script adds, by running the full suite (collection
# order matters for the RNG-cascade reasons in
# docs/journal/2026-09-04-trace-and-verdict-redesign.md) once per model per
# epoch and then pulling one scenario's story out of each run.
set -euo pipefail

EPOCHS="${1:-5}"
SCENARIO="${2:-add.basmati_rice_in_different_unit}"
SCENARIO_NODEID="${3:-evals/test_add.py::test_basmati_rice_in_different_unit}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run sumac models pull

OUT_DIRS=()
mapfile -t MODELS < <(uv run sumac models list --names-only)

for model in "${MODELS[@]}"; do
  out_dir="runs/epochs/verify-${model}-default"
  OUT_DIRS+=("$out_dir")
  mkdir -p "$out_dir"
  for seed in $(seq 1 "$EPOCHS"); do
    echo "==> ${model} epoch ${seed}/${EPOCHS}"
    uv run pytest evals --eval-model "$model" --eval-seed "$seed" \
      --eval-json "${out_dir}/epoch-$(printf '%02d' "$seed").json" || true
  done
done

echo "==> summary (all models)"
uv run python -m evals.epoch_report "${OUT_DIRS[@]}"

for model in "${MODELS[@]}"; do
  out_dir="runs/epochs/verify-${model}-default"
  echo "==> ${model}: ${SCENARIO}, every epoch"
  for f in "${out_dir}"/epoch-*.json; do
    log="${f%.json}.log.jsonl"
    jq -n --arg scenario "$SCENARIO" --arg nodeid "$SCENARIO_NODEID" \
      --slurpfile meta "$f" \
      --argjson log "$(jq -c --arg s "$SCENARIO" 'select(.scenario == $s)' "$log")" '
      $meta[0] as $m
      | {
          model: $m.model,
          seed: $m.seed,
          order_position: ($m.scenario_order | index($nodeid)),
          verdict: ($m.results[] | select(.scenario == $scenario) | .verdict),
          terminal: $log.terminal,
          nudge_fired: $log.nudge_fired,
          trace_names: [$log.trace[].name]
        }
    '
  done | jq -s .
done
