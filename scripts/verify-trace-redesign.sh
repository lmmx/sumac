#!/usr/bin/env bash
# Verifies the trace/verdict redesign and the RNG-cascade shuffle actually
# earn their keep, on the current `default` prompt only — no variant
# comparison here (`PROMPT_VARIANTS` is back down to just "default"; see
# docs/journal/2026-09-04-trace-and-verdict-redesign.md's "Decision"). This
# is not a re-attempt at the original wording question — that needs a real
# candidate variant back in the registry first, a separate decision from
# this one.
#
# Point of this run: confirm the per-scenario `log` sidecar actually shows
# what happened on the target scenario (default: the original
# add.multiple_products_with_omitted_amounts regression) — specifically
# whether `_maybe_force_action`'s nudge fired and what the model did after
# it — instead of that having to be inferred from `checks.writes` after
# the fact, which is what cost a night of GPU time last time with nothing
# to show for it. Also runs with `--eval-seed` set every epoch, so
# `pytest_collection_modifyitems`'s reproducible shuffle is live
# throughout.
#
# The verdict/metrics file (`epoch-NN.json`) and the execution-log sidecar
# (`epoch-NN.log.jsonl`, one JSON object per line, joined back by
# `scenario`) are separate files as of the trace/verdict redesign's JSONL
# split — see docs/journal/2026-09-04-trace-and-verdict-redesign.md — so
# pulling one scenario's story means reading both and joining on `scenario`,
# which is what the loop below does.
set -euo pipefail

EPOCHS="${1:-5}"
SCENARIO="${2:-add.multiple_products_with_omitted_amounts}"
SCENARIO_NODEID="${3:-evals/test_add.py::test_multiple_products_with_omitted_amounts}"
MODEL="qwen3.5-9b"
OUT_DIR="runs/epochs/verify-${MODEL}-default"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run sumac models pull

mkdir -p "$OUT_DIR"
for seed in $(seq 1 "$EPOCHS"); do
  echo "==> epoch ${seed}/${EPOCHS}"
  uv run pytest evals --eval-model "$MODEL" --eval-seed "$seed" \
    --eval-json "${OUT_DIR}/epoch-$(printf '%02d' "$seed").json" || true
done

echo "==> summary"
uv run python -m evals.epoch_report "$OUT_DIR"

echo "==> ${SCENARIO}, every epoch"
for f in "${OUT_DIR}"/epoch-*.json; do
  log="${f%.json}.log.jsonl"
  jq -n --arg scenario "$SCENARIO" --arg nodeid "$SCENARIO_NODEID" \
    --slurpfile meta "$f" \
    --argjson log "$(jq -c --arg s "$SCENARIO" 'select(.scenario == $s)' "$log")" '
    $meta[0] as $m
    | {
        seed: $m.seed,
        order_position: ($m.scenario_order | index($nodeid)),
        verdict: ($m.results[] | select(.scenario == $scenario) | .verdict),
        terminal: $log.terminal,
        nudge_fired: $log.nudge_fired,
        trace_names: [$log.trace[].name]
      }
  '
done | jq -s .
