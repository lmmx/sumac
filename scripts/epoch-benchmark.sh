#!/usr/bin/env bash
# One epoch = one pytest session at its own --eval-seed, for every model in
# the registry. Deliberately N separate processes, not N repetitions inside
# one session sharing a loaded model: mistralrs.ChatCompletionRequest has no
# seed field, only Runner.__init__ does — sampling from one shared Runner
# across repeated scenarios means the RNG stream position at attempt k
# depends on how many tokens every prior attempt generated, so reordering or
# filtering scenarios would silently change results. A fresh process per
# epoch is what makes each one reproduce exactly from its own seed alone —
# see docs/journal/2026-09-02-eval-suite.md's original "Epochs are separate
# pytest sessions" section, from before this same machinery (along with a
# lot more that never earned its keep at this suite's size) was deleted.
#
# Writes one JSON per epoch under runs/epochs/<model>/, so one bad epoch
# doesn't lose the rest of the run, then prints the aggregate.
set -euo pipefail

EPOCHS="${1:?usage: scripts/epoch-benchmark.sh N_EPOCHS}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run sumac models pull

uv run sumac models list --names-only | while read -r name; do
  out_dir="runs/epochs/${name}"
  mkdir -p "$out_dir"
  for seed in $(seq 1 "$EPOCHS"); do
    echo "==> ${name} epoch ${seed}/${EPOCHS}"
    uv run pytest evals --eval-model "$name" --eval-seed "$seed" \
      --eval-json "${out_dir}/epoch-$(printf '%02d' "$seed").json" || true
  done
done

echo "==> summary"
uv run python -m evals.epoch_report runs/epochs/*/
