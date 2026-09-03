#!/usr/bin/env bash
# Runs the eval suite once per registry preset (`sumac models list`) and
# prints a pass-rate/latency comparison table. Replaces the old by-hand loop
# (manually editing DEFAULT_MODEL_PRESET, running `sumac ask` once per model
# to prime the HF cache, then a for-loop of `pytest --eval-model`) documented
# in docs/journal/2026-09-02-eval-suite.md's 2026-09-03 entries — see there
# for what each field in the printed table means and how it was chosen.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> pulling any uncached presets"
uv run sumac models pull

mkdir -p runs
while IFS= read -r name; do
  echo "==> benchmarking $name"
  # A scenario failing is an expected, informative outcome here (e.g.
  # add.basmati_rice_in_different_unit is deliberately left failing — see
  # the eval README's Blind Spots) — pytest exits 1 for that, which would
  # otherwise trip `set -e` and abort the rest of this loop.
  uv run pytest evals --eval-model "$name" --eval-json "runs/${name}.json" || true
done < <(uv run sumac models list --names-only)

echo "==> summary"
jq -c -s -f evals/report.jq runs/*.json
