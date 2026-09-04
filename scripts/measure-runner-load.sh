#!/usr/bin/env bash
# Measures how long one AgentRunner._build_runner(model) load takes, to
# decide between RNG-cascade fix option A (rotate collection order — free)
# and option B (one Runner per scenario — costs one model load per
# scenario). See "Open question, not decided" in
# docs/journal/2026-09-04-trace-and-verdict-redesign.md.
#
# Loads the model once first (warms the OS page cache reading the GGUF
# file), then times a second load. Compare the printed "N loads" estimate
# against an existing epoch file's total_duration_s (e.g.
# runs/epochs/qwen3.5-9b-default/epoch-01.json) to see what fraction of a
# full epoch's wall clock option B would add.
set -euo pipefail

MODEL="${1:?usage: scripts/measure-runner-load.sh MODEL_NAME [N_SCENARIOS]}"
N="${2:-25}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run python -c "
import time
from sumac import llm

model = llm.model_preset('${MODEL}')

print(f'loading {model.quantized_model_id} once to warm the page cache...')
llm._build_runner(model, seed=0)

print('timing a second load...')
start = time.perf_counter()
llm._build_runner(model, seed=0)
elapsed = time.perf_counter() - start

print(f'one _build_runner load: {elapsed:.2f}s')
print(f'{$N} loads (one per scenario, option B): {elapsed * $N:.1f}s')
print('compare against total_duration_s in an existing runs/epochs/*/epoch-*.json')
"
