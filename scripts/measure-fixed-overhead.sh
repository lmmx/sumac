#!/usr/bin/env bash
# Idea 20's micro-benchmark (docs/journal/2026-09-06-ask-latency-round-two.md):
# prices the fixed per-request cost mistral.rs pays before the first
# completion token, isolated from decode. Three probes, from Python, no Rust
# changes needed:
#
#   1. Same prompt, max_tokens=1, sent 50x after a warmup. The minimum is
#      handoff + scheduler + tokenise + one forward pass, with prefill fully
#      cached.
#   2. The same, with growing filler text prepended to the system prompt
#      (each length prefilled once to warm the cache first). If the
#      per-request floor grows with prompt length even though prefill is
#      cached, that's re-tokenisation cost, not engine overhead.
#   3. The same request with and without a grammar constraint, to isolate
#      per-request llguidance parser construction.
#
# Talks to `_LocalMistralRsBackend` directly — no `AgentRunner`, no vault, no
# tool schemas — so a request here costs nothing but the engine's own fixed
# overhead. Loads the real model: run this yourself, it is not run by an
# agent on your behalf.
#
# Usage: scripts/measure-fixed-overhead.sh MODEL
# Example: scripts/measure-fixed-overhead.sh qwen3.5-4b
set -euo pipefail

MODEL="${1:?usage: scripts/measure-fixed-overhead.sh MODEL}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run python -c "
import time
from sumac import llm

model = llm.model_preset('${MODEL}')
print(f'loading {model.quantized_model_id}...')
backend = llm._build_runner(model, seed=0)

def send(messages, max_tokens=1, grammar=None, grammar_type=None):
    request = {
        'messages': messages,
        'model': model.quantized_model_id,
        'tool_schemas': [],
        'tool_choice': 'auto',
        'enable_thinking': False,
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': None,
        'min_p': None,
        'max_tokens': max_tokens,
        'grammar': grammar,
        'grammar_type': grammar_type,
        'seed': None,
    }
    start = time.perf_counter()
    backend.send_chat_completion_request(request)
    return time.perf_counter() - start

SYSTEM = 'You are a helpful household inventory assistant.'
PROMPT = 'Say hello.'
FILLER_UNIT = 'the quick brown fox jumps over the lazy dog. '

print()
print('--- 1. cache-hit floor: same prompt, max_tokens=1, min of 50 ---')
base_messages = [
    {'role': 'system', 'content': SYSTEM},
    {'role': 'user', 'content': PROMPT},
]
send(base_messages)  # warm the prefix cache
times = sorted(send(base_messages) for _ in range(50))
print(f'min={times[0]*1000:.2f}ms median={times[25]*1000:.2f}ms max={times[-1]*1000:.2f}ms')

print()
print('--- 2. does the floor grow with prompt length? (prefill stays cached) ---')
for n_repeat in (5, 50, 200, 400, 800):
    filler = FILLER_UNIT * n_repeat
    messages = [
        {'role': 'system', 'content': SYSTEM + ' ' + filler},
        {'role': 'user', 'content': PROMPT},
    ]
    send(messages)  # warm this length's prefill once
    reps = sorted(send(messages) for _ in range(10))
    print(f'~{len(filler):6d} filler chars: min={reps[0]*1000:.2f}ms median={reps[5]*1000:.2f}ms')

print()
print('--- 3. grammar construction: same request, with vs without a grammar ---')
plain = sorted(send(base_messages) for _ in range(20))
grammared = sorted(
    send(base_messages, max_tokens=8, grammar=llm._CLASSIFY_GRAMMAR, grammar_type='regex')
    for _ in range(20)
)
print(f'no grammar:   min={plain[0]*1000:.2f}ms median={plain[10]*1000:.2f}ms')
print(f'with grammar: min={grammared[0]*1000:.2f}ms median={grammared[10]*1000:.2f}ms')

print()
print('Re-run with RUST_LOG=info uv run ... to see mistral.rs\'s own per-request')
print('breakdown alongside these numbers.')
"
