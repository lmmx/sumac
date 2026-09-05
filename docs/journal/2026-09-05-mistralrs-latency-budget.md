# sumac: `mistralrs` Latency Budget — Where `sumac ask` Actually Spends Its Time

**Status:** the entry itself (everything through "Missing / open threads") is measurement and
analysis only, no code — nothing in `src/sumac/llm.py` changed to produce it. Findings 1, 2, and
7/10 were subsequently implemented in `llm.py`, in a follow-up session on the same branch; see
"Confirmed effects" below for what shipped and what it measured. Follows from
`docs/journal/2026-09-04-modal-remote-inference-backend.md`, which framed local `mistralrs` epoch
wall-clock (~15 min for one 20-epoch prompt-variant verdict) as the iteration bottleneck and
proposed remote inference as the answer. This entry asks the prior question that entry skipped —
*what is the 15 minutes made of* — and answers it from data already committed to this repository.

**Provenance:** every number under "The measured budget" and "Findings" is derived from
`runs/epochs/verify-qwen3.5-4b-default-20/` (20 epochs × 22 scenarios = 440 scenarios, 1,922 engine
requests), the `qwen3.5-4b` / `default` run recorded by
`docs/journal/2026-09-04-trace-and-verdict-redesign.md`'s per-round `usage_history`. No model was
loaded and no inference was run for this entry — the analysis is arithmetic over
`epoch-*.log.jsonl` and `epoch-*.json`. Sections marked **(verified against committed run data)**
are reproducible from those files with the snippets given inline. Sections marked **(verified
against the `mistralrs` stub)** were checked against `mistralrs-pyo3/mistralrs.pyi` at the
`v0.9.2` tag — the version `pyproject.toml`'s `ask`/`ask-cuda` groups pin. Sections marked
**(from mistral.rs `master`, not `v0.9.2`)** were read from upstream `master` documentation and
source and have not been checked against the pinned version.

## The measured budget **(verified against committed run data)**

Per-request latency across all 1,922 requests fits `t = overhead + new_prompt_tokens/P +
completion_tokens/D` at R² = 0.951, where `new_prompt_tokens` is each request's `prompt_tokens`
minus the previous request's in the same `_run_loop` conversation (the portion mistral.rs's prefix
cache cannot serve). The same coefficients recover independently from minimum observed latency
bucketed by `completion_tokens`, restricted to requests whose new-prompt count is ≤ 200 —
intercept 67 ms, slope 8.17 ms/token, over 1,062 such requests.

| component | rate | share of 861.0 s engine time |
| --- | --- | --- |
| decode | 122 tok/s (8.17 ms/token) | 669 s — 78 % |
| fixed per-request | 67 ms × 1,922 requests | 130 s — 15 % |
| prefill | ~6,100 tok/s median | 63 s — 7 % |

Wall clock across the same 20 epochs (summed `total_duration_s`) is 921.6 s against 861.0 s of
engine-reported `total_time_sec` — 60.5 s, 6.6 %, is spent in Python outside `mistralrs`.

Reproduce the split:

```python
import json, glob
pts = []
for f in sorted(glob.glob('runs/epochs/verify-qwen3.5-4b-default-20/epoch-*.log.jsonl')):
    for line in open(f):
        s = json.loads(line); prev = 0
        for i, u in enumerate(s['usage_history']):
            new = u['prompt_tokens'] if (u['round'] == 0 or i == 1) else max(0, u['prompt_tokens'] - prev)
            pts.append((u['completion_tokens'], new, u['total_time_sec'])); prev = u['prompt_tokens']
```

### Prefill is already cheap because the prefix cache is being hit

Prompt tokens across the run total 1,841,989 against 75,948 completion tokens — a 24.3:1 ratio.
Only 625,229 of those prompt tokens (34.0 %) are new relative to the preceding request in the same
conversation. Fitting against the full `prompt_tokens` column instead of the new-token column
yields an implied prefill rate of 58,302 tok/s and a worse fit (R² = 0.932), which is the
signature of a working prefix cache rather than a genuinely fast prefill path.

`_run_loop` (`src/sumac/llm.py:1417-1487`) appends to `self._messages` and never rewrites earlier
entries, so each round's rendered prompt is a strict extension of the previous round's;
`Runner`'s `prefix_cache_n` default of 16 (`mistralrs.pyi`, `v0.9.2`) holds both the classifier
conversation and the domain conversation resident simultaneously. `_maybe_self_review`
(`llm.py:1508-1523`) and `revise()` (`llm.py:1544-1553`) both append rather than rebuild, and
preserve the property.

### Prompt size is close to free; generated tokens and round-trips are not

At 122 tok/s decode against ~6,100 tok/s prefill, one generated token costs the same as ~50
prompt tokens, and one extra round-trip costs a flat 67 ms before any token is generated. Request
count per scenario averages 4.37, by category:

| category | scenarios | mean requests/scenario |
| --- | --- | --- |
| `reject` | 60 | 1.00 |
| `find` | 100 | 3.00 |
| `remove` | 80 | 5.00 |
| `add` | 200 | 5.81 |

Tool-call counts across the 440 scenarios: `classify_request` 440, `sumac_find_inventory` 440,
`sumac_discover_inventory` 280, `sumac_consume_inventory` 40, `sumac_move_inventory` 40.

## Findings

### 1. The self-review pass changes the plan in 1 scenario out of 280 **(verified against committed run data)**

`SELF_REVIEW_ROUNDS = 1` (`llm.py:153`) makes `_maybe_self_review` (`llm.py:1508-1523`) append
`_SELF_REVIEW_MESSAGE` (`llm.py:450`) and re-enter `_run_loop` on every plan carrying writes. It
ran on 280 of 440 scenarios. It emitted a further `<tool_call>` in **1** of those 280 (0.4 %).
`_maybe_self_review:1521-1522` discards the reviewed plan and keeps the original whenever the
review round produces no writes, so in the other 279 the round decodes a mean 29.2 completion
tokens of plain-text agreement and cannot alter the outcome.

Segmenting `usage_history` on `round` number resets (each `_run_loop` call restarts `round_num` at
1, `llm.py:1428`) separates the cost:

- Requests after the first `_run_loop` returns: 348 requests, 12,808 completion tokens, 152.13 s —
  17.7 % of engine time.
- Restricted to the 417 scenarios where `nudge_fired` is `False`, so the segment is self-review
  alone with no `_maybe_force_action` re-run mixed in: 257 requests, 7,704 completion tokens,
  99.03 s — 11.5 % of engine time.
- The remaining 91 requests / 53.09 s belong to the 23 scenarios where `nudge_fired` is `True` and
  mix the nudge re-run with the self-review that follows it.

`_maybe_force_action` (`llm.py:1489-1506`) fired on 23 of 440 scenarios (5.2 %).

### 2. The classifier spends 27 completion tokens to carry a four-way choice **(verified against committed run data)**

Round 0 across 440 scenarios: 440 requests, 11,880 completion tokens (mean 27.0), 144.40 s —
16.8 % of engine time. `_classify` (`llm.py:1368-1415`) obtains the answer as a tool call against
`_CLASSIFY_SCHEMA`, so each response carries the full `<tool_call>` envelope:

```
<tool_call>
{"name": "classify_request", "arguments": {"kind":"remove"}}
</tool_call>
```

Of the 27 tokens, the `kind` value is one. Mean round-0 latency is 0.328 s, against the 0.075 s a
one-token response would cost at the measured 67 ms overhead + 8.17 ms/token.

The classifier's system prompt (`CLASSIFIER_PROMPT`, `llm.py:211`) is byte-identical across every
scenario, so its 395-token prefix is prefix-cache-resident after the first request in a session —
the 144.40 s is decode and per-request overhead, not prefill.

### 3. Generated output is 67–71 % verbatim copy from context **(verified against committed run data)**

Measured over the first 5 epochs (110 scenarios) by greedy longest-substring coverage: for each
assistant message, the fraction of its characters covered by spans of ≥ 8 characters that already
appear in the concatenated `content` of every preceding message in `messages`.

| assistant message kind | characters | covered by ≥8-char spans already in context |
| --- | --- | --- |
| containing `<tool_call>` | 25,481 | 66.8 % |
| plain text | 19,388 | 71.5 % |

The tool-call figure follows from the schemas: `sumac_consume_inventory`, `sumac_move_inventory`
and `sumac_discover_inventory` all require `product_id`, `unit` and location ids that
`_FIND_INVENTORY_SCHEMA`'s description (`llm.py:228-251`) instructs the model to reuse "verbatim"
from a preceding `sumac_find_inventory` result, and `_propose_write` (`llm.py:1174-1297`) rejects
invented ones. The tool name, the JSON key names and the `<tool_call>` envelope are themselves
present in the request's `tool_schemas` and in earlier rounds' assistant turns.

Plain-text replies average 113 characters (~28 tokens) over 171 measured replies.

### 4. `_build_request` sends 9 of `ChatCompletionRequest`'s ~40 fields **(verified against the `mistralrs` stub)**

`_build_request` (`llm.py:1310-1338`) emits `messages`, `model`, `tool_schemas`, `tool_choice`,
`enable_thinking`, `temperature`, `top_p`, `max_tokens`, `seed`;
`_LocalMistralRsBackend.send_chat_completion_request` (`llm.py:783-806`) forwards eight of those
and drops `seed`. `mistralrs.pyi` at `v0.9.2` declares `ChatCompletionRequest` fields that no
`sumac` call site sets, among them:

- `grammar: str | None` and `grammar_type: str | None` — constrained decoding.
- `logit_bias: dict[int, float] | None`, `logprobs: bool`, `top_logprobs: int | None`.
- `top_k: int | None`, `min_p: float | None`, `stop_seqs: list[str] | None`, `ignore_eos: bool`.
- `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `dry_multiplier`, `dry_base`,
  `dry_allowed_length`, `dry_sequence_breakers`.
- `max_tool_rounds: int | None`, `tool_dispatch_url: str | None` — the server-side tool loop the
  module docstring (`llm.py:31-44`) records as deliberately unused.

`docs/journal/2026-09-04-modal-remote-inference-backend.md`'s open thread 6 raises the unset
sampling fields as a Modal/local *parity* question. The same unset fields are also the entire
constrained-decoding surface, which finding 2 bears on.

### 5. `Runner` is constructed with `which` and `seed` only **(verified against the `mistralrs` stub)**

`_build_runner` (`llm.py:809-828`) passes `which=mistralrs.Which.GGUF(...)` and `seed=`, leaving
every other `Runner.__init__` parameter at its default. `mistralrs.pyi` at `v0.9.2` declares,
among others: `max_seqs=16`, `no_kv_cache=False`, `prefix_cache_n=16`, `mtp_model=None`,
`mtp_n_predict=None`, `num_device_layers=None`, `in_situ_quant=None`, `pa_gpu_mem=None`,
`pa_gpu_mem_usage=None`, `pa_ctxt_len=None`, `pa_blk_size=None`, `pa_cache_type=None`,
`no_paged_attn=False`, `paged_attn=False`.

`master`'s stub documents `paged_attn` as "enables PagedAttention on Metal" and `no_paged_attn` as
"disables PagedAttention on CUDA" — PagedAttention is on by default under the `ask-cuda` build
(`scripts/build-mistralrs-cuda.sh`) with the current all-defaults construction. **(from
mistral.rs `master`, not `v0.9.2`)**

### 6. `mistralrs-pyo3` exposes no way to enable a model's built-in MTP head **(from mistral.rs `master`, not `v0.9.2`)**

Upstream's speculative-decoding guide records two paths: an external assistant checkpoint, and a
built-in multi-token-prediction head shipped inside Qwen3.5/Qwen3.8 checkpoints as `mtp.*` weights
sharing the target's embeddings and `lm_head`, enabled by a `--mtp` flag with no separate
assistant model. `mtp_n_predict` defaults to 2 for built-in heads and 6 for external assistants.
The guide states the target must run with PagedAttention (non-paged KV-cache MTP is disabled),
that every accepted token is verified by the target model before emission, and that acceptance is
content-dependent.

`mistralrs-server-core/src/mistralrs_for_server_builder.rs` on `master` carries
`mtp_config: Option<MtpConfig>`, calls `.with_mtp(self.mtp_config.as_ref().is_some_and(MtpConfig::is_builtin))`
on the loader, and attaches `SpeculativeConfig::Mtp(...)` post-load.

`Runner.__init__` in `mistralrs-pyo3/mistralrs.pyi` exposes only `mtp_model: str | None`
("attaches an MTP assistant from a model id or path") and `mtp_n_predict: int | None`. No boolean
or sentinel corresponding to `MtpConfig::is_builtin` appears in the Python signature, on `master`
or at `v0.9.2`. The built-in-head path is reachable from the CLI and the Rust builder and not from
the Python `Runner` this repository uses.

Upstream also documents that UQFF artifacts must include the `mtp.*` tensors for `--mtp` to work.
Whether `unsloth/Qwen3.5-4B-GGUF`'s `Qwen3.5-4B-Q4_K_M.gguf` (`MODEL_PRESETS`, `llm.py:110-111`)
carries them was not checked in this entry.

`EricLBuehler/mistral.rs#2125` reports GGUF loading failing for Qwen3.5 with `Unknown GGUF
architecture qwen35`. This repository's own committed epoch runs load
`Qwen3.5-4B-Q4_K_M.gguf` under `mistralrs==0.9.2` and produce completions, so the reported failure
does not reproduce on the pinned version with this file.

### 7. `ledger.build_inventory` re-reads the sealed log on every search **(verified against source)**

`_sumac_find_inventory` (`llm.py:1110-1172`) calls `ledger.build_inventory` (`llm.py:1138`) and
`ledger.load_locations_or_empty` (`llm.py:1139`) on every non-repeated query, and `_propose_write`
calls `ledger.build_inventory` again (`llm.py:1205`). `self._searched` (`llm.py:1122-1136`) caches
the *rendered result* per query string within one `propose`/`revise` call, so a repeated identical
query skips the rebuild; a different query string against unchanged data does not. Nothing between
`propose()` and `commit()` writes to the store — `commit()` (`llm.py:1555-1583`) is the only
`store.append` call site in the module, and it reloads state itself (`llm.py:1564-1565`).

440 `sumac_find_inventory` calls plus 360 `_propose_write` calls fall inside the 60.5 s of
non-engine wall clock measured above.

### 8. Two prompt instructions encode retry logic as extra round-trips **(verified against source)**

`_ADD_PROMPT` (`llm.py:380-407`) instructs, at `llm.py:390-395`: "search for it twice if the first
search finds nothing: once with the full name, then with the brand dropped and only the product
itself kept". Each such second search is one `_run_loop` iteration — 67 ms overhead plus the
tokens to emit the call. `sumac_find_inventory` fires 440 times across 380 non-`reject` scenarios
(mean 1.16 per scenario), so the second search occurs on roughly 60 of 440 scenarios.

`_FIND_INVENTORY_SCHEMA`'s description (`llm.py:228-251`) instructs "Search for the place before
writing to it if you do not already have its id" — the location-lookup half of the same shape,
already partly absorbed into the tool by `_location_candidates` (`llm.py:711-728`) and by
`_sumac_find_inventory` returning `locations` alongside `products` (`llm.py:1162-1172`).

## Projected effect of the findings, if acted on

Arithmetic only, applying the measured 67 ms/request and 8.17 ms/token to the same 440 scenarios.
No variant of this has been run; every row is a prediction, not a measurement, and each depends on
the `evals/` pass rate holding at its current 100 %.

| change | engine time |
| --- | --- |
| baseline (measured) | 861 s |
| self-review removed or gated (finding 1) | ~725 s |
| classifier reduced to a 1–2 token response (finding 2) | ~614 s |
| speculative decode at 1.6× on the remainder (findings 3, 6) | ~449 s |

The 1.6× figure is an assumption applied to findings 3 and 6 together, not a measured acceptance
rate for this workload on this model.

Finding 7 falls in the 60.5 s of non-engine wall clock, not in the engine budget above.

## Confirmed effects **(measured against `evals/`, not the run analyzed above)**

Findings 1, 2, and 7/10 shipped in `llm.py`, each measured against the 22-scenario `evals/` suite
(`uv run pytest evals`) — a different, much smaller harness than the 440-scenario
`runs/epochs/verify-qwen3.5-4b-default-20/` run this entry otherwise analyzes. These numbers
confirm direction and that pass rate holds; they are not comparable in magnitude to the
"Projected effect" table above, which was fit against the larger run.

| change | `evals/` wall clock | pass rate |
| --- | --- | --- |
| baseline | ~45 s | 22/22 |
| self-review gated (finding 1) + `build_inventory` memoized (finding 7/10) | 41 s | 22/22 |
| classifier switched to `grammar`/`grammar_type` (finding 2) | 33 s, then 36 s on a rerun | 22/22 |

Each shipped as a narrower or differently-shaped change than the finding above describes verbatim:

- **Finding 1** was gated, not removed. `_maybe_self_review` skips the review round only when the
  plan has exactly one write and that write's `product_id`/`unit`/`from_location` all trace back to
  a prior `sumac_find_inventory` result (`_write_is_grounded`, `llm.py`). A plan with more than one
  write, or any write not grounded in a search, still gets reviewed. Narrower than open thread 4's
  `SELF_REVIEW_ROUNDS = 0` by design — it targets the specific 0.4 %-effective case finding 1
  measured, rather than removing the check for every plan.
- **Finding 2** used `grammar`/`grammar_type="regex"` (`_CLASSIFY_GRAMMAR = "find|add|remove|reject"`,
  `max_tokens=8`), not `logit_bias` + `max_tokens=1` as open thread 6 also considered. `_classify`
  now reads `message.content` directly instead of parsing a `classify_request` tool call;
  `_CLASSIFY_SCHEMA` is gone, and its per-kind `enum` descriptions moved into `CLASSIFIER_PROMPT`
  so that guidance isn't lost along with the schema. `classify_messages`' recorded assistant turn
  changed shape (a bare word, not a `<tool_call>` envelope) — nothing in `evals/` or
  `tests/test_llm.py` asserted on that exact shape, only on `.name`/role, so nothing downstream
  broke. `modal_backend.py`'s vLLM translation was **not** updated to forward `grammar`/
  `grammar_type` — a Modal-backed classify call would have run unconstrained (existing no-tools
  code path, doesn't crash, just isn't grammar-enforced) — left as-is at the time, since the Modal
  backend wasn't the active path and was already the slower of the two. Moot as of the same
  session: the Modal backend was removed entirely, `modal_backend.py` included — see
  docs/journal/2026-09-04-modal-remote-inference-backend.md's retraction note.
- **Finding 7/10** (`build_inventory` memoization) shipped as designed: one read per
  `propose()`/`revise()` call, cached on `AgentRunner` and reset at the top of each; `commit()`
  untouched, since it deliberately re-reads state per write.

Finding 8 (the brand-drop retry) was **not** implemented. "Drop the first word" was rejected as a
heuristic — a multi-word brand ("Ben & Jerry's") breaks it — and the alternative (reversing
`search_inventory`'s substring-match direction so a *stored* product name can match as a substring
of a *longer* query, rather than only the reverse) changes shared matching semantics `sumac find`
and `evals/` both depend on. Left open pending a design decision, not a heuristic fix; open thread
9 below is unchanged.

## Missing / open threads, ranked by expected value

Original ranking, kept for the reasoning; status noted per item after the "Confirmed effects"
implementation session. Live candidates for a next session are 7, 8, and 11.

1. **[declined — not pursuing mistral.rs/llama.cpp internals]** The roofline measurement that
   decides whether contributing to mistral.rs is worth anything here. Decode measures 122 tok/s for
   `Qwen3.5-4B-Q4_K_M.gguf`. Running the same file on the same GPU through `llama.cpp` bounds the
   gap: a materially higher `llama.cpp` number localises it to candle's k-quant dequantisation
   kernels against `llama.cpp`'s MMQ/mmvq path and puts a ceiling on what a kernel contribution
   could return; a comparable number means the workload is memory-bandwidth-bound and no kernel
   change helps, leaving only token-count reduction. Cheap and gates every item below it. Not run —
   no GPU in this container, and a deliberate choice not to go down the mistral.rs-internals path
   right now.
2. **[blocked on item 1, declined with it]** Whether `unsloth/Qwen3.5-4B-GGUF`'s Q4_K_M file
   carries the `mtp.*` tensors (finding 6). Readable from the GGUF tensor index without loading the
   model. If absent, the built-in-head path requires a `Which.Plain`/`Which.MultimodalPlain` preset
   over safetensors with `in_situ_quant=`, which is a `MODEL_PRESETS` addition (`llm.py:107-117`)
   rather than upstream work — and changes the quantisation kernel path at the same time,
   confounding it with item 1 unless measured separately.
3. **[blocked on item 2, declined with it]** Surfacing `MtpConfig::is_builtin` through
   `mistralrs-pyo3`'s `Runner` (finding 6) — the binding gap between what `mistralrs-server-core`
   supports and what the Python `Runner` accepts. Blocked on item 2 being answered affirmatively,
   or on item 2's safetensors fallback.
4. **[superseded — see "Confirmed effects"]** Whether `SELF_REVIEW_ROUNDS = 0` holds the eval suite
   at 100 % (finding 1), and if not, which of the 440 scenarios regress. Shipped instead as item 5
   below, a narrower gate rather than a blanket removal.
5. **[shipped — see "Confirmed effects"]** A cheap predicate that keeps self-review only where it
   could act (finding 1) — restricted to plans whose write arguments are not verbatim echoes of a
   preceding `sumac_find_inventory` result. Implemented as `_write_is_grounded`.
6. **[shipped — see "Confirmed effects"]** Classifier via `grammar`/`grammar_type` or `logit_bias` +
   `max_tokens=1` (findings 2, 4). Implemented via `grammar`/`grammar_type="regex"`, not
   `logit_bias`; `classify_messages`' recorded shape did change, and nothing downstream broke.
7. **Whether `temperature=0` or a set `top_k` changes the 8.17 ms/token slope** (finding 4).
   mistral.rs samples over Qwen's ~151k vocabulary; the current `DEFAULT_TEMPERATURE = 0.2` /
   `DEFAULT_TOP_P = 0.95` (`llm.py:163-164`) with no `top_k` is the configuration most likely to
   take a full-sort path. Untested, and it moves sampling behaviour, so it cannot be measured
   independently of the eval verdict. Still open — cheap to try against the now-33s `evals/`
   baseline.
8. **What the 67 ms fixed per-request cost is** (measured budget) — 15 % of engine time across a
   4.37-request pipeline. Candidates not distinguished this session: the pyo3 request handoff, the
   engine scheduler tick, per-request sampler construction, minijinja chat-template re-render, and
   full-prompt re-tokenisation. Measurable from Python with a minimal request (short prompt,
   `max_tokens=1`) before touching Rust. Still open, and doesn't require touching mistral.rs's own
   source — only timing requests from the Python side.
9. **[open — heuristic rejected, not re-attempted]** Absorbing `_ADD_PROMPT`'s brand-drop retry
   into `_sumac_find_inventory` (finding 8). "Drop the first word" was rejected (breaks on
   multi-word brands); see "Confirmed effects" for why the alternative considered (reversing the
   substring-match direction) wasn't taken up either. Needs a design decision on shared matching
   semantics before either heuristic returns.
10. **[shipped — see "Confirmed effects"]** Memoising `ledger.build_inventory` for the lifetime of
    one `propose`/`revise` (finding 7). `commit()`'s deliberate reload (`llm.py:1555-1565`) is
    untouched.
11. **Whether the terminal plain-text reply round is load-bearing for `add`/`remove`** — 682
    plain-text assistant messages across the run, mean ~28 tokens. `render.py:511-512` and
    `cli.py:1057-1058`, `cli.py:1203-1204` print `plan.reply_text` alongside the plan table, so it
    reaches the person; what is open is whether a person reviewing the table needs it. Still open —
    unlike 1–3, this is app-level and doesn't need mistral.rs internals, but it's a UX change (the
    person loses the narration line), not a pure performance one, so it needs a decision on that
    trade before it ships, not just a measurement.
