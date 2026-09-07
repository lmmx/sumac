# sumac: Fast-Forward Tokens — Motivation, CPU Prototype, and the Patch Plan

**Status:** investigation and a standalone prototype outside the `mistral.rs` tree. Nothing in
`src/sumac/llm.py` or in any vendored/patched wheel changed. This entry follows directly from idea
14 of `docs/journal/2026-09-06-ask-latency-round-two.md` ("Fast-forward tokens in the
constrained-decode path"), which predicted the mechanism and its ceiling from source reading alone
but left three questions open: whether `llguidance`'s fast-forward API actually behaves as
documented, whether Qwen3.5's tokenizer is eligible for it, and what the KV-cache-bookkeeping side
of a mistral.rs patch would actually touch. This entry answers all three empirically, in a CPU-only
container with no GPU, and lays out why the fix has to live inside the Rust engine rather than in
`sumac` itself.

**Provenance markers**, same convention as the previous two entries:

- **(measured, CPU container)** — timed directly in this session, on a 4-vCPU/15GB container with
  no CUDA. `unsloth/Qwen3.5-4B-GGUF` (`Qwen3.5-4B-Q4_K_M.gguf`), the same preset `sumac ask` uses.
  Decode throughput here (~4-6 tok/s) is not representative of the GPU this project targets — these
  numbers validate the *mechanism*, not the previous entries' GPU-relative projections.
- **(verified against mistral.rs `v0.9.2` source)** — read from the pinned tag, cloned locally to
  `.build/mistral.rs` (gitignored, not part of this commit).
- **(verified against llguidance source and a working prototype)** — read from
  `guidance-ai/llguidance` at the pinned constraint (`^1.2.0`, resolved to `1.8.0` by cargo; the
  `Matcher` API in question is unchanged across that range) and exercised directly in a small Rust
  binary against the real crate, not read off GitHub.

---

## Motivation

PR #10 shipped two changes off the back of `2026-09-06-ask-latency-round-two.md`: `top_k=20` as the
default sampling configuration (28-30% decode throughput gain, no eval regression), and a
constrained `DONE|CONTINUE` decision after writes so the loop terminator costs one grammar-picked
token instead of a full unconstrained generation. Both are Tier 1/2 changes in that entry's
language — engine configuration and application-level token-count surgery, no upstream work
required.

Idea 14 in the same entry is different in kind: it is the single largest lever identified across
both latency entries (idea 14's own projection: -30% to -44% at the baseline, "with no application
change at all"), and it is not reachable from `sumac`'s side at all. The measurement behind it:
77-87% of every tool call `sumac ask` emits, by character, is grammar-forced scaffold —
`<tool_call>\n{"name": "sumac_discover_inventory", "arguments": {"product_id":"` has no branch
point in it once the tool name is chosen, and neither does `","amount":"`, `","unit":"`, or the
closing `"}}\n</tool_call>`. mistral.rs already knows this — `llguidance::Matcher`, the type it
holds per-sequence for grammar-constrained decoding, computes a mask over the vocabulary before
every sampled token, and when a grammar admits exactly one continuation that mask has exactly one
bit set. What idea 14 established from `sampling.rs` alone is that mistral.rs still runs a full
model forward pass to discover the token that mask already determined.

This entry exists to turn that source-reading finding into three empirical answers, and to produce
a patch plan concrete enough that building it is the only thing left.

## Why this can't be done from the `sumac` side, in Python

The natural first question, given every other latency win in this series (handles, `top_k`,
`DONE|CONTINUE`, prefix caching) was reachable from `src/sumac/llm.py`: could fast-forwarding be
implemented as a wrapper around the existing `mistralrs.Runner` calls, without touching mistral.rs
itself?

No, for a reason confirmed by direct inspection rather than assumed. `mistralrs`'s Python surface
(`.venv/lib/python3.12/site-packages/mistralrs/__init__.pyi` in this container) is opaque at the
token level: `Runner.send_chat_completion_request` takes a `ChatCompletionRequest` — `messages`,
`grammar`, `grammar_type`, sampling parameters — and returns a complete `ChatCompletionResponse`.
There is no streaming callback that exposes the grammar matcher, no per-token hook, and no access
to `llguidance::Matcher` at all from Python — it is compiled into `mistralrs.abi3.so` and never
re-exported. `pip list` in this container confirms `llguidance` is not even installed as a separate
package; it is a Rust-only transitive dependency, statically linked. Sumac's own
`_LocalMistralRsBackend.send_chat_completion_request` (`llm.py:764-790`, read in the previous
entry) is a thin wrapper over exactly this same opaque call — there is no lower layer to intercept
from the Python side without dropping to FFI against a private symbol, which is a worse position
than patching the Rust source directly.

The mechanism this entry needs — inspect the grammar matcher's forced-token set *before* running
the model, and skip the forward pass when it is non-empty — has to live in the same process and the
same language as the forward pass itself, which is `mistralrs-core`, in Rust. This rules out the
approach every other idea in the latency series used.

## Why not other approaches already considered or already reverted

- **Handles instead of real ids** (idea 6 of the previous entry) was implemented and reverted
  earlier in this project's history — it produced correctness mistakes and real ids remain the
  model-facing representation. Fast-forwarding is compatible with real ids or handles equally: it
  operates on whatever bytes the grammar has already committed to, regardless of what those bytes
  represent. It is not a replacement for idea 6 and does not depend on it either way.
- **Speculative decoding / MTP** (ideas 15-17 of the previous entry) were explicitly excluded from
  this work by request. This matters more than it might look, because mistral.rs's existing
  multi-token-per-forward plumbing (Section 3 below) was built *for* speculative decoding's staged
  draft-model proposals — the design in this entry reuses that plumbing's mechanical shape (one
  windowed forward over N new positions) without reusing its verification logic or touching the
  `speculative::` module at all. Grammar-forced tokens are never proposals that can be rejected;
  they are guaranteed by the grammar, so there is nothing to verify against a draft model or a
  second forward pass. Conflating the two would be a correctness risk for no benefit.
- **A second, smaller "fast" model to draft tokens** was not considered as an alternative, because
  it is speculative decoding under a different name and carries the same exclusion.
- **Doing this in the sampler only** (`mistralrs-core/src/sampler.rs`) — biasing logits so the
  forced token is always chosen — was ruled out early because it still pays the forward pass; it
  changes *what* is computed, not *whether* it's computed. This is the status quo idea 14 already
  identified as the problem, not a fix for it.

## 1. CPU baseline: what `sumac ask` actually costs on this hardware

Set up fresh — `uv sync --group ask` fails in this container because `uv.lock` still points at a
machine-local CUDA wheel path (`vendor/wheels/mistralrs-0.9.2-...whl`, from
`scripts/build-mistralrs-cuda.sh`) that does not exist here; worked around with a plain
`uv pip install -e . mistralrs` against the PyPI CPU wheel, no lockfile involved. Seeded a scratch
vault (fridge, pantry, one purchase) and ran the example query the confirmation-UX and latency
entries both use:

```
sumac ask "where is the butter?" --stats
```

**(measured, CPU container)**

| round | prompt tok | completion tok | tok/s | wall |
|---|---:|---:|---:|---:|
| 0 — classify | 112 | 2 | 4.3 | 8.5 s |
| 1 — tool call (`sumac_find_inventory`) | 746 | 28 | 5.8 | 25.8 s |
| 2 — terminal reply | 849 | 8 | 6.0 | 25.6 s |

~60 s wall time for a 3-round find, at 4-6 tok/s. The answer was correct ("The butter is in the
fridge.") — this run exists to establish that the harness (mistral.rs + llguidance + the tool-call
loop) runs correctly end to end in CPU-only mode in this container, and to give the rest of this
entry a real reference query, not to re-derive the previous entries' GPU-relative figures.

## 2. Constrained decoding buys nothing today, confirmed live

Idea 14 established from `sampling.rs` source that mistral.rs calls `compute_mask_or_eos` and
`consume_token`, never the `ff` family. A three-case standalone test against the running model
confirms the *consequence* of that gap on real hardware, not just its cause in the source:

**(measured, CPU container)**, `top_k=20`, `temperature=0.0`, same model:

| case | completion tok | wall | tok/s |
|---|---:|---:|---:|
| free decoding ("say butter ten times") | 12 | 3.46 s | 3.47 |
| fully grammar-forced single legal string | 21 | 4.79 s | 4.39 |
| the same string, tokenized with no model calls | 21 | 0.009 ms | — |

The forced case is not faster than free decoding — same order of magnitude, same cost structure,
because both pay one forward pass per token regardless of whether the grammar already knew the
answer. The tokenize-only row is the ceiling idea 14 projected from the character-scaffold
measurement, shown directly here: 21 tokens of a fully-determined completion cost 4.79 seconds
today and would cost close to nothing if the forward pass for them were skippable.

## 3. The fast-forward mechanism, proven against the real dependency

Rather than patch mistral.rs blind, drove `llguidance::Matcher` directly in a throwaway Rust binary
(`Cargo.toml` pinning the same `llguidance = { version = "1.2.0", features = ["lark"] }` mistral.rs
`v0.9.2` uses), following the exact loop from llguidance's own `sample_parser/src/minimal.rs`:

```rust
loop {
    let mask = m.compute_mask()?;
    m.consume_token(sampled_token)?;
    let splice = m.compute_ff_tokens();
    if !splice.is_empty() {
        m.consume_tokens(&splice)?;   // whole forced run, one call
    }
}
```

Three grammars, each asserting the fast-forward path reproduces the reference token stream exactly
(`tokens[idx..idx+splice.len()] == splice`, or the run panics):

**(verified against llguidance source and a working prototype)**

| grammar | generated tok | forced tok | forced runs | forwards today | forwards with FF |
|---|---:|---:|---:|---:|---:|
| `{"query":"[a-z]+"}` (mirrors sumac's tool-call shape) | 18 | 10 | 2 | 18 | 8 |
| `DONE` (fully forced literal) | 4 | 3 | 1 | 4 | 1 |
| `(DONE\|CONTINUE)`, sampled `CONTINUE` | 8 | 7 | 1 | 8 | 1 |

For the tool-call-shaped case, 10 of 18 tokens (56%) are forced, and fast-forwarding would cut
model forwards from 18 to 8 — a 2.25x reduction for that round alone, consistent with idea 14's
55-80% scaffold-share projection. The `DONE`/`CONTINUE` case is the sharper result for sumac
specifically: this is exactly the loop-terminator grammar PR #10 already ships, so once the grammar
commits to one branch (i.e. `CONTINUE` was the model's real choice, and everything after that is
fixed by the alternation having been resolved), fast-forwarding collapses the remaining decode into
a single forward regardless of how long the literal is.

`m.stop_reason()` reported `NoExtension` (clean terminal state) in every case — no dangling matcher
state, no assertion failures. This is the strongest evidence obtainable without a model in the
loop that the mechanism is real: the crate does exactly what its doc comment says, using the
identical version mistral.rs already vendors, exercised end-to-end rather than read off GitHub.

The prototype used `llguidance::toktrie::ApproximateTokEnv::single_byte_env()` (llguidance's own
quick byte-level tokenizer, also from `sample_parser/src/minimal.rs`), not Qwen3.5's real GGUF
tokenizer — this sidesteps rather than answers idea 14's first blocking question
("whether Qwen3.5's BPE registers as canonical under llguidance's definition", since
`compute_ff_tokens` documents itself as returning `[]` for non-canonical tokenizers). Checking that
directly against the real tokenizer is listed as open work below, not closed by this prototype.

## 4. The patch plan, anchored to `v0.9.2`

**(verified against mistral.rs `v0.9.2` source, cloned to `.build/mistral.rs`)**

### Where the grammar lives today

`mistralrs-core/src/sequence.rs`:

```rust
pub enum SequenceRecognizer {
    Llguidance(Box<llguidance::Matcher>),
    None,
}
```

`mistralrs-core/src/pipeline/sampling.rs:sample_sequence` (~line 774-928) is the per-token site.
After sampling, it calls `llg.consume_token(second_logprobs_response.token)` at line 909 and never
the `ff` family — matching what idea 14 read from source, now pinned to an exact line and confirmed
by the prototype's `consume_tokens` call to be a drop-in extension point.

### The multi-token-per-forward path already exists

This answers idea 14's open question about "where the real work is" in the KV-cache bookkeeping —
it turns out the hard part is already built, just wired to a different caller.
`mistralrs-core/src/pipeline/inputs_processor.rs`:

- `get_completion_input` (line 2012) hardcodes one new token per decode step:
  `make_completion_chunk(toks, ..., 1)`.
- `get_completion_input_windowed` (line 2054), doc'd *"for models that consume more than one new
  token per decode step (e.g. block diffusion...)"*, already takes a `decode_window: usize`.
- `make_completion_chunk` (line 1387) already appends a per-sequence token window from
  `seq.active_staged_speculative_tokens()` (line 1417-1425) before building `seqlen_offsets`,
  `context_lens`, `position_ids`, and paged-attention slot mappings for the whole window in one
  pass — this is speculative decoding's existing "propose N, verify N, one forward" plumbing,
  cited above as the reason not to touch the `speculative::` module directly while still reusing
  its mechanical shape.

Fast-forwarding does not need a new engine primitive. It needs `compute_ff_tokens()`'s output
routed through this same windowed-decode path instead of a draft model's proposals, with no
verification step — forced tokens are grammar-guaranteed, never rejected, unlike speculative
proposals.

### The five-step patch

1. **`sequence.rs`** — new field `pending_ff_tokens: Vec<u32>` on `Sequence`, separate from
   `active_staged_speculative_tokens` (different acceptance semantics: draft proposals can be
   rejected, forced tokens cannot — conflating the two fields would blur that and risk pulling in
   the verifier's assumptions by accident).
2. **`sampling.rs:sample_sequence`, after line 909** — `llg.compute_ff_tokens()`; if non-empty,
   `llg.consume_tokens(&splice)` and stash `splice` on `seq.pending_ff_tokens`. This is the exact
   call proven in Section 3 — `compute_ff_tokens`/`consume_tokens` returned byte-identical splices
   to the reference decode in every test case there.
3. **`inputs_processor.rs:get_completion_input`** — `decode_window = 1 + seq.pending_ff_tokens.len()`
   per batch, and extend `make_completion_chunk`'s per-sequence loop to append
   `pending_ff_tokens` the same way it appends `staged_speculative`. The existing code's own
   comment at line 1410-1414 already states the constraint to preserve: a batch needs one shared
   window width, so a batch mixing forced and non-forced sequences falls back to width 1 — moot for
   sumac (one sequence at a time regardless of `max_seqs`), load-bearing for the general engine.
4. **`pipeline/mod.rs:step` (~1394-1478)** — only the *last* window position's logits feed
   `sample_sequence`; the forced tokens plus the newly-sampled one all get appended to
   `seq.get_toks()`, with KV cache already correctly extended for all of them by step 3.
5. **Clear `pending_ff_tokens`** at the same point `clear_staged_speculative_tokens` already runs
   (`mod.rs:1408`), so a queue from a grammar that has since deactivated (e.g. `DONE|CONTINUE`
   handing back to the domain loop, per PR #10's design) cannot leak into the next window. This is
   the one correctness trap worth a dedicated test: forced tokens must be scoped to the grammar
   instance that produced them, not just to the sequence.

Nothing in `sumac/src/sumac/llm.py` changes — `_build_request`'s `grammar`/`grammar_type` fields
reach `mistralrs.ChatCompletionRequest` exactly as today. The win is entirely inside the engine a
patched wheel provides, the same deployment shape `ask-cuda`'s `vendor/wheels/` local build already
uses (`scripts/build-mistralrs-cuda.sh`) — a CPU build is simpler, needing only `rustc >= 1.94` and
`maturin`, no CUDA toolchain, which is why this session's validation could happen at all in a
container without a GPU.

### What is not yet answered

- **Canonical-tokenizer status for Qwen3.5's BPE**, idea 14's first blocking question, not answered
  by Section 3's prototype (see above — it used a byte-level stand-in tokenizer). A prerequisite
  before trusting any token-count projection against the real model.
- **Paged-attention and flash-attn interaction** with a variable, forced-token-driven window width,
  beyond the fixed-width case `make_completion_chunk` already handles for staged speculative
  tokens.
- **EOS landing mid-window** — if a forced run's last token is EOS, the window still needs to stop
  cleanly rather than run past it.
- **Whether upstream wants it** — unasked. This is a general-purpose constrained-decoding
  optimization, not sumac-specific; a one-paragraph issue against `EricLBuehler/mistral.rs` is
  cheap and would surface any of the above that a maintainer already knows the answer to.

None of these were attempted against the real `mistral.rs` clone this session — the correctness
surface (cache offsets under paged-attn, mixed-batch fallback, EOS-mid-window) is large enough that
a rushed first pass risks a build that compiles and looks right while silently corrupting KV cache
state, which is a worse failure mode than "still slow" and specifically the one this project cannot
tolerate: fast-forwarding must reproduce exactly the same generated output as the unpatched decode
path, or it is not worth having regardless of speed.

Recommended next-session scope: steps 1-5 above, against a real build, validated with `evals/` for
output-identity before any latency number is trusted — the standing rule from every entry in this
series: **prove forwards are avoided before optimizing for wall clock, and prove output is
unchanged before either.**

---

## Appendix: reproducing the prototype

```bash
git clone --depth 1 --branch v0.9.2 https://github.com/EricLBuehler/mistral.rs.git .build/mistral.rs
git clone --depth 1 --branch v1.2.0 https://github.com/guidance-ai/llguidance.git .build/llguidance
# see .build/llguidance/sample_parser/src/minimal.rs for the reference loop this entry's
# ff_prototype/src/main.rs is adapted from (three grammars instead of one, and it counts
# forwards-with-vs-without fast-forward rather than just asserting equivalence).
```

No files under `.build/` are committed; both are gitignored scratch clones, same convention as
`scripts/build-mistralrs-cuda.sh`'s own `.build/`.

---

## Current State

- `sumac ask "where is the butter?" --stats` runs end to end on a 4-vCPU/15GB CPU-only container with no CUDA, using `unsloth/Qwen3.5-4B-GGUF` (`Qwen3.5-4B-Q4_K_M.gguf`), the preset `src/sumac/llm.py`'s `ModelPreset("qwen3.5-4b", ...)` names — round 0 (classify) 8.5s at 4.3 tok/s, round 1 (`sumac_find_inventory` tool call) 25.8s at 5.8 tok/s, round 2 (terminal reply) 25.6s at 6.0 tok/s, answer "The butter is in the fridge."
- `mistralrs-core/src/pipeline/sampling.rs:774-928` (`sample_sequence`, `v0.9.2` tag) calls `llg.compute_mask_or_eos()` and `llg.consume_token(...)` at line 909 — a full model forward pass runs for every decoded token regardless of whether `SequenceRecognizer::Llguidance` (`sequence.rs`) has already narrowed the grammar to one legal continuation.
- A CPU timing comparison in this container shows a fully grammar-constrained single-legal-string completion (21 tokens, one legal continuation throughout) decoding at 4.39 tok/s, the same order as an unconstrained "say butter ten times" completion (12 tokens) at 3.47 tok/s — the constraint changes which token is sampled, not how many forward passes the completion costs.
- `llguidance::Matcher` at the version `mistralrs-core`'s workspace `Cargo.toml` pins (`llguidance = { version = "1.2.0", features = ["lark"] }`, resolved to 1.8.0 by cargo in a standalone build) exposes `compute_ff_tokens() -> Vec<TokenId>` and `consume_ff_tokens() -> Vec<TokenId>`, exercised directly (no mistral.rs, no model) in a throwaway crate at `.build/ff_prototype/src/main.rs` against the loop pattern in `llguidance`'s own `sample_parser/src/minimal.rs:43-71`.
- The `.build/ff_prototype` run against grammar `\{"query":"[a-z]+"\}` (target `{"query":"butter"}`, the shape of a sumac tool call) produces 18 tokens of which 10 are returned by `compute_ff_tokens` across 2 forced-token runs; `consume_tokens(&splice)` advances the matcher past all of them in one call each time, and every returned splice matches the reference token stream at the assertion in `main.rs:46-49` (`tokens[idx..idx+splice.len()] == splice`).
- The same run against grammar `(DONE|CONTINUE)` sampling `CONTINUE` produces 8 tokens of which 7 are returned by one `compute_ff_tokens` call after the branch point is resolved; against grammar `DONE` (no branch at all) 3 of 4 tokens are returned by one call. `m.stop_reason()` reports `NoExtension` (clean terminal state) in all three cases.
- `mistralrs-core/src/pipeline/inputs_processor.rs:1387-1470` (`make_completion_chunk`) already builds a per-sequence decode input wider than one token: it appends `seq.active_staged_speculative_tokens()` (line 1417-1425) to the per-sequence context before computing `seqlen_offsets`, `context_lens`, `position_ids`, and paged-attention slot mappings for the whole window in one forward call — this path exists for `crate::speculative::driver`'s staged-proposal verification, not for grammar-forced tokens.
- `mistralrs-core/src/pipeline/inputs_processor.rs:2054-2067` (`get_completion_input_windowed`) takes a `decode_window: usize` parameter and is documented as "for models that consume more than one new token per decode step (e.g. block diffusion...)"; `get_completion_input` (line 2012-2049), the path `mistralrs-core/src/pipeline/mod.rs:1394` (`step`) drives for ordinary decode, calls `make_completion_chunk(..., 1)` with the window fixed at 1.

## Stubbed

- None — no fast-forward code exists in any form in `src/sumac/` or in the vendored `mistralrs` wheel; `.build/ff_prototype` is a standalone crate outside both trees, not a stub of the real integration.

## Missing

- `mistralrs-core/src/sequence.rs`'s `Sequence` struct carries `active_staged_speculative_tokens` (consumed by `make_completion_chunk`) but no equivalent field for grammar-forced tokens — `compute_ff_tokens()`'s output has no field to land in.
- `sampling.rs:sample_sequence` (line 903-914) calls `llg.consume_token` after sampling but never `llg.compute_ff_tokens()` or `llg.consume_tokens()` — the call this entry's Section 4 proposes adding at this site does not exist.
- `get_completion_input` (`inputs_processor.rs:2012`) has no branch that raises `decode_window` above 1 for a sequence carrying pending forced tokens — the windowed-decode path `get_completion_input_windowed` provides is wired only to the speculative-decoding and block-diffusion callers, not to the ordinary text-decode path `mistralrs-core/src/pipeline/mod.rs:1394` uses.
- No code checks whether Qwen3.5's GGUF-embedded BPE tokenizer registers as "canonical" under llguidance's definition — `compute_ff_tokens`'s own doc comment states it returns `[]` for non-canonical tokenizers, and `.build/ff_prototype` sidesteps the question by using `llguidance::toktrie::ApproximateTokEnv::single_byte_env()` rather than the real GGUF tokenizer.
- No test or code path in this session covers EOS landing mid-fast-forward-window, or a batch mixing forced and non-forced sequences (the fallback-to-width-1 case `make_completion_chunk`'s comment at `inputs_processor.rs:1410-1414` documents for the existing staged-speculative case).
- No build of a patched `mistral.rs` wheel exists — `uv sync --group ask` in this container fails before any patch work starts, because `uv.lock` references `vendor/wheels/mistralrs-0.9.2-cp310-abi3-manylinux_2_39_x86_64.whl`, a path that does not exist outside the machine that ran `scripts/build-mistralrs-cuda.sh`; this session's CPU runs use a plain `uv pip install -e . mistralrs` against the PyPI wheel instead, bypassing the lockfile.

## Divergence

- None recorded — this entry is standalone investigation and a prototype outside the `sumac` and vendored-wheel trees; no README or prior journal claim describes fast-forward token support as present.
