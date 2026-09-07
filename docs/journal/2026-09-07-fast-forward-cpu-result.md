# sumac: Fast-Forward Tokens — Patch Implemented, CPU Result Measured Slower

**Status:** the patch plan from `docs/journal/2026-09-07-fast-forward-prototype.md` Section 4
implemented against a real `mistral.rs` `v0.9.2` clone, built into a CPU wheel, and run
end-to-end through `sumac ask` in this container. Output is byte-identical to the unpatched
decode path in every run. Wall-clock time is not improved — it is slower, on this hardware, for
this model. The patch is left in the source tree with its capability flag defaulting to `false`
in both pipelines it could apply to; nothing is enabled or shipped as a result of this entry.

**Provenance markers**, same convention as the two entries this follows:

- **(measured, CPU container)** — timed directly in this session, same 4-vCPU/15GB CPU-only
  container as the prototype entry, `unsloth/Qwen3.5-4B-GGUF` (`Qwen3.5-4B-Q4_K_M.gguf`).
- **(verified against mistral.rs `v0.9.2` source)** — read and edited directly in a clone at
  `.build/mistralrs-cpu-ff/src` (gitignored, not part of this commit); the diff against that
  clone is `docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch`, committed for reference.

---

## 1. The patch, as actually built (differs from the plan in one load-bearing way)

The prototype entry's Section 4 five-step plan is implemented, with one correction the plan got
wrong about `mistralrs-core`'s existing `decode_window` parameter (established by reading the
real source in this session, not assumed from the plan): `get_completion_input`'s call into
`make_completion_chunk` stays at `decode_window=1` unconditionally. The window growth comes
entirely from `make_completion_chunk`'s existing pattern of *appending* extra tokens on top of
that fixed width — the same mechanism `staged_speculative_tokens` already used for speculative
decoding's verification forward, now mirrored for `pending_ff_tokens`. Setting `decode_window`
itself to `1 + N` as the plan describes would have double-counted the forced tokens.

**(verified against mistral.rs `v0.9.2` source)**, the five pieces:

1. `mistralrs-core/src/sequence.rs` — `pending_ff_tokens: Vec<u32>` field on `Sequence`, plus
   `active_pending_ff_tokens`/`set_pending_ff_tokens`/`take_pending_ff_tokens`, mirroring the
   existing `staged_speculative_tokens` accessors exactly.
2. `mistralrs-core/src/pipeline/sampling.rs`, `sample_sequence` — after the existing
   `llg.consume_token(...)` call, `llg.compute_ff_tokens()` runs when the pipeline's
   `supports_grammar_fast_forward` metadata flag is set (new, see point 5); a non-empty splice
   is consumed into the matcher and stashed on the sequence via `set_pending_ff_tokens`.
3. `mistralrs-core/src/pipeline/sampling.rs`, `sample_and_add_toks` — captures each sequence's
   `pending_ff_tokens` *before* calling `sample_sequence` (which stages a new splice for the
   *next* step onto the same field), then replays the captured splice one token at a time
   through the normal `finish_or_add_toks_to_seq` completion path via a new
   `apply_pending_ff_tokens` helper, so tool-call detection, mid-stream grammar activation, and
   stop handling all run exactly as for a sampled token. A stop firing partway through the
   splice — legitimate, not just an error path, e.g. a stop string matching mid-scaffold or the
   context limit landing inside it — is checked after every token and discards the rest of the
   splice and the real token sampled after it, exactly as the "prove output is unchanged" rule
   from the previous entry requires.
4. `mistralrs-core/src/pipeline/inputs_processor.rs`, `make_completion_chunk` — extends the
   per-sequence context with `pending_ff_tokens` the same way it already extends it with
   `staged_speculative_tokens` (new `pending_ff_batch_width` helper, same homogeneous-width
   fallback semantics as `speculative::staging::staged_batch_width`), and narrows
   `context_lens` to just the final window position when a forced splice is present — the
   model's `extract_logits`/lm_head projection only needs to run for the one position whose
   token is still unknown, not the whole window. `DecodePagedRows.query_len` — a paged-attention
   metadata field this session's CPU run never exercises — is fixed to read the true window
   width from a new parallel `query_lens` vector rather than deriving it from the now-narrower
   `context_lens`, since the previous code's derivation assumed a caller-and-only-caller who
   always keeps the two synchronized. The two mechanisms cannot both be active for a real
   pipeline in this codebase today (GGUF/Normal decode never populate `staged_speculative_tokens`
   at all — see Section 2), but `context_lens` is left at full width whenever both are non-empty
   on the same sequence regardless, so a future pipeline that combines them does not silently
   lose speculative verification's per-position logits.
5. **New beyond the plan**: `GeneralMetadata::supports_grammar_fast_forward: bool`, one field
   added to `mistralrs-core/src/pipeline/mod.rs`'s metadata struct and to all seven pipeline
   constructors (`gguf.rs`, `normal.rs`, `ggml.rs`, `multimodal.rs`, `diffusion.rs`,
   `embedding.rs`, `speech.rs`). The plan's own prototype only drove `llguidance::Matcher`
   directly and never asked which pipelines' input builders would actually consume a forced
   splice if one appeared — checked in this session (Section 2) and found to be exactly two,
   both using the same shared `text_models_inputs_processor::TextInputsProcessor`. Every other
   pipeline builds its own input tensors (`vision_models/*/inputs_processor.rs` and similar) and
   would silently corrupt `Sequence::get_toks()` by replaying tokens whose forward pass never
   ran, had `sample_sequence` staged a splice for them unconditionally. The flag is the guard
   against that: `sample_sequence` only calls `compute_ff_tokens` when it is `true`, and it
   defaults `false` everywhere including the two pipelines that can safely turn it on (Section 3
   explains why they stay off in this commit).

## 2. Which pipeline actually decodes `unsloth/Qwen3.5-4B-GGUF` — not the one the plan read

The prototype entry's Section 4 was anchored entirely to `mistralrs-core/src/models/quantized_llama.rs`
and the `GGUFPipeline` struct in `pipeline/gguf.rs` (`enum Model { XLoraLlama(..), XLoraPhi3(..) }`,
line 129), because that is what `sampling.rs`'s call site read from source. That struct is real
and is the actual pipeline for a GGUF file whose architecture resolves to Llama or Phi3 family. It
is not what sumac's model uses.

**(measured, CPU container)**, `RUST_LOG=mistralrs_core::pipeline::gguf=debug`:

```
DEBUG mistralrs_core::pipeline::gguf: GGUF file(s) [".../Qwen3.5-4B-Q4_K_M.gguf"]
DEBUG mistralrs_core::pipeline::gguf: Loading GGUF architecture `qwen35` through native Qwen3_5
      (SingleCandidate, layouts [Direct, PerLayerInventory, Qwen35SplitQkvzSplitBetaAlpha, ShiftedRmsNorm])
```

`qwen35`'s GGUF architecture resolves (`mistralrs-core/src/gguf/normal_registry.rs`) to a
*native* implementation — `vision_models/qwen3_5::text::Qwen3_5TextModel` — loaded through
`pipeline/normal.rs`'s `NormalPipeline`, not `pipeline/gguf.rs`'s `GGUFPipeline`. This was not
apparent from source reading alone in the prototype entry; it took three failed attempts in this
session, each showing `supports_fast_forward=false` inside `sample_sequence` despite the field
being set `true` at the `GGUFPipeline` construction site edited first, before adding a debug log
directly at `this.name()` in `sample_and_add_toks` (which correctly printed
`unsloth/Qwen3.5-4B-GGUF`) proved the *metadata* wasn't coming from the pipeline that name implied
— and the `gguf=debug` log above named the real one.

Both `GGUFPipeline` and `NormalPipeline` reach `make_completion_chunk` the same way: neither
overrides `PreProcessingMixin::get_processor`, so both fall through to the trait's default
`BasicProcessor`, whose `inputs_processor()` returns `text_models_inputs_processor::TextInputsProcessor`
— the struct Section 1 point 4 patches. This is why enabling the flag on `normal.rs` (Section 3)
needed no further plumbing changes once the routing was understood.

## 3. The GDN blocker, and why the fix is low-risk despite touching an SSM kernel

**(measured, CPU container)**, first attempt with `supports_grammar_fast_forward: true` set only
on `gguf.rs` (the wrong pipeline, harmless) then correctly on `normal.rs`:

```
✗ error: Agent error: GDN decode expects a single-token query.
   1: mistralrs_core::gdn::backend::causal_conv1d
   2: mistralrs_core::gdn::layer::GatedDeltaNet::forward_with_stash
   3: mistralrs_core::vision_models::qwen3_5::text::Qwen3_5TextModel::forward_embeds
```

Qwen3.5 is a hybrid architecture: some layers are ordinary attention, others are Gated DeltaNet
(GDN) — a linear-attention/recurrent layer with an internal depthwise-conv "state" (`GdnLayerCache.conv_state`)
updated one step at a time during decode. `gdn/backend.rs:causal_conv1d`'s dispatch
(**verified against mistral.rs `v0.9.2` source**) hard-asserted exactly one query token whenever
`RecurrentBatchKind::Decode` was in effect — the kind every plain decode step in this codebase
uses (`pipeline/normal.rs:1818` and three other call sites) — before this session's window
widening ever produced more than one.

This looked like a dead end for Qwen3.5 specifically, but the same file already contains the fix
in a different, working shape:

- `GatedDeltaNet::advance_state_from_stash` (`gdn/layer.rs:129`, pre-existing, doc'd for "replay
  an accepted prefix after speculative verification rejected the tail of a multi-token step")
  calls `causal_conv1d` with `RecurrentBatchKind::Prefill` for an arbitrary `rows` count of
  already-known tokens — the exact "extend cached state by N already-known tokens" operation a
  forced splice needs, just written for a different caller (speculative decoding's accepted-prefix
  replay, not grammar fast-forward).
- `causal_conv1d_full` (reached via any non-`Decode` kind) is not a fresh-prompt-only
  computation: it unconditionally reads `cache.conv_state` as prior context to prepend
  (`gdn/backend.rs:874`), narrows the new tail back into `cache.conv_state` afterward
  (`gdn/backend.rs:876-880`), and has its own CPU F32 fused fast path
  (`causal_conv1d_full_cpu_f32`). A test named `causal_conv1d_prefill_continues_from_existing_state`
  already exists for this property.
- `mistralrs-core/src/models/lfm2.rs:754` — a different hybrid-recurrent model in the same
  codebase — already draws exactly this line: `matches!(batch_kind, Decode) && seq_len == 1`
  picks the single-token fast path; anything else (including `Decode` with `seq_len > 1`) falls
  back to the width-general computation.

The fix mirrors `lfm2.rs`'s established pattern in `causal_conv1d`'s own dispatch, three lines
changed:

```rust
if matches!(batch_kind, RecurrentBatchKind::Decode) && seq_len == 1 {
    causal_conv1d_update(x, conv1d_weight, dims, cache)
} else {
    causal_conv1d_full(x, conv1d_weight, dims, cache)
}
```

No change to `apply_recurrence_from_convolved` (the actual delta-rule recurrence, called after
the conv) was needed — its signature never took a `batch_kind` in the first place and already
processes whatever `seq_len` it is given, which is how `advance_state_from_stash` could call it
directly for multi-token replay without any prior patch.

After this fix, the same `sumac ask "where is the butter?" --stats` query completes all three
rounds with no error, on both the debug-logged and clean (no `RUST_LOG`) runs.

## 4. Correctness: byte-identical, every run that completed

**(measured, CPU container)**, `sumac ask "where is the butter?" --stats`, patched wheel,
`supports_grammar_fast_forward: true`:

```
round 1: 746 prompt + 28 completion tokens, 6.0 tok/s, 39.2s — tool call:
sumac_find_inventory({"query":"butter"})
round 2: 849 prompt + 8 completion tokens, 4.4 tok/s, 42.5s — The butter is in the fridge.
```

Identical tool call and final text to the unpatched baseline in Section 5. This holds across
every completed run in this session (three, after the GDN fix) — the one case that did not match
was the pre-GDN-fix crash itself, a loud `Agent error`, not silent wrong output.

## 5. Wall-clock: slower with the mechanism on, on this hardware, for this model

**(measured, CPU container)**, same query, same container, back to back:

| run | round 0 | round 1 (tool call, 28 tok) | round 2 (reply, 8 tok) | total |
|---|---:|---:|---:|---:|
| baseline (`supports_grammar_fast_forward` absent from this build) | 4.7s | 31.4s | 33.4s | 69.5s |
| patched, flag `true`, `RUST_LOG` debug logging on | 4.9s | 37.6s | 38.3s | 80.8s |
| patched, flag `true`, no logging | 5.5s | 39.2s | 42.5s | 87.2s |

The no-logging run rules out `tracing::debug!` overhead as the explanation — it is slower than
the logged run, not faster; both are slower than baseline. `RUST_LOG=mistralrs_core::pipeline::sampling=debug`
during the no-logging-run's predecessor recorded every `compute_ff_tokens` call for round 1's 18
grammar-constrained tokens: 15 empty, 2 of length 1, 1 of length 6. That one length-6 splice
collapses what would have been 7 forward passes into 1 — the exact mechanism idea 14 and the
prototype entry predicted — and the round is still 25% slower than baseline.

Two things this session can measure, and one it cannot:

- **Real BPE forced-run lengths are much shorter than the prototype's byte-level tokenizer
  suggested.** The prototype's `ApproximateTokEnv::single_byte_env()` made every forced byte its
  own token, so a 10-byte forced span was 10 forceable tokens. Against Qwen3.5's real tokenizer,
  the same grammar produces mostly zero- or one-token splices, with one six-token run in an
  18-token constrained stretch — the "canonical tokenizer" open question from the prototype
  entry is answered (`toktrie::TokenizerEnv::tokenize_is_canonical` defaults `true` and mistral.rs's
  `ByteTokenizerEnv` never overrides it, so the mechanism is *eligible*), but eligibility and
  actual BPE token-boundary alignment are different questions, and this session only measured the
  latter as low for this specific grammar shape.
- **A wider decode window is not free on this CPU.** Collapsing 7 forward calls into 1 reduced
  round 1's total forward-pass count by roughly 8 of 28 (~29%), yet wall time rose ~25%. The
  extra query positions in a widened window cost close to their proportional share of compute on
  this 4-vCPU container rather than being absorbed into memory-bandwidth headroom the way
  single-token decode's per-step weight reload is. Idea 14's projection, and this whole series'
  framing of CPU as "worst case," assumed decode is memory-bandwidth-bound the way it typically
  is on a GPU — this session's numbers are consistent with that assumption not holding on this
  particular CPU, not with the mechanism being broken.
- **What this session cannot measure**: whether the same patch wins on the CUDA target this
  project actually deploys to. GPU decode's memory-bandwidth-boundedness is the premise idea 14
  was built on and this container has no GPU to check it against. The patch is CPU-portable code
  (ordinary candle `Tensor` ops, no CUDA-specific paths touched) but its wall-clock case rests
  entirely on unverified-here GPU numbers.

## 6. Disposition

`supports_grammar_fast_forward` defaults `false` in every pipeline construction site, including
`gguf.rs` and `normal.rs` where it is mechanically safe to enable. Nothing in this commit changes
`sumac`'s runtime behavior — `src/sumac/llm.py` is untouched, and no build of this patched wheel
is wired into `pyproject.toml` or `vendor/wheels/`, unlike `scripts/build-mistralrs-cuda.sh`'s
committed CUDA build. The patch against the `v0.9.2` clone is
`docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch`, kept for whoever next has GPU access
to check Section 5's open question — flipping the flag to `true` and rerunning the eval suite's
latency measurement is the entire next step, no further Rust changes anticipated unless the GPU
number also disappoints, in which case the per-window overhead itself (not just its CPU-vs-GPU
cost model) would need profiling.

---

## Appendix: reproducing this session's build and result

```bash
git clone --depth 1 --branch v0.9.2 https://github.com/EricLBuehler/mistral.rs.git .build/mistralrs-cpu-ff/src
cd .build/mistralrs-cpu-ff/src
git apply /path/to/sumac/docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch
# flip both `supports_grammar_fast_forward: false` lines to `true` (pipeline/gguf.rs, pipeline/normal.rs)
# to reproduce Section 4/5's numbers rather than the shipped disabled state
uv venv --python 3.12 build-venv && source build-venv/bin/activate && uv pip install "maturin[patchelf]"
cd mistralrs-pyo3 && maturin build --release -o ../wheels-raw   # no --features cuda: default features are CPU
```

No files under `.build/` are committed, same convention as the prototype entry and
`scripts/build-mistralrs-cuda.sh`'s own `.build/`.

---

## Current State

- `docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch` (14 files, 168 insertions) implements grammar fast-forward tokens against a `mistral.rs` `v0.9.2` clone, applying and compiling cleanly (`cargo check -p mistralrs-core`, zero errors) in this session's container.
- `Sequence::pending_ff_tokens` (`mistralrs-core/src/sequence.rs`), `sample_sequence`'s post-`consume_token` `compute_ff_tokens` call and `sample_and_add_toks`'s capture-before-resample-and-replay loop (`mistralrs-core/src/pipeline/sampling.rs`), and `make_completion_chunk`'s window extension plus narrowed `context_lens` (`mistralrs-core/src/pipeline/inputs_processor.rs`) together implement the mechanism the prototype entry's Section 4 planned, with the `decode_window` correction described in Section 1 of this entry.
- `GeneralMetadata::supports_grammar_fast_forward` (`mistralrs-core/src/pipeline/mod.rs`) gates the mechanism per-pipeline; set in all seven `GeneralMetadata` construction sites in the tree (`gguf.rs`, `normal.rs`, `ggml.rs`, `multimodal.rs`, `diffusion.rs`, `embedding.rs`, `speech.rs`), `false` in every one as shipped in this commit.
- `unsloth/Qwen3.5-4B-GGUF`'s GGUF architecture `qwen35` loads through `pipeline/normal.rs`'s `NormalPipeline` running `vision_models/qwen3_5::text::Qwen3_5TextModel`, confirmed by `RUST_LOG=mistralrs_core::pipeline::gguf=debug` logging `Loading GGUF architecture qwen35 through native Qwen3_5` — not through `pipeline/gguf.rs`'s `GGUFPipeline`/`quantized_llama.rs`, which the prototype entry's Section 4 was anchored to and which is a real, correct pipeline for GGUF files whose architecture resolves to plain Llama or Phi3 family instead.
- `mistralrs-core/src/gdn/backend.rs:causal_conv1d`'s dispatch condition changed from `if matches!(batch_kind, Decode)` to `if matches!(batch_kind, Decode) && seq_len == 1`, letting a multi-token decode window reach `causal_conv1d_full` (previously reachable only via `Prefill`) instead of hitting `causal_conv1d_update`'s hard-coded `seq_len != 1` bail — required because Qwen3.5 is a hybrid architecture with Gated DeltaNet layers whose depthwise-conv decode state update was single-token-only.
- `sumac ask "where is the butter?" --stats` run against the patched wheel with `supports_grammar_fast_forward: true` on `pipeline/normal.rs` produces byte-identical tool-call arguments (`sumac_find_inventory({"query":"butter"})`) and final text (`The butter is in the fridge.`) to the unpatched baseline, across three runs in this session (one with debug logging, one without, both post-GDN-fix).
- Same query, same 4-vCPU/15GB CPU-only container, back to back **(measured, CPU container)**: baseline 69.5s total (4.7s + 31.4s + 33.4s across the three agent-loop rounds); patched (flag `true`, no logging) 87.2s total (5.5s + 39.2s + 42.5s) — about 25% slower, not faster, despite one window in round 1 correctly collapsing 7 forward passes into 1 (`RUST_LOG=mistralrs_core::pipeline::sampling=debug` recorded round 1's 18 grammar-constrained `compute_ff_tokens` calls as 15 empty, 2 of length 1, 1 of length 6).

## Stubbed

- None — the patch is a complete, compiling implementation of the planned mechanism, not a stub; it is disabled by configuration (`supports_grammar_fast_forward: false`), not incomplete.

## Missing

- No GPU/CUDA measurement of this patch exists anywhere in this project. Idea 14's original speedup projection, and this whole three-entry series' framing of the CPU numbers as "worst case," rests on decode being memory-bandwidth-bound the way it typically is on GPU — this session's CPU numbers (Section 5) are consistent with that not holding on a 4-vCPU CPU, but say nothing about the CUDA target `scripts/build-mistralrs-cuda.sh` builds for.
- No accounting exists for how much of the patched run's slowdown is the widened forward pass itself versus this session's added per-token bookkeeping replay (`apply_pending_ff_tokens` calling the full `finish_or_add_toks_to_seq` completion path — tool-call state parsing, streaming checks, mid-stream grammar activation — once per forced token instead of once per step). Section 5 rules out `tracing` logging overhead specifically but not this.
- `supports_grammar_fast_forward` is not enabled for `pipeline/ggml.rs`'s `GGMLPipeline`, despite it plausibly sharing the same `text_models_inputs_processor::TextInputsProcessor` as `gguf.rs` and `normal.rs` (not checked directly in this session) — left `false` along with the vision/embedding/diffusion/speech pipelines, which are confirmed unsafe to enable (Section 1, point 5) rather than merely unchecked.
- No test in `mistral.rs`'s own suite or in this repository's `evals/` covers the fast-forward path — this session's validation is three manual `sumac ask` runs compared by eye against a manual baseline run, not an automated regression.

## Divergence

- None recorded against README or prior journal claims — `docs/journal/2026-09-07-fast-forward-prototype.md`'s own "Current State" already describes itself as investigation and a standalone prototype, not a shipped capability, and this entry's disposition (mechanism implemented, disabled by default) does not contradict anything it or the README claimed.
