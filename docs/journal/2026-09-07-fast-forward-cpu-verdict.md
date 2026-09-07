# sumac: Fast-Forward Tokens on CPU — Full Reasoning Chain and Final Verdict

**Status:** every hypothesis raised in this session's live discussion about *why* fast-forward
tokens measured slower on CPU (`docs/journal/2026-09-07-fast-forward-cpu-result.md`) is checked
against direct measurement in this entry, in the order they were raised and discarded. The
session's own prior claim (~25% slower, that entry's Section 5) does not reproduce under a clean
controlled test run in this entry (Section 4). The mechanism is closed out as measured-neutral on
this hardware, for a structural reason common to every measurement below (Section 6), not a
remaining implementation bug.

**Provenance markers**, same convention as the four entries this follows:

- **(measured, CPU container)** — timed directly in this session, same 4-vCPU/15GB CPU-only
  container as every prior entry in this series, `unsloth/Qwen3.5-4B-GGUF` (`Qwen3.5-4B-Q4_K_M.gguf`).
- **(verified against mistral.rs `v0.9.2` source)** — read directly in the clone at
  `.build/mistralrs-cpu-ff/src` (gitignored); every diagnostic addition described below was
  reverted from that clone after its measurement, confirmed byte-identical to
  `docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch` via `diff` before moving to the
  next step, so nothing in this entry changes what that patch file contains.

---

## 1. The wrong hypothesis: blaming GDN's general recurrence path

Section 5 of the prior entry left one open question: whether the ~25% slowdown came from the
widened forward pass itself or from this session's replay bookkeeping. Asked directly why
fast-forward would be slower at all when it strictly reduces forward-pass count, this session's
first answer (in chat, not committed to any file) was a guess: that `causal_conv1d_full` and the
general (non-decode) branch of `apply_recurrence_from_convolved` (`gdn/backend.rs`) are a less-tuned
code path than the single-token decode fast path (`causal_conv1d_update_cpu`,
`decode_recurrence_cpu_from_convolved`), and that a forced multi-token window pays for that
mismatch. This claim was stated without measuring it.

## 2. Measuring GDN internals in isolation — the guess was backwards

**(measured, CPU container)**, a `#[cfg(test)]` added directly to `gdn/backend.rs` (reverted after
measurement, never committed) constructed real Qwen3.5-4B GDN dimensions read from the GGUF file's
own metadata — `qwen35.ssm.state_size=128`, `ssm.group_count=16`, `ssm.time_step_rank=32`,
`ssm.inner_size=4096`, `ssm.conv_kernel=4` — the same mapping `gguf/normal_config.rs:1844-1894`
(`build_qwen3_5`) uses to derive `linear_key_head_dim`/`linear_num_key_heads`/etc, giving
`GdnDims { num_k_heads: 16, num_v_heads: 32, head_k_dim: 128, head_v_dim: 128, conv_kernel_size: 4,
v_head_layout: Tiled }`. The test called `causal_conv1d` and `apply_recurrence_from_convolved`
directly (the same functions the real forward pass calls, bypassing only the input projections),
comparing 7 sequential `seq_len=1` decode calls against 1 batched `seq_len=7` call, 30-run median,
3-run warmup discarded, fresh zeroed cache per run so allocation cost stays outside the timed
region:

| approach | total (30-run median) | per token |
|---|---:|---:|
| 7x sequential `seq_len=1` decode calls | 8.14ms | 1.16ms/tok |
| 1x `seq_len=7` widened call | 2.03ms | 0.29ms/tok |

Re-run once more for confirmation: 8.03ms vs 1.88ms — same ~4.3x gap, not noise. The widened call
is **faster** per token, not slower. Section 1's guess was wrong on the specific mechanism: GDN's
general path is not a slower, less-tuned fallback — batching amortizes fixed per-call `rayon`
dispatch overhead (`barrier_pool().execute_chunked`, `par_chunks_mut`) that is paid once per
`causal_conv1d`/`apply_recurrence_from_convolved` call regardless of how many tokens that call
covers, so 7 separate calls pay that fixed cost 7 times and 1 call pays it once. GDN internals are
not the source of the measured end-to-end slowdown; in isolation they are a net win when batched.

## 3. Measuring the actual candidate the prior entry left open: matcher and replay overhead

With GDN cleared, the next candidate was the one piece of the mechanism that runs unconditionally
on every constrained token regardless of outcome: `llg.compute_ff_tokens()` (`pipeline/sampling.rs`,
inside `sample_sequence`), and the per-forced-token replay loop (`apply_pending_ff_tokens`, same
file) that runs the full `finish_or_add_toks_to_seq` completion path once per token in a splice.
Both were instrumented directly with `std::time::Instant` and `tracing::info!` (reverted after
measurement — the working tree diffs against `2026-09-07-fast-forward-mistralrs-v0.9.2.patch`
identically before and after this section's work), then exercised with a real
`sumac ask "where is the butter?" --stats` run against a rebuilt CPU wheel with
`supports_grammar_fast_forward: true` on `pipeline/normal.rs`, `RUST_LOG=mistralrs_core::pipeline::sampling=info`.

**(measured, CPU container)**, round 1 (28 completion tokens, 18 of them grammar-constrained),
every `ff_tokens_step`/`ff_tokens_consume_splice`/`ff_tokens_replay` log line summed:

| quantity | total across 18 constrained tokens |
|---|---:|
| `llg.consume_token` (baseline matcher cost, runs with or without this patch) | 174us |
| `llg.compute_ff_tokens()` (new: the fast-forward check itself) | 1187us |
| `llg.consume_tokens(&splice)` (new: only on a non-empty splice) | 8us |
| `apply_pending_ff_tokens` replay loop (new: 3 splices, lengths 6/1/1) | 191us |
| **total new overhead this patch adds to round 1** | **1386us (1.386ms)** |

Round 1 takes 30-31 seconds end to end. 1.386ms is roughly 0.0045% of that. `compute_ff_tokens()`
being called on all 18 constrained tokens (15 of them returning an empty splice) does not explain
a multi-second regression — the total cost of doing so, useful or not, is immeasurably small next
to the round's wall-clock time.

## 4. Re-testing the original ~25% finding — it does not reproduce

With both candidate causes measured and cleared, the only remaining possibility was that the
original finding itself (`2026-09-07-fast-forward-cpu-result.md` Section 5: 87.2s patched vs 69.5s
baseline, one run each, not interleaved) reflected container-performance variance rather than a
reproducible cost of the patch — that entry's own round 0 timings already ranged 4.7-5.9s across
three different builds with no code difference between them, an 8-25% swing attributable to nothing
in the code under test.

**(measured, CPU container)**, this entry: two wheels built from the identical instrumented source
(Section 3), one with `supports_grammar_fast_forward: true`, one `false`, installed into separate
venvs, run interleaved (ON, OFF, ON, OFF) against the same vault and query back to back:

| run | round 0 | round 1 (28 tok) | round 2 (8 tok) | total |
|---|---:|---:|---:|---:|
| ON #1 | 8.4s | 31.4s (7.0 tok/s) | 30.9s | 70.7s |
| OFF #1 | 8.2s | 31.0s (6.8 tok/s) | 30.5s | 69.7s |
| ON #2 | 7.9s | 30.6s (6.8 tok/s) | 30.8s | 69.3s |
| OFF #2 | 8.3s | 30.7s (6.8 tok/s) | 30.7s | 69.7s |

ON and OFF land within 69.3-70.7s of each other, a ~2% spread indistinguishable from ordinary
run-to-run noise. The prior entry's ~25% finding does not reproduce on this container today, under
an interleaved same-session comparison designed specifically to control for the variance the prior
single-run-each comparison could not.

## 5. Testing whether longer forceable runs would change the verdict

Real BPE splices measured in Section 3's round 1 were short (15 empty, 2 length-1, 1 length-6 out
of 18 constrained tokens) — a property of sumac's tool-call grammar's literal-span structure, not
of the fast-forward mechanism. Before closing this out, the mechanism's one measured real win
(Section 2's dispatch-overhead amortization) needed checking against splice length: does batching
more tokens keep buying proportionally more, or does the curve flatten.

**(measured, CPU container)**, the same isolated GDN harness from Section 2, widened to a
seven-point sweep (1, 2, 4, 7, 14, 28, 56 tokens), 20-run median each, 3-run warmup discarded,
`RecurrentBatchKind::Prefill` at every width so the comparison isolates the general path's own
width-scaling rather than mixing in the separate `Decode`-vs-`Prefill` dispatch question Section 2
already answered:

| width | per-token time |
|---:|---:|
| 1 | 1687.95us |
| 2 | 591.58us |
| 4 | 361.32us |
| 7 | 265.98us |
| 14 | 306.19us |
| 28 | 274.39us |
| 56 | 259.82us |

Per-token cost drops steeply from width 1 to width 7, then flattens completely from width 7 through
width 56 — 266, 306, 274, 260us/tok, all within run-to-run noise of each other. The dispatch-overhead
amortization Section 2 measured is a fixed pool that is already almost entirely captured by a
6-7-token splice; a longer forceable literal run would not unlock a larger discount per token. A
grammar or schema change that produced longer literal spans would extend today's already-tiny
per-token saving (Section 3 context: ~0.29ms/tok against a ~278ms/tok forward pass, roughly 0.1%)
to more tokens, not make the per-token saving itself bigger — the ceiling is set by GDN's real
compute cost, which fast-forward does not reduce, not by how many tokens a single splice covers.

## 6. Why the mechanism is structurally capped here, not merely unlucky

Fast-forward tokens save wall-clock time only where forward-pass cost is either memory-bandwidth-bound
(the GPU-typical case `docs/journal/2026-09-07-fast-forward-prototype.md`'s original projection was
built on — a call streams the same weights from HBM whether it computes 1 token or 7, so skipping
calls skips real time) or dominated by fixed per-call overhead unrelated to FLOPs. Neither holds on
this 4-vCPU CPU container:

- Attention and MLP matmuls (`mistralrs-quant`'s `GgufMatMul`/`QMatMul`, AVX2/AVX-512F dispatch —
  this container's `/proc/cpuinfo` reports `avx2 avx512f fma`, no `avx512vnni`) scale with the
  number of query positions processed regardless of how many calls that spans across — batching
  does not remove this work, only the call-dispatch overhead around it.
- The one place real dispatch overhead exists — GDN's `rayon` parallel dispatch, Section 2 — is
  worth amortizing, but its own real compute is 71.5-85.7% of total forward-pass time
  (`docs/journal/2026-09-07-fast-forward-cpu-profile.md` Section 4), so the amortizable *overhead*
  slice of that is small relative to the whole forward pass: Section 2's measured saving is
  ~0.87ms/tok against a ~278ms/tok forward pass, about 0.3%.
- Section 5 shows that ceiling does not move with splice length — there is no larger reservoir of
  overhead elsewhere in the pipeline waiting to be amortized by a longer forced run.

The technique is correctly implemented (this series' earlier entries) and measurably real at the
GDN-dispatch layer (Section 2) but the quantity it can reclaim on this specific hardware is capped
near zero by the pipeline being compute-bound rather than bandwidth-bound or dispatch-bound at the
scale that matters. This is a property of this CPU (4 cores, no AVX-512VNNI) and this architecture
(GDN-dominated hybrid decode), not of the fast-forward implementation.

---

## Current State

- `docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch` is unaffected by this entry — every diagnostic addition described above (Sections 2, 3, 5) was written directly into `.build/mistralrs-cpu-ff/src` (gitignored), measured, and reverted, verified via `diff` to be byte-identical to the committed patch both before and after each section's work.
- `gdn/backend.rs`'s `causal_conv1d` and `apply_recurrence_from_convolved`, called directly with real Qwen3.5-4B dimensions (`num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel_size=4`, read from the GGUF file's `qwen35.ssm.*` metadata keys), run a batched `seq_len=7` call in 2.03ms median against 8.14ms median for 7 sequential `seq_len=1` calls — a ~4x per-token improvement from batching, confirmed on a second run (1.88ms vs 8.03ms).
- The same harness swept over widths 1/2/4/7/14/28/56 shows per-token time falling from 1687.95us (width 1) to 265.98us (width 7) and staying flat through width 56 (259.82-306.19us across widths 7-56) — the dispatch-overhead-amortization gain is fully captured by width 7 and does not grow with longer forced runs.
- `pipeline/sampling.rs`'s `compute_ff_tokens()` call and `apply_pending_ff_tokens`'s replay loop, instrumented directly and exercised via a real `sumac ask "where is the butter?" --stats` run (`supports_grammar_fast_forward: true` on `pipeline/normal.rs`), add 1.386ms total measured overhead across round 1's 28-token completion (18 grammar-constrained), against a 30-31 second round.
- Two wheels built from otherwise-identical source, one with `supports_grammar_fast_forward: true` and one `false`, run interleaved (ON, OFF, ON, OFF) against the same vault and query on this container measure 69.3-70.7 seconds total across all four runs, a spread indistinguishable from run-to-run noise — no measurable difference between the flag enabled and disabled.
- This container's CPU reports `avx2 avx512f fma` in `/proc/cpuinfo` and 4 cores (`nproc`), no `avx512vnni` — read directly, not previously recorded in this journal series.

## Stubbed

- None — every measurement in this entry used the already-complete patch from `2026-09-07-fast-forward-cpu-result.md`; no new mechanism code exists as a placeholder.

## Missing

- No measurement exists of whether a longer, real (not synthetic) literal span in sumac's actual tool-call grammar would still show a flat per-token GDN curve past width 7 in the full pipeline, as opposed to the isolated GDN-only harness Section 5 used — the isolated harness answers the GDN-specific question directly but does not exercise attention layers, embeddings, or lm_head at a widened width.
- No SIMD-level investigation exists of GDN's scalar recurrence inner loop (`decode_recurrence_cpu_from_convolved`, `gated_delta_rule_recurrence` in `gdn/backend.rs`) despite this entry establishing GDN as 71.5-85.7% of forward-pass time and this container reporting AVX2/AVX-512F/FMA support that the current scalar-loop-plus-`rayon` implementation does not use directly (only `mistralrs-quant`'s separate quantized-matmul kernels use explicit SIMD dispatch on this codepath).
- No GPU measurement exists anywhere in this project of this patch — Section 6's bandwidth-bound-vs-compute-bound distinction is argued from this CPU container's own measurements and general knowledge of GPU decode characteristics, not from a GPU run of this same patch.

## Divergence

- `2026-09-07-fast-forward-cpu-result.md` Section 5 states "Wall-clock: slower with the mechanism on, on this hardware, for this model" (87.2s patched vs 69.5s baseline, one run each). This entry's Section 4 measures 69.3-70.7s across two interleaved runs each of ON and OFF on the same container and finds no reproducible difference. That prior entry is left unedited per this project's journal convention (corrections are recorded in new entries, not retroactive edits); this entry is the correction of record — the ~25% figure does not reproduce and is attributed here to container-performance variance between non-interleaved single runs, not to a property of the patch.
