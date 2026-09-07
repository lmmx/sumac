# sumac: Fast-Forward Tokens — Correction, ISQ Detour, and Where the CPU Time Actually Goes

**Status:** follows directly from `docs/journal/2026-09-07-fast-forward-cpu-result.md`. That entry's
"weights are running dequantized at F32" claim was wrong; this entry corrects it, traces the actual
loader code that proves it wrong, and profiles the real forward pass to answer the question that
entry should have asked instead: not "why isn't the model quantized" (it is) but "where does the
time actually go." It goes almost entirely into the model's Gated DeltaNet layers, not into
lm_head's large vocabulary and not into quantization format. A second, unrelated finding surfaces
along the way: this session's from-source builds are measurably slower than the published PyPI
wheel, independent of everything else investigated, for a reason not isolated here.

**Provenance markers**, same convention as the two entries before this one:

- **(measured, CPU container)** — timed directly in this session, same 4-vCPU/15GB CPU-only
  container, `unsloth/Qwen3.5-4B-GGUF` (`Qwen3.5-4B-Q4_K_M.gguf`).
- **(verified against mistral.rs `v0.9.2` source)** — read directly in the clone at
  `.build/mistralrs-cpu-ff/src` (gitignored). Corrections here that touch prior claims cite the
  exact function or line that establishes them, not RSS numbers or log-line inference.

---

## 1. The correction: weights are quantized, not dequantized to F32

The prior entry's Section 5 misread two signals — `DType selected is F32` in the load log, and a
9.2GB resident-memory figure for a 2.7GB Q4_K_M file — as evidence the model was silently
dequantizing its weights to F32 and losing the point of quantization. Both readings were wrong,
and tracing the actual loader code shows why:

- `DType selected is F32` (`mistralrs-core/src/utils/normal.rs:64`) is the *activation* dtype
  selector, not a weight-format decision. `determine_auto_dtype_all` (same file, line 121) picks
  it deliberately for all-CPU device sets, with its own comment explaining why: *"f16 passes the
  probe matmul on CPU but its range is too small for modern residual streams... bf16 runs
  correctly but taxes every elementwise op with converts while ISQ + the f16 KV cache already bank
  the memory wins, so f32 activations stay the default until a native bf16 op pass lands."* This
  is a considered choice already made by mistral.rs's maintainers, not an oversight.
- The weight path is separate and does not go through this at all. `linear_no_bias`
  (`mistralrs-quant/src/lib.rs:2192`) checks `base_vb.weight_source()` first (line 2199) — GGUF
  tensors carry one — and calls `source.load_linear(...)`, which for `GgufWeightSource`
  (`mistralrs-quant/src/gguf/weight_source.rs:769`) branches on the GGUF tensor's raw dtype: types
  0/1/30 (F32/F16/BF16, i.e. tensors the GGUF file itself never quantized — norms, small vectors)
  go through `load_dense_linear`; everything else — the actual Q4_K attention and MLP
  projections — goes through `load_direct_linear` (line 524), which ends in
  `Ok(Arc::new(GgufMatMul::from_qtensor(weight, bias)))` (line 562). `GgufMatMul::new`
  (`mistralrs-quant/src/gguf/mod.rs:403-413`) wraps that in `QMatMul::from_arc` — candle-core's
  real quantized matmul type, the one whose `vec_dot` kernels
  (`candle-core/src/quantized/avx.rs`, `.../repack_x86.rs`) dispatch to AVX2/AVX-512/AVX-512VNNI
  at runtime via `is_x86_feature_detected!`, confirmed present and checked in this session's
  earlier read of that source. The 9.2GB figure is real but is not "the weights are F32" — it
  reflects load-time peak allocation (holding source and destination briefly), KV cache, a 248,320-
  entry vocabulary's embedding/output tables (GGUF conventionally keeps these at higher precision
  even in a Q4_K_M file, unrelated to the transformer layers' quantization), and ordinary
  process/runtime overhead — not measured apart in this session, but not the single dominant
  explanation the prior entry implied either.

The correction was prompted directly by the user disputing the claim in conversation ("the weights
are already quantised") — right, and confirmed by source rather than taken on assertion.

## 2. The ISQ detour this correction explains

The prior entry's `in_situ_quant="Q4K"` test (**measured, CPU container**: 46.4s load vs ~5s
normal, 2.79 tok/s generation) is now explained rather than mysterious: `in_situ_quant` tells
mistral.rs to *requantize* GGUF weights during loading (`isq_flow::plan` logs "Quantizing model
weights to q4k, with sensitive tensors using q6k" — **measured, CPU container**, this session).
Passed against a GGUF file whose weights are already Q4_K, this pays a real dequantize-then-
requantize round trip at load time for a target format the loader would have produced anyway on
the default path — pure overhead, no compute-path change, which is exactly why generation didn't
speed up. Section 1's finding makes this legible: there was never a "run it unquantized" state for
`in_situ_quant` to fix.

## 3. A second, orthogonal discovery: this session's from-source builds are slower than PyPI's wheel

While chasing a `target-cpu=native` rebuild as a separate lever (a pure compiler flag, tried
because this container's CPU has AVX-512 and AVX-512VNNI available per `/proc/cpuinfo`, unused by
a generically-compiled wheel), inspection of the actual `rustc` invocation
(**measured, CPU container**: `ps aux` during a build) turned up `-C target-cpu=native` already
present — not from anything this session added at that point. The cause:
`mistral.rs`'s own repository ships `.cargo/config.toml` at its root with

```toml
[build]
rustflags = ["-C", "target-cpu=native"]

[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]
```

committed as part of the `v0.9.2` tag. Every from-source build in this session and the prior
entry — the correctness checks, the fast-forward mechanism tests, the GDN fix validation — carried
this flag from the first `cargo check` on, without any deliberate action. The one build in this
whole investigation that does *not* carry it is the very first comparison point: the PyPI wheel
(`uv pip install mistralrs`), which is compiled for broad portability and does not set
`target-cpu=native`.

This means the prior entry's "patched, flag `true`" vs "baseline" comparison was confounded: the
patched side had a compiler-flag advantage the baseline never got. To isolate it, this session
built a third reference point — pristine `v0.9.2`, zero patches applied, same native
`.cargo/config.toml` left untouched (`git stash` on the patched clone, build, test, `git stash
pop` to restore the patches afterward):

**(measured, CPU container)**, same query, same container, back to back:

| build | round 0 | round 1 (28 tok) | round 2 (8 tok) | total |
|---|---:|---:|---:|---:|
| PyPI wheel (no `target-cpu=native`) | 4.7s | 31.4s (5.6 tok/s) | 33.4s (6.0 tok/s) | 69.5s |
| pristine `v0.9.2` from source, `target-cpu=native` | 5.9s | 48.2s (3.3 tok/s) | 45.3s (3.9 tok/s) | 99.4s |

The from-source build is **~43% slower**, with zero of this project's patches applied. This rules
out the specific worry that motivated the rebuild — that the patched build's "no speedup" verdict
in the prior entry was really "no speedup beyond a native-codegen win already banked" — because
the *unpatched* from-source reference is itself far slower than PyPI's wheel, before the
fast-forward mechanism enters the picture at all. It does not, on its own, prove `target-cpu=native`
is the cause (a pristine build *without* the flag was not tested to isolate it from other possible
differences — a different pinned `candle-core` revision between whatever PyPI's publish pipeline
used and this session's fresh `Cargo.lock`, or other profile settings). What it does establish
cleanly: the prior entry's internal comparison (fast-forward on vs off, both from the same
from-source, same-flags build) remains a fair like-for-like test even though neither side of it
should be read as representative of PyPI-wheel-level performance.

## 4. Profiling: where the forward pass actually spends its time

Instrumented `Qwen3_5TextModel::forward_embeds`
(`mistralrs-core/src/vision_models/qwen3_5/text.rs`) directly with `std::time::Instant`
accumulators around the per-layer match arms (`LayerType::FullAttention` vs
`LayerType::LinearAttention`) and around `self.lm_head.forward(&xs)`, logged once per forward call
via `tracing::info!`. Not committed — a one-off diagnostic, reverted after collecting the numbers
below, matching this journal series' convention of not shipping throwaway measurement code (the
prior two entries' own prototypes were likewise never committed).

**(measured, CPU container)**, `RUST_LOG=mistralrs_core::vision_models::qwen3_5::text=info`,
full `sumac ask "where is the butter?" --stats` run, patched build (`supports_grammar_fast_forward`
still `false`, so this is the shipped decode path, not the fast-forward path):

Prefill (one call per round, `seq_len` = prompt length):

| round | seq_len | attn | gdn | lm_head | total |
|---|---:|---:|---:|---:|---:|
| 0 (classify) | 112 | 870ms (14.3%) | 5177ms (85.2%) | 29ms (0.5%) | 6078ms |
| 1 (tool call) | 746 | 5644ms (15.1%) | 31629ms (84.8%) | 45ms (0.1%) | 37321ms |
| 2 (reply) | 849 | 6360ms (14.2%) | 38282ms (85.7%) | 38ms (0.1%) | 44684ms |

Decode (`seq_len=1`, steady state — 35 samples across rounds 1-2, first cold call after model load
excluded):

| component | avg | share |
|---|---:|---:|
| attention | 47.13ms | 17.0% |
| GDN (linear attention) | 198.50ms | 71.5% |
| norm | 0.011ms | ~0.0% |
| logits selection | 0.003ms | ~0.0% |
| lm_head (248,320-way) | 32.14ms | 11.6% |
| **total** | **277.79ms** | |

277.79ms/token implies 3.60 tok/s from the profiled components alone, consistent with this run's
observed 3.7-3.9 tok/s (the small gap is bookkeeping/sampling overhead outside `forward_embeds`).

**GDN dominates in both prefill and decode — 71.5-85.7% of every forward call.** This directly
answers the question posed at the end of the prior entry and refutes the large-vocabulary
hypothesis floated there: `lm_head`'s 248,320-way output projection is a real but minor cost
(11.6% of decode, under 0.5% of prefill, where it only ever runs once per chunk against
`ctx.logits()`'s already-narrowed hidden state — see the fast-forward entries' own `context_lens`
narrowing for the mechanism this reuses).

### Is GDN more expensive per layer, or are there just more GDN layers?

`Qwen3.5TextConfig::layer_types()` (`mistralrs-core/src/vision_models/qwen3_5/config.rs:171`)
places a `FullAttention` layer only where `(i + 1) % full_attention_interval == 0`. The GGUF file's
own metadata settles the count directly (**measured, CPU container**, read straight from the
`.gguf` header): `qwen35.block_count = 32`, `qwen35.full_attention_interval = 4` — 8 attention
layers, 24 GDN layers, a 1:3 ratio.

Dividing the decode averages by layer count: GDN costs ~8.27ms/layer (198.50ms / 24), attention
~5.89ms/layer (47.13ms / 8) — GDN is about **1.40x an attention layer's cost per layer**, real but
modest. The 71.5%-vs-17.0% split in aggregate is explained roughly 3 parts "there are three times
as many GDN layers" to 1 part "each one is somewhat more expensive," not by GDN being dramatically
more expensive per invocation than ordinary attention.

## 5. Disposition

No code changes result from this entry — Section 1's correction changes nothing about the shipped
patch (`supports_grammar_fast_forward` still defaults `false` everywhere, per the prior entry), and
Section 4's profiling instrumentation was diagnostic-only and was reverted (`git checkout --
mistralrs-core/src/vision_models/qwen3_5/text.rs`) before this entry was written; the committed
`docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch` is unchanged and still matches the
working tree exactly (`cargo check -p mistralrs-core`, clean, verified after reverting).

What this entry adds is a corrected and load-bearing account of *why* `sumac ask` is slow on this
CPU, for whoever picks this up next: not missing quantization (Section 1), not ISQ configuration
(Section 2), and — per Section 3 — not fully explained by this session's own from-source build
setup either, which itself underperforms the shipped PyPI wheel for a reason not isolated here.
The one lever squarely implicated by direct measurement is Section 4's: three-quarters of every
forward pass is Gated DeltaNet compute, on a model where GDN layers outnumber attention layers
3:1 and cost moderately more per layer besides. Fast-forward tokens (the mechanism this whole
three-entry series set out to validate) skip forward *passes*, not layers within a pass — it was
never going to touch this cost, which is orthogonal to why the mechanism didn't pay off on CPU.
Any future CPU speedup effort for this model should start from GDN's per-layer cost, not from the
decode-loop-level ideas the earlier `2026-09-06-ask-latency-round-two.md` entry catalogued.

---

## Current State

- `mistralrs-quant/src/gguf/weight_source.rs:524-563` (`load_direct_linear`) and
  `mistralrs-quant/src/gguf/mod.rs:403-425` (`GgufMatMul::new`) together load a GGUF file's
  quantized tensors (Q4_K etc.) into `QMatMul::from_arc`, candle-core's real quantized matmul type
  with runtime AVX2/AVX-512/AVX-512VNNI dispatch (`candle-core/src/quantized/avx.rs`,
  `repack_x86.rs`) — confirmed by direct code trace in this session, not by inference from logs or
  memory figures; `unsloth/Qwen3.5-4B-GGUF`'s default (non-ISQ) load path runs its Q4_K attention
  and MLP projections quantized, contradicting the prior journal entry's claim.
- `mistralrs-core/src/utils/normal.rs:64,121-133` (`determine_auto_dtype_all`) selects F32 as the
  *activation* dtype on all-CPU device sets deliberately (documented in its own code comment as a
  considered tradeoff against bf16's elementwise-convert cost and f16's insufficient range for
  some models' residual streams), unrelated to weight storage format.
- `.cargo/config.toml` at the root of the `v0.9.2` tag sets `-C target-cpu=native` as the default
  `[build]` rustflags for the whole `mistral.rs` workspace, applying to every from-source build in
  this session (including the prior two journal entries') without any deliberate action by this
  session; the PyPI wheel used as this series' original comparison point does not carry it.
- **(measured, CPU container)** a pristine, unpatched `v0.9.2` built from source with this native
  config produces `sumac ask "where is the butter?" --stats` total wall time of 99.4s (5.9s + 48.2s
  + 45.3s across the three agent-loop rounds), ~43% slower than the PyPI wheel's 69.5s baseline
  from the prior entry, with zero of this project's patches applied.
- **(measured, CPU container)** direct instrumentation of `Qwen3_5TextModel::forward_embeds`
  (35 steady-state decode-step samples, plus one prefill call per round) shows Gated DeltaNet
  (linear-attention) layers consuming 71.5% of decode time and 84.8-85.7% of prefill time, versus
  17.0%/14.2-15.1% for ordinary attention and 11.6%/under 0.5% for the 248,320-entry-vocabulary
  `lm_head` projection; the GGUF file's own metadata (`qwen35.full_attention_interval = 4`,
  `qwen35.block_count = 32`) gives an 8:24 (1:3) attention:GDN layer ratio, and dividing by that
  ratio shows GDN costing roughly 1.40x an attention layer per layer — the aggregate dominance is
  mostly (not purely) a layer-count effect.

## Stubbed

- None — this entry's own profiling instrumentation was a diagnostic, not a stub; it was written,
  used to produce the numbers above, and reverted (`git checkout -- .../qwen3_5/text.rs`) before
  this entry was committed, matching the convention the prior two entries in this series already
  established for one-off measurement code.

## Missing

- No isolation of *why* this session's from-source builds underperform the PyPI wheel by ~43%
  (Section 3) beyond ruling out the fast-forward mechanism itself as the cause — `target-cpu=native`
  is the most visible remaining difference but was not tested in isolation (a pristine build with
  that flag explicitly disabled, to compare directly against the flagged pristine build in this
  entry's Section 3 table, would settle whether it helps, hurts, or is neutral here). A pinned
  `candle-core` git revision mismatch between whatever built the published PyPI wheel and this
  session's fresh `Cargo.lock` resolution is an equally live, equally untested candidate.
- No GDN-specific profiling below the layer level exists yet — Section 4 establishes that GDN
  layers dominate and are ~1.40x an attention layer's cost each, but not which part of a GDN
  layer's own work (the depthwise causal conv, the delta-rule recurrence in
  `apply_recurrence_from_convolved`, the input/output projections) accounts for that cost, which
  is the next question a real CPU speedup attempt would need answered.
- No GPU measurement exists anywhere in this three-entry series, as recorded in the prior entry's
  own Missing section and still true here — everything in this entry is CPU-specific and silent on
  whether GDN's cost profile, or the from-source-build slowdown, look the same on the CUDA target
  this project actually deploys to.

## Divergence

- The prior entry, `docs/journal/2026-09-07-fast-forward-cpu-result.md`, states as fact in its
  Section 5 and its own "Current State"/"Missing" sections that GGUF weights get dequantized to
  F32 and that `in_situ_quant="Q4K"` "correctly" requantizes them — both wrong, per Section 1 and
  Section 2 of this entry. That entry is left uncorrected in place (per this project's journal
  convention of not rewriting prior entries) rather than edited; this entry is the correction of
  record. Everything else in that entry — the patch implementation, the GDN `causal_conv1d`
  dispatch fix, the byte-identical-output correctness validation, and the measured
  fast-forward-mechanism-is-slower-not-faster verdict — is unaffected by this correction and
  still stands.
