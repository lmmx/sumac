# sumac: Fast-Forward Tokens on CPU, Reopened — Mechanism Confirmed Firing, 6.4x on a Fully-Forceable Completion

**Status:** follows `docs/journal/2026-09-07-fast-forward-cpu-verdict.md`, which closed the series out
as "measured-neutral, structurally capped near zero" on CPU. That verdict's own §6 arithmetic
divides a single-GDN-layer isolated-harness saving (§2 of that entry) by the whole 24-GDN-layer
model's forward-pass cost (`docs/journal/2026-09-07-fast-forward-cpu-profile.md` §4) — a unit
mismatch, independently re-derived by re-reading both entries in this session before running
anything. This entry does not re-argue that point; it runs the two measurements the verdict skipped
(did the mechanism fire at all; what does it do on a completion where most tokens are forceable) and
reports what they show: the mechanism fires, and on a fully-forceable completion it is **6.4x
faster**, not neutral. On sumac's own tool-call queries, where only a few tokens per response are
forceable, the same mechanism's win dilutes to roughly 3% — real, but easy to mistake for noise,
which is what happened in the prior entries.

**Provenance markers**, same convention as the four entries before this one:

- **(measured, this container)** — timed directly in this session, in a container distinct from the
  four prior entries' 4-vCPU/15GB/AVX-512F box: `nproc` reports 20 cores, `/proc/cpuinfo` reports
  `avx avx2` only (no `avx512f`, no `avx512vnni`), and the container's memory cgroup started at
  4GiB (`cat /sys/fs/cgroup/memory.max` — insufficient to even load the 2.7GB Q4_K_M GGUF without
  being OOM-killed mid-load, confirmed three times via `dmesg`'s `Memory cgroup out of memory:
  Killed process ... (sumac)` before the user raised it to 64GiB mid-session). Absolute wall-clock
  numbers in this entry are not comparable to the four prior entries' numbers for that reason; the
  ON-vs-OFF deltas measured within this container are.
- **(verified against mistral.rs `v0.9.2` source)** — read directly in a fresh clone at
  `.build/mistralrs-cpu-ff/src` (gitignored), same tag as every prior entry.

---

## 1. Rebuilding the two-wheel setup on a new container

Rust toolchain and Python tooling were not preinstalled in this container; both were set up from
scratch (`rustup`/`cargo` 1.98.0 already present read-only under a mounted `/mnt/.cargo`,
`/mnt/.rustup`; `uv` installed fresh under `$HOME`, which is itself ephemeral and had to be
reinstalled once after a mid-session container reload). Two build-time issues neither prior entry
hit, both specific to this container, not to the patch:

- The mounted `/mnt/.cargo/config.toml` sets `[target.x86_64-unknown-linux-gnu] linker = "clang"`
  with `rustflags = ["-C", "link-arg=-fuse-ld=mold"]` globally for this user, and this container has
  neither `clang` nor `mold` installed (only `gcc`/`cc` and GNU `ld`/`ld.gold`). Overridden per-build
  with `RUSTFLAGS="-C linker=cc -C link-arg=-fuse-ld=gold"`, not by editing the global config.
- The GGUF weights were already cached (`/mnt/.cache/huggingface`, populated in a prior session) but
  mistral.rs's Rust-side `hf-hub` crate resolves the default cache at `$HOME/.cache/huggingface`,
  not `$HF_HOME`'s value alone in every code path exercised here — resolved with a symlink
  (`ln -s /mnt/.cache/huggingface ~/.cache/huggingface`) rather than chasing the exact resolution
  order, since the mount survives container reloads and `$HOME` does not.

Same method as `cpu-verdict.md` §4 for the two wheels: one clone, the committed patch
(`docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch`) applied once, two `maturin build
--release` passes that differ only in the single `supports_grammar_fast_forward: true` vs `false`
literal in `mistralrs-core/src/pipeline/normal.rs` (`git diff --stat` against the OFF build confirmed
one file, one line, before each ON build).

## 2. A venv-copy trap that produced silent false negatives

The first attempt at E0 (below) used `cp -r .venv .venv-on`/`.venv-off` to avoid re-resolving
sumac's own dependencies twice, then `uv pip install --python .venv-X/bin/python --reinstall
<wheel>` to swap in each patched build. Every run against both copied venvs showed **zero**
`forward_embeds` calls — not fewer calls, none — despite `strings` on the installed `.so` confirming
the instrumentation (below) was compiled in at the right file and line
(`mistralrs-core/src/vision_models/qwen3_5/text.rs:944`, symbol demangling to
`Qwen3_5TextModel::forward_embeds::FF_PROBE_CALLS`). Tracing the actual GGUF-load dispatch path
(`pipeline/gguf.rs`'s `load_model_from_path` → `load_native_normal` → `resolve_native_adapter` →
`NormalLoaderType::Qwen3_5` → `Qwen3_5TextLoader::load`, `pipeline/loaders/normal_loaders.rs:5918-5944`)
confirmed the callsite should run on every decode step; a second, unconditional `eprintln!` placed at
the top of `Qwen3_5TextLoader::load` itself — a one-shot, load-time call, immune to any tracing
subscriber or `RUST_LOG` misconfiguration — also produced no output. The actual cause: `cp -r` copies
a venv's `bin/sumac` launcher script byte-for-byte, including its absolute-path shebang
(`#!/workspace/.venv/bin/python`), so `.venv-off/bin/sumac` and `.venv-on/bin/sumac` were both
silently executing under the *original*, untouched base venv's Python the entire time — never
importing either patched wheel. Fixed with `sed -i "1s|.venv/bin/python|.venv-X/bin/python|"` on each
copied `bin/sumac`; every measurement below post-dates that fix and confirmed the fix worked before
being treated as data. Recorded here because it is a generic trap for any venv-copy-based ON/OFF
comparison method, not specific to this patch.

## 3. E0 — the mechanism fires, confirmed on a real `sumac ask` query

**(measured, this container)**, a `static AtomicUsize` counter plus an unconditional
`eprintln!("FF_PROBE call={n} seq_len={seq_len}")` and a matching `tracing::info!` added to the top
of `Qwen3_5TextModel::forward_embeds` (`vision_models/qwen3_5/text.rs:930`, before this function's
existing body; reverted from the working clone after collecting the numbers below, same
not-committed convention as every prior entry's diagnostic instrumentation). Ran
`sumac ask "where is the rice?" --stats --dry-run` once against each of the fixed, corrected
`.venv-off`/`.venv-on`:

| build | total `forward_embeds` calls | decode calls at width > 1 |
|---|---:|---|
| OFF (`supports_grammar_fast_forward: false`) | 38 | none — all 35 decode calls width 1 |
| ON (`supports_grammar_fast_forward: true`) | 31 | 3 — widths 7, 2, 2 |

Both runs' three prefill calls (widths 112, 746, 847 — round 0/1/2's prompt lengths) are identical
between builds, confirming the ON/OFF split is real and not a load-order artifact. The OFF run
decoded 35 tokens across 35 calls; the ON run decoded 36 tokens (one more — completion length is not
seeded here, so a one-token stochastic difference between separate runs is expected, not a bug)
across 28 calls, 3 of which cover 11 of those 36 tokens in one dispatch each. This directly closes
`cpu-verdict.md`'s own §4 gap: that entry's four ON/OFF timing runs never checked whether the ON
build's forward-pass count actually dropped. Here it does, by exactly the forced-splice widths
`llg.compute_ff_tokens()` reports.

## 4. E3 — real `sumac ask` query, interleaved, interference-free

With both background builds finished (E0's numbers above were measured while a build was still
running on the other CPUs, so not used for timing), five interleaved (ON, OFF) pairs of
`sumac ask "where is the butter?" --stats --dry-run`, no other process competing for CPU, wall time
summed from each run's own three `--stats` round lines (prefill + decode, not a wrapper-script timer,
so Python/shell startup overhead is excluded from the compared quantity):

| pair | ON | OFF | ON's margin |
|---:|---:|---:|---:|
| 1 | 32.5s | 33.1s | 0.6s |
| 2 | 31.6s | 32.8s | 1.2s |
| 3 | 31.1s | 33.2s | 2.1s |
| 4 | 30.8s | 31.8s | 1.0s |
| 5 | 31.2s | 31.5s | 0.3s |

ON faster in 5 of 5 pairs (median 31.2s vs 32.8s, mean margin 1.04s, ~3.2%). Under a null hypothesis
of no real difference, 5/5 same-direction pairs has probability 1/32 by a sign test — small effect,
but not noise, and the direction matches E0's mechanism confirmation and E3's own arithmetic:
sumac's tool-call completions are mostly free-form JSON values the grammar cannot force, with only a
handful of literal scaffold tokens (the widths-7/2/2 splices from §3) actually forceable per
response. This is the same order of magnitude as `cpu-verdict.md` §6's own (mis-scaled) estimate,
now measured directly rather than derived from a mismatched division.

## 5. The clean measurement: what the same mechanism does when most tokens are forceable

Sumac's own grammar dilutes the effect by construction — most of what it asks the model to generate
is free-form (a location name, a quantity) that no grammar can pre-determine. To see the mechanism's
effect without that dilution, a minimal script outside sumac
(`.build/ff_bench.py`, gitignored, not committed) calls `mistralrs.Runner`/`ChatCompletionRequest`
directly against the same cached `unsloth/Qwen3.5-4B-GGUF` `Qwen3.5-4B-Q4_K_M.gguf` sumac uses, with
`grammar_type="regex"` and `grammar=re.escape(<fixed ~73-token passage>)` — a regex matching exactly
one string, so every token after the first is fully determined by the grammar alone.
`enable_thinking=False` is required for this to work at all: left at its default (thinking enabled
whenever neither `enable_thinking` nor `reasoning_effort` is set, per `ChatCompletionRequest`'s own
docstring in `mistralrs/__init__.pyi:148-151`), the model spends its whole token budget on freeform
`<think>` content the grammar does not constrain, and the constrained answer never appears in the
completion at all — confirmed directly: the first attempt without this flag returned an empty
`message.content` at `max_tokens=60`. Ten timed repeats plus two discarded warmup calls, one process
per build (model loaded once, so per-repeat timing excludes model load), `temperature=0.0`:

| | OFF | ON |
|---|---:|---:|
| completion tokens, every repeat | 74 | 73-74 |
| median wall time | 10.94s | **1.70s** |
| median engine-reported `total_time_sec` | 10.89s | **1.69s** |
| range across 10 repeats | 10.6s - 13.1s | 1.53s - 1.90s |
| `forward_embeds` calls per repeat | 73 (all width 1) | **2** (one prefill width 18, one decode width 72) |

**6.4x faster** (10.94s / 1.70s), with the two distributions not overlapping across the measured
repeats — no interleaving or sign test needed to argue this one is real. `grep`ping every
`FF_PROBE` line across all 12 runs (10 measured + 2 warmup) of each process confirms the mechanism:
OFF logs 877 decode calls, every one width 1 (877 ≈ 73 tokens × 12 runs); ON logs 12 decode calls
total, one per run, each covering the entire ~72-73-token forced completion in a single batched
`forward_embeds` invocation. This is the case `cpu-verdict.md` §6 argued could not exist on this
class of hardware ("the pipeline being compute-bound rather than bandwidth-bound or dispatch-bound
at the scale that matters") — it exists, and the difference from §3/§4's small real win is entirely
the forceable-token fraction, not the hardware.

## 6. What this changes about the prior verdict

`cpu-result.md` §5's "~25% slower" (one run each, not interleaved) was already retracted by
`cpu-verdict.md` §4's interleaved comparison; nothing here revisits that retraction. What this entry
adds is the two things `cpu-verdict.md` needed and did not have: direct confirmation the mechanism
fires (§3, closing that entry's own "Missing" note), and a measurement of what it does when the
workload actually exercises it (§5), rather than reasoning about "GDN's real compute cost" in the
abstract. §6 of the prior entry's "structurally capped near zero... a property of this CPU... not of
the fast-forward implementation" does not hold: the same CPU, same model, same mistral.rs build
produces a 6.4x win once the completion is forceable enough. The cap is on sumac's current grammar
shape, not on the hardware or the mechanism.

## Next steps

Two independent pieces of work, only one of which touches mistral.rs:

- **sumac-only, no mistralrs change needed beyond having the flag on:** raise the forceable fraction
  of a real tool-call completion. §3's real query forced 11 of 36 decode tokens (three short
  scaffold spans); §5's synthetic query forced all of it. The gap between them is sumac's own JSON
  tool-call grammar and prompt shape (`src/sumac/llm.py`), not mistral.rs — longer fixed key names,
  enum-valued fields already pinned to one legal value by the schema, more literal scaffolding
  between the free-form slots that need the model's judgment. This is the E4 idea from the
  session that reopened this investigation, not yet measured here; §5 gives an upper bound on what
  it is worth (up to 6.4x on the fully-forced end of the spectrum) but not what a realistic
  schema change actually buys.
- **mistralrs-side, but does not require new mistral.rs functionality:** the mechanism is fully
  implemented in the `v0.9.2` tag already (the committed patch only touches the GDN batching-width
  fast path and the width-narrowing plumbing; it does not add the fast-forward computation itself).
  It ships gated off by a hardcoded `false` literal, once per pipeline type (`normal.rs`,
  `gguf.rs`, `multimodal.rs`, `embedding.rs`, `ggml.rs`, `speech.rs`, `diffusion.rs` — six
  `supports_grammar_fast_forward: false` sites plus `normal.rs`'s one flippable site), with no
  `RUST_LOG`-style env var, CLI flag, or Python-API field reading it — confirmed by grepping every
  reference to the field in `mistralrs-core/src/`. Getting it into what sumac actually runs is a
  build-and-vendor problem, not a missing-feature problem: `scripts/build-mistralrs-cuda.sh`
  already does exactly this shape of thing for the CUDA path (clone, patch, `maturin build`, vendor
  into `vendor/wheels/`, `uv add` the local wheel) — a CPU/Metal sibling script following the same
  pattern needs no upstream involvement at all. An upstream PR (flipping the default, or better,
  exposing a real config knob instead of a hardcoded literal) would reach the plain `pip install
  mistralrs` path every non-source-building user is on, which a vendored wheel does not, but per
  this entry's own §1-§2, mistral.rs's own build tooling and source layout are straightforward to
  work with directly — the vendored-wheel path is available now, upstreaming is a slower option to
  reach for later if the vendored path proves to be a maintenance burden, not a blocker to shipping
  this to sumac's own CPU (or CUDA) users first.

---

## Current State

- `docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch` is unchanged by this entry — every
  diagnostic addition described above (the `forward_embeds` counter in §3, `ff_bench.py` in §5) was
  written into `.build/mistralrs-cpu-ff/src` (gitignored) or `.build/ff_bench.py` (gitignored), never
  committed, matching this series' existing convention for one-off measurement code.
- `mistralrs-core/src/pipeline/normal.rs:235`'s `supports_grammar_fast_forward` literal is the only
  read-write toggle for the mechanism on the code path sumac's GGUF loading actually uses
  (`pipeline/loaders/normal_loaders.rs:5918`'s `Qwen3_5TextLoader`, confirmed by direct trace in §2);
  five sibling pipeline files carry their own hardcoded `false` for pipelines sumac does not use
  (embedding, GGML, speech, diffusion, the legacy-adapter GGUF path in `gguf.rs:1405`, and the
  `--mmproj` multimodal path in `multimodal.rs`), none of them reachable from a text-only GGUF load.
- This container (20 cores, `avx`/`avx2` only, memory cgroup raised to 64GiB mid-session) builds
  both ON and OFF wheels from the committed patch in about 1.5-3 minutes per incremental build once
  the dependency graph is warm, using `RUSTFLAGS="-C linker=cc -C link-arg=-fuse-ld=gold"` to route
  around this container's missing `clang`/`mold` (the mounted global cargo config assumes both are
  present; this container has only `gcc`/`cc` and GNU `ld`/`ld.gold`).
- `.build/ff_bench.py` (gitignored, not committed) demonstrates the fully-forceable case directly
  against `mistralrs.Runner`/`ChatCompletionRequest` with `grammar_type="regex"` and
  `enable_thinking=False`, independent of sumac's own agent/tool-call code entirely.

## Stubbed

- None — every measurement in this entry used the already-complete patch from
  `2026-09-07-fast-forward-cpu-result.md`; the instrumentation added to observe it (§3) was
  diagnostic-only and reverted, same as every prior entry in this series.

## Missing

- No measurement of the actual forceable-fraction gain achievable by editing sumac's own tool-call
  grammar (the "Next steps" sumac-side item above) — §5 measures the ceiling (a grammar that forces
  100% of the completion), not what a realistic schema change to `src/sumac/llm.py` would buy.
- No seed/determinism control on the `sumac ask` queries used in §3/§4 (`AgentRunner` does not
  expose one for interactive use, unlike the eval suite's `--eval-seed` —
  `docs/journal/2026-09-02-eval-suite.md`), so completion lengths vary by a token or two run to run;
  noted explicitly in §3 rather than treated as a discrepancy.
- No CPU/Metal build-and-vendor script exists yet (`scripts/build-mistralrs-cuda.sh` has no sibling
  for the non-CUDA path) — the mechanism this entry measures is not available to any sumac install
  that does not build its own patched wheel by hand, as this session's did.
- No GPU (CUDA) measurement exists anywhere in this series, as recorded in every prior entry's own
  Missing section and still true here.
- No macOS/Metal measurement exists — whether GDN has a Metal kernel at all, and whether this
  container's `avx`/`avx2`-only, no-`avx512f`/`vnni` profile is closer to or further from Apple
  silicon's compute/bandwidth balance than the prior entries' AVX-512F container, is unmeasured in
  every entry in this series including this one.

## Divergence

- `docs/journal/2026-09-07-fast-forward-cpu-verdict.md` states in its Status line and §6 that the
  mechanism is "measured-neutral" and "structurally capped near zero... a property of this CPU...
  not of the fast-forward implementation." §5 of this entry measures a 6.4x speedup from the same
  mechanism, same model, on CPU, once the completion is fully grammar-forceable — the cap that entry
  describes does not hold in general; it holds for the specific, low-forceable-fraction workload
  (`sumac ask`'s own tool-call grammar) both entries happened to test end-to-end. That prior entry is
  left unedited per this project's journal convention; this entry is the correction of record for
  its closing claim. `cpu-verdict.md`'s §4 interleaved-comparison retraction of `cpu-result.md`'s
  "~25% slower" finding is unaffected by this entry and still stands — this entry's own §4 gives a
  third, independent measurement (a real but small ~3% win) under a similar interleaved-control
  method, on different hardware, and agrees with the retraction that a naive single-run comparison
  is not powered to detect an effect this size.
