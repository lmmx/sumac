# sumac: The Precise Gap Blocking an Upstream Win, and a Drafted Fix

**Status:** follows `docs/journal/2026-09-08-fast-forward-cpu-reopened.md`. That entry measured what
the fast-forward mechanism does (6.4x on a fully-forceable completion, ~3% on sumac's own diluted
tool-call grammar) but left open exactly what stands between "measured, working, validated" and
"something sumac's users actually get." This entry states that gap precisely, as a spec for an
upstream `mistral.rs` PR, and drafts and locally tests the smallest fix for it — a runtime opt-in,
not a default change, so it does not depend on anyone re-litigating whether fast-forward should be
on by default. `docs/journal/2026-09-08-fast-forward-runtime-toggle.patch` is the actual diff,
applying on top of the already-committed
`docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch`.

---

## 1. What is actually wrong, stated precisely

Not a correctness bug in the mechanism — `cpu-result.md` already validated byte-identical output,
and `2026-09-08-fast-forward-cpu-reopened.md` §3 confirmed it measurably reduces forward-pass count
on real queries. Not the `false` default either — an author choosing caution for an unmeasured code
path is a reasonable default, not a defect.

The defect is narrower and more mechanical: **`GeneralMetadata::supports_grammar_fast_forward`
(`mistralrs-core/src/pipeline/mod.rs:940`) is a plain `bool` struct field, set to a hardcoded literal
at exactly seven call sites, and nothing anywhere in the crate reads an environment variable, a CLI
flag, an HTTP request field, or a Python API parameter to decide what that literal should be**:

| file:line | value |
|---|---|
| `pipeline/normal.rs:235` | `false` (the only pipeline with the mechanism actually wired up and measured — `cpu-verdict.md` §4, `cpu-reopened.md` §3/§5) |
| `pipeline/embedding.rs:681` | `false` |
| `pipeline/speech.rs:330` | `false` |
| `pipeline/ggml.rs:400` | `false` |
| `pipeline/diffusion.rs:252` | `false` |
| `pipeline/gguf.rs:1405` | `false` (legacy GGUF adapter path, distinct from the native `NormalPipeline` path GGUF text models actually use — see `2026-09-08-fast-forward-cpu-reopened.md` §2's trace through `load_native_normal`) |
| `pipeline/multimodal.rs:1312` | `false` |

Confirmed by grepping every reference to the field name in `mistralrs-core/src/` — the seven
initializer sites above and the two reader sites in `pipeline/sampling.rs:591,830` (which just take
whatever `bool` `GeneralMetadata` was constructed with) are the entire set of places this value is
touched. There is no eighth site anywhere that could plausibly be a config parser, an env lookup, or
an HTTP field mapping.

**The practical consequence:** every consumer of `mistral.rs` — the published `mistralrs` Python
wheel, the `mistralrs` Rust crate, the HTTP server, the CLI — inherits `false` unconditionally and
permanently. Nobody using the project as a dependency, including someone who has explicitly measured
the mechanism and wants it on, has any way to enable it short of cloning the source, editing this
literal by hand, and building their own wheel — which is exactly what
`2026-09-08-fast-forward-cpu-reopened.md`'s entire measurement process had to do. A capability that
was implemented, validated, and measured, is unreachable by every actual user of the published
artifact. That gap — not the mechanism, not the default — is what this entry's patch closes.

## 2. Why a runtime toggle, not a default flip

`2026-09-08-fast-forward-cpu-reopened.md` §4/§5 measured a small real win on typical tool-call
grammars (~3%) and a large one on grammars with long forceable spans (6.4x). Flipping the shared
default to `true` bets that the typical case is positive for every caller's grammar shape on every
device this pipeline serves, on the evidence of one CPU container and one model. An opt-in flag
makes no such bet: default behavior is unchanged (still `false`, still exactly what every existing
caller already gets), and anyone who has measured their own workload can turn it on. This is a
strictly smaller, strictly safer change to ask a maintainer to merge than "change what everyone gets
by default" — nothing about existing behavior moves unless the new environment variable is set.

## 3. The fix, drafted and locally tested

`docs/journal/2026-09-08-fast-forward-runtime-toggle.patch` (apply after the base patch), two files,
touches only the one pipeline the mechanism is actually validated for:

- `pipeline/mod.rs`: adds `pub(crate) fn grammar_fast_forward_env_enabled() -> bool`, a
  `std::sync::OnceLock`-cached read of `MISTRALRS_GRAMMAR_FAST_FORWARD`, checked once per process.
  Same shape as this crate's own existing precedent for exactly this kind of toggle —
  `MISTRALRS_FORCE_AVX2` (`attention/backends/cpu/avx.rs:629`, `OnceLock`-cached, `== Ok("1")`) and
  `MISTRALRS_CPU_KV_F32` (`kv_cache/mod.rs:80`) are the same pattern, already shipping in this crate,
  not a new convention introduced by this patch.
- `pipeline/normal.rs`: the one flippable call site (`:235`) now reads
  `supports_grammar_fast_forward: grammar_fast_forward_env_enabled()` instead of the literal
  `false`, plus an updated doc comment pointing at both CPU measurement entries and stating the
  opt-in explicitly. The six other `false` sites are untouched — `pipeline/mod.rs:935-939`'s own
  doc comment on the field states leaving it `false` for vision/embedding/diffusion/speech pipelines
  is a correctness requirement (`sample_sequence` only stages a splice when the flag is true, and
  those pipelines never consume `Sequence::pending_ff_tokens`), not just an unmeasured-caution
  default, so this patch does not touch them.

**(measured, this container)**, `cargo check -p mistralrs-core` against the patched clone: clean,
zero errors, ~40s incremental. **(measured, this container)** functional test — one `maturin build
--release` wheel (not two; the whole point is one binary, two behaviors), same fully-forceable regex
grammar as `cpu-reopened.md` §5 (shorter fixture text, ~15 tokens, for a quick check rather than a
full timed comparison):

| `MISTRALRS_GRAMMAR_FAST_FORWARD` | wall time | output |
|---|---:|---|
| unset | 2.61s | exact match |
| `1` | 0.78s | exact match |

Same wheel, same process type, only the environment variable differs — 3.3x faster with it set,
output identical either way. This is the toggle actually working, not just compiling: a maintainer
reviewing this patch does not have to take the mechanism's correctness or effect on faith from the
prior entries, this one line of config visibly changes measured behavior on one build.

## 4. What this patch deliberately does not do

- Does not touch the six other `false` sites (embedding/speech/ggml/diffusion/legacy-GGUF-adapter/
  multimodal) — those pipelines' correctness requirement (§3) makes them out of scope for a minimal
  PR, not merely unmeasured.
- Does not add a per-request override (an HTTP field or a `ChatCompletionRequest`/`Runner`
  constructor parameter) — a process-wide env var is the smallest surface that makes the mechanism
  reachable at all; a finer-grained per-request toggle is a larger API surface for a maintainer to
  review and is not needed to unblock sumac, which would set the env var once, process-wide, exactly
  like it already could set `MISTRALRS_CPU_KV_F32` or `RUST_LOG` today.
- Does not change the default anywhere. Every existing caller's behavior is bit-for-bit identical to
  before this patch unless they set the new variable.
- Does not include GPU/CUDA or macOS/Metal measurements — `2026-09-08-fast-forward-cpu-reopened.md`'s
  own Missing section already records neither exists in this series; this patch's CPU-only scope
  matches what has actually been measured, and the toggle itself is device-agnostic (whatever
  device runs `NormalPipeline` reads the same env var).

---

## Current State

- `docs/journal/2026-09-08-fast-forward-runtime-toggle.patch` applies cleanly on top of
  `docs/journal/2026-09-07-fast-forward-mistralrs-v0.9.2.patch` against the `v0.9.2` tag, verified
  in this session's clone (`cargo check -p mistralrs-core` clean).
- `grammar_fast_forward_env_enabled()` (`pipeline/mod.rs`, new in this patch) is read by exactly one
  call site (`pipeline/normal.rs:235`) — the only pipeline `2026-09-08-fast-forward-cpu-reopened.md`
  measured the mechanism against.
- One wheel built from this patch, tested with `MISTRALRS_GRAMMAR_FAST_FORWARD` unset and set to
  `1` against the identical process type and grammar: 2.61s vs 0.78s, both producing byte-identical
  output to the fixed regex-forced passage.
- Not yet submitted anywhere — no PR opened against `EricLBuehler/mistral.rs` as of this entry;
  the patch and this spec exist only in this session's clone and `docs/journal/`.

## Stubbed

- None — the toggle function and its one call site are complete, not placeholders; `cargo check`
  and the functional wall-clock test in §3 both exercise the real code path, not a stub.

## Missing

- No per-request override (§4) — every request within a process gets the same, process-wide
  setting; a finer-grained toggle is unbuilt and, per §4, deliberately out of this patch's scope.
- No HTTP server or CLI documentation of the new environment variable — this patch only touches
  `mistralrs-core`; `mistralrs-server-core`'s routes and the CLI's own help text do not mention it,
  since neither needed to change for the variable to take effect (both already construct pipelines
  through `pipeline/normal.rs`'s `build_normal_pipeline`).
- No upstream review has happened — this is a locally drafted and tested patch, not a submitted or
  accepted one; "the maintainer would probably accept a minimal opt-in over a default flip" is this
  entry's own argument (§2), not a decision made by anyone at `mistral.rs`.
- No macOS/Metal or CUDA measurement of this specific patch — inherits
  `2026-09-08-fast-forward-cpu-reopened.md`'s own Missing section; the toggle is architecturally
  device-agnostic (it gates a `bool` read at pipeline-construction time, before any device-specific
  code runs) but that claim itself is unmeasured on non-CPU devices.

## Divergence

- None — this entry does not correct a claim made in a prior entry; it specifies and drafts new
  work the prior entries identified as needed (`2026-09-08-fast-forward-cpu-reopened.md`'s "Next
  steps" section) but did not do.
