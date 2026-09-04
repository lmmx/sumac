# sumac: Trace/Verdict Redesign — Handoff

**Status:** diagnosis and decision only — nothing implemented yet. Written as a handoff for a
fresh session to pick up and build. Follows directly from
`docs/journal/2026-09-02-eval-suite.md`'s prompt-variant entries (PromptVariant mechanism, the
overnight nudge-v2/v3/v4 experiment) — read that file's later entries for the full run-up if more
context is needed than this entry restates.

## Context

Investigating a real 9B write regression (`add.multiple_products_with_omitted_amounts`) led to
building `PromptVariant` — a named-registry mechanism for A/B testing `AgentRunner`'s prompt
constants, mirroring `ModelPreset`'s existing pattern — and running an overnight comparison across
`default`/`nudge-v2`/`nudge-v3`/`nudge-v4` on both registered models, 15 full-suite epochs each.

That experiment's results are being discarded as inconclusive (see Decision below), for a reason
that turned out to be more fundamental than "not enough epochs": **the eval suite's trace format
cannot answer the question this kind of experiment needs answered.** Two specific, confirmed gaps,
found by actually reading the raw JSON for two of the night's most surprising results:

1. **Nothing records what's sent to the LLM.** `ToolCallRecord` (`src/sumac/llm.py`) captures a
   dispatched tool call's `name`/`arguments`/`result` — never the system prompt, the injected
   `_EMPTY_PLAN_NUDGE` text, or the message history at the point a request was sent. For a suite
   whose current purpose is comparing prompt wording, the experimental variable itself is
   invisible in the data. Confirmed directly: pulled the raw trace for a
   `add.multiple_products_with_omitted_amounts` failure under `qwen3.5-9b`/`nudge-v4` — three
   entries (`classify_request`, two `sumac_find_inventory` calls), then nothing. The forced-retry
   round that `_maybe_force_action` triggers (appending `empty_plan_nudge`'s text and calling
   `_run_loop()` again) definitely ran — `checks.writes: false` on an ADD-classified scenario
   makes that certain, by reading `_maybe_force_action`'s guard clause, not from anything in the
   data — but what the model was told, and what it said back, is gone. A round that ends in plain
   text with no tool call adds nothing to `self._trace`/`self._trace_history` at all (`_run_loop`
   returns immediately in that branch, before ever calling `_record_call`).
2. **The per-scenario JSON record conflates an execution log with a final verdict.** `duration_s`,
   `tokens_per_sec`, `trace`, `checks`, `passed`, `failures` are all flat sibling fields of one
   `EvalResult`. `trace` is fundamentally log-shaped — an ordered sequence of things that happened
   *during* a run. `checks`/`passed`/`failures` are verdict-shaped — a judgment computed once,
   *after* the run, by comparing final state against expectations. Bundling both as flat fields of
   one object is a data-model mismatch, independent of file format (this isn't a JSON-vs-JSONL
   question) — confirmed as a real, felt problem tonight, not just a theoretical one: every ad hoc
   `jq` query written to debug a specific scenario had to reach through `trace` for behavior and
   `checks`/`failures` for outcome as if they were the same kind of thing.

A third, separate methodological finding worth recording even though it's not primarily a trace
problem: `mistralrs.Runner`'s RNG stream is seeded once per session and shared by every request
that session sends (no per-request seed exists in the SDK). This was already known and was why
"one epoch = one separate process" was designed the first place
(`docs/journal/2026-09-02-eval-suite.md`'s original "Epochs are separate pytest sessions"
section) — but it turned out to bite twice more, in subtler forms, this session:

- Using `pytest -k` to narrow a session to just the two target scenarios (for faster iteration)
  runs them essentially first in the session instead of after ~10 preceding ADD scenarios — same
  nominal `--eval-seed`, genuinely different point in the RNG stream. Produced a misleadingly clean
  10/10-both-conditions result that had zero discriminating power. Fixed mid-session by scoping to
  the whole `evals/test_add.py` file instead of `-k`-selecting two tests, which preserves collection
  order for everything that matters.
- Even running the *full* suite in identical order across two variant-sessions doesn't fully
  control for this: if the nudge wording changes behavior on some *other*, earlier-in-collection-
  order scenario (because it also triggers `_maybe_force_action`), that changes how many tokens
  that earlier scenario generates, which shifts the RNG stream position for every scenario after
  it — including ones with no logical connection to the nudge at all. Confirmed directly:
  `add.existing_item_explicit_location` (a trivial, always-15/15-everywhere scenario) dropped to
  6/15 under `qwen3.5-4b`/`nudge-v3` only. Traced it — `_maybe_force_action` never fires in any of
  the failing runs (the write succeeds on the very first round); the failure is a plain
  amount-hallucination (model said 2, expected 1) with no mechanism connecting it to the nudge
  text at all. The most plausible explanation is exactly this RNG-cascade effect, not a real
  wording-caused regression.

## Decision

- The overnight `default`/`nudge-v2`/`nudge-v3`/`nudge-v4` results are discarded as inconclusive —
  not because the effect sizes were small, but because at least one of the disagreeing scenarios
  (`existing_item_explicit_location`) demonstrably wasn't caused by the thing being varied, which
  undermines confidence in attributing any of the others to it either without being able to see
  what was actually sent/received.
- `nudge-v2`/`nudge-v3`/`nudge-v4` are removed from `PROMPT_VARIANTS` (`src/sumac/llm.py`) —
  rolled back to just `PromptVariant("default")`, mirroring how `MODEL_PRESETS` entries that didn't
  pan out get removed rather than left in the registry as unverified options. `scripts/run-nudge-
  overnight.sh` (referenced the now-removed variants) deleted.
- The `PromptVariant` *mechanism* itself (the dataclass, registry/lookup/default pattern,
  `--eval-prompt-variant` CLI flag, `AgentRunner` threading, `epoch_report.py`'s
  variant-aware grouping) is **kept** — it's validated, general infrastructure independent of any
  specific variant's fate, same category as `ModelPreset` surviving repeated prunings of specific
  model entries. The next real prompt-wording experiment should use it, once it can actually answer
  the question.
- `runs/epochs/full-*` and the earlier `runs/epochs/qwen3.5-9b-{default,nudge-v2}` reconnaissance
  directories are stale (the variants they reference no longer exist in the registry) — left for
  the user's own cleanup, per standing preference this session (`runs/` is disposable, regenerable
  from the registry).

## What's missing — the actual handoff task

Redesign the per-round and per-scenario data captured by the eval harness so that a future
prompt-variant comparison can actually be verified against what happened, not inferred from
`llm.py`'s control flow after the fact. Concretely:

1. **Capture what's sent, not just tool-call results.** Each round `AgentRunner._run_loop`
   (and `_classify`) sends currently disappears except for its outcome. At minimum, capture: the
   system prompt / active `PromptVariant` fields once per scenario (not once per round — this is
   how to avoid the size blowup from repeating a multi-hundred-token system prompt on every
   round), and each round's *incremental* addition to the conversation (what got appended to
   `self._messages` since the last request) rather than the full growing history duplicated every
   time.
2. **Capture what's received, including a plain-text-only reply.** Right now a round with no tool
   call is invisible — `_run_loop` returns without touching `self._trace`/`self._trace_history` in
   that branch. This is the single most acute gap: it's exactly the round that
   `_maybe_force_action` cares about (successful nudge recovery = the *next* round makes a tool
   call; failed recovery = it doesn't), and it's currently the one round type never recorded.
3. **Restructure the record to separate log from verdict**, roughly:
   ```json
   {
     "scenario": "...",
     "verdict": {"passed": false, "checks": {...}, "failures": [...]},
     "metrics": {"duration_s": ..., "tokens_per_sec": ...},
     "rounds": [
       {"sent": {...delta or reference...}, "received": {"tool_call": {...}} | {"text": "..."}},
       ...
     ]
   }
   ```
   Exact shape is a design decision for the implementing session, not dictated here — the
   structural requirement is separating an ordered execution log from a one-shot final judgment,
   not any particular field naming.
4. This touches, at minimum: `src/sumac/llm.py` (`ToolCallRecord` likely becomes something richer,
   or gets a sibling type for text-only rounds; `AgentRunner` needs to record sent content
   somewhere it doesn't today), `evals/evaluators.py` (`EvalResult`'s shape), `evals/conftest.py`
   (the `agent`/`result` fixtures' capture, the `--eval-json` payload), and probably
   `evals/report.jq`/`evals/epoch_report.py` if either needs to read the new shape (neither
   currently parses `trace` contents, only counts/aggregates scenario-level fields, so this may be
   less invasive downstream than it sounds).

**Explicitly not wanted, so it isn't re-litigated:** JSONL event streams, content-addressed blob
storage, a manifest file, a SQLite index. That architecture (proposed once this session, via an
external design review) solves for multi-MB artifacts worth deduplicating, concurrent/sub-agent-
spawning runs needing a `parent_event_id` DAG, and crash-resilience on long-running processes —
none of which describe this system (one `AgentRunner`, one loop, no sub-agents, tool results are a
few hundred bytes of JSON, a full epoch file is tens of KB). The fix is a data-model correction
(separate log from verdict; record inputs, not just outputs) within the existing "one JSON per
scenario, checked into `runs/`" approach, not a new storage architecture.

## Open question, not decided

Once inputs/outputs are actually visible, does the RNG-stream-cascade problem (any earlier
scenario's behavior change shifting the sampling point for everything after it, within one
session) still need its own fix — e.g., something that decorrelates variant-condition sessions
more directly — or is being able to *see* when/why two sessions' traces diverge sufficient to
reason about it by inspection, without changing how sampling itself works? Not resolved this
session; worth deciding once the trace redesign lands and a real comparison is attempted again.

## Missing

- Nothing here has been implemented — this entire entry is the design/diagnosis handoff, not a
  change log.
- The original motivating question (is `qwen3.5-9b`'s occasional write-omission on multi-item ADD
  requests fixable via prompt wording, and does any fix avoid trading it for a different failure)
  is still open. Revisit after the redesign, not before — another blind wording iteration without
  being able to see what's actually happening would repeat tonight's mistake.

---

# 2026-09-04: Trace/Verdict Redesign — Review Addendum

## Context

Written after checking the handoff above against `src/sumac/llm.py`, `evals/evaluators.py` and
`evals/conftest.py` line by line, and after the user supplied two AI-drafted treatments of
"seed the RNG or randomise it when testing an agent" (one ChatGPT, one Claude) to be filtered
against this harness rather than adopted — the same handling
`docs/journal/2026-09-02-eval-suite.md`'s "Repeated-Epoch Comparison" entry gave the two pasted
epoch-benchmark designs.

## Correction: the sent content reaches `self._messages` and is never exported

- `_run_loop` appends a text-only reply to `self._messages` before returning
  (`src/sumac/llm.py:993`), `_maybe_force_action` appends `empty_plan_nudge`'s text to
  `self._messages` (`src/sumac/llm.py:1040`), and `propose()` seats the system prompt at
  `self._messages[0]` (`src/sumac/llm.py:1071`) — at the end of a scenario `self._messages` holds
  the whole conversation, including the model-family-specific rendered tool calls
  (`_render_tool_call`, `src/sumac/llm.py:1009-1014`) that are closer to the wire format than a
  structured record.
- The handoff's "what the model was told, and what it said back, is gone" holds for `--eval-json`
  and not for the runner — the `agent` fixture copies `trace_history` only
  (`evals/conftest.py:193-195`) and never reads `self._messages`, which has no public accessor.
- Requirements 1 and 2 therefore reduce to a `messages` property mirroring `trace_history`
  (`src/sumac/llm.py:739-747`) plus one more line in the `agent` fixture — a smaller change than
  the four-file list in item 4 describes.
- Two holes survive that reduction: `_classify` builds its messages as a local
  (`src/sumac/llm.py:938-941`) and never puts them on `self._messages`, so the classifier round
  needs its own capture; and a REJECT-classified request sets `self._messages = None`
  (`src/sumac/llm.py:1067`) before returning, discarding the transcript for exactly the three
  `evals/test_reject.py` scenarios.
- Capturing the final message list once per scenario removes the size pressure that item 1's
  "incremental addition since the last request" design answers — one system prompt per scenario
  rather than one per round falls out of the message list's own shape, with no delta machinery to
  build or verify.
- Epoch files are ~33KB today (`runs/epochs/full-qwen3.5-9b-default/epoch-02.json`, 25 scenarios)
  and tool results would appear in both `messages` and `trace` — the duplication argues for
  `messages` as the primary record with `trace` kept as an index into it, not for suppressing
  either.

## Capture gaps not in the handoff's list

- `_run_loop` exhausting `MAX_TOOL_ROUNDS` returns `reply_text=""` (`src/sumac/llm.py:1020-1022`),
  which in the recorded output is indistinguishable from a model that replied with empty text — a
  terminal-reason field separates the two.
- `_record_usage` folds each round's tokens into session-level running totals and discards the
  per-round numbers (`src/sumac/llm.py:925-931`); per-round completion-token counts are the
  quantity the RNG-cascade argument in the entry above is made of, and are unrecoverable after the
  fact.
- `_maybe_force_action` firing is currently derivable only by reading its guard clause against
  `checks.writes` (`src/sumac/llm.py:1038-1041`) — recording a marker where the nudge is appended
  is two lines and removes the specific after-the-fact inference the whole redesign exists to end.

## Item 3 is a serializer change, not a shape change

- `EvalResult` already separates verdict fields (`checks`, `failures`, `passed`) from execution
  fields (`duration_s`, `tokens_per_sec`, `trace`) as distinct members
  (`evals/evaluators.py:63-77`) — the flatness the entry objects to is introduced only by the
  per-result dict literal in `pytest_sessionfinish` (`evals/conftest.py:277-291`).
- Grouping into `verdict`/`metrics`/`rounds` therefore lands in that dict literal plus
  `evals/report.jq`; `evals/evaluators.py` needs no change, contrary to item 4's file list.

## The open question, answered: tracing is necessary and not sufficient

- The principle that names the confirmed bug, from the pasted ChatGPT material: a scenario's result
  depends on its own case and its own seed, not on which scenarios ran before it. `--eval-seed`
  satisfies that between epochs and violates it within one.
- Being able to see two sessions' traces diverge explains a difference after the fact; it does not
  restore discriminating power, because a variant that changes token counts anywhere shifts the
  sampling position of every later scenario, so the comparison unit stays confounded however
  legible the traces are. `add.existing_item_explicit_location`'s 6/15 under
  `qwen3.5-4b`/`nudge-v3` is the confirmed instance.
- `mistralrs.Runner` exposes `seed` on `__init__` only, with no reseed method and no per-request
  seed (`mistralrs/__init__.pyi:729-758`) — one Runner per scenario, meaning one model load per
  scenario, is the only construction that gives a scenario its own stream.
- Three options, with the cost that separates them unmeasured:
  - **A. Accept, rotate collection order per epoch.** Cheap (a seeded shuffle keyed on epoch
    index). Converts the cascade from a fixed per-scenario bias into noise that averages out over
    15 epochs. Leaves single-scenario reproduction and `pytest -k` iteration broken.
  - **B. One `Runner` per scenario, seeded from `hash(scenario_id, epoch)`.** Restores per-scenario
    reproduction, makes `pytest -k` safe, and unblocks the save-the-failing-seed-and-replay loop
    that is currently impossible at scenario granularity. Costs one model load per scenario:
    25 loads per epoch against the current one, on epochs already running ~10 minutes.
  - **C. Both shapes at once** — per-scenario seeds when reproducing one failure, shared Runner for
    full-suite throughput — reproduces the exact trap already hit this session, where a `-k`-scoped
    run and a full-suite run disagree by construction. Recorded as rejected, not as a fallback.
- The measurement that decides between A and B: time one `_build_runner(qwen3.5-9b)` from a warm
  page cache, multiply by 25, compare against the per-epoch wall clock already recorded in
  `runs/epochs/*/epoch-*.json`'s `total_duration_s`. Not run this session.
- Pairing is already structurally present and is not what broke: `scripts/epoch-benchmark.sh` runs
  seeds 1..N for every model, so scenario *k* at epoch *j* is compared against the same scenario at
  the same nominal seed across models — the cascade is what desynchronises the two, not the design.

## A second order-dependence channel: the prefix cache

- `mistralrs.Runner.__init__` defaults `prefix_cache_n=16` (`mistralrs/__init__.pyi:729-758`) and
  `_build_runner` passes only `which` and `seed` (`src/sumac/llm.py:652`), so the default is live in
  every eval run.
- Every scenario in a session sends the same system prompt as its prefix, so cross-scenario prefix
  cache hits within one session are expected rather than incidental — a coupling between scenarios
  independent of the RNG stream position.
- Whether a prefix cache hit changes sampled output at all is untested here — hypothesis, not a
  confirmed finding. `prefix_cache_n=0` is the lever if it needs isolating; option B above removes
  the channel as a side effect by ending the shared Runner.

## From the pasted material: what applies

- Test-case-plus-seed independence, and saving a failing seed to replay it deterministically — both
  land on the open question above; the second is blocked today at scenario granularity.
- Investing in trace capture rather than replay-by-reseeding, on the grounds that a one-token
  divergence early makes a long trajectory unrecognisable later — this is an argument *for* the
  redesign this entry hands off, and against treating better seeding as a substitute for it.
- Rotating seeds between the tuning runs and the confirming run (seeds 1-15 to develop, a different
  range to confirm), against overfitting a prompt variant to one seed set. One flag's worth of
  change to `scripts/epoch-benchmark.sh`.

## From the pasted material: what does not apply, recorded so it isn't re-litigated

- **Per-component RNG streams** (separate seeds for environment, tools, injected failures, test-data
  generation): there is one stochastic component in this harness, the model sampler. The inventory
  is a fixed seeded fixture (`evals/fixtures.py`), the four tool callbacks are deterministic
  functions over that store (`src/sumac/llm.py:755-899`), scenarios are hand-written rather than
  generated, and no latency or failure is injected. Nothing to split.
- **Batch-invariance and batch-shape nondeterminism**: one local process, one Runner, requests sent
  sequentially at batch size 1 (`src/sumac/llm.py:979`). Not a noise source here.
- **A hermetic record/replay tier below the evals**: `tests/test_llm.py` already drives
  `AgentRunner` through a scripted fake `SendsCompletions` over a real encrypted `data_dir`, with
  no model and no GGUF — the tier exists and is not what the eval suite is for.
- **Pinned model snapshots over floating aliases**: `ModelPreset` already pins repo id plus exact
  GGUF filename (`src/sumac/llm.py:99-115`).
- **Cluster bootstraps, IQM/`rliable`, sequential testing, pass@k versus pass^k, shrinking,
  adversarial seed search, OpenTelemetry GenAI conventions**: the paired-McNemar/cluster-bootstrap
  version of this was designed and deleted in the reduction pass at 25 hand-written scenarios
  (`docs/journal/2026-09-02-eval-suite.md`, "Repeated-Epoch Comparison"), and
  `evals/epoch_report.py`'s per-scenario `p/N` count table is already the pass^k-shaped view. The
  entry above discards an experiment for a confounding it can name, not for a missing statistic.

## Sequencing

- Three separable changes, in order, one testable at a time: (a) capture — `messages` property,
  classifier round, nudge marker, terminal reason, per-round usage, after which a prompt-variant
  comparison is runnable again; (b) the `verdict`/`metrics`/`rounds` reshape, in the payload
  literal and `report.jq`; (c) the A-versus-B RNG decision, after its one measurement.
- (a) alone reopens the original `add.multiple_products_with_omitted_amounts` question; (b) and (c)
  are not prerequisites for it.
