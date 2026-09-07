# sumac: `sumac ask` Latency, Round Two — Twenty Ideas, Costed

**Status:** analysis only, no code. Nothing in `src/sumac/llm.py` changed to produce this entry.
Follows directly from `docs/journal/2026-09-05-mistralrs-latency-budget.md`, which measured the
per-request budget and shipped three changes off the back of it (gated self-review, a
grammar-constrained classifier, a memoized `build_inventory`). That entry closed with eleven open
threads; this one answers three of them from primary sources, adds a measurement the first entry
did not take, and works through every remaining idea in enough detail to be implemented or
rejected without re-deriving it.

**Provenance and verification markers**, same convention as the previous entry:

- **(verified against committed run data)** — arithmetic over
  `runs/epochs/verify-qwen3.5-4b-default-20/` (20 epochs × 22 scenarios = 440 scenarios, 1,922
  engine requests). Reproduction snippets inline. No model was loaded and no inference was run for
  this entry.
- **(verified against mistral.rs `v0.9.2` source)** — read from the pinned tag on GitHub, the
  version `pyproject.toml`'s `ask`/`ask-cuda` groups install. The previous entry could only mark
  several of these **(from mistral.rs `master`, not `v0.9.2`)** or leave them unchecked; three are
  now settled at the pinned version.
- **(projection)** — arithmetic applying the previous entry's measured fit (67 ms fixed per
  request, 8.17 ms per completion token, ~6,100 tok/s prefill) to counts measured here. Every such
  number is a prediction, not a measurement, and each assumes the `evals/` pass rate holds at its
  current 100 %.
- **(estimate, tokenizer not available)** — character counts are measured; the conversion to
  tokens is estimated from the measured chars-per-token density of the same message class. No
  tokenizer was loaded in the container this analysis ran in. Every such figure should be firmed
  up against the GGUF's own embedded tokenizer before it is trusted to a second significant
  figure.

---

## Executive summary

### Where the time goes now

The previous entry split the budget by *component* — decode 78 %, fixed per-request overhead 15 %,
prefill 7 %. More actionable is a split by *what each request was for*. Re-segmenting the same 440
scenarios, restricted to the first `_run_loop` segment (so the now-gated self-review rounds are
excluded) and costing the classifier at the 1 token its grammar constraint now permits:

| what the request was for | n | mean completion tok | est. sec | % |
|---|---:|---:|---:|---:|
| write call (`discover`/`move`/`consume`) | 314 | **86.9** | 260.8 | **43.7** |
| first `sumac_find_inventory` | 380 | 29.5 | 161.3 | **27.0** |
| terminal plain-text reply | 380 | 29.0 | 123.7 | **20.7** |
| classifier (grammar-constrained) | 440 | 1.0 | 33.1 | 5.5 |
| second `sumac_find_inventory` | 60 | 29.0 | 18.7 | 3.1 |

**≈ 598 s across 440 scenarios = 1,358 ms of engine time per ask, over 3.58 requests each.**

Then the measurement the first entry did not take. Splitting every tool call the model emitted
into the bytes fixed by the schema and the bytes carrying actual information:

| tool | fixed scaffold | variable payload | % scaffold |
|---|---:|---:|---:|
| `sumac_find_inventory` | 84.0 ch | 12.1 ch | **87 %** |
| `sumac_consume_inventory` | 133.0 ch | 32.6 ch | **80 %** |
| `sumac_discover_inventory` | 132.0 ch | 39.3 ch | **77 %** |
| `sumac_move_inventory` | 147.0 ch | 33.5 ch | **81 %** |

**About 74 % of engine time is spent emitting tool-call syntax, and roughly 80 % of that (by
character) is bytes determined the moment the tool name is chosen.** Not narration waste — 0 of
800 tool-call messages carried any preamble text. Effective density on write calls is 1.98
chars/token against 3.26 on the find call, because `pantry-white-unit-r2c3` and `","amount":"`
tokenize badly. That, not the model and not the quantisation, is the target.

### The twenty ideas, in the order this entry works through them

**Tier 1 — engine and process configuration, no behaviour change intended**

1. Set `top_k`, or go greedy. `sampler.rs` full-sorts the ~151 k vocabulary once per token when
   `top_k` is unset and `top_p` is set — which is exactly the current configuration. Highest
   expected value per line changed on this page.
2. `_LocalMistralRsBackend` silently drops `top_k`, `min_p` and `stop_seqs`; they cannot be set
   without touching it.
3. `max_seqs` is at its default of 16 for a single-user CLI.
4. PagedAttention is on by default under the CUDA build and has never been A/B'd against
   `no_paged_attn=True` for single-sequence decode.
5. Build the inventory on a background thread during the classifier round — `send_chat_completion_request`
   releases the GIL, so this is genuinely free parallelism.

**Tier 2 — token-count surgery, application level, no upstream work**

6. Replace long ids in write-call arguments with short handles minted by the search result.
7. Shorten tool names and argument keys; keep the semantics in the descriptions, which are prompt.
8. Stop paying for the terminal narration round on `add`/`remove` — either delete it, or reduce
   the "am I finished?" decision to one grammar-constrained token.
9. Put the location tree in the system prompt, where it is prefix-cached and free.
10. The aggressive version: put the whole product catalog there too, and delete the search round.
11. Fuse the classifier into round 1 with a Lark grammar, subject to a prefix-cache constraint.

**Tier 3 — constrained decoding available today**

12. `grammar` and `tool_schemas` are parsed independently and can both be set on one request. Tool
    calls can be JSON-schema- or regex-constrained now, with ids drawn from a closed alternation.
13. The caveat that governs the whole tier: it will not make anything faster today.

**Tier 4 — the mistral.rs contributions worth making**

14. Fast-forward tokens in the constrained-decode path. The single largest lever identified in
    either entry, and it is Rust glue, not kernels.
15. Expose `MtpConfig::is_builtin` through `mistralrs-pyo3`'s `Runner` (the previous entry's
    finding 6).
16. Prompt-lookup / n-gram speculative decoding.
17. The five-minute prerequisite for 15: read the GGUF tensor index over an HTTP range request.

**Tier 5 — axes outside the per-request budget**

18. Run eval scenarios concurrently against one `Runner`; `max_seqs` is already 16.
19. A resident process, because one-shot CLI latency is dominated by GGUF load, not inference.
20. What the 67 ms fixed per-request cost actually is (the previous entry's open thread 8),
    with a measurement recipe.

Plus three measured negative results and one cross-cutting constraint, recorded so they are not
re-investigated: no preamble text to suppress, `enable_thinking` already correct, `stop_seqs`
buys nothing here, and every proposal must preserve the strict-extension property that keeps the
prefix cache hitting.

### Combined projections

Two orthogonal multipliers. **Idea 1** scales the decode rate and leaves token counts alone.
**Ideas 6–11** cut token counts and request counts at the current rate. Applied as a ladder, each
row on top of the previous **(projection)**:

| after | est. engine s / 440 scenarios | ms per ask |
|---|---:|---:|
| baseline (as measured, re-segmented) | 598 | 1,358 |
| + handles on write arguments (6) | 444 | 1,009 |
| + short tool names and keys (7) | 395 | 898 |
| + deterministic terminal reply (8, hard variant) | 271 | 616 |
| + classifier fused into round 1 (11) | 211 | 480 |
| + location tree in system prompt (9) | 195 | 442 |
| + full catalog injection (10, gated) | 159 | 360 |

The ladder is **not** the sum of the sections' individual figures, and each section states both
numbers where they differ. Ideas 6 and 7 shrink the find call that idea 11 then deletes, so 11 is
worth less here than standalone; idea 11 moves the search off the engine entirely, so idea 10 —
which deletes the same search — is worth much less after it than before. Applying the sections'
standalone savings additively would overstate the total by about a third.

And separately, **idea 14** applied to the baseline with no application change at all:
**≈ 335–417 s (−30 % to −44 %)**, because it converts the 77–87 % scaffold from decode into
prefill.

---

## The measurement this entry adds **(verified against committed run data)**

The previous entry fitted `t = overhead + new_prompt_tokens/P + completion_tokens/D` across all
1,922 requests and reported the three components. It did not ask which *requests* the decode time
belonged to. That question has a different answer from "decode is 78 %", because the four kinds of
request in a `sumac ask` pipeline have completion-token counts differing by two orders of
magnitude.

### Segmenting by role

Each scenario's `usage_history` is in request order, with `round=0` for the classifier and
`round` restarting at 1 on each `_run_loop` entry (`llm.py:1418`). Each domain round's assistant
message is at the corresponding index of the `messages` list filtered to `role == "assistant"`, so
a round can be labelled by what the model actually emitted: which tool it called, or a plain-text
reply.

```python
import json, glob, collections

files = sorted(glob.glob("runs/epochs/verify-qwen3.5-4b-default-20/epoch-*.log.jsonl"))
scen = [json.loads(l) for f in files for l in open(f)]
cat = collections.Counter(); n = collections.Counter(); tok = collections.Counter()
for s in scen:
    uh = s["usage_history"]
    assts = [m for m in (s["messages"] or []) if m["role"] == "assistant"]
    dom = [u for u in uh if u["round"] >= 1]
    first = []                       # first _run_loop segment only
    for u in dom:
        if first and u["round"] <= first[-1]["round"]:
            break
        first.append(u)
    for i, (u, a) in enumerate(zip(first, assts)):
        c = a["content"] or ""
        if "<tool_call>" in c:
            nm = json.loads(c.split("<tool_call>\n")[1].split("\n</tool_call>")[0])["name"]
            lab = "1st_find" if (i == 0 and nm == "sumac_find_inventory") else nm
        else:
            lab = "terminal_reply"
        cat[lab] += u["total_time_sec"]; n[lab] += 1; tok[lab] += u["completion_tokens"]
    cat["classify(grammar,est)"] += 0.067 + 0.00817   # 1 token under _CLASSIFY_GRAMMAR
    n["classify(grammar,est)"] += 1; tok["classify(grammar,est)"] += 1
```

Two adjustments make this an estimate of *today's* budget rather than a measurement of the
recorded run: only the first `_run_loop` segment is counted (the recorded run predates
`_write_is_grounded`, so it carries self-review rounds that no longer fire on a single grounded
write), and the classifier is costed at 1 token rather than the 27 it spent emitting a
`classify_request` envelope before `_CLASSIFY_GRAMMAR` shipped. Everything else is the recorded
per-round `total_time_sec`.

The same segmentation over the **full** recorded run, for reference — this is the measurement, with
no adjustments:

| role | n | completion tok | mean | sec | % of 861.0 s |
|---|---:|---:|---:|---:|---:|
| plain-text reply | 682 | 20,383 | 29.9 | 240.7 | 28.0 |
| `sumac_discover_inventory` | 280 | 24,204 | 86.4 | 233.8 | 27.2 |
| first `sumac_find_inventory` | 380 | 11,218 | 29.5 | 161.3 | 18.7 |
| `classify_request` (pre-grammar) | 440 | 11,880 | 27.0 | 144.4 | 16.8 |
| `sumac_move_inventory` | 40 | 3,500 | 87.5 | 33.0 | 3.8 |
| `sumac_consume_inventory` | 40 | 3,023 | 75.6 | 29.1 | 3.4 |
| second `sumac_find_inventory` | 60 | 1,740 | 29.0 | 18.7 | 2.2 |

682 plain-text replies against 440 scenarios is the self-review pass showing up: a review round
that agrees produces a plain-text reply and nothing else, which is exactly the 279-of-280 case
finding 1 of the previous entry measured. Gating it removes 302 of those 682.

### The first domain tool is `sumac_find_inventory`, always

```
first domain tool: {'sumac_find_inventory': 380}
```

380 of 380 non-`reject` scenarios. No exceptions across 20 epochs. This is load-bearing for idea
11 (the fused router): the pipeline's second request is not a decision the model needs to make,
it is a query string it needs to produce.

### The scaffold measurement

For each tool-call message, the fixed portion is the whole message minus the argument *values*:

```python
body = c.split("<tool_call>\n")[1].split("\n</tool_call>")[0]
d = json.loads(body)
variable = sum(len(str(x)) for x in d["arguments"].values())
fixed = len(c) - variable
```

| tool | n | fixed | variable | % scaffold | mean chars | mean tok | chars/tok |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sumac_find_inventory` | 440 | 84.0 | 12.1 | 87 % | 96.1 | 29.5 | 3.26 |
| `sumac_consume_inventory` | 40 | 133.0 | 32.6 | 80 % | 165.7 | 75.6 | 2.19 |
| `sumac_discover_inventory` | 280 | 132.0 | 39.3 | 77 % | 171.3 | 86.4 | 1.98 |
| `sumac_move_inventory` | 40 | 147.0 | 33.5 | 81 % | 180.5 | 87.5 | 2.06 |

Location values across 400 write calls average **16.5 characters** (`pantry-white-unit-r2c3` is
22). The chars-per-token column is the interesting one: the write calls are 40 % less
token-efficient per character than the find call, because hyphenated internal ids fragment under
BPE while `"strawberry jam"` does not.

**Caveat on the scaffold split.** The 77–87 % figures are by *character*. The token split is
probably lower, because the scaffold is repeated JSON punctuation and a repeated tool name — both
of which tokenize efficiently — while the variable payload is the part that fragments. Everything
downstream that depends on the token share is given as a range for this reason, and marked
**(estimate, tokenizer not available)**. Reading the GGUF's embedded tokenizer and re-running this
split on tokens rather than characters is the single cheapest thing that would tighten this entry.

### Negative result: there is no preamble to suppress

```
sumac_find_inventory       n= 440 preamble=   0 ( 0.0%)
sumac_consume_inventory    n=  40 preamble=   0 ( 0.0%)
sumac_discover_inventory   n= 280 preamble=   0 ( 0.0%)
sumac_move_inventory       n=  40 preamble=   0 ( 0.0%)
```

Zero of 800 tool-call messages emitted any text before `<tool_call>`. A grammar forcing the call to
start immediately, or a `stop_seqs` entry to cut trailing text after `</tool_call>`, would save
nothing. Maximum message lengths (189 chars for `discover` against a 171.3 mean) confirm there is
no trailing junk either. Recorded so it is not tried.

### Where the 598 s sits by component

Applying the previous entry's fit to the counts above: 51,700 completion tokens × 8.17 ms =
422.4 s decode; 1,574 requests × 67 ms = 105.5 s overhead; remainder ≈ 70 s prefill.

| component | est. sec | % |
|---|---:|---:|
| decode | 422 | 70.6 |
| fixed per-request | 105 | 17.6 |
| prefill | 70 | 11.7 |

Prefill's share has risen from 7 % to 12 % because the shipped changes removed decode, not
prefill. It is still the cheapest thing in the pipeline per token — which is the whole basis of
ideas 9, 10 and 14, all of which trade decoded tokens for prompt tokens at roughly 50:1.

---

# Tier 1 — engine and process configuration

Five changes that are not meant to alter what the model decides, only how fast it decides it. Idea
1 is the exception: greedy sampling does change behaviour, deliberately, and that is most of its
value.

## 1. `top_k` is unset, so mistral.rs full-sorts a 151 k vocabulary once per token

**(verified against mistral.rs `v0.9.2` source)**

`_build_request` (`llm.py:1302`) sends `temperature=0.2` and `top_p=0.95` from
`DEFAULT_TEMPERATURE`/`DEFAULT_TOP_P` (`llm.py:163-164`), and sends no `top_k`.
`_LocalMistralRsBackend.send_chat_completion_request` (`llm.py:764-790`) does not forward one
either — `mistralrs.ChatCompletionRequest` is constructed with ten named fields and `top_k` is not
among them, so the engine sees its default.

`mistralrs-core/src/sampler.rs` at `v0.9.2` dispatches like this:

```rust
let temperature = if temperature.is_none_or(|v| v < 1e-7) { None } else { temperature };
match self.temperature {
    None => self.sample_argmax(logits, return_logprobs)?,
    Some(temperature) => {
        let logits = (&logits / temperature)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        // -> sample_top_kp_min_p
    }
}
```

and inside `sample_top_kp_min_p`:

```rust
let k = if top_k > 0 { top_k as usize } else { sampling_probs.len() };
```

`top_k` is stored as `i64` on the `Sampler`, with negative or zero meaning "no limit". With no
limit, `k` becomes the vocabulary size and `partial_sort_top_k`'s
`select_nth_unstable_by(k - 1, …)` partition-then-sort degenerates into a **full sort of the entire
probability vector**. There is no fast path for `top_p >= 1.0`; the only short-circuit in the
function is

```rust
if top_k <= 0 && !(top_p > 0.0 && top_p < 1.0) && !(min_p > 0.0 && min_p < 1.0) { /* plain multinomial */ }
```

which the current configuration misses by exactly one condition — `top_p` is 0.95, strictly inside
`(0, 1)`, so the branch is not taken.

### Why this is likely to be large

Qwen's vocabulary is ~151 k entries. A comparison sort of 151 k `f32` is on the order of 2.6 M
comparisons. Measured decode on this model is **8.17 ms per token**. A 4B model at Q4_K_M is about
2.4 GB of weights, and decode is weight-bandwidth-bound: on any modern card that is a 2–5 ms per
token floor, which leaves several milliseconds per token unexplained. A per-token full sort on the
CPU is a plausible occupant of that gap, and it is the *only* candidate identified so far that is
both large and free to remove.

This is open thread 7 of the previous entry ("whether `temperature=0` or a set `top_k` changes the
8.17 ms/token slope"), which guessed that the current configuration "is the configuration most
likely to take a full-sort path". The source now confirms the guess. The magnitude is still
unmeasured.

### Two variants

**`top_k=20`** (or 40) keeps stochastic sampling and turns the full sort into an O(n) partition
plus a 20-element sort. Behaviour changes only in that the tail beyond rank 20 becomes
unreachable, which at `temperature=0.2` it effectively already is.

**`temperature=0.0`** takes the `sample_argmax` path, a single O(n) pass, and **ignores `top_k`
and `top_p` entirely**. Fastest available. This is the variant to try first, for a reason beyond
speed.

### The determinism corollary, which may be worth more than the speed

`docs/journal/2026-09-04-trace-and-verdict-redesign.md` recorded that `mistralrs.Runner`'s RNG
stream position drifts based on everything that ran earlier in the same session, so two runs
differing in prompt wording are not cleanly comparable. That is why the `nudge-v2/v3/v4` prompt
variants had to be rolled back as unverifiable, and why `PROMPT_VARIANTS` (`llm.py:462-482`) currently
contains exactly one entry. It is also why `evals/conftest.py:66-101` carries
`_shuffle_model_scenarios` and a seed-keyed collection order, and why `--eval-seed` exists at all.

Under `sample_argmax` there is no RNG draw. The stream never advances, so its position cannot
drift, so a prompt-variant comparison becomes exact: two runs differing only in prompt text differ
only because of the prompt text. The entire apparatus built to work around the cascade becomes
unnecessary rather than merely better-controlled.

Whether the eval suite still passes 22/22 under greedy is the open question, and it is a single
`uv run pytest evals` away. Greedy decoding on a 4B tool-calling model is not obviously worse —
`DEFAULT_TEMPERATURE = 0.2` was chosen on the reasoning that "low temperature favours tool-call
reliability" (`llm.py:155-165`), and 0.0 is the limit of that argument, not a departure from it.

One thing greedy does remove: the "regenerate" retry in the `sumac ask` confirmation UX gets its
variety from resampling (`_build_runner`'s docstring, `llm.py:792-825`). Under greedy, regenerating
an unchanged conversation returns the identical plan. That needs a decision — either regenerate
switches to `temperature > 0` for that one request, or it stops being offered, or it becomes
"regenerate with a nudge" and appends something to the conversation.

### Experiment

Three `pytest evals` runs: baseline, `top_k=20`, `temperature=0.0`. Compare
`AgentRunner.tokens_per_sec` and total wall clock, and check the pass rate. If the sort hypothesis
is right, `avg_compl_tok_per_sec` in the per-round usage lines moves visibly and immediately. If it
is wrong, the number barely moves and this whole section costs half an hour.

Note that the eval suite's own comparability problem applies to the *first* two runs and not the
third: baseline and `top_k=20` are both stochastic. Run greedy first, and use it as the reference
point for everything else in this entry.

## 2. `_LocalMistralRsBackend` drops three sampling fields on the floor

**(verified against source, both sides)**

`ChatCompletionRequest` at `v0.9.2` declares, among ~40 fields:

```python
logit_bias: dict[int, float] | None = None
top_k: int | None = None
min_p: float | None = None
stop_seqs: list[str] | None = None
ignore_eos: bool = False
presence_penalty / frequency_penalty / repetition_penalty
dry_multiplier / dry_base / dry_allowed_length / dry_sequence_breakers
```

`_LocalMistralRsBackend.send_chat_completion_request` (`llm.py:770-782`) constructs the request
with exactly ten: `messages`, `model`, `tool_schemas`, `tool_choice`, `enable_thinking`,
`temperature`, `top_p`, `max_tokens`, `grammar`, `grammar_type`. `_build_request` also puts `seed`
in the dict and the backend ignores it (documented, `llm.py:1338-1345`).

This is a prerequisite for idea 1 and not independently interesting, but it is worth noting the
shape: `_build_request`'s dict is the contract every `SendsCompletions` implementation consumes,
so adding a key there is the cheap half, and threading it through the one real backend is the
other half. Both halves are needed; adding `top_k` to the dict alone would silently do nothing,
which is exactly the failure mode the previous entry recorded for `grammar` on the (since-removed)
Modal backend.

Of the remaining fields, `min_p` is worth a look alongside `top_k` (it is a cheaper filter and
composes), and the `dry_*` / penalty family is not — repetition is not a failure mode in this
workload, and the repeated-call problems that *have* occurred (`already_proposed` at `llm.py:1266-1289`,
`repeated_query` at `llm.py:1119-1133`) are handled structurally, in the tool results, which is
the right place for them.

## 3. `max_seqs=16` for a single-user CLI

**(verified against the `mistralrs` stub)**

`_build_runner` (`llm.py:792-811`) constructs `mistralrs.Runner(which=which, seed=seed)` and leaves
every other parameter at its default. `Runner.__init__`'s full signature at `v0.9.2` is 26
parameters; the relevant defaults are `max_seqs=16` ("how many sequences may be running at any
time"), `no_kv_cache=False`, `prefix_cache_n=16` ("sets the number of sequences to hold in the
device prefix cache; older ones are evicted and re-prefilled on a later match").

`sumac ask` runs exactly one sequence. `max_seqs=16` sizes the scheduler's bookkeeping and, under
PagedAttention, the KV block reservation for sixteen. Dropping it to 2 is unlikely to be a large
win and is trivially reversible; it is listed because it costs one keyword argument to test and
because it interacts with idea 18 in the opposite direction (the eval suite wants `max_seqs`
high, the interactive CLI wants it low — which argues for making it a `ModelPreset` field or a
`_build_runner` argument rather than a constant).

`prefix_cache_n=16` should be left alone. The previous entry established it is doing real work:
it holds the classifier conversation and the domain conversation resident simultaneously, which is
why only 34 % of the run's 1.84 M prompt tokens were new. Idea 11 increases the number of distinct
conversations in flight, so if anything this wants to go up, not down.

## 4. PagedAttention has never been A/B'd here

**(from mistral.rs `master`, not `v0.9.2`, for the semantics; verified against the `v0.9.2` stub
for the parameters)**

`Runner.__init__` exposes `no_paged_attn: bool = False` ("disables PagedAttention on CUDA"),
`paged_attn: bool = False` ("enables PagedAttention on Metal"), plus `pa_gpu_mem`,
`pa_gpu_mem_usage`, `pa_ctxt_len`, `pa_blk_size`, `pa_cache_type`. With the all-defaults
construction in `_build_runner`, **PagedAttention is on** under the `ask-cuda` build.

PagedAttention is a throughput feature: it exists so that many concurrent sequences can share a KV
allocator without fragmentation. Its cost is a level of indirection on every attention read — a
block table lookup per block per layer. With a single sequence and no memory pressure there is
nothing for it to win and the indirection is pure overhead. How much is an empirical question and
plausibly small, but `no_paged_attn=True` is one keyword argument.

Two things to hold in mind while testing it. First, it interacts with idea 15: upstream documents
that built-in-MTP speculative decoding *requires* PagedAttention (non-paged KV-cache MTP is
disabled), so if the MTP thread is ever revived this switch has to go back. Second, it interacts
with idea 18 in the same direction as `max_seqs` — concurrent eval scenarios are precisely the
workload PagedAttention is for.

## 5. Build the inventory on a background thread during the classifier round

The previous entry measured 60.5 s of the 921.6 s wall clock — 6.6 % — outside `mistralrs`, and
attributed it to the 440 `sumac_find_inventory` calls plus 360 `_propose_write` calls (finding 7).
Memoizing `build_inventory` for the lifetime of one `propose()`/`revise()` shipped
(`_build_inventory`, `llm.py:1095-1101`), which collapses N reads to one. It does not overlap that one read with
anything.

`Runner.send_chat_completion_request` is a blocking pyo3 call into Rust. pyo3 releases the GIL
around blocking work, so a Python thread started before it genuinely runs in parallel with the
engine rather than time-slicing against it. The classifier round is ~90 ms of wall clock during
which the Python side does nothing at all.

`propose()` (`llm.py:1557-1573`) sets `self._inventory_cache = None` and then immediately calls
`self._classify(prompt)`. Starting `ledger.build_inventory` in a `ThreadPoolExecutor` before the
classify call, and having `_build_inventory` resolve the future instead of calling directly, hides
the whole vault decrypt behind a request that was going to happen anyway. On the eval fixture the
vault is small and this is worth little; on a real household log that has been appended to for a
year it is the difference between the decrypt being on the critical path and not.

Two details. The `REJECT` path returns without ever touching the inventory, so the thread's work is
wasted on 60 of 440 scenarios — acceptable, since it is wasted in parallel. And `passphrase.get_key`
caches the derived key in a process global (`src/sumac/passphrase.py:14,24-29`), so the key
derivation is not repeated per thread; only the log read and fold are.

`commit()` (`llm.py:1587-1615`) deliberately re-reads per write and must not use this. The comment
at `llm.py:978-981` already says so.

---

# Tier 2 — token-count surgery

Six changes at the application layer, none of which require anything from upstream. They share one
premise: at 8.17 ms per decoded token against ~6,100 tok/s prefill, **one generated token costs
about fifty prompt tokens**, and one extra round-trip costs a flat 67 ms before any token is
generated. Every idea here is a trade in that direction — move information from the completion
into the prompt, or delete a round-trip.

They also share one constraint, stated once here because it governs all of them:

> **The prefix-cache constraint.** `_run_loop` (`llm.py:1418-1487`) appends to `self._messages`
> and never rewrites earlier entries, so each round's rendered prompt is a strict extension of the
> previous round's and `prefix_cache_n=16` serves the shared prefix for free. Any change that
> alters an *earlier* message between rounds — swapping the system prompt, rewriting a tool result,
> renumbering handles — forces a full re-prefill of everything from the change point onward. At
> ~1,000 prompt tokens and 6,100 tok/s that is ~160 ms, which is enough to eat the entire saving
> of several ideas below. Each proposal that touches message history says explicitly how it
> preserves the property.

## 6. Handles instead of ids in write-call arguments

**The largest application-level win identified, and it is a correctness change as much as a
latency one.**

### The problem, quantified

Every write call transcribes internal identifiers the model read out of a search result a round
earlier:

```
<tool_call>
{"name": "sumac_discover_inventory", "arguments": {"product_id":"Ocado Italian Chopped Tomatoes","amount":"1","unit":"cans","to_location":"pantry-white-unit-r2c3"}}
</tool_call>
```

Location values average 16.5 characters across 400 write calls, and hyphenated ids fragment under
BPE — `pantry-white-unit-r2c3` is on the order of a dozen tokens on its own. Write calls average
86.9 completion tokens at 1.98 chars/token, against 3.26 for the find call whose only variable
content is natural-language words.

Finding 3 of the previous entry measured this from the other direction: 66.8 % of the characters in
tool-call messages are covered by ≥8-character spans already present verbatim in the conversation.
The model is being paid 8.17 ms/token to copy.

### The change

Have `_sumac_find_inventory` (`llm.py:1102-1165`) mint a short handle for every product and every
location it returns, and have the write tools accept handles instead of ids:

```json
{"products":[{"h":"p1","product_id":"Ocado Italian Chopped Tomatoes",
              "locations":[{"h":"l1","location_path":"Pantry > White Unit R2C3",
                            "amount":"3","unit":"cans"}]}],
 "locations":[{"h":"l7","location_path":"Fridge > Bottle Rack"}]}
```

```
<tool_call>
{"name":"add","arguments":{"p":"p1","a":"1"}}
</tool_call>
```

`_propose_write` (`llm.py:1166-1290`) resolves handles back to real ids before calling
`decide.decide_change`, which is unchanged. The handle table lives on `AgentRunner` next to
`self._searched` (`llm.py:975`) and has exactly the same lifetime: reset at the top of
`propose()` and `revise()`.

### Three effects, not one

**Latency.** Write call 86.9 → ~27 tokens **(estimate, tokenizer not available)**. Across 314 write
calls: ≈18,800 fewer tokens × 8.17 ms ≈ **−154 s, −25.7 %** **(projection)**.

**Ungrounded ids become unrepresentable.** A handle that was never minted does not resolve, so the
tool result is a rejection naming the valid handles rather than a `decide.Rejected` for an unknown
location. More importantly, the model *cannot* invent a plausible-looking `product_id` for a
consume or a move, which is the failure class `_FIND_INVENTORY_SCHEMA`'s description currently
spends four sentences of prompt trying to prevent ("use them, verbatim… never invent one",
`llm.py:236-240`).

**`_write_is_grounded` becomes structural.** That predicate (`llm.py:1509-1531`) exists to decide
whether a plan is trustworthy enough to skip self-review, by checking whether the write's
`product_id`/`unit`/`from_location` all trace back to a prior search result. With handles, a
resolved write is grounded by construction and the predicate collapses to "were all arguments
handles?". The gated self-review shipped in the previous session gets simpler and fires less.

There is a fourth, smaller effect: `unit` mostly disappears from the argument list, because the
handle identifies a specific product-at-location whose unit is already known. `_location_candidates`
(`llm.py:694-717`) and the `unknown_location` rejection branch (`llm.py:1218-1223`) mostly stop
firing.

### What still needs free text

`sumac_discover_inventory` for a genuinely new product — the `ABSENT_PRODUCT` case in
`evals/fixtures.py`, and the "different brand of the same basic product" reasoning `_ADD_PROMPT`
(`llm.py:368-395`) is built around. There the `product_id` is a name the model is inventing on
purpose, in the catalog's own Title Case style, and it must stay a string. The destination can
still be a handle. So `discover` takes either `{"p": "p3", ...}` or `{"new": "Heinz Baked Beans",
"u": "tin", "l": "l4"}`, which is also a cleaner signal to the tool about which of its two jobs it
is doing than the current single shape.

### Risks

The handles must be stable across rounds within one `propose()`, or the prefix-cache constraint is
violated and — worse — a handle in an earlier tool result means something different by the time the
model uses it. Mint once, never renumber, never reuse within a call. A second search that returns
an already-seen product returns its existing handle.

The human-facing rendering must not leak handles. `render.print_plan` reads `ProposedWrite`, which
carries resolved ids, so it is unaffected; but the `trace` shown by `--eval-debug` and the
confirmation UX's grounding badges (`review.review_write`) both read tool arguments, and would
show `p1` where they used to show a product name. That is arguably an improvement for debugging
(it makes the grounding chain explicit) but it is a change to `review.py`'s `ungrounded` check,
which currently reasons about names.

## 7. Shorter tool names and argument keys

The tool name is decoded on every call. `sumac_discover_inventory` is ~6 tokens; `add` is 1. The
argument keys are decoded too: `"product_id"`, `"from_location"`, `"to_location"` are perhaps 3–4
tokens each, three or four times per write call.

The semantics do not live in the names. They live in the `description` fields of the schemas
(`llm.py:212-336`), which are **prompt**, sent in `tool_schemas` on every request, prefix-cached
after the first, and therefore priced at roughly one-fiftieth of a decoded token. Lengthening a
description to compensate for a shortened name is close to free; the current descriptions are
already 40–90 words each and could be longer without anyone noticing in the budget.

Proposed: `f`/`add`/`use`/`mv` for the four tools, `p`/`a`/`u`/`l`/`from`/`to` for the arguments.
Roughly 8 tokens saved per tool call **(estimate, tokenizer not available)**; across 754 tool calls
in the current-budget segment, ≈ **−49 s, −8 %** **(projection)**.

Two caveats worth stating, because this is the idea most likely to be waved through without
thought.

**It is not free of quality risk.** A 4B model's prior over `{"name": "sumac_discover_inventory"}`
is not the same as over `{"name": "add"}`, and short generic names collide with the model's
pre-training notion of what an `add` tool does. `_KIND_BY_TOOL` (`llm.py:339-343`) and
`_TOOL_NAMES_BY_KIND` (`llm.py:426-429`) are the only places the names are structural, so the
change is mechanically small — but it is exactly the kind of change that moves an eval pass rate
for reasons nobody predicted, so it should be measured on its own and not bundled with idea 6.

**It should be measured after idea 1, not before.** Under stochastic sampling this is a ~8 %
effect on a metric whose run-to-run noise the previous entry could not bound. Under greedy it is
directly readable.

## 8. The terminal narration round — 20.7 % of the budget

This is open thread 11 of the previous entry, and the re-segmentation promotes it from "app-level,
needs a UX decision" to the third-largest line item.

### What is being paid for

380 requests, 29.0 completion tokens each, 123.7 s. What they buy:

> *"I've added 1 box of Pizza Express Margherita Pizza to the fridge bottle rack."*
>
> *"I've recorded that you finished the jar of strawberry jam from the Pantry > White Unit R3C1."*
>
> *"The move of 2 tubs of Ragu from the Big Freezer > Drawer 3 to the fridge has been proposed."*

`render.print_plan` already renders, from `ProposedWrite` and its `effects` tuple: the change kind,
the product, the amount, the unit, both endpoints, and the per-location before/after quantities.
The narration restates a strict subset of what is on screen two lines below it, in prose, having
cost a 67 ms round-trip and 29 decoded tokens.

For `find`-classified requests the plain-text reply is not narration — it *is* the answer, and it
carries the judgment the `_FIND_PROMPT` (`llm.py:352-365`) asks for ("Name only the products that
actually answer the request"). Nothing below applies to `find`.

### Variant A — hard stop

For `ADD`/`REMOVE`, terminate `_run_loop` as soon as `self._pending` is non-empty, and synthesize
`reply_text` from the writes. **−123.7 s, −20.7 %** **(projection)**.

What it costs: compound requests. "Add the beans and finish off the jam" produces one write and
stops. The previous entry noted `MAX_TOOL_ROUNDS = 20` is sized for "one write per round… not a cap
on how compound a single request may be" (`llm.py:146-150`), and this variant contradicts that
design intent directly. The recorded run offers no evidence either way, because no scenario in
`evals/` is compound — which is itself a gap worth noting: **the eval suite cannot currently detect
this regression.** Adding a compound scenario is a prerequisite for shipping variant A, not a
follow-up.

### Variant B — the one-token terminator

Keep the round, but constrain it. After a write tool returns, send the next request with
`grammar_type="regex"` and a grammar admitting either the literal `DONE` or a tool call:

```
DONE|<tool_call>[\s\S]*
```

The "am I finished?" decision costs 1 token instead of 29. On `DONE`, synthesize `reply_text` from
the plan; on a tool call, continue the loop exactly as now. 380 × (67 ms + 8.17 ms) = 28.6 s
against 123.7 s: **−95 s, −15.9 %** **(projection)**, and compound requests still work.

This is the same trick already shipped for the classifier (`_CLASSIFY_GRAMMAR`, `llm.py:190`),
applied to the loop terminator instead of the router. It is strictly the safer of the two and
gives up only 5 percentage points.

A subtlety: the grammar must be applied only on rounds *after* a write has been proposed, not on
every round, or the model can never emit the free-text reply that a `find` needs and that an
`add`/`remove` uses to explain a failure. `_run_loop` currently sends one request shape for every
round; this makes the request shape depend on `self._pending`, which is a small change to
`llm.py:1430` and worth calling out because it is the first time the loop's request differs
between rounds.

### The UX decision underneath both

Either variant means the person stops seeing a sentence and starts seeing only the table. That is
a real change to `sumac ask`'s character — the confirmation UX designed in
`docs/journal/2026-09-04-ask-confirmation-ux.md` deliberately shows an effect-preview *and* a
narration line, and `render.py:511-512`, `cli.py:1057-1058` and `cli.py:1203-1204` all print
`plan.reply_text` alongside the table. A synthesized sentence built from the `ProposedWrite` can
reproduce most of it ("Recorded consumption of 1 jar Strawberry Jam" is already `commit()`'s own
summary format, `llm.py:1614`), but it will read as generated rather than written, because it will
be.

This wants a decision from a person looking at the two side by side, not a measurement.

## 9. Put the location tree in the system prompt

Locations are static within a session, small (25 in `evals/fixtures.py`; a real household perhaps
40), and — placed in the system prompt — they sit in the **cache prefix**. They are prefilled once
per session at ~6,100 tok/s and cost nothing on every subsequent request.

What they buy:

- The model never needs a round-trip to turn "the fridge bottle rack" or "the top shelf" into an
  id. Today that is either a second `sumac_find_inventory` (60 of them in the recorded run, 18.7 s)
  or a rejected write that comes back with `_location_candidates`' suggestions and has to be
  retried.
- The location half of `_FIND_INVENTORY_SCHEMA`'s description retires — "Search for the place
  before writing to it if you do not already have its id" (`llm.py:223-226`) — as does the
  `locations`/`location_match_count` half of the search result (`llm.py:1154-1163`), which shrinks
  every search result in the conversation.
- `_MAX_LOCATION_MATCHES = 20` (`llm.py:691`) and the truncation reasoning around it become
  unnecessary; the whole tree is always in view.

**≈ −19 s, −3.1 %** on the second-search line alone at the baseline, **≈ −16 s** at its position in
the ladder once idea 7 has shortened those calls **(projection)** — plus an unmeasured saving on the
rejection-retry path and a smaller prompt per search result.

The reason to do it despite the modest headline number is that it is the cheapest possible test of
idea 10's premise. If the model handles a 40-entry location tree in-prompt as well as it handles a
search result, that is direct evidence about the catalog case; if it degrades, idea 10 is dead and
the test cost almost nothing.

Rendering matters. The tree should go in as paths, not as an id/parent adjacency list —
`config.location_path` already produces `Fridge > Main Shelves > Shelf 2`, and a flat list of
paths is both shorter and closer to how a person names a place. Ids only need to appear if the
model must emit them; under idea 6 it emits handles instead, and the ids never need to be in the
prompt at all.

## 10. The aggressive version: inject the whole catalog, delete the search round

The first `sumac_find_inventory` is 27.0 % of the budget: 380 requests, 29.5 tokens each, plus a
67 ms round-trip each. It exists to answer a question — "what is in the inventory that resembles
this?" — whose complete answer is small enough to fit in the prompt.

For the eval fixture: 9 products across 25 locations. For a real household: perhaps 200 products,
which at ~15 tokens per line is ~3,000 prompt tokens — half a second of prefill, *once per
session*, prefix-cached thereafter, against 27 % of every request thereafter.

Combined with idea 11, this takes the pipeline to **one request per ask** for `find`, and two for
`add`/`remove`.

The honest projection is much smaller than that framing suggests, because **idea 11 has already
claimed most of it**. Once the router is fused, the search round is no longer an engine request at
all — `AgentRunner` runs `_sumac_find_inventory` in Python and injects the result. What idea 10
removes on top of that is only the query string the router still emits (~8 tokens × 440 ≈ 29 s)
and the per-request prefill of the search result now sitting in the conversation (~10 s), against
the added one-time prefill of the catalog itself (~3 s). **≈ −36 s beyond the ladder's earlier
rows, taking the total to ~159 s / 360 ms per ask** **(projection)**.

Applied *instead of* idea 11 rather than after it, it is worth far more — the whole 161.3 s
first-find line. The two are substitutes for each other more than they are complements, and idea
11 is the safer of the two by a wide margin (see the risk below). That is the main argument for
doing 11 and treating 10 as optional.

### Why this is the one idea here that could genuinely cost eval points

`ledger.search_inventory` is not a substring filter. It classifies matches into tiers — exact,
whole-word, substring — and `_sumac_find_inventory` surfaces the top tier as `is_exact_match`
(`llm.py:1136-1140`). The prompts are then built on that signal: `_FIND_PROMPT` spends a paragraph
on how to weigh an exact match against a non-exact one, and the `find.shared_word_picks_right_product`
scenario exists specifically to test that "Butter Beans" is not what someone asking about butter
meant.

Injecting the raw catalog moves that entire judgment from Python into a 4B model's attention over a
200-line table. The tiering signal is gone unless it is recomputed — and it cannot be recomputed
without knowing the query, which is the thing the search round was for.

Three ways to keep it, in increasing order of how much they give up:

1. **Size-gate it.** Inject the catalog when it is under N products (N ≈ 50?), keep the search tool
   above. Correct behaviour at both ends, two code paths to maintain, and the eval fixture only
   ever exercises one of them — which means the untested path is the one real households use.
2. **Inject and keep the tool.** The catalog is context; `sumac_find_inventory` remains available
   for when the model wants the tiering. Saves the round-trip only when the model chooses to skip
   it, which is not controllable and not measurable in advance.
3. **Locations only** — idea 9. Keeps every product-matching semantic exactly as it is.

Recommendation: do idea 9, measure, and treat idea 10 as gated on that result plus a deliberate
expansion of `evals/` with more products and more near-miss names. The current 9-product fixture
cannot distinguish "the model reads a table fine" from "the model reads a 9-line table fine".

## 11. Fuse the classifier into round 1

### The premise

Two facts, both measured above. The classifier is a four-way choice costing a full request. And the
first domain tool is `sumac_find_inventory` in **380 of 380** non-`reject` scenarios — the model is
not deciding *what* to do in round 1, it is producing a query string.

So the pipeline's first two requests are, jointly: a four-way label plus a short string.

### The change

One request against a shared router prompt, with `grammar_type="lark"` (verified available at
`v0.9.2`, see idea 12):

```lark
start: kind " " query
kind: "find" | "add" | "remove" | "reject"
query: /[^\n]{0,60}/
```

~10 tokens. `AgentRunner` then reads the kind, runs `_sumac_find_inventory` itself in Python,
fabricates round 1's assistant turn and tool result into `self._messages`, and enters `_run_loop`
at round 2 with search results already in context. `reject` short-circuits before the search, as
`propose()` already does (`llm.py:1562-1565`).

Old cost: 33.1 s (classify) + 161.3 s (first find) = 194.4 s. New cost: 440 × (67 ms + 10 × 8.17 ms)
+ the same prefill ≈ 109 s. **≈ −85 s, −14 %, applied at the baseline** **(projection)** — and it
removes a round-trip from every single ask, including the `reject` path, which currently pays a
full request to produce one word.

**Applied after ideas 6 and 7 it is worth less**: those shrink the find call from 29.5 to ~21.5
tokens, so the pair being replaced costs ~169 s rather than ~194 s, and the fused router still
costs ~109 s. **≈ −60 s at that position in the ladder** **(projection)**. The saving that
survives is mostly the deleted round-trip, not the deleted tokens — which is the right way to
think about it, and the reason idea 20 (pricing the 67 ms) should be measured before this one is
implemented.

### The prefix-cache trap, and how to avoid it

The obvious implementation switches system prompts between the router request and the domain loop:
router prompt for request 1, `_PROMPT_BY_KIND[kind]` for request 2. That changes message index 0,
so request 2 shares *no* prefix with request 1 and re-prefills ~1,000 tokens — ~160 ms, which is
most of the 194 ms/scenario this idea saves. Implemented naively, it is worth approximately
nothing.

The fix is structural: make the router prompt a literal prefix of all three per-kind prompts, and
append the kind-specific text as a later message rather than replacing the first one. Then request
2 is a strict extension of request 1 and the cache holds. Concretely, `_PROMPT_BY_KIND`
(`llm.py:416-420`) becomes `_ROUTER_PROMPT + _KIND_SUFFIX[kind]`, and the loop's messages become
`[system: router, user: request, assistant: kind+query, tool: results, system-or-user: kind
suffix]`.

This also has a pleasant side effect on `PromptVariant` (`llm.py:462`): `classifier_prompt` and
`prompt_by_kind` stop being independent axes that can drift apart, because the first is a prefix of
the second by construction.

### What is given up

The previous entry's query-classifier design (`docs/journal/2026-09-02-query-classifier.md`)
established that one prompt covering every tool degraded on all three tasks, which is why the
per-kind split exists. This idea does **not** undo that: the domain loop still runs against a
kind-specific prompt with a kind-scoped tool schema. What is shared is only the router preamble,
which contains no tool guidance at all — it describes the four kinds, which `CLASSIFIER_PROMPT`
(`llm.py:193-204`) already does in five lines.

The risk that is real: the query string is now extracted by a prompt that does not know which kind
it is (the grammar emits kind and query in one pass, so the query cannot be conditioned on the kind
having been chosen). If `add` and `find` want materially different first queries from the same
sentence, this loses something. The recorded traces suggest they do not — the first query is
almost always the product noun phrase, for every kind — but that is an impression from reading
traces, not a measurement, and it is checkable from the committed data before writing any code.

---

# Tier 3 — constrained decoding available today

## 12. `grammar` and `tool_schemas` can be set on the same request

**(verified against mistral.rs `v0.9.2` source)**

The previous entry's finding 4 listed `grammar`/`grammar_type` among the fields no `sumac` call
site sets, and finding 2's implementation used them only on the classifier request, which sends
`tool_schemas=[]`. Whether they could be combined with real tool schemas was never established.
The module docstring's constraint — that 0.9.2's bindings register `tool_callbacks` with an empty
schema and reject a request that also declares a real `tool_schemas` entry for the same name — made
it reasonable to assume a similar interaction here.

There is none. `mistralrs-pyo3/src/lib.rs` builds the two independently:

```rust
match grammar_type.unwrap() {
    "regex"       => Constraint::Regex(grammar.to_string()),
    "lark"        => Constraint::Lark(grammar.to_string()),
    "json_schema" => { /* parse -> */ Constraint::JsonSchema(value) }
    "llguidance"  => { /* parse -> */ Constraint::Llguidance(value) }
    _ => return Err(...)
}
```

```rust
let tools = if let Some(tools) = &request.tool_schemas {
    let mut new_tools = Vec::new();
    for schema in tools { new_tools.push(serde_json::from_str::<Tool>(schema)?); }
    Some(new_tools)
} else { None };
```

Both land as separate fields on `NormalRequest`. `mistralrs-core/src/request.rs` confirms the enum:

```rust
/// Control the constraint with llguidance.
pub enum Constraint {
    Regex(String),
    Lark(String),
    JsonSchema(serde_json::Value),
    Llguidance(LlguidanceGrammar),
    None,
}
```

So **four grammar types are available at the pinned version**, not just the `regex` currently in
use, and any of them can accompany a real tool schema.

### What that unlocks

**Every tool-call round can be made unfalsifiable.** A JSON-schema constraint over the union of the
kind's allowed tools makes a malformed call impossible to sample. That retires a whole family of
defensive code paths that exist because a small model can emit a wrong shape:

- the `missing_required_argument` rejection and `_REQUIRED_ARGS_BY_TOOL` (`llm.py:1166-1181`,
  `llm.py:431-435`), written after an LFM2.5 run produced `_amount` instead of `amount`;
- the `tool_not_available` branch in `_run_loop` (`llm.py:1457-1464`), for a name outside the
  schemas actually sent;
- the `invalid_amount` rejection (`llm.py:1192-1195`) for a value `Decimal` cannot parse — a regex
  of `[0-9]+(\.[0-9]+)?` makes it unreachable;
- `_classify`'s `except ValueError: return QueryKind.REJECT` fallback (`llm.py:1414-1416`), which
  is already unreachable under `_CLASSIFY_GRAMMAR` for the local backend and only exists for a
  hypothetical backend that ignores the grammar keys.

**Ids can be constrained to a closed set.** A regex alternation over exactly the `product_id`s and
`location_id`s the preceding search returned makes an invented id unsamplable rather than merely
rejected. That is the same guarantee idea 6 provides by a different route, and the two compose:
handles shrink the alternation from a list of 30-character names to a list of two-character
tokens, which makes the grammar cheap to compile and impossible to get wrong.

Compilation cost is not a concern at this scale — llguidance's own documented overhead is ~50 µs
per token for a 128 k tokenizer, against 8,170 µs of decode.

## 13. …but it will not make anything faster today

**(verified against mistral.rs `v0.9.2` source)**

This is the caveat that governs the whole tier, and it is the finding that determines Tier 4.

`mistralrs-core/src/sequence.rs` holds the grammar state:

```rust
pub enum SequenceRecognizer {
    Llguidance(Box<llguidance::Matcher>),
    None,
}
```

and `mistralrs-core/src/pipeline/sampling.rs` drives it:

```rust
let mask = llg.compute_mask_or_eos()?;                  // mask the logits
// ... sample one token under the mask ...
llg.consume_token(second_logprobs_response.token)?;     // advance by that one token
```

Mask, sample one, advance one. Every step. There is no path in that file by which more than one
token joins the sequence per decode step.

So a grammar that forces the next forty tokens exactly — which is what a JSON-schema constraint
over a tool call does for the `<tool_call>\n{"name": "sumac_discover_inventory", "arguments":
{"product_id":"` prefix — still runs forty full forward passes to emit them. Constrained decoding
at `v0.9.2` buys **correctness, not speed**.

There is a second, smaller consequence worth recording: the mask is applied to the logits *before*
sampling, but the sampler's cost (idea 1) does not depend on how many entries the mask left alive.
`sample_top_kp_min_p` sorts `sampling_probs.len()`, which is the full vocabulary regardless. So a
tight grammar does not mitigate the full-sort problem, and idea 1 remains independently necessary.

---

# Tier 4 — the mistral.rs contributions worth making

The previous entry's open thread 1 framed the upstream question as a roofline measurement against
`llama.cpp` to decide whether contributing to candle's k-quant kernels was worth anything, and
declined it: no GPU in the container, and a deliberate choice not to go down the
mistral.rs-internals path. That framing assumed the contribution on offer was CUDA kernel work with
an unknown ceiling.

Idea 13 says there is a different contribution on offer, with a *known* ceiling, in Rust glue.

## 14. Fast-forward tokens in the constrained-decode path

**(verified against mistral.rs `v0.9.2` source and llguidance's public API)**

### What is missing

`llguidance::Matcher` — the exact type mistral.rs already holds in `SequenceRecognizer` — exposes:

```rust
pub fn compute_ff_tokens(&mut self) -> Vec<TokenId>   // "This will always return [] for non-canonical tokenizers."
pub fn consume_ff_tokens(&mut self) -> Vec<TokenId>
pub fn compute_ff_bytes(&mut self) -> Vec<u8>         // "Return any bytes that are forced by the current parser
                                                      //  state. This also works for non-canonical tokenizers."
pub fn compute_mask(&mut self) -> Result<SimpleVob>
pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob>
pub fn consume_tokens(&mut self, tokens: &[TokenId]) -> Result<()>
pub fn validate_tokens(&mut self, tokens: &[TokenId]) -> Result<usize>
pub fn rollback(&mut self, num_tokens: usize) -> Result<()>
```

The fast-forward family is the point of the API: **when the grammar admits exactly one
continuation, llguidance hands you those tokens and no forward pass is needed to produce them.**
mistral.rs calls `compute_mask_or_eos` and `consume_token`, and none of the three `ff` methods.

### Why it matters so much for *this* workload specifically

77–87 % of every tool call this application emits, by character, is grammar-forced scaffold. Under
a JSON-schema constraint, `<tool_call>\n{"name": "sumac_discover_inventory", "arguments":
{"product_id":"` is one forced span with no branch point in it: once the tool name is chosen, every
byte through to the first argument value is determined. Same for `","amount":"` and `","unit":"`
and the closing `"}}\n</tool_call>`. With fast-forward, those spans cost one prefill each — at
~6,100 tok/s against 122 tok/s, a **50× price cut on 80 % of the output**.

Finding 3 of the previous entry (67–71 % of generated characters are verbatim copies of spans
already in context) is the same observation from the other side. A grammar over the closed set of
ids converts "copy" into "forced", and forced is free.

### Projected effect

Tool calls account for 40,232 of the 51,700 completion tokens in the current-budget segment
(78 %). Taking the scaffold share by token as 55–80 % — a range, because the character measurement
overstates it (see the caveat in the measurement section) — 22,000 to 32,000 tokens move from
decode at 8.17 ms to prefill at ~0.16 ms.

**≈ −181 s to −263 s, taking 598 s to 335–417 s: −30 % to −44 %, with no application change at
all** **(projection)**. Stacked with ideas 6 and 7, which shrink the *variable* remainder, the two
compose rather than overlap — handles make the forced spans a larger fraction of a smaller total.

### Shape of the change

The work is in `mistralrs-core/src/pipeline/sampling.rs` and `sequence.rs`, not in any kernel:

1. After `consume_token`, call `compute_ff_tokens()` (or `consume_ff_tokens()`).
2. Append the returned tokens to the sequence.
3. Advance position/KV bookkeeping for them — they still need to enter the KV cache, which is a
   prefill of a short span, not a decode step each.
4. Respect `max_tokens` and stop conditions across the batch, not just per token.
5. Emit them correctly in the streaming path, which currently assumes one token per step.

Points 3 and 5 are where the real work is; 1 and 2 are trivial. The blocking questions to answer
before starting:

- **Canonical tokenizer.** `compute_ff_tokens` returns `[]` for non-canonical tokenizers. Whether
  Qwen3.5's BPE registers as canonical under llguidance's definition is the first thing to check,
  because if it does not, the token-level API is inert and the implementation has to go through
  `compute_ff_bytes` and re-tokenize — which is doable (that method is documented to work
  generally) but is a materially different patch.
- **Interaction with PagedAttention** and with the batching scheduler, since appending N tokens
  outside the decode loop touches block allocation.
- **Whether upstream wants it**, which is a one-paragraph issue rather than a speculative PR.

This is the contribution. It is bounded, it benefits every mistral.rs user doing structured output
rather than just this application, it requires no CUDA, and — unlike the roofline thread — its
ceiling is known in advance from the measurement above.

## 15. Expose `MtpConfig::is_builtin` through `mistralrs-pyo3`'s `Runner`

The previous entry's finding 6, unchanged and still accurate. `mistralrs-server-core`'s
`mistralrs_for_server_builder.rs` on `master` carries `mtp_config: Option<MtpConfig>` and calls
`.with_mtp(self.mtp_config.as_ref().is_some_and(MtpConfig::is_builtin))`. The Python `Runner`
exposes only:

```python
mtp_model: str | None = None       # "attaches an MTP assistant from a model id or path"
mtp_n_predict: int | None = None   # "controls the number of assistant tokens proposed per
                                   #  speculative step. If unset, the assistant generation
                                   #  config is used"
```

No boolean or sentinel corresponding to `is_builtin`. The built-in-head path is reachable from the
CLI and the Rust builder and not from Python.

Two notes the previous entry did not make. First, `mtp_model` taking "a model id or path" raises
the question of whether an arbitrary small same-tokenizer checkpoint — a Qwen3.5-0.6B — can be
attached as a draft, which would be classic speculative decoding through an existing binding rather
than a new one. Given finding 3's copy rate, acceptance would likely be high. Worth ten minutes of
reading `mistralrs-pyo3`'s handling of that parameter before assuming it only accepts a real MTP
head. Second, the built-in path requires PagedAttention on (upstream: non-paged KV-cache MTP is
disabled), which directly conflicts with idea 4 — they cannot both be tested in the same
configuration.

Ranked below idea 14 because it is more plumbing for less certain benefit, and because it is
blocked on idea 17.

## 16. Prompt-lookup / n-gram speculative decoding

No draft model, no extra memory, no second set of weights: draft the next *n* tokens by matching
the last *k* generated tokens against the prompt and copying whatever followed the match there,
then verify with one forward pass. Acceptance is high exactly when the output copies the input —
which finding 3 measured at 67–71 % for this workload.

It is the textbook fit, and it is listed third rather than first for two reasons. The surface is
larger than idea 14 (it needs a draft/verify loop, not just a splice into an existing one), and
idea 14 subsumes most of its benefit for grammar-constrained spans, which is where the copying
actually happens here. If idea 14 ships, the residual that prompt-lookup would catch is the
*unconstrained* copying — mostly the terminal narration round, which ideas 8 and 10 are separately
trying to delete.

Do it third, or not at all.

## 17. Read the GGUF tensor index over an HTTP range request

The previous entry's open thread 2 asked whether `unsloth/Qwen3.5-4B-GGUF`'s `Qwen3.5-4B-Q4_K_M.gguf`
carries the `mtp.*` tensors that `--mtp` requires, and marked it "readable from the GGUF tensor
index without loading the model" — then left it blocked behind open thread 1, which was declined,
which killed the whole MTP branch.

It does not need to be blocked. GGUF puts its complete tensor index in the file header, before any
weight data. A single HTTP range request for the first few hundred kilobytes of the file on the Hub,
plus a header parse, lists every tensor name. If `mtp.*` is absent, ideas 15 and 16's built-in-head
variant are dead and can be struck from the list permanently. If present, idea 15 becomes a
concrete binding patch with a known payoff.

Five minutes, no GPU, no download, and it resolves a thread that has been blocked across two
entries.

---

# Tier 5 — axes outside the per-request budget

Everything above optimises the engine time of one `sumac ask`. Two things matter more than any of
it for two different audiences, and neither appears anywhere in the per-request budget.

## 18. Run eval scenarios concurrently — the iteration-speed axis

`docs/journal/2026-09-04-modal-remote-inference-backend.md` framed local `mistralrs` epoch wall
clock (~15 min for one 20-epoch prompt-variant verdict) as *the* iteration bottleneck, and proposed
remote inference as the answer — a proposal that entry has since retracted, with the Modal backend
removed entirely. The previous latency entry then asked what the 15 minutes was made of and
answered it in terms of single-request latency.

Neither asked the throughput question. `evals/conftest.py` runs scenarios strictly serially: one
`AgentRunner` per scenario (`conftest.py:195-210`), sharing one `base_runner`, each awaiting the
previous. Meanwhile `Runner` is constructed with `max_seqs=16` and mistral.rs does continuous
batching.

Decode is weight-bandwidth-bound. Reading 2.4 GB of Q4_K_M weights to produce one token produces N
tokens just as easily if N sequences are in flight — the weight read is amortised, only the
per-sequence KV read and the attention scale with N. Batching 4–8 scenarios should approach linear
speedup on wall clock while leaving *per-request* latency roughly unchanged.

15-minute epochs become 2–4 minutes. That is the number that governs how many ideas from this entry
actually get measured.

Four things to check before assuming it works:

- **Prefix cache pressure.** `prefix_cache_n=16` against N concurrent conversations plus N
  classifier conversations. At N=8 that is 16 live conversations against 16 slots — exactly at the
  limit, so it wants raising.
- **`max_seqs`** must stay at 16 or go up for the eval runner, which argues against idea 3's
  reduction being a global constant (make it a `_build_runner` parameter).
- **Reproducibility.** Under stochastic sampling, concurrent execution makes the RNG-stream
  position depend on interleaving, which is worse than the seed-order cascade
  `_shuffle_model_scenarios` already works around. Under greedy (idea 1) the objection vanishes
  entirely — which makes idea 1 a prerequisite for this one, not merely adjacent to it.
- **The fixture.** `evals/conftest.py` builds one seeded inventory per session; concurrent
  scenarios that only read it are fine, but anything reaching `commit()` would need isolation.
  Nothing in `evals/` commits today.

## 19. A resident process — the actual user-facing latency

For someone typing `sumac ask "where is the jam"` once, at a shell, the dominant cost is not any
line in the budget above. It is loading a 2.4 GB GGUF.

`--loop` amortises it — "the model is loaded once, on the first request, and reused for the rest of
the session" (README) — and `shared_runner` (`llm.py:826-854`) makes that work across
`AgentRunner` instances. A single invocation gets none of that. `scripts/measure-runner-load.sh`
exists precisely to time this and was written for a different question (whether to build one
`Runner` per eval scenario).

A resident process — `sumac ask --serve` holding the `Runner`, a unix socket in the data dir, and
the CLI becoming a thin client that falls back to in-process loading when no server is up — makes
every ask a warm ask. Against a cold-start cost measured in seconds, the entire contents of this
entry is a rounding error for that user.

Two design notes. The passphrase-derived key is already cached in a process global
(`src/sumac/passphrase.py:14,24-29`), so a resident server holds vault key material in memory for
its lifetime — that is a threat-model change, and `docs/FORMAT.md` is the place it has to be argued,
not here. And the server must hold *no* conversation state between requests: `sumac ask --loop`
already builds a fresh `AgentRunner` per request deliberately, and the socket protocol should
preserve that property rather than accidentally introduce a session.

## 20. What the 67 ms fixed per-request cost actually is

Open thread 8 of the previous entry, unresolved, and now 17.6 % of the estimated current budget —
up from 15 %, because the shipped changes removed decode while leaving request count untouched. At
3.58 requests per ask it is ~240 ms of every ask.

Candidates, none distinguished:

- **pyo3 request handoff** and the engine's channel/scheduler tick.
- **Full-prompt re-tokenisation.** mistral.rs re-renders and re-tokenizes the whole conversation
  per request. At ~2–8 k tokens and HF `tokenizers` throughput this should be single-digit
  milliseconds — probably not the answer, but it is the easiest to bound.
- **minijinja chat-template render.** The README already documents a
  `RUST_LOG=mistralrs_core::gguf::chat_template=off` filter, which implies it is logged per
  request. Rendering ten messages is sub-millisecond.
- **Per-request sampler and grammar construction.** Building a token-trie mask over a 151 k
  vocabulary for a fresh llguidance parser is the most plausible multi-millisecond item on this
  list, and it would apply only to grammar-bearing requests — which, after ideas 8, 11 and 12,
  becomes *all* of them.
- **Prefix-cache matching**, a token-by-token comparison against up to 16 cached sequences.
- **Kernel launch overhead and a device sync** for a 36-ish-layer model's first forward pass.

A measurement that separates them, from Python, without touching Rust:

1. Same prompt, `max_tokens=1`, sent twice in a row. The second is a total cache hit with no new
   prompt tokens; its latency is handoff + scheduler + tokenise + one forward. Repeat 50 times and
   take the minimum.
2. The same, varying prompt length from 100 to 8,000 tokens with the *same* suffix, so prefill is
   cached and only tokenisation scales. The slope isolates re-tokenisation.
3. The same, with and without a `grammar`, to isolate parser construction.
4. `RUST_LOG=info` on all three, to see what the engine itself reports between receipt and first
   token.

None of it requires reading mistral.rs's source, and the outcome determines whether idea 14's
prefill-instead-of-decode trade is as good as the 50:1 rate suggests — because if a meaningful part
of the 67 ms is per-request grammar construction, a heavily grammared pipeline pays it more often.

---

# Cross-cutting notes

## Measured negative results

Recorded so they are not re-investigated.

**No preamble to suppress.** 0 of 800 tool-call messages emitted text before `<tool_call>`. A
grammar forcing an immediate call, or a `stop_seqs` entry cutting trailing text after
`</tool_call>`, saves nothing. Maximum message lengths are within 10 % of means, so there is no
trailing junk either.

**`enable_thinking=False` is already correct** and set on every request (`llm.py:1334`). There is
no thinking-block overhead to remove.

**`DEFAULT_MAX_TOKENS = 1024`** (`llm.py:165`) does not cost anything. Under PagedAttention it does
not pre-reserve, and no round in the recorded run approached it — the largest single round in the
design journal's real runs was 432 completion tokens, and the largest role mean here is 87.5.

**The `_searched` repeat-query cache** (`llm.py:1119-1133`) and the `already_proposed` guard
(`llm.py:1266-1289`) are both already doing their job; neither shows up as a cost in the recorded
run.

## Risk register

Ordered by how likely each is to move the `evals/` pass rate off 100 %.

| idea | risk | detectable by current `evals/`? |
|---|---|---|
| 10 catalog injection | product-match judgment moves from `search_inventory` into the model | **No** — 9 products is too small to distinguish |
| 8A hard-stop reply | compound requests lose their second write | **No** — no compound scenario exists |
| 1 greedy | tool-call reliability at `temperature=0` unmeasured | Yes |
| 7 short tool names | the model's prior over `add` differs from `sumac_discover_inventory` | Yes |
| 11 fused router | query extraction no longer conditioned on kind | Yes |
| 6 handles | model must indirect through a handle table | Yes |
| 9 location tree | 40-line table in prompt vs a search result | Yes |
| 2, 3, 4, 5, 18, 19, 20 | none intended | n/a |

**Two gaps in `evals/` block two ideas outright.** A compound request scenario ("add the beans and
finish the jam") is a prerequisite for idea 8A. A larger product fixture with more near-miss names
is a prerequisite for idea 10. Both are fixture work, not model work, and both should land before
the ideas they gate rather than after.

## Measurement protocol

The previous entry's rolled-back `nudge-v2/v3/v4` comparison is the cautionary tale: two runs
differing in prompt wording were not cleanly comparable because the RNG stream position drifts on
everything that ran earlier in the session, and the trace format could not show what was actually
sent. The trace format was fixed in
`docs/journal/2026-09-04-trace-and-verdict-redesign.md`. The RNG cascade was not.

So the order matters:

1. **Idea 1 first, in its `temperature=0.0` form.** It is the only change on this page that makes
   every subsequent comparison exact. Measure it against the current stochastic baseline knowing
   that *that* comparison is the noisy one; everything after it is clean.
2. **Idea 20's micro-benchmark second**, because it prices the round-trip and therefore the value of
   ideas 8, 9, 10 and 11, all of which trade round-trips.
3. **One idea per run thereafter**, `--eval-json` per run, `evals/epoch_report.py` to compare. The
   temptation to bundle ideas 6 and 7 (both are "shorten the tool call") should be resisted: 7 is
   the one with a behavioural mechanism nobody can predict.
4. **Re-run the character-vs-token scaffold split with the real tokenizer** before trusting any
   single-token-count projection in this entry to a second significant figure.

## Ideas from the previous entry, restated with their current status

| previous open thread | status after this entry |
|---|---|
| 1. roofline vs `llama.cpp` | still declined — and idea 14 makes it less necessary, since the identified upstream work is not kernel work |
| 2. does the GGUF carry `mtp.*` | **unblocked** — idea 17, an HTTP range request, no GPU needed |
| 3. surface `MtpConfig::is_builtin` | unchanged — idea 15, still blocked on 17 |
| 4. `SELF_REVIEW_ROUNDS = 0` | superseded, shipped as the narrower `_write_is_grounded` gate; idea 6 makes that gate structural |
| 5. grounded-write predicate | shipped; idea 6 subsumes it |
| 6. grammar-constrained classifier | shipped; idea 11 replaces it with something better |
| 7. does `temperature=0`/`top_k` move the slope | **answered as a mechanism** — idea 1, confirmed from `sampler.rs`; magnitude still unmeasured |
| 8. what the 67 ms is | unchanged — idea 20, with a concrete recipe |
| 9. brand-drop retry | **dissolved** — ideas 9 and 10 remove the need for a second search rather than making the heuristic work |
| 10. memoise `build_inventory` | shipped; idea 5 overlaps it with the classifier round |
| 11. is the terminal reply load-bearing | **promoted** — idea 8, now measured at 20.7 % of the budget, with a variant (8B) that keeps the round at 1 token |

---

# Ranked next steps

By expected value per hour of work, with prerequisites noted.

1. **Idea 1 — `temperature=0.0`, then `top_k=20`.** One line each (plus idea 2's plumbing). Largest
   uncertain upside on the page, and it makes every other measurement exact. Prerequisite for 18.
2. **Idea 17 — GGUF header range request.** Five minutes. Resolves a thread blocked across two
   entries, either way.
3. **Idea 20 — the 67 ms micro-benchmark.** Half an hour, no Rust. Prices ideas 8–11.
4. **Idea 6 — handles.** The largest application-level win (−26 % projected), and it converts two
   prompt-enforced invariants into structural ones.
5. **Idea 8B — the one-token terminator.** −16 % projected, no UX loss, no new eval scenario needed
   (unlike 8A).
6. **Idea 9 — location tree in the system prompt.** Small direct win; its real value is as the
   cheap test of idea 10's premise.
7. **Idea 11 — fused router.** −14 % projected, but only if the shared-prefix construction is done
   correctly; naive implementation is worth nothing.
8. **Idea 5 — background inventory build.** Free, small on the eval fixture, larger on a real vault.
9. **Idea 18 — concurrent evals.** Gated on 1. Changes iteration speed more than it changes latency,
   which may make it the highest-value item on this list in practice.
10. **Idea 14 — the mistral.rs ff-token patch.** The largest single lever measured anywhere in
    either entry (−30 % to −44 % with no application change), and the answer to "is contributing
    upstream worth it". Ranked tenth only because it is upstream work with a review cycle, not
    because it is small. Open the issue early; the patch can follow.
11. **Ideas 7, 3, 4, 12** — cheap, individually modest, measure one at a time.
12. **Idea 19 — resident process.** Not a latency optimisation in the sense of this entry, but the
    one that a person typing `sumac ask` would notice most. Needs a threat-model paragraph in
    `docs/FORMAT.md` first.
13. **Ideas 10, 15, 16** — each gated on something above (an expanded fixture, idea 17, idea 14
    respectively).

## Sources

- mistral.rs `v0.9.2`: `mistralrs-core/src/sampler.rs`, `mistralrs-core/src/request.rs`,
  `mistralrs-core/src/sequence.rs`, `mistralrs-core/src/pipeline/sampling.rs`,
  `mistralrs-pyo3/src/lib.rs`, `mistralrs-pyo3/mistralrs.pyi` —
  <https://github.com/EricLBuehler/mistral.rs>
- llguidance `Matcher` API — <https://docs.rs/llguidance/latest/llguidance/>
- llguidance grammar syntax (Lark variant, `%json`) —
  <https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md>
- The PR that introduced llguidance into mistral.rs —
  <https://github.com/EricLBuehler/mistral.rs/pull/899>
- This repository: `runs/epochs/verify-qwen3.5-4b-default-20/`, `src/sumac/llm.py`,
  `evals/conftest.py`, `evals/fixtures.py`, and the four preceding journal entries.

---

# Addendum — turning the backlog into an execution plan

**Added after review. Nothing above this line was changed.** The review's verdict, near enough
verbatim: *"The document is excellent as a research backlog. It is not yet a good execution plan.
It has effectively ranked by technical upside / conceptual importance, not by what you should
actually do next."*

That criticism is correct, and it is worth naming precisely what went wrong with "Ranked next
steps" above. That list is ordered by *expected value per hour of work*, which sounds like an
execution order and is not one. It ignores two things an execution order must respect: whether an
experiment's result changes what you do next, and whether an experiment is cheap to *undo*. Idea
14 sits at position 10 with the largest measured upside on the page — which is precisely the shape
of a research backlog entry and precisely the wrong thing to look at when deciding what to do on
a Tuesday.

## The reframing: three optimisation classes, currently mixed together

The twenty ideas are not one list. They are three lists that have been interleaved, and they have
different costs, different risks, and different consumers.

| class | what it changes | ideas |
|---|---|---|
| **A — make inference itself faster** | tokens/second, at fixed token count | 1 greedy, 1 `top_k`, 3 `max_seqs`, 4 PagedAttention, 14 ff-tokens, 15/16 MTP and speculative decoding |
| **B — make the agent generate less** | token count and request count, at fixed tokens/second | 6 handles, 7 short names, 8 terminal reply, 9 location tree, 10 catalog, 11 fused router, 12 constrained tool calls |
| **C — make experimentation faster** | how many of A and B you can afford to test | 18 concurrent evals, 19 resident process, 20 the 67 ms micro-benchmark |

The classes differ in a way that decides the order:

- **A is configuration.** One keyword argument, reversible in seconds, no protocol change, no eval
  fixture work, and the result is a single number.
- **B is protocol.** It changes tool schemas, message shapes, or what the person sees. Each item
  needs its own eval run, two of them need new eval scenarios first, and none of them is trivially
  revertible once downstream code depends on the new shape.
- **C is neither** — it changes the cost of every A and B experiment that follows, which means its
  value compounds and it should run *alongside* the early work rather than after it.

Stated that way, the ordering falls out: **do all of A first, because it is cheap and its results
reprice B; run C in parallel, because it makes the rest affordable; do B in order of
number-produced-per-unit-of-protocol-churn; and touch the upstream items last, because they are
the only ones with a review cycle outside this repository.**

## The revised order

The review's list, adopted, with one item struck (see below):

| # | do | class | produces |
|---|---|---|---|
| 1 | greedy (`temperature=0.0`) | A | a decode-rate number, and exact comparability for everything after it |
| 2 | `top_k=20` | A | whether the win is the sort or the sampling path |
| 3 | the 67 ms micro-benchmark (idea 20) | C | the price of a round-trip, which reprices all of B |
| ~~4~~ | ~~`llama.cpp` same-model/GPU comparison~~ | ~~A~~ | **struck — see below** |
| 5 | one-token terminal decision (idea 8B) | B | −16 % projected, no protocol change, no new fixture |
| 6 | handles (idea 6) | B | −26 % projected, real protocol change |
| 7 | location tree (idea 9) | B | small win, and the cheap test of idea 10's premise |
| 8 | fused classifier/router (idea 11) | B | −14 % standalone, gated on item 3's number |
| 9 | concurrent evals (idea 18) | C | **run in parallel from the start**, gated only on item 1 |
| 10 | everything else | — | ff-tokens, MTP, constrained-decoding internals |

Three changes from "Ranked next steps" above, each with a reason worth recording.

**Idea 8B moves above idea 6.** The review calls it "sneakily excellent" and the numbers support
it: 20.7 % of engine time for a sentence Python can synthesize, and the one-token `DONE`
variant preserves compound requests, so it needs no new eval scenario and changes no tool schema.
It is the cleanest experiment in class B — the largest win available without touching the
protocol. Handles are the better *architecture* and remain so; they are simply not the better
*next move*.

**Idea 6 is P1, not P0.** The reasoning: *"How fast can this model actually run when we aren't
spending so many tokens on sampling?"* is a question you want answered **before** changing the
protocol, because the answer changes how much the protocol change is worth. If class A turns out
to double the decode rate, every class-B saving is halved in absolute terms and the case for a
disruptive change weakens. If class A does nothing, class B is the whole game. Doing A first is
not caution, it is sequencing: A's result is an input to B's cost/benefit.

There is a nice secondary argument for handles that the entry above buried under the latency
figure and the review surfaced properly: **the model should not be responsible for faithfully
copying opaque identifiers around in the first place.** That is a design argument, independent of
tokens per second, and it is the reason handles stay high on the list even if class A goes well.

**Idea 17 (the GGUF range request) drops off the front.** It is five minutes, but MTP is not on
the critical path, and five minutes spent on a branch that cannot affect today's latency is five
minutes not spent on the empirical fork. Do it later, once something upstream is actually being
pursued.

## The struck item, and what replaces it

The review moved the `llama.cpp` comparison to position 4 and argued it hard: 122 tok/s should be
interrogated against a second engine, because *"it tells us which universe we're in — if
`llama.cpp` gives ~120 tok/s too, stop thinking about kernel optimisation; if it gives ~180 tok/s,
we have an engine problem."* That is a sound argument in the abstract and it is the same question
the previous entry's open thread 1 asked.

**It is struck by decision, not by disagreement.** `llama.cpp` is not an engine this project wants
to run, and the repository owner has ruled out going down that path — including as a
benchmark-only exercise, on the grounds that a benchmark is how these things start. Open thread 1
was already declined once; this records that the decline extends to the comparison, and that it
should not be re-proposed.

What is lost: a clean external bound on whether 122 tok/s is a mistral.rs problem or a hardware
problem. What can substitute, none of it requiring another engine:

1. **A spec-sheet roofline.** Q4_K_M at 4B is ~2.4 GB of weights, read once per token. Divide by
   the card's rated memory bandwidth. That gives a floor in milliseconds per token with no
   software involved at all, and comparing 8.17 ms against it says immediately whether there is a
   4× gap to explain or a 1.3× one. This is arithmetic, not a benchmark.
2. **Items 1 and 2 answer most of the same question.** If greedy or `top_k=20` moves the decode
   rate materially, the gap was the sampler — an engine-side problem, already located, already
   fixed, and no kernel work implied. If neither moves it, the gap is elsewhere and the roofline
   from (1) says whether "elsewhere" is the hardware.
3. **Item 3 bounds the per-request half** independently of the per-token half.

Between them, those three answer the "which universe" question well enough to decide whether Tier
4 is ever worth opening, without installing a second inference engine. If they leave it genuinely
ambiguous, that ambiguity is itself the finding, and the decision at that point is to accept
mistral.rs's decode rate as given and spend everything on class B — which is a legitimate outcome
and arguably the expected one.

## The decision tree

Every node produces a number or kills a branch. Kill conditions are stated so that a branch can be
closed by evidence rather than by attrition.

```
START ──┬── [1] greedy (temperature=0.0)          -> decode rate, pass rate, exact comparability
        ├── [2] top_k=20                          -> is the win the sort, or the sampling path?
        └── [3] 67 ms micro-benchmark             -> price of a round-trip

        (in parallel, from the start)
        └── [9] concurrent evals                  -> gated on [1] only; changes the cost of all below
```

**Branch A resolves at this point.**

- Decode rate improves materially under [1] or [2] → the sampler was the cost. Class A is **done**;
  apply the roofline arithmetic above to confirm nothing large remains, and do not open Tier 4.
- Decode rate barely moves and the roofline says 8.17 ms is near the bandwidth floor → the workload
  is memory-bound, **no engine change helps**, and Tier 4's ff-token work is the only remaining
  lever — which is a class-B lever in disguise (it removes tokens; it does not speed up decode).
- Pass rate drops under greedy → keep `top_k=20`, accept stochastic sampling, and note that the
  eval-comparability benefit is lost, which raises the cost of every experiment below.

**[3] gates the ordering within class B.** If the round-trip is expensive (67 ms confirmed or
worse), the ideas that delete *requests* — 8, 11 — outrank the ideas that delete *tokens*. If it is
cheap (the fit's intercept was an artefact), the reverse, and idea 11's value collapses to almost
nothing since it saves only ~10 tokens once ideas 6 and 7 have run.

```
Then, in order, each behind its own eval run:

    [5] one-token terminal DONE (8B)
         ├─ pass rate holds, ~-16%      -> keep; consider 8A only if a compound scenario exists
         └─ pass rate drops             -> revert; the narration round is load-bearing, record why
              ↓
    [6] handles (6)
         ├─ pass rate holds, ~-26%      -> keep; _write_is_grounded becomes structural, simplify it
         └─ pass rate drops             -> revert; the indirection is too much for a 4B model,
                                           which also kills the closed-alternation half of idea 12
              ↓
    [7] location tree (9)
         ├─ pass rate holds             -> keep; this is also the go/no-go for idea 10
         └─ pass rate drops             -> KILLS idea 10 outright, no further test needed
              ↓
    [8] fused router (11)
         ├─ pass rate holds             -> keep; must use the shared-prefix construction or it is
         │                                 worth nothing (see idea 11)
         └─ pass rate drops             -> revert; and note idea 10 becomes the substitute rather
                                           than the complement it is elsewhere
```

**Only after all of the above:** ff-tokens (14), MTP (15) and its prerequisite (17), speculative
decoding (16), and the constrained-decoding internals of Tier 3. Every one of them is upstream
work or has an upstream dependency, and none should start before the local questions are
answered — if class A resolves as "memory-bound, nothing to do" and class B lands its four
experiments, the pipeline is at roughly 400–500 ms per ask and the case for opening a PR against
mistral.rs is a different, better-informed conversation than it is today.

**Two fixture prerequisites, unchanged from the risk register above**, both of which sit *before*
the ideas they gate rather than after: a compound-request scenario before idea 8A (not 8B, which
is why 8B is the one in the tree), and a larger product fixture before idea 10.

## What this addendum does not change

The analysis above stands as written — the measurements, the source verifications, the
projections, the risk register, and the per-idea sections. What changes is only the ordering and
the framing: **"Ranked next steps" is the research backlog, and this section is the execution
plan.** Where they disagree, this section wins. Where the two agree — idea 1 first, ideas 15/16
last, idea 10 gated on idea 9 — the agreement is worth noting, because it means the disagreement
is narrow and about sequencing rather than about substance.

---

## Current State

- `docs/journal/2026-09-06-ask-latency-round-two.md` records twenty proposals against `sumac ask`
  latency, none of which is implemented — the commit carrying the entry changes no file outside
  `docs/journal/`.
- The per-request budget re-segmented by request role reads 598 s of engine time across the 440
  scenarios of `runs/epochs/verify-qwen3.5-4b-default-20/`, at 3.58 requests per scenario: write
  calls 43.7 %, first `sumac_find_inventory` 27.0 %, terminal plain-text reply 20.7 %, classifier
  5.5 %, second `sumac_find_inventory` 3.1 %.
- The 598 s figure counts only each scenario's first `_run_loop` segment and prices the classifier
  at one completion token — two adjustments that model the gated `_maybe_self_review`
  (`llm.py:1530-1556`) and the `_CLASSIFY_GRAMMAR` constraint (`llm.py:190`) shipped since
  `runs/epochs/verify-qwen3.5-4b-default-20/` was recorded.
- Tool-call messages in `runs/epochs/verify-qwen3.5-4b-default-20/` carry 84.0 fixed characters
  against 12.1 variable for `sumac_find_inventory`, 133.0 against 32.6 for
  `sumac_consume_inventory`, 132.0 against 39.3 for `sumac_discover_inventory`, and 147.0 against
  33.5 for `sumac_move_inventory` — 77 % to 87 % of each call is determined by the schema rather
  than by the request.
- Write-call messages in the same run decode at 1.98 characters per token against 3.26 for
  `sumac_find_inventory` calls — location values in write calls average 16.5 characters across 400
  calls.
- 0 of the 800 tool-call assistant messages in `runs/epochs/verify-qwen3.5-4b-default-20/` carry
  text before `<tool_call>`.
- The first domain tool call is `sumac_find_inventory` in 380 of the 380 non-`reject` scenarios of
  `runs/epochs/verify-qwen3.5-4b-default-20/`.
- `_build_request` sends `temperature`, `top_p` and no `top_k` (`llm.py:1302-1348`), with
  `DEFAULT_TEMPERATURE = 0.2` and `DEFAULT_TOP_P = 0.95` (`llm.py:163-164`).
- `mistralrs-core/src/sampler.rs` at `v0.9.2` sets the partial-sort bound to the full vocabulary
  length when `top_k` is not positive, and takes `sample_argmax` only when temperature is `None` or
  below `1e-7`.
- `_LocalMistralRsBackend.send_chat_completion_request` constructs
  `mistralrs.ChatCompletionRequest` with ten named fields (`llm.py:770-782`), omitting the
  `top_k`, `min_p`, `stop_seqs`, `logit_bias` and penalty fields `mistralrs-pyo3/mistralrs.pyi`
  declares at `v0.9.2`.
- `_build_runner` constructs `mistralrs.Runner` with `which` and `seed` (`llm.py:810`), leaving
  `max_seqs`, `prefix_cache_n`, `no_paged_attn`, `paged_attn`, `mtp_model` and the `pa_*` family at
  their defaults.
- `mistralrs-pyo3/src/lib.rs` at `v0.9.2` maps `grammar_type` to `Constraint::Regex`,
  `Constraint::Lark`, `Constraint::JsonSchema` and `Constraint::Llguidance`, and parses
  `tool_schemas` into `NormalRequest` independently of the constraint — a single request carries
  both.
- `mistralrs-core/src/pipeline/sampling.rs` at `v0.9.2` calls `compute_mask_or_eos` and
  `consume_token` on the `llguidance::Matcher` held by `SequenceRecognizer`
  (`mistralrs-core/src/sequence.rs`), and calls none of `compute_ff_tokens`, `consume_ff_tokens` or
  `compute_ff_bytes`.
- The entry's addendum orders the twenty proposals as greedy sampling, `top_k=20`, the 67 ms
  micro-benchmark, the one-token terminal decision, handles, the location tree, the fused router,
  and upstream work last, with concurrent evals running alongside from the start.
- The addendum records the `llama.cpp` decode-rate comparison as struck by decision, benchmark-only
  runs included, and names a spec-sheet bandwidth roofline as the substitute that answers the same
  question without a second inference engine.

## Stubbed

- None found — the entry adds no code.

## Missing

- No measurement exists for the effect of any of the twenty proposals; every figure in the entry's
  projection tables is arithmetic over the previous entry's 67 ms-per-request and
  8.17 ms-per-completion-token fit.
- No tokenizer was loaded to produce the entry — the 77 %-to-87 % scaffold split is by character,
  and the token counts derived from it are estimated from measured chars-per-token density.
- No `evals/` run exists at `temperature=0.0` or with `top_k` set, so the decode-rate effect of
  `mistralrs-core/src/sampler.rs`'s full-vocabulary sort is unquantified.
- No micro-benchmark of the 67 ms fixed per-request cost has been run, and the six candidate causes
  the entry lists are undistinguished.
- `evals/` contains no compound-request scenario, so a hard-stop terminal reply that drops a second
  write in one request passes the suite unchanged.
- `evals/fixtures.py` seeds 9 products across 25 locations, which does not distinguish a model
  reading a full product catalog in-prompt from a model reading a 9-line one.
- No tensor index has been read from `unsloth/Qwen3.5-4B-GGUF`'s `Qwen3.5-4B-Q4_K_M.gguf`, so
  whether the file carries `mtp.*` tensors is unrecorded.
- No issue or pull request has been opened against `EricLBuehler/mistral.rs` for fast-forward token
  support in the constrained-decode path.
- `evals/conftest.py` runs `pytest.mark.model` scenarios one at a time (`_shuffle_model_scenarios`,
  `evals/conftest.py:66-101`); no code runs scenarios concurrently against one `mistralrs.Runner`.
- No resident-process or socket entry point exists for `sumac ask`; `shared_runner`
  (`llm.py:826-854`) reuses a backend within one process only.

## Divergence

- None found against `README.md`, which documents `sumac ask --loop`'s single model load and the
  `ask-cuda` wheel build without making claims about sampling configuration, per-request latency,
  or tokens per second.
