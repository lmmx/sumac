# sumac: Modal Remote Inference Backend — Spec, Not Implemented

**Status:** specification only, per explicit request — no code in this entry, meant to be handed
to a future implementing session as a planning doc. Follows directly from this session's
`add-amount-delta` `PromptVariant` work (`docs/journal/2026-09-04-basmati-rice-unit-mismatch.md`'s
sibling failure, `add.discriminator_variant_not_confused`) and hands off a second, independent
problem the prompt-iteration loop surfaced: local `mistralrs` epochs are too slow to iterate
against.

**Revision history:** first drafted from a pasted ChatGPT engineering brief, reviewed against the
actual codebase. Substantially revised after a second, much deeper review (pasted from Opus) that
found real fidelity risks the first draft missed entirely — most of what follows is that second
pass, filtered against the actual `mistralrs` package installed in this repo's own environment
and the actual `src/sumac/llm.py` call sites, not taken on faith. Sections marked **(verified this
session)** were checked directly against `mistralrs==0.9.2`'s runtime behavior or `llm.py`'s real
code; sections marked **(unverified, from review)** are carried over from the pasted review as-is
because this session had no way to check them (no Modal account access, no vLLM instance) — treat
those as strong leads to confirm, not settled facts.

## Motivation, with real numbers from this session

Getting a trustworthy read on whether `add-amount-delta` actually helps required stepping past 5
epochs — this session's own `qwen3.5-9b` disagreement (`existing_item_explicit_location` reading
5/5 in one baseline run and 3/5 in a supposedly-identical rerun, no code difference) showed 5
epochs sits inside the noise floor, not above it. The user's own follow-up run was
`scripts/eval-prompt-variant.sh qwen3.5-4b add-amount-delta 20` — 20 full-suite epochs, one
`qwen3.5-4b` model.

That run's own cost is measurable from this session's epoch files:
`runs/epochs/verify-qwen3.5-4b-default/epoch-01.json`'s `total_duration_s` is 40.7s for the
22-scenario suite (warm-cache model load is ~2.7s of that, per
`docs/journal/2026-09-04-trace-and-verdict-redesign.md`'s own A-vs-B measurement — negligible
against generation time). 20 epochs × ~42s ≈ **14 minutes wall-clock for one prompt-wording
guess on the fastest of the two registered models** — the "~15 min run time... for this one
simple 4b query" the user flagged. Multiply that by however many wording iterations a real
prompt-fix cycle takes, and it's the actual bottleneck, not model quality.

## What already exists — don't build the swap seam, it's already there

`AgentRunner` is already backend-agnostic. `src/sumac/llm.py:578-585` declares:

```python
class SendsCompletions(Protocol):
    def send_chat_completion_request(
        self, request: mistralrs.ChatCompletionRequest, model_id: str | None = None
    ) -> mistralrs.ChatCompletionResponse: ...
```

and `AgentRunner.__init__` (`llm.py:707-756`) already takes `runner: SendsCompletions | None`,
falling back to a real `mistralrs.Runner` only when `runner` is `None`. `tests/test_llm.py`'s
`FakeRunner` already proves the seam works end to end with no real model. **This task is "write
a second `SendsCompletions` implementation and plug it in somewhere," not "make `AgentRunner`
swappable" — that part is already done and already tested.**

The one production call site that needs a new branch is `evals/conftest.py:195`'s
`agent_runner_factory` fixture:

```python
base_runner = llm._build_runner(model, seed=seed_value)
```

This is session-scoped (`conftest.py:169`, one load per pytest session, matching the
"one epoch = one process" design `docs/journal/2026-09-02-eval-suite.md` already established)
and is the only place a real backend gets constructed for the eval suite. `cli.py`'s interactive
`sumac ask` path is a separate call site — out of scope here; the ask is eval-iteration speed,
not an interactive Modal-backed assistant.

## The `SendsCompletions` typing fork — resolved, not just flagged **(verified this session)**

The first draft of this entry raised, as an open question, whether a Modal backend could read
fields off the real `mistralrs.ChatCompletionRequest` object `_build_request` constructs
(`llm.py:996-1008`). That question has an answer, checked directly against the `mistralrs==0.9.2`
wheel installed in this repo's own dependency groups:

```
$ uv run python -c "
import mistralrs as m
r = m.ChatCompletionRequest(model='x', messages=[{'role':'user','content':'hi'}], max_tokens=1,
                             temperature=0.2, top_p=0.95, tool_schemas=['{}'],
                             tool_choice=m.ToolChoice.Auto, enable_thinking=False)
print(type(r))
print([a for a in dir(r) if not a.startswith('_')])
print('messages:', getattr(r, 'messages', 'NO GETTER'))
"
<class 'builtins.ChatCompletionRequest'>
[]
messages: NO GETTER
```

`mistralrs/__init__.pyi` (the type stub, what `ty`/IDEs read) lists `ChatCompletionRequest` as a
`@dataclass` with 30-odd named, defaulted fields (`messages`, `temperature`, `top_p`, `top_k`,
`tool_choice`, `enable_thinking`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`,
`min_p`, `dry_*`, ...) — but the *runtime* class is a PyO3 extension type
(`<class 'builtins.ChatCompletionRequest'>`, not `sumac`'s or Python's own) with **zero readable
attributes and no dataclass machinery** (`dataclasses.is_dataclass(r)` is `False`). The `.pyi`
describes the intended construction shape for type-checking; it does not mean the object is
introspectable once built. **Reading fields off a constructed `ChatCompletionRequest` is not
possible — confirmed, not "likely dead."**

The response side turned out to have its own, different wrinkle, also checked directly:

```
$ uv run python -c "
import mistralrs as m
for name in ('ChatCompletionResponse','Choice','ResponseMessage','Usage'):
    try:
        getattr(m, name).__new__(getattr(m, name))
        print(name, 'constructed')
    except Exception as e:
        print(name, 'FAILED:', e)
"
ChatCompletionResponse FAILED: No constructor defined for ChatCompletionResponse
Choice FAILED: No constructor defined for Choice
ResponseMessage FAILED: No constructor defined for ResponseMessage
Usage FAILED: No constructor defined for Usage
```

These types can only ever be produced by the Rust engine itself — a Modal backend can never
return a real `mistralrs.ChatCompletionResponse` even if it wanted to. This isn't a problem: the
existing production code already only ever *reads* these (never constructs them —
`_record_usage`/`_round_preview`/`_run_loop` all just call attribute access on whatever came back
from `send_chat_completion_request`), and `SendsCompletions` is a `Protocol`, which Python never
enforces with `isinstance` unless something explicitly asks it to (nothing here does) — so a
plain local dataclass with matching attribute names satisfies every real call site today, and
would satisfy `ty` too once the Protocol's declared return type is loosened to match.

**Resulting design — smaller and more mechanical than either fork first considered:**

1. **Request side:** change `_build_request` (`llm.py:996-1008`) to return a plain
   `dict`/small local dataclass of the same kwargs, instead of constructing
   `mistralrs.ChatCompletionRequest` directly. The *local* `mistralrs`-backed `SendsCompletions`
   implementation becomes the one place that does
   `mistralrs.ChatCompletionRequest(**request_kwargs)`, immediately before calling the engine.
   The Modal-backed implementation instead serializes the same dict to an OpenAI-style JSON body.
   Both backends provably see the identical kwargs — a unit test can assert the dict is unchanged
   field-for-field without touching either engine.
2. **Response side:** narrow `SendsCompletions`'s declared return type from
   `mistralrs.ChatCompletionResponse` to a small local `Protocol`/dataclass covering exactly
   what `llm.py` reads today (enumerated below) — a Modal backend constructs an instance of it;
   the real `mistralrs.Runner` already satisfies it structurally with zero wrapping, since its
   real responses already expose exactly this shape.

This is the one genuine, explicit `llm.py` diff this task requires outside pure addition —
call it out to the user rather than absorbing it silently, per the original brief's own "stop
and explain" instruction, but it's now a concrete, scoped, two-part change rather than an open
question.

### The exact minimal response shape needed **(verified this session, grep'd from `llm.py`)**

Every attribute `llm.py` actually reads off a response, across `_classify` (`llm.py:1037-1074`),
`_run_loop` (`llm.py:1076-...`), `_print_usage`/`_record_usage` (`llm.py:662-675`,
`1017-1035`) — nothing else is touched:

```
response.usage                                   # optional; None already tolerated today
    .prompt_tokens: int
    .completion_tokens: int
    .avg_compl_tok_per_sec: float
    .total_time_sec: float
response.choices[0].message
    .content: str | None
    .tool_calls: list[...] | None
        [0].function.name: str
        [0].function.arguments: str               # a JSON string, parsed with json.loads
```

That's the entire contract. Nothing else — no `finish_reason`, `logprobs`, `id`,
`system_fingerprint`, `reasoning_content` — is read anywhere in `llm.py`. Keep the local response
shape this small; don't pad it out to match the full `mistralrs.ChatCompletionResponse`/OpenAI
shape "for completeness."

**No knock-on change needed in `render.py`** — checked directly:
`print_agent_request`/`print_agent_response`/`print_agent_message`/`print_agent_content`/
`print_agent_tool_calls` (`render.py:316-336`) all take a bare `object` parameter and pretty-print
it with `rich.Pretty`, agnostic to the actual type. Debug mode (`AgentRunner(debug=True)`) keeps
working unchanged regardless of which backend or request/response shape is live.

## Where backend selection plugs in

Precedent already exists for an envvar-backed CLI option:
`cli.py:53`'s `typer.Option("--data-dir", envvar="SUMAC_DATA_DIR", ...)`. The natural mirror is
a `--eval-backend`/`SUMAC_LLM`-style option added to `evals/conftest.py:26-...`'s
`pytest_addoption` alongside `--eval-model`/`--eval-prompt-variant`, read at
`conftest.py:195`'s one call site to choose `llm._build_runner(...)` vs. a new
`llm._build_modal_runner(...)`-shaped constructor. `--eval-model qwen3.5-4b` should keep meaning
"which `ModelPreset`" regardless of backend — for Modal that likely resolves to "ask the Modal
deployment for its Qwen3.5-4B endpoint," not a second, parallel model-name vocabulary.

**Local `mistralrs` stays the default in every case; Modal is opt-in.** Matches this repo's own
established pattern of prior variants (`nudge-v2/v3/v4`, now removed) staying non-default until
proven, and the fact that the entire premise of this backend is "faster, not authoritative."

## Usage accounting will degrade quietly across backends **(unverified, from review — plausible and checked against the mistralrs stub)**

`mistralrs.Usage` (`__init__.pyi:1185-1194`) carries `completion_tokens`, `prompt_tokens`,
`total_tokens`, `avg_tok_per_sec`, `avg_prompt_tok_per_sec`, `avg_compl_tok_per_sec`,
`total_time_sec`, `total_prompt_time_sec`, `total_completion_time_sec` — an OpenAI-compatible
serving stack's usage object typically returns only the three token counts and nothing else
(unverified against a live vLLM instance, but consistent with the OpenAI `usage` schema this
class of server usually mirrors). `_record_usage` (`llm.py:1017-1035`) already tolerates
`usage is None` outright — that path exists today for `FakeRunner` in tests — but it does **not**
tolerate `usage` being present with `avg_compl_tok_per_sec` missing; `_print_usage` (`llm.py:662-675`)
would raise `AttributeError` on that field specifically.

Decide explicitly, don't let it surface as a crash mid-eval: either the Modal adapter populates a
`Usage`-shaped object with `None`/computed-from-wall-clock timing fields (`total_time_sec` is
knowable from the HTTP round-trip even if the server doesn't report it), or `_print_usage`/
`_record_usage` become attribute-tolerant (`getattr(usage, "avg_compl_tok_per_sec", None)`).
**Never compare `tok/s` across backends either way** — mistral.rs's numbers are engine-internal
generation time; a Modal adapter's equivalent would be wall-clock including HTTP round-trip and
(when cold) container scheduling, not a comparable quantity.

## `tool_choice` — the fidelity risk is real in general, but currently latent in this codebase **(verified this session)**

The pasted review's warning is right in general: an OpenAI-compatible server implementing
`tool_choice: "required"` or a named-function choice typically does it via constrained decoding —
masking logits during sampling, which is a materially different sampling distribution from "same
prompt, hope it calls a tool." If sumac ever set that, "same temperature, same prompt" would stop
implying "comparable behavior" on exactly the turns that matter most.

**Checked against the actual code: this risk doesn't bite today.** `_build_request` (`llm.py:1003`)
hardcodes `tool_choice=mistralrs.ToolChoice.Auto` at its one call site — there is no code path in
`llm.py` that ever requests anything else. `mistralrs.ToolChoice` (`__init__.pyi:73-75`) itself
only has two variants, `Auto` and `NoTools` — not the richer OpenAI vocabulary (`"none"`/
`"auto"`/`"required"`/named-function). `_maybe_force_action` (`llm.py:1141`) — the mechanism that
"forces" a retry after an empty plan — does it purely by appending `_EMPTY_PLAN_NUDGE` text to the
conversation and calling `_run_loop()` again; it never touches `tool_choice`. So there is
currently exactly one `tool_choice` value to map (`Auto` → OpenAI `"auto"`), not a table.

Still worth doing cheaply: record the resolved `tool_choice` value per turn in the trace (one
field, trivial given the request is already being restructured into a plain dict/dataclass per
the section above) so this stays visible if a future change ever does start varying it, rather
than becoming another thing that's only discoverable by reading `llm.py`'s source after the fact.

## The full sampling vector must be pinned explicitly, not assumed **(partially verified this session)**

Checked directly against `_build_request` (`llm.py:996-1008`): sumac's own request sets
`temperature`, `top_p`, `max_tokens`, `enable_thinking=False`, `tool_choice=Auto`, and
`tool_schemas`/`messages`/`model` — **nothing else**. Every other field
`mistralrs.ChatCompletionRequest` accepts (`top_k`, `min_p`, `presence_penalty`,
`frequency_penalty`, `repetition_penalty`, `logit_bias`, `stop_seqs`, the `dry_*` fields) is left
at the Python-level default of `None`, meaning **mistral.rs's own Rust-side default applies, and
what that default actually is isn't visible from the Python stub** — worth checking (or logging
what mistral.rs actually samples with) before assuming "unset" means "off" the way it typically
does for an OpenAI-compatible server. The pasted review names `repeat_last_n` as another knob
mistral.rs exposes "directly in its `Which.GGUF` config" — not found in this installed 0.9.2
stub's `Which.GGUF` fields (`__init__.pyi:555-570`); either a different mistral.rs version, a
Rust-CLI-only flag not surfaced to Python, or a misremembered name. Don't take it on faith;
grep the actual installed stub for whatever version ships when this is implemented.

Instruction for the implementing session either way: **enumerate the full sampling vector,
set every field explicitly on both backends (not just the three sumac currently pins), and write
the resolved vector into `--eval-json`.** Unset is not a value — it's a silent, per-backend
variable.

Worth knowing sumac is already off whatever a model card recommends, deliberately: Qwen's own
non-thinking guidance for the Qwen3.5 family is commonly temperature 0.7 / top_p 0.8 /
presence_penalty 1.5 / top_k 20 **(unverified, from review — not checked against Qwen's actual
model card this session)**. Sumac's 0.2/0.95 is a deliberate, lower-variance pin appropriate for a
tool-calling eval, already shipped — the point isn't to match the model card, it's to apply
whatever's pinned identically on both backends rather than accidentally inheriting whichever
engine's own defaults differ from sumac's explicit choices.

## `enable_thinking` — mechanics differ per stack, and there's a specific interaction to test for **(unverified, from review)**

`_build_request` already always sends `enable_thinking=False` (`llm.py:1004`) — the mistral.rs
Python binding takes it as a first-class request field. An OpenAI-compatible HTTP server
typically expects the equivalent as a chat-template kwarg buried in `extra_body`
(`{"chat_template_kwargs": {"enable_thinking": false}}`), not a top-level sampling param — that
distinction has to survive the Modal adapter's translation intact, or thinking silently turns
back on.

The review flags a specific, named failure mode worth testing for directly rather than trusting
config: with a reasoning parser enabled alongside a tool-call parser on vLLM, Qwen3.5 has been
reported to return tool calls as XML text inside `content` instead of the structured `tool_calls`
array when thinking is *on*; disabling thinking has been reported to fix it, switching the tool
parser alone has not **(unverified — a specific bug report cited in review, not independently
confirmed here)**. Two concrete, cheap mitigations either way: don't enable a reasoning parser at
all if thinking is meant to be off, and make the smoke test (below) assert `content` contains no
`<think>` block and no XML-shaped tool-call fragment — a parse-time assertion, not just a "did a
request come back" check.

## Tool-call parser choice is a silent-failure fork, and the gate that catches it **(unverified serving-stack specifics, from review; the local-side confirmation mechanism below is grounded in verified code)**

An OpenAI-compatible server serving Qwen with tool-calling needs to be told which parser
understands Qwen's tool-call format (e.g. vLLM's `--enable-auto-tool-choice
--tool-call-parser <name>`) — the review cites reports of the wrong parser choice for Qwen3.5
(`hermes` vs `qwen3_xml` vs `qwen3_coder`) producing silently empty `tool_calls`, with the model's
actual tool call landing as unparsed text in `content` instead. **In this harness that reads as
"the model didn't call the tool" — indistinguishable from a real prompt regression** unless
something checks for it explicitly.

Two concrete gates, cheap and specific to what's actually being built here:

1. **Deploy-time gate.** Before any eval run is allowed to start, send one fixed, canonical
   request (e.g. the same shape as a real `sumac_find_inventory` call) through the Modal
   endpoint and assert the response comes back with a non-empty `tool_calls` array whose
   `arguments` field parses as JSON. Fail loudly and refuse to run evals if it doesn't — never
   let a misconfigured parser present as "a prompt change hurt every ADD scenario."
2. **Rendered-prompt diff, before trusting any comparison.** The review's suggested mechanism —
   `mistralrs.Runner.tokenize_text` on one side, an OpenAI server's `/tokenize` on the other,
   diff the token IDs for one fixed conversation — needs a correction: `tokenize_text`
   (`__init__.pyi:909-924`) tokenizes a raw string, it does **not** render a chat template from a
   messages list (its own docstring: "raw text tokenization does not render a chat template").
   So this specific API doesn't do what the review's script sketch implies on its own. What's
   actually needed: independently obtain the *fully chat-template-rendered* prompt string on the
   mistral.rs side (the GGUF's embedded `tokenizer.chat_template` metadata, rendered locally with
   Jinja2 against the same messages/tools — mistral.rs's Rust CLI may log this in verbose mode,
   worth checking before hand-rolling it) and compare that against whatever the OpenAI-compatible
   server's own template renders for the identical input, ideally at the token-ID level via
   `tokenize_text`/`/tokenize` once both rendered strings are in hand. **The underlying point
   stands and is the single highest-value pre-flight check for this whole effort** — the local
   GGUF's embedded chat template and the upstream HF repo's `tokenizer_config.json` template are
   different files with no guarantee of agreement, and if the two backends render a different
   prompt string for the same conversation, no amount of sampling-parameter matching fixes it —
   just don't assume the exact API calls sketched above are correct without checking the
   installed mistral.rs version's actual capabilities first.

## Determinism — the local baseline itself is the bigger open question **(review's reframe kept; the observation it's reframing is this session's own verified finding)**

The originally-drafted version of this entry said "don't pretend Modal has identical RNG to local
`mistralrs`" as a one-way caution. The review points out this likely has the framing backwards.
vLLM ships a batch-invariance mode (`VLLM_BATCH_INVARIANT=1`, paired with a fixed per-request
`seed`) specifically to make output independent of batch composition and request order —
explicitly motivated by the same class of problem this session hit **(unverified specifics —
exact flag name/requirements not checked against a live vLLM instance this session, but the
motivation matches exactly)**. mistral.rs schedules and batches requests too, and this session's
own `existing_item_explicit_location` flip (5/5 → 3/5, same nominal seed, no code difference
between the two runs) has exactly the shape of batch-composition nondeterminism, not RNG-stream
drift from a code change (the RNG-cascade mechanism `docs/journal/2026-09-04-trace-and-verdict-redesign.md`
already diagnosed and fixed via per-epoch shuffling is a different, already-understood effect —
this is a second, distinct wobble on top of it that was flagged but never chased down).

**Reframe worth writing into the README once this ships:** the honest framing isn't "Modal won't
match local's RNG," it's "the remote backend may turn out to be *more* reproducible than the
local one, if it runs with batch invariance on — a Modal-vs-local divergence is at least as
likely to be pre-existing local nondeterminism as anything Modal introduces." If diagnosing a
divergence is ever the actual goal (not just speed), running the Modal deployment with batch
invariance on is a legitimate debugging configuration, trading throughput for eliminating one
whole noise source. Cheaper 80% version if the batch-invariant kernels aren't worth the
performance cost for routine use: run one epoch at concurrency 1, which makes batch composition
trivially constant without needing the dedicated kernel mode at all.

## Concurrency is the actual 14-minute unlock, and it directly fights the determinism fix above

22 scenarios sequentially, ~40s total, is the current shape. A warm remote endpoint serving a 4B
model can likely handle many concurrent requests at near-flat per-request latency — the real win
here is running epochs *concurrently against one warm container*, not a faster single-stream
decode. But `evals/conftest.py`'s harness is one-pytest-session-per-epoch, sequential within an
epoch, specifically *because* of the RNG-cascade reasoning in
`docs/journal/2026-09-04-trace-and-verdict-redesign.md` — a naive Modal port that keeps that
shape only gets whatever raw per-request latency improvement the remote GPU provides (the review
estimates 2-3×, likely disappointing relative to the premise of "rip through evals faster"), and
running multiple epochs *concurrently* is exactly what reintroduces the batch-composition
nondeterminism the determinism section above is trying to eliminate.

**Don't resolve this by picking one mode — build two, and record which produced each epoch
file:**

- **`fast`** — high concurrency (many epochs as concurrent clients against one warm container),
  batch invariance off. For "does this prompt change do anything at all" — a coarse filter,
  explicitly not meant to be directly comparable epoch-for-epoch to a local run.
- **`comparable`** — concurrency 1, batch invariance on (or the concurrency-1 approximation of
  it), fixed per-request seed. For "is this real" — slower, closer in spirit to what the local
  benchmark already guarantees.

Measure single-request latency and aggregate throughput at concurrency 1/4/16 before deciding
where `fast` mode's default concurrency should sit, rather than guessing. **(unverified, from
review)** Modal's own examples repository reportedly ships a load-testing script for
OpenAI-compatible endpoints (`openai_compatible/load_test.py`, locust-based,
`modal run openai_compatible/load_test.py`) — worth using that rather than hand-rolling one, if
it exists and fits, but confirm it's still current before relying on it.

## Modal deployment specifics **(unverified, from review — not checked against Modal's own examples repo this session, no Modal account access)**

Carried over because they're specific, plausible, and exactly the kind of thing that turns into a
multi-minute cold-start surprise if skipped — verify each against Modal's current docs/examples at
implementation time rather than trusting this list blind:

- Start from Modal's own `llm-serving`/vLLM example rather than hand-writing a FastAPI wrapper, if
  one still exists in their examples repo in a similar shape.
- **Two volumes, not one**: cache HF weights (`/root/.cache/huggingface`) and the serving
  engine's own JIT-compiled kernel cache (e.g. `/root/.cache/vllm`) separately — skipping the
  second is reported as a common cause of multi-minute cold starts even with weights already
  cached.
- **Boot vs. throughput tradeoff**: an "eager"/no-compile mode boots in seconds but generates
  slower; full compilation costs a slow first boot (reported: tens of seconds to a few minutes,
  faster once its own cache is warm) in exchange for better steady-state throughput. For a session
  firing many hundreds of eval requests, the slow-boot/fast-throughput side is very likely the
  right trade — confirm against whatever the actual measured boot-cache-hit time turns out to be.
- **Pin the model revision** to a specific HF commit hash, not a floating repo reference — the
  remote equivalent of `ModelPreset` already pinning an exact GGUF filename
  (`docs/journal/2026-09-04-trace-and-verdict-redesign.md`'s note on this). An unpinned upstream
  repo updating mid-experiment silently changes the benchmark model.
- **Cold-start floor management**: a scale-to-zero container takes real time to come back;
  something has to keep at least one instance warm for the duration of an iteration session (a
  minimum-container floor, toggleable without a full redeploy) and something has to release that
  floor afterward so an idle GPU isn't paid for indefinitely. A small `scripts/modal-warm.sh
  on|off` wrapper around whatever Modal's current autoscaler-override mechanism is would cover
  this.
- **Handle request-against-cold-container failures explicitly**: a request against a
  scaled-to-zero deployment is expected to fail or queue, not silently hang — whatever client code
  calls the endpoint needs an explicit "wait for it to come up" loop (poll a health endpoint,
  treat "not ready yet" distinctly from a real failure) rather than treating the first request
  after idle as a scenario failure.
- **GPU sizing**: a 4B model in bf16 is on the order of 8GB of weights — a small/mid-tier GPU is
  almost certainly enough; don't reach for whatever the example template happens to default to
  without checking it's proportionate to a 4B model specifically.

## Quantization parity — the direction of the bias matters more than its existence

Local is a quantized GGUF under mistral.rs; a Modal/vLLM deployment would most naturally serve
bf16, fp8, or an AWQ/GPTQ quantization — not the identical GGUF quant level (vLLM's GGUF support,
per the review, is second-class and slow — probably not worth reaching for on a
speed-motivated backend) **(unverified — vLLM's current GGUF support maturity not checked this
session)**. The original brief's framing — "if the exact quantization can't be reproduced,
document it, that's acceptable" — undersells the actual risk.

At 4B, quantization plausibly affects tool-call *formatting* reliability specifically (a
lower-precision model dropping a brace or malforming JSON args is a known failure shape) more than
it affects general reasoning quality. If that holds, **the remote model is likely to be *better*
at tool-calling than the local one** — which makes Modal a generous proxy, the wrong direction for
a fast filter. A fast filter should be pessimistic (anything that passes remotely should also
pass locally); as specified, this setup could instead produce prompts that score well remotely and
regress locally, which actively trains distrust of the fast loop rather than earning it.

**Sharpen this against the actual live variant**: `add-amount-delta`
(`src/sumac/llm.py`'s `PROMPT_VARIANTS`) is specifically about the model getting the `amount`
*argument value* right — not whether it calls the right tool, whether it correctly computes a
delta and passes that number as a string argument. That's argument-formatting-adjacent, plausibly
one of the exact axes quantization perturbs most. **The local/remote gap for this specific
experiment could be largest on precisely the thing being measured.**

Pick one response deliberately, not by default: **(i)** accept the generous-proxy bias and use
Modal only as a coarse "did this obviously break something" filter, never for reading a small
(1-2 scenario) delta — the way the review frames it, appropriate for `fast` mode above; or
**(ii)** find a remote quantization closer to the local GGUF's precision (int4/AWQ) to narrow the
gap, accepting whatever serving-stack cost that carries. Don't silently assume (i) works for
every future use of this backend just because it's cheaper to set up.

## Statistical hygiene — worth more than the backend itself

The whole point of this backend is buying more epochs. Make sure the epochs actually buy
inference, not just more numbers to eyeball.

This session's own 5/5-vs-3/5 flip isn't evidence of anything on its own: treated as a 2×2, a
Fisher's exact test on that specific pair comes out non-significant (consistent with one
underlying ~80% pass rate and pure noise, not a real change) **(unverified — the exact p-value
cited in review, ≈0.44, not independently recomputed this session, but the qualitative point —
n=5 has essentially no power to distinguish this — is uncontroversial)**. With 22 scenarios per
epoch, *some* scenario disagreeing between two runs is the expected outcome at small epoch counts,
not an anomaly worth chasing every time — eyeballing `epoch_report.py`'s per-scenario table for
"did anything change" is a multiple-comparisons trap that will manufacture an apparent regression
on nearly every run.

Four concrete, cheap additions to `evals/epoch_report.py` (`_print_report`,
`epoch_report.py:40-117`) that would pay for themselves faster than the Modal backend does:

1. **Wilson confidence intervals per scenario**, not bare `p/N` counts. A 5-epoch 5/5 has a wide
   interval (roughly 0.57-1.0 at 95% confidence) — printing that instead of a bare fraction kills
   the temptation to read "5/5" as "solid."
2. **Paired comparison**: run baseline and variant against the *same* seeds and (given the
   RNG-cascade shuffle is seed-keyed) the same resulting scenario order, then difference within
   pairs rather than comparing independent rate estimates — removes the shared-noise component
   for free, worth several × in effective sample size.
3. **Pre-register what's being compared** before a run starts: the suite-level mean, or one named
   scenario — not "look at all 22 rows afterward and see what stands out." Decide the question
   before seeing the answer.
4. **A stopping rule instead of a fixed epoch count**: run a smaller batch, stop early if the
   effect is already unambiguous, escalate to a much larger batch if it isn't — a fixed "20
   epochs" leaves the actual value of a cheap-epoch backend on the table if the real answer needs
   200, or wastes a coarse-filter run's time if 10 would have already been clear.

Fold this into the same `--eval-json` requirement already in this spec: extend it into a
**provenance block** — backend (local/modal), serving stack + version, model revision hash,
quantization, the full resolved sampling vector (per the section above), tool-call parser,
`enable_thinking` value, concurrency mode (`fast`/`comparable`), batch-invariance flag. This
backend doubles the number of ways two epoch files can differ from each other; if the file itself
doesn't say which axis produced a given result, the growing epoch corpus stops being comparable
across exactly the dimension just introduced. `epoch_report.py`'s existing `_group_label`
(`epoch_report.py:35-37`, currently `model` + `prompt_variant`) is the natural place this
provenance also needs to be visible in the printed report, not just the raw JSON.

## Two non-negotiable instructions the original brief didn't have at all

**Transport failures must never become scenario verdicts.** A remote backend introduces an entire
new failure class local `mistralrs` doesn't have — HTTP timeouts, connection resets, 503s from a
cold container, rate limiting. The sloppy default implementation catches whatever exception and
records the scenario as a failed verdict, which silently corrupts the eval (a transport blip
reads identically to "the model got the answer wrong"). Rule: **transport-layer errors raise and
abort the epoch; they are never caught and recorded as a model/verdict failure.** No blanket
retry-and-swallow wrappers. If retries exist at all, they must be explicit, bounded, and their
count recorded in the provenance block above — not silent.

**Record the raw pre-parse response body in the trace, not just the parsed tool call.**
`ToolCallRecord` (`llm.py:558-568`) today only carries `name`/`arguments`/`result` — the already-
parsed shape. When a tool-call parser mismatch (the section above) makes `tool_calls` come back
empty because the model's actual call landed as unparsed text, the raw response body *is* the
entire diagnosis — a parsed-only trace throws that evidence away before it can ever be inspected.
This needs either a new field on `ToolCallRecord` or a sibling record type capturing the raw HTTP
response alongside the parsed one, at least for the Modal path (local `mistralrs` has no
comparable "raw wire format" to capture in the same way, so this is asymmetric by necessity, not
an oversight).

## Architecture (kept from the original brief — it matches the actual code)

```text
Local eval process (AgentRunner, unchanged)
    │  system prompt, messages, tool schemas, tool results
    ▼
SendsCompletions (existing seam, llm.py:578)
    │
    ├── mistralrs.Runner ──────────► local GGUF, current default
    └── new Modal-backed impl ─────► HTTP ──► Modal GPU ──► Qwen3.5-4B
```

Modal must not own `sumac_find_inventory`/`sumac_discover_inventory`/`decide.py`/eval
scenarios/verdict logic/prompt variants/tool execution/inventory writes — all of that is
already local-only and inference-independent (`AgentRunner.tool_callbacks`, `llm.py:742-747`,
dispatches every tool client-side; `_run_loop`, `llm.py:1076`, is the one loop, already backend-
blind). Modal is inference only. An OpenAI-compatible HTTP endpoint is the right target *if* the
chosen serving stack supports it without extra weight — check current recommended serving options
for Qwen3.5-4B before picking a framework; don't add an `openai` client package (none of
`sealedlog`/`pydantic`/`rich`/`typer`/`mistralrs` in `pyproject.toml` pull one in today) merely
for familiarity if a plain HTTP call covers the actual request shape (the minimal shape enumerated
above).

## Scope boundaries

Do not modify `decide.py`, inventory semantics, evaluator expectations, scenario definitions,
prompt wording/variants, `_maybe_force_action`, self-review behavior, RNG architecture, or tool
semantics, beyond the one explicit, scoped `SendsCompletions`/`_build_request` change resolved
above. If anything else outside the backend/configuration layer turns out to be genuinely
required, stop and explain that specific diff rather than absorbing it silently.

**Modal replaces inference, not the agent.**

## Sequencing

This is infrastructure for the *next* round of prompt iteration, not a blocker on the current
`add-amount-delta` variant — that variant's verdict should still be reached (or left
inconclusive) on local `mistralrs`, per [[sumac-ship-one-testable-change-at-a-time]]. Local
`mistralrs` remains the authoritative benchmark permanently, not just until Modal is trusted — a
prompt that scores well on Modal (especially in `fast` mode) still needs a local confirming run
before being treated as a verified improvement, doubly so given the generous-proxy risk above.

## Missing / open threads, ranked by expected value

Roughly in the order a follow-up session (planning or implementing) would get the most out of
picking one up:

1. **The local-nondeterminism question is arguably the bigger prize, and predates Modal
   entirely.** If `mistralrs.Runner(seed=N)` isn't actually reproducible run-to-run (this
   session's `existing_item_explicit_location` flip suggests it might not be), the "authoritative"
   local benchmark has an unquantified noise floor underneath every verdict reached so far,
   including the basmati fix and the `add-amount-delta` variant itself. Clean experiment design:
   same seed, same scenario order, N repeats, measure the per-scenario flip rate with nothing else
   changed. Worth running before investing further in Modal-vs-local comparisons, since it bounds
   how much precision either backend can offer.
2. **The chat-template-diff gate.** The mechanism sketched above (compare mistral.rs's actual
   rendered prompt against the remote server's) needs its exact implementation worked out — what
   this session confirmed is that `tokenize_text` alone doesn't do it; what's still open is what
   does, on the current `mistralrs` version.
3. **The `epoch_report.py` statistics layer** (Wilson intervals, paired differencing,
   pre-registration, a stopping rule) — independently valuable even if Modal is never built, and
   arguably should happen regardless.
4. **The generous-proxy resolution** — whether a closer remote quantization is worth the added
   serving complexity, or whether `fast`-mode Modal should just be documented and used as a coarse
   filter only, never for small deltas.
5. **Concurrency architecture** — how `fast` mode actually runs N epochs as N concurrent clients
   without either abandoning the one-process-per-epoch RNG design
   `docs/journal/2026-09-04-trace-and-verdict-redesign.md` established, or quietly reintroducing
   the exact cross-scenario cascade that design was built to eliminate.
6. Whether `mistralrs.ChatCompletionRequest`'s currently-unset sampling fields (`top_k`, `min_p`,
   `presence_penalty`, `frequency_penalty`, `repetition_penalty`, ...) have Rust-side defaults
   that differ meaningfully from an OpenAI-compatible server's own defaults — not visible from the
   Python stub, would need checking mistral.rs's own source/docs or empirically logging sampled
   output differences.
7. Modal free-tier ceiling (compute-seconds/GPU-type limits) against this account's actual
   remaining credits — assumed adequate by the user, not independently verified.
8. Auth/endpoint config naming (`SUMAC_MODAL_ENDPOINT` or similar) and the exact new
   `evals/conftest.py` option name/shape — not decided; pick whichever reads closest to the
   existing `--eval-*`/`SUMAC_DATA_DIR` family.
9. Every item under "Modal deployment specifics" above is carried from the pasted review
   unverified against Modal's actual current docs/examples — re-confirm at implementation time,
   Modal's own recommended patterns change.
