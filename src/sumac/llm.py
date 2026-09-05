"""In-process agent for `sumac ask`: mistral.rs `Runner` wrapping the domain
layer's existing `ledger`/`decide` entry points.

Background on the mistral.rs SDK and the client-side tool-calling loop this
module uses instead of the server-side one is in
docs/journal/2026-09-01-ask-agent-design.md. The query-classifier design this
module currently implements — routing a request to a small, task-specific
prompt instead of one prompt covering every tool — is in
docs/journal/2026-09-02-query-classifier.md.

`AgentRunner.propose` first classifies the request into a `QueryKind`
(`FIND`/`ADD`/`REMOVE`/`REJECT`) with a small, single-purpose call, then runs
the agentic tool-calling loop against a system prompt and tool schema scoped
to that one kind — a `find`-classified request never sees the
consume/move/discover schemas, and vice versa. Every mutating tool callback
stops short of `store.append`; it calls `decide.decide_change` to validate
the resolved call and records it in `self._pending`, which is what makes
dry-run and preview-then-accept the same mechanism. `AgentRunner.commit` is
only reachable after a human has reviewed an `AgentPlan` and accepted it —
it re-decides each `ProposedWrite` against freshly reloaded state rather
than replaying what `propose`/`revise` resolved, since real time may have
passed since the preview was shown.

`Rejected` raised inside a tool callback during propose/revise is expected
and is converted to a `{"status": "rejected", ...}` tool result — it never
escapes `propose`/`revise`. `Rejected` raised during `commit` is not a
modeled outcome (the human is no longer in the loop) and propagates
normally, the same way `cli.py`'s `add` command already lets it reach
`cli.main`'s top-level handler.

**Client-side, not server-side tool-calling loop.** mistral.rs's Python SDK
offers two loop shapes; this module uses the client-side one deliberately,
not by default. Registering `tool_callbacks` on `Runner` and letting
mistral.rs's server-side loop dispatch them is the documented alternative,
but the installed 0.9.2 Python bindings register each callback with an empty
schema and separately reject a request that also declares a real
`tool_schemas` entry for the same name — there is no way to give a
server-side-dispatched tool a real, `strict`-mode schema through this SDK
version. `Runner` is therefore built with no `tool_callbacks` at all;
`tool_schemas` carries the real schemas on every request, and `AgentRunner`
inspects `response.choices[0].message.tool_calls` itself, dispatches the
matching Python function, appends the result as a new message, and re-sends.
Full traces and upstream references are in the design journal above.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import mistralrs

from sumac import config, decide, ledger, paths, render, store
from sumac.errors import Rejected
from sumac.models import ChangeKind


class ToolCallFormat(StrEnum):
    """Which chat-template convention the loaded GGUF expects a completed
    tool call replayed in, as the assistant's own prior turn — see
    `_render_tool_call`. Not one universal format: mistralrs-pyo3 0.9.2
    drops the structured `tool_calls` field when building a message from a
    plain dict (module docstring) and passes only `content` through, so this
    module has to hand-render exactly what each model's own chat template
    would have produced from a real `tool_calls` field — and that rendering
    is template-specific, not generic. Qwen3 uses a `<tool_call>` JSON
    block; LFM2.5 uses a different, Pythonic `<|tool_call_start|>[name(...)]
    <|tool_call_end|>` syntax. See the design journal for how each was
    worked out against a real GGUF-embedded chat template.

    `GEMMA` (Gemma 4's `<|tool_call>call:name{...}<tool_call|>` syntax) is
    the odd one out here — reconstructed from published research, not from
    reading a real chat template byte-for-byte the way QWEN and LFM were
    (see `_render_tool_call`). Treat any benchmark built on it as
    unverified until checked against a real `--eval-debug` trace."""

    QWEN = "qwen"
    LFM = "lfm"
    GEMMA = "gemma"


# --- Configuration -----------------------------------------------------------
# Not empirically checked against every tool schema below for every model —
# switching is meant to be cheap precisely because reliability varies
# request to request (see the design journal's real-run comparisons); a
# `ModelPreset` bundles a GGUF repo/file with the wire format its chat
# template expects, so picking one can't leave `ToolCallFormat` pointing
# at the wrong syntax the way the old comment/uncomment blocks could.


@dataclass(frozen=True, slots=True)
class ModelPreset:
    name: str
    quantized_model_id: str
    quantized_filename: str
    tool_call_format: ToolCallFormat


MODEL_PRESETS: tuple[ModelPreset, ...] = (
    # The winner of a real multi-model/multi-quant comparison
    # See docs/journal/2026-09-02-eval-suite.md
    ModelPreset("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF",
                "Qwen3.5-4B-Q4_K_M.gguf", ToolCallFormat.QWEN),
    # Same family/tool_call_format, one size up — kept on hand for a later
    # comparison, not benchmarked yet. See docs/journal/2026-09-02-eval-suite.md.
    ModelPreset("qwen3.5-9b", "unsloth/Qwen3.5-9B-GGUF",
                "Qwen3.5-9B-Q4_K_M.gguf", ToolCallFormat.QWEN),
)  # fmt: skip

_MODEL_PRESETS_BY_NAME: dict[str, ModelPreset] = {p.name: p for p in MODEL_PRESETS}


def model_preset(name: str) -> ModelPreset:
    """Raises `KeyError` for an unknown name — callers taking a name from a
    person (a CLI retry prompt) should catch it and show `MODEL_PRESETS`'
    own names back, not let a typo surface as a raw traceback."""
    return _MODEL_PRESETS_BY_NAME[name]


DEFAULT_MODEL_PRESET = MODEL_PRESETS[0]


def is_cached(model: ModelPreset) -> bool:
    """Whether `model.quantized_filename` is already present in the local
    Hugging Face Hub cache, checked without ever constructing a real
    `mistralrs.Runner` — a first load of an uncached preset downloads a
    multi-gigabyte GGUF file over the network (`_build_runner`'s own log
    line: "first run downloads it; may take a while"). `sumac models
    pull`/`list` and the eval suite's `agent_runner_factory` fixture both
    use this to decide whether to trigger that download, so it lives here
    rather than duplicated in `evals/conftest.py`."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / ("models--" + model.quantized_model_id.replace("/", "--"))
    if not repo_dir.exists():
        return False
    return any(repo_dir.rglob(model.quantized_filename))


# A termination guarantee sized for "one write per round" (every sequential
# search-then-act step a compound request could produce), not a cap on how
# compound a single request may be.
MAX_TOOL_ROUNDS = 20

# One self-check pass by default; 0 disables it.
SELF_REVIEW_ROUNDS = 1

# Sampling defaults for every request `AgentRunner` sends. Previously unset —
# `_build_request` passed no sampling field at all, so a run inherited
# whatever mistral.rs itself defaults to, which can move under a dependency
# bump with nothing catching it. Low temperature favours tool-call
# reliability on a 1-4B model; the classifier in particular is a four-way
# decision that should not be sampled loosely. `DEFAULT_MAX_TOKENS` is sized
# above the largest single round recorded in the design journal's real runs
# (432 completion tokens), not tuned further.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 1024


# --- Query classification ----------------------------------------------------
# The first step of handling any request: decide which of a small, fixed set
# of task shapes it is, before the model sees any domain tool at all. See the
# design journal for why — in short, one system prompt trying to cover
# search-relevance judgment, add-vs-already-exists reasoning, and
# consume-vs-move disambiguation all at once degraded on all three.


class QueryKind(StrEnum):
    FIND = "find"
    ADD = "add"
    REMOVE = "remove"
    REJECT = "reject"


_CLASSIFY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "classify_request",
        "description": "Classify the person's request. Call this exactly once, with no other text.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["find", "add", "remove", "reject"],
                    "description": (
                        "find: locate or ask about something already in inventory, "
                        "nothing changes. add: record new or additional stock of a "
                        "product. remove: record that stock was used, thrown away, "
                        "or moved elsewhere. reject: anything else, or too vague to "
                        "act on."
                    ),
                }
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
    },
}
_CLASSIFY_SCHEMA_JSON = json.dumps(_CLASSIFY_SCHEMA)

CLASSIFIER_PROMPT = """\
You classify one household inventory request. Call classify_request exactly
once with the single best-fitting kind. Do not answer the request itself.
"""

_REJECT_REPLY = "This doesn't look like a request to find, add, or remove something from inventory."


# --- Tool schemas -------------------------------------------------------------
# `amount` is `"string"`, not `"number"` — matches `cli.py`'s own `add`
# command, which parses `amount: str` through `Decimal` rather than trusting
# JSON's number type to carry an exact decimal string.

_FIND_INVENTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sumac_find_inventory",
        "description": (
            "Search current inventory by product name, and the location "
            "layout by location name, case-insensitive. "
            'Returns {"products": [...], "locations": [...]}. '
            '"locations" holds every place whose name or path matched — each '
            'with "location_id" and "location_path" — and is how a place '
            'named in plain words ("the fridge", "the top shelf") becomes a '
            "location_id you can pass to another tool. Search for the place "
            "before writing to it if you do not already have its id from a "
            "result in this conversation. "
            '"products" has one entry per distinct product '
            "name that matched — never one row per location. Each entry "
            'has "product_id" (its exact spelling as stored), '
            '"is_exact_match" (true only when the product\'s name exactly '
            'matches the search), and "locations" (every location holding '
            'it, each with "location_id", "location_path", "amount", and '
            '"unit"). "is_exact_match": false means the name only partly '
            "matches — it may still be the same basic kind of product the "
            "search was for, or it may be a different, established product "
            "that merely shares a word; judge each one the way a person "
            "reading the name would. product_id and location_id are "
            "internal identifiers — use them, verbatim, only in a later "
            "consume/move/discover call; never invent one and never "
            "mention one in a reply to the person."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": 'Product name or partial name, e.g. "ragu" or "jam".',
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

_CONSUME_INVENTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sumac_consume_inventory",
        "description": (
            "Propose recording that some quantity of a product was used, eaten, or "
            "consumed from one location. product_id, unit, and from_location must be "
            "exact values already seen in a sumac_find_inventory result — never invented. "
            "Does not immediately alter inventory; the proposal is shown to the "
            "person for review before anything is written."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "amount": {
                    "type": "string",
                    "description": 'Plain decimal string, e.g. "1" or "0.5".',
                },
                "unit": {"type": "string"},
                "from_location": {"type": "string"},
            },
            "required": ["product_id", "amount", "unit", "from_location"],
            "additionalProperties": False,
        },
    },
}

_MOVE_INVENTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sumac_move_inventory",
        "description": (
            "Propose recording that some quantity of a product was relocated from "
            "one location to another, unchanged — same product, same unit, only the "
            "location differs. If the product's identity, container, or unit changed "
            "(cooked, decanted, repackaged), use sumac_discover_inventory for the result "
            "instead of sumac_move_inventory."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "amount": {"type": "string"},
                "unit": {"type": "string"},
                "from_location": {"type": "string"},
                "to_location": {"type": "string"},
            },
            "required": ["product_id", "amount", "unit", "from_location", "to_location"],
            "additionalProperties": False,
        },
    },
}

_DISCOVER_INVENTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sumac_discover_inventory",
        "description": (
            "Propose recording that some quantity of a product now exists at a "
            "location. Covers both a genuinely new or additional product (a search "
            "finding no match is expected, not a blocker) and one whose identity, "
            "container, or unit changed since it was taken from inventory — a cooked "
            "dish, a decanted portion, a repackaged remainder. product_id may be a "
            "new name distinct from anything currently in inventory."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "amount": {"type": "string"},
                "unit": {"type": "string"},
                "to_location": {"type": "string"},
            },
            "required": ["product_id", "amount", "unit", "to_location"],
            "additionalProperties": False,
        },
    },
}

_KIND_BY_TOOL = {
    "sumac_consume_inventory": ChangeKind.CONSUMPTION,
    "sumac_move_inventory": ChangeKind.MOVEMENT,
    "sumac_discover_inventory": ChangeKind.DISCOVERY,
}


# --- Per-kind prompts and tool scoping ---------------------------------------
# Each prompt below covers exactly the reasoning its own kind needs, and no
# other kind's tools are on the request at all — a `find` request never sees
# the consume/move/discover schemas, so there's nothing for it to reason
# about wanting to use them, and vice versa.

_FIND_PROMPT = """\
You are a household grocery inventory assistant. You have one tool,
sumac_find_inventory. Use it to answer the request.

"is_exact_match": true is strong evidence for what the person means, but
does not rule out the other, non-exact matches — a modifier on the search
word (a flavor, variant, or preparation) is often still the same basic
product; a name where the search word is only part of a different,
established product's name usually isn't. Judge each match the way a
person reading a shopping list would.

Name only the products that actually answer the request — don't just list
every search result. Call one tool at a time; when nothing more is needed,
answer in plain text with no further tool call.
"""

_ADD_PROMPT = """\
You are a household grocery inventory assistant. You have two tools,
sumac_find_inventory and sumac_discover_inventory.

If the person's wording ties the location to something already in
inventory — "the same place as", "with the other X", "existing stock",
"the usual spot" — call sumac_find_inventory for that other product
first and use its location; never guess a location string from their
own wording in this case.

Before assuming a product is new, search for it twice if the first search
finds nothing: once with the full name, then with the brand dropped and
only the product itself kept — not the other way round (after "Heinz
Baked Beans" finds nothing, search "Baked Beans", not "Heinz"). A
different brand of the same basic product is stock to add to, not a new
product. Use sumac_discover_inventory to record the new or additional
stock once that broader search also finds nothing plausible, using a new
product_id in this catalog's own style — Title Case, brand name if the
person gave one, no underscores (e.g. "Heinz Baked Beans", not
"heinz_baked_beans").

Call one tool at a time; when nothing more is needed, answer in plain text
with no further tool call.
"""

# `add-amount-delta` PromptVariant candidate (not yet default) — see
# docs/journal/2026-09-04-basmati-rice-unit-mismatch.md's sibling failure,
# add.discriminator_variant_not_confused: a real qwen3.5-4b trace worked out
# the correct resulting total in its own reply text ("added 2 more, total is
# now 4") but then passed that total as sumac_discover_inventory's amount
# instead of the delta actually requested — decide.py/the fold already add
# the delta to what's on record, so passing the total double-counts it.
# Deliberately no worked numeric example (see
# docs/journal/2026-09-01-ask-agent-design.md's worked-example-bias finding:
# a concrete instance in the prompt teaches the model that one instance, not
# the general rule) — states the invariant abstractly instead.
_ADD_PROMPT_AMOUNT_DELTA = """\
You are a household grocery inventory assistant. You have two tools,
sumac_find_inventory and sumac_discover_inventory.

If the person's wording ties the location to something already in
inventory — "the same place as", "with the other X", "existing stock",
"the usual spot" — call sumac_find_inventory for that other product
first and use its location; never guess a location string from their
own wording in this case.

Before assuming a product is new, search for it twice if the first search
finds nothing: once with the full name, then with the brand dropped and
only the product itself kept — not the other way round (after "Heinz
Baked Beans" finds nothing, search "Baked Beans", not "Heinz"). A
different brand of the same basic product is stock to add to, not a new
product. Use sumac_discover_inventory to record the new or additional
stock once that broader search also finds nothing plausible, using a new
product_id in this catalog's own style — Title Case, brand name if the
person gave one, no underscores (e.g. "Heinz Baked Beans", not
"heinz_baked_beans").

sumac_discover_inventory's amount is the quantity this request is adding
by itself, not the total that will exist once it's applied to what's
already on record — don't work out the new grand total and pass that.

Call one tool at a time; when nothing more is needed, answer in plain text
with no further tool call.
"""

_REMOVE_PROMPT = """\
You are a household grocery inventory assistant. You have three tools:
sumac_find_inventory, sumac_consume_inventory, and sumac_move_inventory.

First call sumac_find_inventory to get the product's exact product_id,
location_id, amount, and unit — never invent these. Then, if the person
names a destination for it, call sumac_move_inventory; otherwise call
sumac_consume_inventory.

Call one tool at a time; when nothing more is needed, answer in plain text
with no further tool call.
"""

_SCHEMA_DICTS_BY_KIND: dict[QueryKind, tuple[dict, ...]] = {
    QueryKind.FIND: (_FIND_INVENTORY_SCHEMA,),
    QueryKind.ADD: (_FIND_INVENTORY_SCHEMA, _DISCOVER_INVENTORY_SCHEMA),
    QueryKind.REMOVE: (_FIND_INVENTORY_SCHEMA, _CONSUME_INVENTORY_SCHEMA, _MOVE_INVENTORY_SCHEMA),
}

_PROMPT_BY_KIND: dict[QueryKind, str] = {
    QueryKind.FIND: _FIND_PROMPT,
    QueryKind.ADD: _ADD_PROMPT,
    QueryKind.REMOVE: _REMOVE_PROMPT,
}

_SCHEMAS_BY_KIND: dict[QueryKind, list[str]] = {
    kind: [json.dumps(s) for s in schemas] for kind, schemas in _SCHEMA_DICTS_BY_KIND.items()
}

_TOOL_NAMES_BY_KIND: dict[QueryKind, frozenset[str]] = {
    kind: frozenset(s["function"]["name"] for s in schemas)
    for kind, schemas in _SCHEMA_DICTS_BY_KIND.items()
}

_REQUIRED_ARGS_BY_TOOL: dict[str, tuple[str, ...]] = {
    s["function"]["name"]: tuple(s["function"]["parameters"]["required"])
    for schemas in _SCHEMA_DICTS_BY_KIND.values()
    for s in schemas
}


_SELF_REVIEW_MESSAGE = (
    "Check the plan above against the original request. If it is correct, "
    "say so with no further tool calls. If not, revise it."
)

_EMPTY_PLAN_NUDGE = (
    "This request needs a change to inventory, but no tool call was made — "
    "describing what you would do is not the same as doing it. Call the "
    "matching tool now."
)


# --- Prompt variants -----------------------------------------------------------
# The same problem `ModelPreset` solves for "which model" — comparing options
# without editing the source and reverting it back — applies to the prompt
# text itself. `PromptVariant` is that same registry/lookup/default shape
# applied to a second, independent axis. Each field here is referenced at
# exactly one call site in `AgentRunner` (`classifier_prompt` in `_classify`,
# `prompt_by_kind` in `propose()`, `empty_plan_nudge` in `_maybe_force_action`)
# — a new variant genuinely cannot miss updating a use site, because there's
# only ever the one. See docs/journal/2026-09-02-eval-suite.md.


@dataclass(frozen=True, slots=True)
class PromptVariant:
    name: str
    classifier_prompt: str = CLASSIFIER_PROMPT
    prompt_by_kind: dict[QueryKind, str] = field(default_factory=lambda: dict(_PROMPT_BY_KIND))
    empty_plan_nudge: str = _EMPTY_PLAN_NUDGE


PROMPT_VARIANTS: tuple[PromptVariant, ...] = (
    PromptVariant("default"),  # every field defaults — reproduces current behavior exactly
    PromptVariant(
        "add-amount-delta",
        prompt_by_kind={**_PROMPT_BY_KIND, QueryKind.ADD: _ADD_PROMPT_AMOUNT_DELTA},
    ),
    # nudge-v2/v3/v4 (a series of `empty_plan_nudge` rewordings aimed at a
    # diagnosed 9B failure) were tried and rolled back — see
    # docs/journal/2026-09-04-trace-and-verdict-redesign.md for why: the
    # comparison was confounded (mistralrs.Runner's RNG stream position
    # drifts based on everything that ran earlier in the same session, so
    # two different-wording sessions aren't cleanly comparable) and the
    # trace format couldn't show what was actually sent to the model or
    # what it replied when a round produced no tool call, so the apparent
    # per-scenario effects couldn't be verified against what really
    # happened. Revisit once that's fixed, not before.
)

_PROMPT_VARIANTS_BY_NAME: dict[str, PromptVariant] = {v.name: v for v in PROMPT_VARIANTS}


def prompt_variant(name: str) -> PromptVariant:
    """Raises `KeyError` for an unknown name — same contract as `model_preset()`."""
    return _PROMPT_VARIANTS_BY_NAME[name]


DEFAULT_PROMPT_VARIANT = PROMPT_VARIANTS[0]


# --- Orchestration types ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocationEffect:
    """What one location holds of one product before and after a single
    proposed write, both numbers taken from `ledger.project`'s fold of the
    records `decide.decide_change` returned for that write — not from
    subtracting an amount from a holding. `None` means the location holds
    none of the product on that side: `before=None` is a genuinely new
    arrival, `after=None` is the last of it leaving."""

    location_id: str
    product_id: str
    unit: str
    before: Decimal | None
    after: Decimal | None


@dataclass(frozen=True, slots=True)
class ProposedWrite:
    kind: ChangeKind
    product_id: str
    amount: Decimal
    unit: str
    from_location: str | None
    to_location: str | None
    warnings: tuple[str, ...] = ()
    # What was already at the relevant location (`to_location` for a
    # discovery, `from_location` otherwise) before this write, captured at
    # propose time — lets the human review a before/after quantity rather
    # than just the delta, without a second inventory query at render time
    # that could reflect a different moment than the one actually decided.
    current_amount: Decimal | None = None
    # Per-location before/after for this write, in endpoint order (a
    # movement's source then its destination). Empty when the write was
    # built without a projection — `render.print_plan` falls back to
    # `current_amount`'s descriptive "already there" line in that case, so
    # a hand-built `ProposedWrite` renders exactly as it did before this
    # field existed.
    effects: tuple[LocationEffect, ...] = ()
    # Which fields a person set by hand, via `sumac ask`'s edit menu. Empty
    # for anything the model proposed. `review.review_write` reads it: its
    # `ungrounded` check asks whether the *model* produced a name from
    # nowhere, and a name typed deliberately by the person reviewing the plan
    # is not that.
    edited_fields: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One tool call `AgentRunner._run_loop` (or the classifier step)
    dispatched, and its raw JSON result — a plain final reply like "the jam
    is in the fridge" otherwise hides the exact `sumac_find_inventory` query
    and match data that produced it, with no way for a human to tell a
    vague answer from a genuinely empty result without seeing the
    underlying call."""

    name: str
    arguments: dict
    result: str
    # The pre-parse response body verbatim, when the active backend
    # provides one — only a remote HTTP backend ever does (local mistral.rs
    # has no comparable "raw wire format" to capture). What a tool-call
    # parser mismatch on the serving side is actually diagnosed from: if a
    # misconfigured parser makes `tool_calls` come back empty because the
    # model's real call landed as unparsed text, this is the only place
    # that evidence survives. See
    # docs/journal/2026-09-04-modal-remote-inference-backend.md.
    raw_response: str | None = None


@dataclass(frozen=True, slots=True)
class AgentPlan:
    reply_text: str
    writes: tuple[ProposedWrite, ...]
    trace: tuple[ToolCallRecord, ...] = ()


class ChatResponseUsage(Protocol):
    """The subset of `mistralrs.Usage` (or an equivalent populated by a
    remote backend) anything in this module ever reads — see the "exact
    minimal response shape" audit in
    docs/journal/2026-09-04-modal-remote-inference-backend.md. Declared as
    read-only `@property` members, not plain attributes: nothing here ever
    writes one back, and a read-only member is covariant, which is what
    lets a real `mistralrs.Usage` (and any other concrete field type)
    satisfy this Protocol structurally without an exact type match."""

    @property
    def prompt_tokens(self) -> int: ...
    @property
    def completion_tokens(self) -> int: ...
    @property
    def avg_compl_tok_per_sec(self) -> float: ...
    @property
    def total_time_sec(self) -> float: ...


class ChatResponseFunctionCall(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def arguments(self) -> str: ...  # a JSON string, parsed with json.loads


class ChatResponseToolCall(Protocol):
    @property
    def function(self) -> ChatResponseFunctionCall: ...


class ChatResponseMessage(Protocol):
    @property
    def content(self) -> str | None: ...
    @property
    def tool_calls(self) -> Sequence[ChatResponseToolCall] | None: ...


class ChatResponseChoice(Protocol):
    @property
    def message(self) -> ChatResponseMessage: ...


class ChatResponse(Protocol):
    """The narrowed response shape every `SendsCompletions` implementation
    must satisfy structurally. A real `mistralrs.ChatCompletionResponse`
    already does, with no wrapping; `usage` is `None`-tolerant, matching
    `_record_usage` below (a fake `SendsCompletions` in tests has none)."""

    @property
    def choices(self) -> Sequence[ChatResponseChoice]: ...
    @property
    def usage(self) -> ChatResponseUsage | None: ...


class SendsCompletions(Protocol):
    """The one method `AgentRunner` actually calls to get a completion —
    a real `mistralrs.Runner` satisfies this only once wrapped in
    `_LocalMistralRsBackend` below, since `mistralrs.ChatCompletionRequest`
    is an opaque PyO3 object no other backend can construct or read back.
    `request` is therefore the plain dict `_build_request` produces, not
    that type — see
    docs/journal/2026-09-04-modal-remote-inference-backend.md for why.
    Tests pass a hand-built fake instead, with no real model or GGUF
    download involved."""

    def send_chat_completion_request(
        self, request: dict, model_id: str | None = None
    ) -> ChatResponse: ...


def _lfm_literal(value: object) -> str:
    """A double-quoted, escaped string literal for LFM2.5's Pythonic
    tool-call syntax, e.g. `query="butter"`. Every argument every tool
    schema in this module defines is string-typed, so this only ever needs
    to quote and escape a plain string, not represent other JSON types."""
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_tool_call(
    name: str, arguments: dict, raw_arguments: str, tool_call_format: ToolCallFormat
) -> str:
    """Hand-renders exactly what the loaded model's own GGUF chat template
    would produce for a completed tool call — see `ToolCallFormat`'s
    docstring for why this can't just pass a real `tool_calls` field
    through. `tool_call_format` (the active `AgentRunner`'s `ModelPreset`)
    selects which template convention to target; `raw_arguments` (the
    model's own emitted JSON string) is only used by the Qwen branch,
    `arguments` (already parsed) only by LFM's and Gemma's."""
    if tool_call_format is ToolCallFormat.LFM:
        # LFM2.5's own documented Pythonic syntax:
        # <|tool_call_start|>[name(key="value", ...)]<|tool_call_end|>
        args_text = ", ".join(f"{key}={_lfm_literal(value)}" for key, value in arguments.items())
        return f"<|tool_call_start|>[{name}({args_text})]<|tool_call_end|>"
    if tool_call_format is ToolCallFormat.GEMMA:
        # Best-effort, NOT a verified byte-exact match (see
        # `ToolCallFormat.GEMMA`'s docstring) — reconstructed as
        # <|tool_call>call:name{key:value,...}<tool_call|>, values passed
        # through unquoted/unescaped. Every tool schema here is
        # string-typed, so there's no other JSON type to represent — but a
        # value containing ',', ':', '{', or '}' will still produce a
        # malformed replay, since the exact escaping rule (if any) wasn't
        # confirmed against a real chat template.
        args_text = ",".join(f"{key}:{value}" for key, value in arguments.items())
        return f"<|tool_call>call:{name}{{{args_text}}}<tool_call|>"
    # QWEN: byte-for-byte what the `{%- if message.tool_calls %}` branch
    # renders from a real `tool_calls` field. `raw_arguments` is spliced in
    # verbatim — already what `{{- tool_call.arguments }}` inserts when
    # `arguments is string` — rather than re-serializing `arguments`, which
    # could reorder keys or whitespace differently.
    return f'<tool_call>\n{{"name": "{name}", "arguments": {raw_arguments}}}\n</tool_call>'


_REJECTION_HINT = (
    "If a fix is obvious from this, retry with the correction — actually "
    "make the retry, don't just describe one and stop. Otherwise explain "
    "why in plain text."
)


# One search result can name a lot of shelves; the count comes back in full
# even when the list is cut, so a too-broad query reads as one to narrow
# rather than as the whole layout.
_MAX_LOCATION_MATCHES = 20


def _location_candidates(cfg: config.Config, value: str, limit: int = 20) -> list[dict[str, str]]:
    """Locations worth offering after `value` failed to resolve: those whose
    id or display path shares a word with it, or — when nothing does — the
    first `limit` of them, so the reply is never "not that one" with no
    indication of what would be."""
    words = {w for w in re.split(r"\W+", value.lower()) if len(w) > 2}
    everything = [
        {"location_id": loc_id, "location_path": config.location_path(cfg.known_locations, loc_id)}
        for loc_id in sorted(cfg.active_locations)
    ]
    matching = [
        candidate
        for candidate in everything
        if any(
            word in candidate["location_id"].lower() or word in candidate["location_path"].lower()
            for word in words
        )
    ]
    return (matching or everything)[:limit]


def _rejected(reason: str, detail: dict) -> str:
    """Every rejected tool result carries its own retry guidance, rather
    than a prompt stating it unconditionally on every request regardless
    of whether a rejection ever happens. A Qwen3 run that hit `rejected`
    with no such instruction anywhere narrated an intention to retry
    ("Let me try again") without making one, then stopped — putting the
    instruction here means it only ever reaches the model at the exact
    moment it's relevant, never spent on a request that succeeds outright."""
    return json.dumps(
        {"status": "rejected", "reason": reason, "detail": detail, "hint": _REJECTION_HINT}
    )


def _round_preview(message: ChatResponseMessage, limit: int = 100) -> str:
    """What the model actually did this round, truncated — a tool call's
    name and arguments, or its plain-text reply."""
    if message.tool_calls:
        call = message.tool_calls[0].function
        text = f"tool call: {call.name}({call.arguments})"
    else:
        text = (message.content or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _print_usage(response: ChatResponse, round_num: int) -> None:
    """Per-round token/timing numbers from mistral.rs's own `Usage`, labeled
    with `round_num` and a preview of what the round actually produced. A
    no-op for a fake `SendsCompletions` in tests, which has no real
    `.usage`."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    preview = _round_preview(response.choices[0].message)
    render.console.print(
        f"[dim]round {round_num}: {usage.prompt_tokens} prompt + "
        f"{usage.completion_tokens} completion tokens, "
        f"{usage.avg_compl_tok_per_sec:.1f} tok/s, {usage.total_time_sec:.1f}s — {preview}[/dim]"
    )


class _LocalMistralRsBackend:
    """Wraps a real `mistralrs.Runner` to satisfy `SendsCompletions`'s
    plain-dict request contract. The one place, for the local backend, that
    turns `_build_request`'s dict back into a real
    `mistralrs.ChatCompletionRequest` — immediately before calling the
    engine. A Modal-backed `SendsCompletions` instead serializes the same
    dict to an OpenAI-style JSON body; both backends see identical kwargs.
    See docs/journal/2026-09-04-modal-remote-inference-backend.md."""

    def __init__(self, runner: mistralrs.Runner) -> None:
        self._runner = runner

    def send_chat_completion_request(
        self, request: dict, model_id: str | None = None
    ) -> mistralrs.ChatCompletionResponse:
        # `tool_choice` is always the string "auto" today (see
        # `_build_request`) — the only `mistralrs.ToolChoice` variant besides
        # `NoTools`, which nothing here ever requests.
        real_request = mistralrs.ChatCompletionRequest(
            messages=request["messages"],
            model=request["model"],
            tool_schemas=request["tool_schemas"],
            tool_choice=mistralrs.ToolChoice.Auto,
            enable_thinking=request["enable_thinking"],
            temperature=request["temperature"],
            top_p=request["top_p"],
            max_tokens=request["max_tokens"],
        )
        # `Runner.send_chat_completion_request` also returns a chunk
        # iterator when `stream=True`; `real_request` never sets it, so
        # narrowing to the non-streaming shape here is safe for every
        # request this module actually sends.
        return cast(
            mistralrs.ChatCompletionResponse,
            self._runner.send_chat_completion_request(real_request, model_id=model_id),
        )


def _build_runner(model: ModelPreset, *, seed: int | None = None) -> SendsCompletions:
    # No `tool_callbacks` here — see the module docstring for why:
    # `AgentRunner` dispatches tool calls itself instead (`_run_loop`).
    # `seed` is `None` for interactive `sumac ask` use (no fixed seed — the
    # existing "regenerate" retry already gets its variety from resampling);
    # an eval run passes an explicit seed so one epoch reproduces exactly
    # from that seed alone. See docs/journal/2026-09-02-eval-suite.md.
    render.console.print(
        f"[dim]Loading {model.quantized_model_id} "
        "(first run downloads it; may take a while)...[/dim]"
    )
    which = mistralrs.Which.GGUF(
        quantized_model_id=model.quantized_model_id,
        quantized_filename=model.quantized_filename,
    )
    # `Which.GGUF` is a nested dataclass, not a `Which` subclass, in the
    # installed 0.9.2 stub — the mismatch below is a stub-modeling gap, not a
    # real one; `Which.GGUF(...)` is mistral.rs's own documented construction.
    runner = mistralrs.Runner(which=which, seed=seed)  # ty: ignore[invalid-argument-type]
    return _LocalMistralRsBackend(runner)


# The most recently built backend, and what it was built for. `sumac ask
# --loop` constructs a fresh `AgentRunner` per request on purpose — each
# request is its own conversation, with no memory of the last — but the model
# behind it need not be reloaded to get that: `AgentRunner` keeps every piece
# of per-request state (`_messages`, `_pending`, `_trace`) on itself, and a
# request carries its whole history in `send_chat_completion_request`, so a
# backend is stateless as far as this module is concerned. `evals/conftest.py`
# has shared one `base_runner` across every scenario in a run since the eval
# suite existed; this is the same reuse for the interactive loop.
_SHARED_RUNNER: tuple[tuple[str, int | None], SendsCompletions] | None = None


def shared_runner(model: ModelPreset, *, seed: int | None = None) -> SendsCompletions:
    """A backend for `model`, reusing the last one when it matches — so a
    `--loop` session loads the model once instead of once per request.

    Holds exactly one, and drops it before building the next: two GGUFs
    resident at once is how switching models mid-session runs a GPU out of
    memory that fits either alone. Dropping this module's reference is all
    this can do — a caller still holding the previous `AgentRunner` keeps its
    backend alive until that goes too.

    Reuse is visible in one way, and it is the same way `evals/` already
    lives with: mistral.rs's RNG stream and prefix cache carry across
    requests, so a request's sampling depends on what ran before it in the
    session (docs/journal/2026-09-04-trace-and-verdict-redesign.md). That
    costs an interactive session nothing — nobody is comparing two runs of
    it — and "regenerate" still resamples, since the stream has moved on.

    Not what `AgentRunner` does by default: it still builds its own backend
    unless handed one, so `evals/` and the benchmark scripts keep deciding
    for themselves when a model is loaded."""
    global _SHARED_RUNNER
    cache_key = (model.name, seed)
    if _SHARED_RUNNER is not None and _SHARED_RUNNER[0] == cache_key:
        return _SHARED_RUNNER[1]
    _SHARED_RUNNER = None
    _SHARED_RUNNER = (cache_key, _build_runner(model, seed=seed))
    return _SHARED_RUNNER[1]


def release_shared_runner() -> None:
    """Drops the cached backend. For a caller that knows it is done with the
    model — nothing in `sumac ask` currently is, since a session ends with
    the process."""
    global _SHARED_RUNNER
    _SHARED_RUNNER = None


def effects(
    inventory: ledger.Inventory,
    cfg: config.Config,
    decided: list[decide.Write],
    product_id: str,
    unit: str,
    from_location: str | None,
    to_location: str | None,
) -> tuple[LocationEffect, ...]:
    """The before/after this write produces at each endpoint it names, read
    off `ledger.project`'s fold of `decided` — the records `decide_change`
    just returned for this one write, which for a consumption exceeding the
    shelf includes §3.5's reconciling `Counted` alongside the `Consumed`.
    Projecting the records rather than subtracting the amount is what makes
    the "after" the same number a commit would produce, for that case as
    much as the ordinary one.

    Endpoint order, source first, so a movement reads the way it happened.
    An unknown location is skipped rather than reported: `decide_change`
    already rejected before returning if an endpoint was invalid, so
    anything reaching here is a location the projection could resolve.

    Public because a hand-edited write needs the same projection recomputed
    for it: `cli._apply_edit` re-decides the edited write and calls this with
    what came back, rather than leaving the preview with the projection of
    the write the model proposed or with none at all."""
    # Log records only. `decide_change` returns config writes alongside them
    # when it auto-registers a product or location (decide.py's
    # `_resolve_product`), and those carry no inventory delta — a config
    # record reaching the fold is not a projection of anything, it's a
    # schema mismatch.
    objs = [w.obj for w in decided if w.stream.startswith("log:")]
    projected = ledger.project(inventory, cfg.known_locations, objs)
    if projected.anomalies:
        # The fold could not apply these records — an endpoint it could not
        # resolve, a unit that would not convert. Whatever the preview showed
        # then would be arithmetic that did not happen: `render.print_plan`
        # falls back to "already there" for a write with no effects, which
        # says less and claims nothing. Reaching here at all means something
        # upstream is wrong, since `decide_change` returned these records.
        return ()
    effects: list[LocationEffect] = []
    for location_id in (from_location, to_location):
        if location_id is None:
            continue
        before = inventory.at(location_id).get(product_id)
        after = projected.at(location_id).get(product_id)
        effects.append(
            LocationEffect(
                location_id=location_id,
                product_id=product_id,
                unit=quantity.unit if (quantity := after or before) is not None else unit,
                before=before.amount if before else None,
                after=after.amount if after else None,
            )
        )
    return tuple(effects)


class AgentRunner:
    """`data_dir`/`key` are closed over by the tool callbacks below,
    matching `cli.py`'s existing `AgentRunner(data_dir, key)` construction.
    Pass `runner` to substitute a fake `SendsCompletions` in tests, or a
    backend of the caller's own, instead of the shared one this otherwise
    builds (`shared_runner`) — which is reused across instances rather than
    loading the model again for each. Tool calls are dispatched by this
    class itself, client-side — see the module docstring — rather than
    registered on the `Runner`."""

    def __init__(
        self,
        data_dir: Path,
        key: bytes,
        *,
        model: ModelPreset = DEFAULT_MODEL_PRESET,
        prompt_variant: PromptVariant = DEFAULT_PROMPT_VARIANT,
        runner: SendsCompletions | None = None,
        debug: bool = False,
        show_usage: bool = True,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        seed: int | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._key = key
        self._model = model
        self._prompt_variant = prompt_variant
        self._debug = debug
        # Per-round token/timing lines. Defaults on, which is what every
        # existing caller (`evals/`, the benchmark scripts) already gets;
        # `sumac ask` passes `--stats` through so the lines aren't printed
        # above a plan someone is trying to read. `usage_history` carries
        # the same numbers programmatically either way, so turning the
        # print off never loses data.
        self._show_usage = show_usage
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        # Session-level for the local backend (`_build_runner(seed=...)`
        # seeds the whole `mistralrs.Runner`, not a per-request field — see
        # `_LocalMistralRsBackend`, which ignores this dict key entirely).
        # Carried into every request dict anyway so a per-request backend
        # (Modal) has something to seed with — previously not threaded
        # through at all, meaning every Modal request ran with vLLM's own
        # unseeded default and no run was reproducible. See
        # docs/journal/2026-09-04-modal-remote-inference-backend.md.
        self._seed = seed
        self._messages: list[dict[str, str]] | None = None
        self._kind: QueryKind | None = None
        self._schemas: list[str] = []
        self._allowed: frozenset[str] = frozenset()
        self._pending: list[ProposedWrite] = []
        self._trace: list[ToolCallRecord] = []
        # Searches already answered in the current `propose`/`revise` call,
        # so an identical repeat comes back labelled rather than looking
        # like a fresh result.
        self._searched: dict[str, dict] = {}
        self._trace_history: list[ToolCallRecord] = []
        self._classify_messages: list[dict[str, str]] | None = None
        self._usage_history: list[dict[str, float | int]] = []
        self._terminal: str = "reply"
        self._nudge_fired: bool = False
        self._completion_tokens = 0
        self._generation_time_sec = 0.0
        self.tool_callbacks: dict[str, Callable[[str, dict], str]] = {
            "sumac_find_inventory": self._sumac_find_inventory,
            "sumac_consume_inventory": self._sumac_consume_inventory,
            "sumac_move_inventory": self._sumac_move_inventory,
            "sumac_discover_inventory": self._sumac_discover_inventory,
        }
        # `shared_runner`, not `_build_runner`: `sumac ask --loop` builds a
        # fresh `AgentRunner` per request — each request is its own
        # conversation — and reloading the GGUF for each of them put seconds
        # of latency in front of a person already waiting. A caller needing
        # a backend of its own passes one; `evals/conftest.py` builds
        # exactly one per run that way and has since the suite existed.
        self._runner: SendsCompletions = (
            runner if runner is not None else shared_runner(model, seed=seed)
        )

    @property
    def model(self) -> ModelPreset:
        """Which `ModelPreset` this instance is running — read by a CLI
        retry prompt to default to "same model as last time" rather than
        forcing a choice on every retry."""
        return self._model

    @property
    def prompt_variant(self) -> PromptVariant:
        """Which `PromptVariant` this instance is running — mirrors `model`."""
        return self._prompt_variant

    @property
    def tokens_per_sec(self) -> float | None:
        """Aggregate completion-token throughput across every round this
        instance has sent so far — the classifier call plus every
        tool-calling round, across every `propose()`/`revise()` call this
        instance has made (never reset; a fresh instance per scenario is
        what scopes it — see `evals/conftest.py`'s `agent` fixture).
        `completion_tokens / generation_time_sec` summed across rounds
        rather than averaging each round's own `avg_compl_tok_per_sec`,
        which would over-weight short rounds. `None` if no round has
        reported real `usage` yet — a fake `SendsCompletions` in tests has
        none, so this stays `None` for the whole test."""
        if self._generation_time_sec <= 0:
            return None
        return self._completion_tokens / self._generation_time_sec

    @property
    def trace_history(self) -> tuple[ToolCallRecord, ...]:
        """Every `ToolCallRecord` this instance has dispatched across every
        `propose()`/`revise()` call it has made so far — unlike
        `AgentPlan.trace`, which is scoped to just the most recent call
        (`self._trace` resets at the top of each), this never resets. A
        fresh `AgentRunner` per scenario (see `evals/conftest.py`'s `agent`
        fixture) is what scopes it to one scenario's full history, feedback
        rounds included."""
        return tuple(self._trace_history)

    @property
    def messages(self) -> tuple[dict[str, str], ...] | None:
        """The full domain-loop conversation `_run_loop` sent/received,
        across every `propose()`/`revise()` call this instance has made —
        `None` before `propose()` has run, or after a REJECT-classified
        request (`propose()` never builds a domain-loop conversation for
        one; see `classify_messages` for what a REJECT still produces).
        `trace_history` records each tool call's outcome; this is the raw
        wire-shaped history — including a plain-text-only reply, which
        never produces a `ToolCallRecord` — it was recorded from. See
        docs/journal/2026-09-04-trace-and-verdict-redesign.md."""
        return tuple(self._messages) if self._messages is not None else None

    @property
    def classify_messages(self) -> tuple[dict[str, str], ...] | None:
        """The classifier round's own system/user/assistant messages.
        `_classify` sends a different system prompt on its own request, so
        its conversation never becomes part of `self._messages` — `None`
        before `propose()` has run, populated even when it rejects."""
        return tuple(self._classify_messages) if self._classify_messages is not None else None

    @property
    def usage_history(self) -> tuple[dict[str, float | int], ...]:
        """Per-round token/timing numbers, in request order, across every
        round this instance has sent (the classifier round tagged
        `round=0`) — unlike `tokens_per_sec`'s running totals, this keeps
        each round's own numbers, which reasoning about the RNG stream's
        position needs and a running total can't reconstruct."""
        return tuple(self._usage_history)

    @property
    def terminal(self) -> str:
        """How the most recently completed `_run_loop` call ended:
        `"reply"` (a round produced no tool call) or `"round_cap"`
        (`MAX_TOOL_ROUNDS` exhausted with no final reply) — both currently
        return `reply_text=""` from `_run_loop`, indistinguishable without
        this."""
        return self._terminal

    @property
    def nudge_fired(self) -> bool:
        """Whether `_maybe_force_action` appended `empty_plan_nudge` and
        re-ran the loop, at any point across this instance's calls —
        previously derivable only by reading `_maybe_force_action`'s guard
        clause against `checks.writes` after the fact."""
        return self._nudge_fired

    def _record_call(self, record: ToolCallRecord) -> None:
        self._trace.append(record)
        self._trace_history.append(record)

    # -- tool callbacks -------------------------------------------------------

    def _sumac_find_inventory(self, _name: str, args: dict) -> str:
        """Groups results by product, not by match tier and not as a flat
        per-location list — every location holding a given product_id is
        collected under that product's one entry, so "two rows, same
        product_id, different location_id" is not bookkeeping the model has
        to notice itself. `ledger.search_inventory`'s classification is the
        only matching logic; this only reshapes its already-computed,
        already-ordered output. `is_exact_match` is `True` only for a
        product whose match_kind is `MatchKind.EXACT`; entries stay in
        `search_inventory`'s tier order (exact, then whole-word, then
        substring) without naming the tiers themselves."""
        query = str(args.get("query", ""))
        if query in self._searched:
            # A real run searched "fridge" seventeen times in a row, each
            # time getting the same empty result, until the round cap
            # stopped it. Repeating the answer is not enough on its own —
            # it is what the model already had — so the answer comes back
            # labelled as the repeat it is.
            return json.dumps(
                {
                    **self._searched[query],
                    "repeated_query": True,
                    "hint": (
                        "This exact search already ran in this request and returned this same "
                        "result. Running it again returns nothing new. Act on what it returned, "
                        "or explain in plain text what you could not find."
                    ),
                }
            )
        inventory = ledger.build_inventory(self._data_dir, self._key)
        locations = ledger.load_locations_or_empty(self._data_dir, self._key)
        products: dict[str, dict] = {}
        for m in ledger.search_inventory(inventory, query):
            entry = products.setdefault(
                m.product_id,
                {
                    "product_id": m.product_id,
                    "is_exact_match": m.match_kind is ledger.MatchKind.EXACT,
                    "locations": [],
                },
            )
            entry["locations"].append(
                {
                    "location_id": m.location_id,
                    "location_path": config.location_path(locations, m.location_id),
                    "amount": str(m.quantity.amount),
                    "unit": m.quantity.unit,
                }
            )
        # Location matches alongside product matches: `query` is whatever the
        # person's sentence named, and a phrase like "the fridge" is a place,
        # not a product. Without this the only way to learn a location id was
        # to guess one, be rejected, and guess again.
        matched = config.search_locations(locations, query)
        result = {
            "products": list(products.values()),
            "locations": [
                {"location_id": loc.id, "location_path": config.location_path(locations, loc.id)}
                for loc in matched[:_MAX_LOCATION_MATCHES]
            ],
            "location_match_count": len(matched),
        }
        self._searched[query] = result
        return json.dumps(result)

    def _propose_write(self, name: str, args: dict) -> str:
        # A real LFM2.5 run produced `_amount` instead of `amount` — the
        # model's own Pythonic tool-call syntax isn't key-constrained the
        # way strict JSON-schema decoding is, so a wrong or missing key
        # reaches here as a plain absence rather than a schema violation
        # caught before generation. Naming exactly what's missing and what
        # was received (rather than a bare "invalid_amount") gives a retry
        # something concrete to correct, instead of repeating the same
        # malformed call.
        missing = [k for k in _REQUIRED_ARGS_BY_TOOL[name] if k not in args]
        if missing:
            return _rejected(
                "missing_required_argument",
                {"missing": missing, "received": sorted(args.keys())},
            )

        kind = _KIND_BY_TOOL[name]
        product_id = str(args.get("product_id", ""))
        unit = str(args.get("unit", ""))
        from_location = args.get("from_location")
        to_location = args.get("to_location")

        try:
            amount = Decimal(str(args["amount"]))
        except InvalidOperation:
            # Not one of decide.py's own rejection-catalogue reasons — this
            # is the tool-callback equivalent of cli.py's `_parse_decimal`
            # rejecting before a value ever reaches `decide_change` at all.
            return _rejected("invalid_amount", {"value": args["amount"]})

        cfg = config.build_config(self._data_dir, self._key)
        inventory = ledger.build_inventory(self._data_dir, self._key)
        try:
            decided, messages = decide.decide_change(
                kind=kind,
                product_id=product_id,
                amount=amount,
                unit=unit,
                from_location=from_location,
                to_location=to_location,
                actor=paths.current_user(),
                occurred_at=datetime.now(UTC),
                inventory=inventory,
                cfg=cfg,
            )
        except Rejected as e:
            # `object` values, not `str`: every `decide` detail stringifies,
            # but the candidate list below is structured on purpose — a
            # model reading `"[{'location_id': ...}]"` as one string has to
            # parse it back out of a Python repr.
            detail: dict[str, object] = {k: str(v) for k, v in e.detail.items()}
            if e.reason == "unknown_location":
                # `decide`'s own `suggestions` are `near_matches` over ids,
                # which a phrase ("top shelf of the fridge") never comes
                # close enough to match. The candidates below are what a
                # person would offer instead: the locations whose path shares
                # a word with what was asked for.
                detail["known_locations"] = _location_candidates(
                    cfg, str(e.detail.get("value", ""))
                )
            return _rejected(e.reason, detail)

        # The ids `decide_change` just resolved these to, not the strings the
        # model supplied. `decide.resolve_location` accepts a display path
        # ("Fridge > Top Shelf") as readily as an id, so a write can be
        # entirely valid — and commit correctly — while its raw endpoint
        # matches nothing in `known_locations`. Recording the raw string made
        # the preview claim a configured location was new and show no
        # before/after for it, since every lookup downstream is by id.
        # Re-resolving cannot raise here: `decide_change` returned, so both
        # already resolved once.
        from_location = decide.resolve_location(from_location, "from", cfg)
        to_location = decide.resolve_location(to_location, "to", cfg)

        # Before/after context for the human reviewing this, not domain
        # logic: a discovery's "current" location is where the product
        # would land (already-there stock this adds to); everything else's
        # is where it's coming from. `None` when nothing is there yet — a
        # genuinely new discovery, not "0 and rising."
        current_location = to_location if kind is ChangeKind.DISCOVERY else from_location
        current_quantity = (
            inventory.at(current_location).get(product_id) if current_location else None
        )

        candidate = ProposedWrite(
            kind=kind,
            product_id=product_id,
            amount=amount,
            unit=unit,
            from_location=from_location,
            to_location=to_location,
            warnings=tuple(messages),
            current_amount=current_quantity.amount if current_quantity else None,
            effects=effects(inventory, cfg, decided, product_id, unit, from_location, to_location),
        )
        if candidate in self._pending:
            # A real LFM2.5 run repeated an already-successful discover call
            # three more times, byte-for-byte — `decide_change` has no
            # memory of what this same `propose`/`revise` call already
            # queued, so each repeat silently became a second, third,
            # fourth full write. Naming the exact quantity already proposed
            # is both the guard against that (never appended twice) and a
            # clearer signal back to the model than repeating "proposed"
            # unchanged, which plausibly reads as "still not done."
            return json.dumps(
                {
                    "status": "already_proposed",
                    "product_id": product_id,
                    "amount": str(amount),
                    "unit": unit,
                    "from_location": from_location,
                    "to_location": to_location,
                }
            )

        self._pending.append(candidate)
        return json.dumps(
            {
                "status": "proposed",
                "product_id": product_id,
                "amount": str(amount),
                "unit": unit,
                "from_location": from_location,
                "to_location": to_location,
                "warnings": messages,
            }
        )

    def _sumac_consume_inventory(self, name: str, args: dict) -> str:
        return self._propose_write(name, args)

    def _sumac_move_inventory(self, name: str, args: dict) -> str:
        return self._propose_write(name, args)

    def _sumac_discover_inventory(self, name: str, args: dict) -> str:
        return self._propose_write(name, args)

    # -- request plumbing (client-side loop — see module docstring) ---------

    def _build_request(self, messages: list[dict[str, str]], schemas: list[str]) -> dict:
        """A plain dict of every kwarg this module's one request shape ever
        sends — not a real `mistralrs.ChatCompletionRequest`, which is an
        opaque PyO3 object no non-mistralrs backend can construct, and which
        can't be read back once built even by mistral.rs's own code (see
        docs/journal/2026-09-04-modal-remote-inference-backend.md). Every
        `SendsCompletions` implementation gets this identical dict;
        `_LocalMistralRsBackend` is the one place that turns it back into a
        real `mistralrs.ChatCompletionRequest`, immediately before calling
        the engine."""
        return {
            "messages": messages,
            "model": self._model.quantized_model_id,
            "tool_schemas": schemas,
            # The only value `mistralrs.ToolChoice` this module ever
            # requests — nothing here ever varies it (see the journal entry
            # above) — carried as a plain string so a non-mistralrs backend
            # can pass it straight through without importing `mistralrs`.
            "tool_choice": "auto",
            "enable_thinking": False,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            # `_LocalMistralRsBackend` ignores this — the local engine is
            # already seeded once at `Runner` construction, not per
            # request. A per-request backend (Modal) uses it for whatever
            # reproducibility a stateless HTTP request can offer.
            "seed": self._seed,
        }

    def classify(self, prompt: str) -> QueryKind:
        """Public alias for `_classify` — the routing-only entry point an
        eval suite outside this module needs (one call per case, not a full
        `propose()`), without reaching into a private name. See
        docs/journal/2026-09-02-eval-suite.md."""
        return self._classify(prompt)

    def _record_usage(self, response: ChatResponse, round_num: int) -> None:
        """Prints the round's usage line (`_print_usage`, when `show_usage`)
        and folds its token/timing numbers into `tokens_per_sec`'s running
        totals — a no-op for the latter beyond the print for a fake
        `SendsCompletions` with no real `.usage`."""
        if self._show_usage:
            _print_usage(response, round_num)
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._completion_tokens += usage.completion_tokens
        self._generation_time_sec += usage.total_time_sec
        self._usage_history.append(
            {
                "round": round_num,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_time_sec": usage.total_time_sec,
            }
        )

    def _classify(self, prompt: str) -> QueryKind:
        """One single-purpose call, made before the model sees any domain
        tool: which of find/add/remove this request is, or reject. See the
        design journal for why this is a separate step rather than folded
        into one prompt covering every tool."""
        messages = [
            {"role": "system", "content": self._prompt_variant.classifier_prompt},
            {"role": "user", "content": prompt},
        ]
        request = self._build_request(messages, [_CLASSIFY_SCHEMA_JSON])
        response = self._runner.send_chat_completion_request(request)
        self._record_usage(response, 0)
        message = response.choices[0].message
        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            self._classify_messages = messages
            self._record_call(
                ToolCallRecord(
                    name="classify_request",
                    arguments={},
                    result=message.content or "",
                    raw_response=getattr(response, "raw_body", None),
                )
            )
            return QueryKind.REJECT
        call = message.tool_calls[0].function
        args = json.loads(call.arguments)
        messages.append(
            {
                "role": "assistant",
                "content": _render_tool_call(
                    call.name, args, call.arguments, self._model.tool_call_format
                ),
            }
        )
        self._classify_messages = messages
        self._record_call(
            ToolCallRecord(
                name="classify_request",
                arguments=args,
                result=json.dumps(args),
                raw_response=getattr(response, "raw_body", None),
            )
        )
        try:
            return QueryKind(args.get("kind"))
        except ValueError:
            return QueryKind.REJECT

    def _run_loop(self) -> AgentPlan:
        """Client-side tool-calling loop — see the module docstring. The
        assistant message for a dispatched call is rendered by
        `_render_tool_call`, per `self._model.tool_call_format` — hand-built because the
        Python bindings drop a real `tool_calls` field but pass `content`
        through unchanged, so this has to be byte-for-byte what the loaded
        model's own chat template would have produced from a real
        `tool_calls` field, and that rendering differs by model family.
        `MAX_TOOL_ROUNDS` is a termination guarantee, not a plan-size cap."""
        assert self._messages is not None
        self._pending = []
        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            request = self._build_request(self._messages, self._schemas)
            if self._debug:
                render.print_agent_messages(self._messages, f"MESSAGES · round {round_num}")
                render.print_agent_request(request, round_num)

            response = self._runner.send_chat_completion_request(request)

            if self._debug:
                render.print_agent_response(response, round_num)

            self._record_usage(response, round_num)
            message = response.choices[0].message

            if self._debug:
                render.print_agent_message(message)
                render.print_agent_content(message.content)
                render.print_agent_tool_calls(message.tool_calls)

            if not message.tool_calls:
                self._messages.append({"role": "assistant", "content": message.content or ""})
                self._terminal = "reply"
                return AgentPlan(reply_text=message.content or "", writes=tuple(self._pending))

            call = message.tool_calls[0].function
            args = json.loads(call.arguments)
            if call.name in self._allowed and call.name in self.tool_callbacks:
                result = self.tool_callbacks[call.name](call.name, args)
            else:
                # A model calling a tool outside the schemas actually sent on
                # this request — not expected under normal operation, but a
                # small model can still emit a name it wasn't given; fail
                # this one call rather than crash or silently dispatch the
                # wrong domain action.
                result = _rejected("tool_not_available", {"name": call.name})
            self._record_call(
                ToolCallRecord(
                    name=call.name,
                    arguments=args,
                    result=result,
                    raw_response=getattr(response, "raw_body", None),
                )
            )
            self._messages.append(
                {
                    "role": "assistant",
                    "content": _render_tool_call(
                        call.name, args, call.arguments, self._model.tool_call_format
                    ),
                }
            )
            self._messages.append({"role": "tool", "content": result})
            if self._debug:
                render.print_agent_messages(self._messages, "MESSAGES FOR NEXT ROUND")

        # Round cap reached with no final reply — the accumulated plan (if
        # any) is still returned rather than raised, since the cap is a
        # termination guarantee, not a success condition.
        self._terminal = "round_cap"
        return AgentPlan(reply_text="", writes=tuple(self._pending))

    def _maybe_force_action(self, plan: AgentPlan) -> AgentPlan:
        """A `find`-classified request producing no writes is the normal,
        correct outcome — a plain question, nothing needs to change. An
        `add`/`remove`-classified request producing no writes is not: the
        classifier already decided something needs to change, so a model
        that stops without ever calling the matching tool described an
        action in place of taking one (the same class of miss the design
        journal's real-run comparisons record for read-only queries,
        happening here on a mutating one instead). One forced follow-up
        round, not unbounded — if the model still doesn't act after this,
        that's the human's call via feedback/regenerate/start over, not
        another retry this method adds on its own."""
        if plan.writes or self._kind not in (QueryKind.ADD, QueryKind.REMOVE):
            return plan
        assert self._messages is not None
        self._messages.append({"role": "user", "content": self._prompt_variant.empty_plan_nudge})
        self._nudge_fired = True
        return self._run_loop()

    def _maybe_self_review(self, plan: AgentPlan) -> AgentPlan:
        """The model checks its own plan against the original request. A
        round that makes no new tool calls means the model is satisfied
        with `plan` as it stands — stop and keep it. A round that does make
        new tool calls replaces `plan` and, if rounds remain, is itself
        reviewed again."""
        assert self._messages is not None
        if not plan.writes:
            return plan
        for _ in range(SELF_REVIEW_ROUNDS):
            self._messages.append({"role": "user", "content": _SELF_REVIEW_MESSAGE})
            reviewed = self._run_loop()
            if not reviewed.writes:
                return plan
            plan = reviewed
        return plan

    # -- public interface -----------------------------------------------------

    def propose(self, prompt: str) -> AgentPlan:
        self._trace = []
        self._searched = {}
        kind = self._classify(prompt)
        self._kind = kind
        if kind is QueryKind.REJECT:
            self._messages = None
            return AgentPlan(reply_text=_REJECT_REPLY, writes=(), trace=tuple(self._trace))
        self._schemas = _SCHEMAS_BY_KIND[kind]
        self._allowed = _TOOL_NAMES_BY_KIND[kind]
        self._messages = [
            {"role": "system", "content": self._prompt_variant.prompt_by_kind[kind]},
            {"role": "user", "content": prompt},
        ]
        plan = self._maybe_self_review(self._maybe_force_action(self._run_loop()))
        return replace(plan, trace=tuple(self._trace))

    def revise(self, feedback: str) -> AgentPlan:
        if self._messages is None:
            raise RuntimeError(
                "AgentRunner.revise() called before propose(), or propose() rejected the request"
            )
        self._messages.append({"role": "user", "content": feedback})
        self._trace = []
        self._searched = {}
        plan = self._maybe_self_review(self._maybe_force_action(self._run_loop()))
        return replace(plan, trace=tuple(self._trace))

    def commit(self, plan: AgentPlan) -> list[str]:
        """Re-decides each write against freshly reloaded state rather than
        replaying the dry run's resolution — real time passed since
        `propose`/`revise` computed it. Errors here are not caught: a
        `Rejected` at this point is not a modeled outcome the model gets to
        react to, and should propagate the same way `cli.py`'s `add`
        command already lets it."""
        summaries: list[str] = []
        for pw in plan.writes:
            cfg = config.build_config(self._data_dir, self._key)
            inventory = ledger.build_inventory(self._data_dir, self._key)
            writes, messages = decide.decide_change(
                kind=pw.kind,
                product_id=pw.product_id,
                amount=pw.amount,
                unit=pw.unit,
                from_location=pw.from_location,
                to_location=pw.to_location,
                actor=paths.current_user(),
                occurred_at=datetime.now(UTC),
                inventory=inventory,
                cfg=cfg,
            )
            for message in messages:
                render.print_warning(message)
            for w in writes:
                store.append(self._data_dir, self._key, w.stream, w.obj)
            summaries.append(f"Recorded {pw.kind.value} of {pw.amount} {pw.unit} {pw.product_id}")
        return summaries
