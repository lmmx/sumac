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
from collections.abc import Callable
from dataclasses import dataclass, replace
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
    worked out against a real GGUF-embedded chat template."""

    QWEN = "qwen"
    LFM = "lfm"


# --- Configuration -----------------------------------------------------------
# Not empirically checked against every tool schema below for every model —
# pick a different GGUF repo/file here if this one doesn't call tools
# reliably. id/filename/format travel together as one block (comment/
# uncomment as a unit) so switching models can't leave TOOL_CALL_FORMAT
# pointing at the wrong wire syntax.

# QUANTIZED_MODEL_ID = "unsloth/LFM2.5-1.2B-Instruct-GGUF"
# QUANTIZED_FILENAME = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
# TOOL_CALL_FORMAT = ToolCallFormat.LFM
QUANTIZED_MODEL_ID = "LiquidAI/LFM2.5-2.6B-GGUF"
QUANTIZED_FILENAME = "LFM2.5-2.6B-Q4_K_M.gguf"
TOOL_CALL_FORMAT = ToolCallFormat.LFM
# QUANTIZED_MODEL_ID = "unsloth/Qwen3-1.7B-GGUF"
# QUANTIZED_FILENAME = "Qwen3-1.7B-Q4_K_M.gguf"
# TOOL_CALL_FORMAT = ToolCallFormat.QWEN
# QUANTIZED_MODEL_ID = "unsloth/Qwen3-0.6B-GGUF"
# QUANTIZED_FILENAME = "Qwen3-0.6B-Q4_K_M.gguf"
# TOOL_CALL_FORMAT = ToolCallFormat.QWEN

# A termination guarantee sized for "one write per round" (every sequential
# search-then-act step a compound request could produce), not a cap on how
# compound a single request may be.
MAX_TOOL_ROUNDS = 20

# One self-check pass by default; 0 disables it.
SELF_REVIEW_ROUNDS = 1

# Passed as ChatCompletionRequest.model — unconfirmed whether this needs to
# match the loaded model's own id, kept equal to it since nothing has shown
# otherwise.
MODEL_ID = QUANTIZED_MODEL_ID


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
            "Search current inventory by product name, case-insensitive. "
            'Returns {"products": [...]}, one entry per distinct product '
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

Use sumac_discover_inventory to record the new or additional stock — a
sumac_find_inventory search turning up no match for the product is
expected, not a reason to stop: use a new product_id describing it, in
this catalog's own style — Title Case, brand name if the person gave one,
no underscores (e.g. "Heinz Baked Beans", "Yeo Valley Natural Yogurt", not
"heinz_baked_beans"). Call sumac_find_inventory first only if you need to
resolve a location the person referred to indirectly (e.g. "with the
other tins of it") rather than named directly.

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


# --- Orchestration types ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProposedWrite:
    kind: ChangeKind
    product_id: str
    amount: Decimal
    unit: str
    from_location: str | None
    to_location: str | None
    warnings: tuple[str, ...] = ()


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


@dataclass(frozen=True, slots=True)
class AgentPlan:
    reply_text: str
    writes: tuple[ProposedWrite, ...]
    trace: tuple[ToolCallRecord, ...] = ()


class SendsCompletions(Protocol):
    """The one `mistralrs.Runner` method `AgentRunner` actually calls — a
    real `Runner` satisfies this structurally; tests pass a hand-built fake
    instead, with no real model or GGUF download involved."""

    def send_chat_completion_request(
        self, request: mistralrs.ChatCompletionRequest, model_id: str | None = None
    ) -> mistralrs.ChatCompletionResponse: ...


def _lfm_literal(value: object) -> str:
    """A double-quoted, escaped string literal for LFM2.5's Pythonic
    tool-call syntax, e.g. `query="butter"`. Every argument every tool
    schema in this module defines is string-typed, so this only ever needs
    to quote and escape a plain string, not represent other JSON types."""
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_tool_call(name: str, arguments: dict, raw_arguments: str) -> str:
    """Hand-renders exactly what the loaded model's own GGUF chat template
    would produce for a completed tool call — see `ToolCallFormat`'s
    docstring for why this can't just pass a real `tool_calls` field
    through. `TOOL_CALL_FORMAT` selects which template convention to
    target; `raw_arguments` (the model's own emitted JSON string) is only
    used by the Qwen branch, `arguments` (already parsed) only by LFM's."""
    if TOOL_CALL_FORMAT is ToolCallFormat.LFM:
        # LFM2.5's own documented Pythonic syntax:
        # <|tool_call_start|>[name(key="value", ...)]<|tool_call_end|>
        args_text = ", ".join(f"{key}={_lfm_literal(value)}" for key, value in arguments.items())
        return f"<|tool_call_start|>[{name}({args_text})]<|tool_call_end|>"
    # QWEN: byte-for-byte what the `{%- if message.tool_calls %}` branch
    # renders from a real `tool_calls` field. `raw_arguments` is spliced in
    # verbatim — already what `{{- tool_call.arguments }}` inserts when
    # `arguments is string` — rather than re-serializing `arguments`, which
    # could reorder keys or whitespace differently.
    return f'<tool_call>\n{{"name": "{name}", "arguments": {raw_arguments}}}\n</tool_call>'


def _round_preview(message: mistralrs.ResponseMessage, limit: int = 100) -> str:
    """What the model actually did this round, truncated — a tool call's
    name and arguments, or its plain-text reply."""
    if message.tool_calls:
        call = message.tool_calls[0].function
        text = f"tool call: {call.name}({call.arguments})"
    else:
        text = (message.content or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _print_usage(response: mistralrs.ChatCompletionResponse, round_num: int) -> None:
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


def _build_runner() -> mistralrs.Runner:
    # No `tool_callbacks` here — see the module docstring for why:
    # `AgentRunner` dispatches tool calls itself instead (`_run_loop`).
    render.console.print(
        f"[dim]Loading {QUANTIZED_MODEL_ID} (first run downloads it; may take a while)...[/dim]"
    )
    which = mistralrs.Which.GGUF(
        quantized_model_id=QUANTIZED_MODEL_ID,
        quantized_filename=QUANTIZED_FILENAME,
    )
    # `Which.GGUF` is a nested dataclass, not a `Which` subclass, in the
    # installed 0.9.2 stub — the mismatch below is a stub-modeling gap, not a
    # real one; `Which.GGUF(...)` is mistral.rs's own documented construction.
    return mistralrs.Runner(which=which)  # ty: ignore[invalid-argument-type]


class AgentRunner:
    """`data_dir`/`key` are closed over by the tool callbacks below,
    matching `cli.py`'s existing `AgentRunner(data_dir, key)` construction.
    Pass `runner` to substitute a fake `SendsCompletions` in tests instead
    of building a real `mistralrs.Runner`. Tool calls are dispatched by
    this class itself, client-side — see the module docstring — rather
    than registered on the `Runner`."""

    def __init__(
        self,
        data_dir: Path,
        key: bytes,
        *,
        runner: SendsCompletions | None = None,
        debug: bool = False,
    ) -> None:
        self._data_dir = data_dir
        self._key = key
        self._debug = debug
        self._messages: list[dict[str, str]] | None = None
        self._kind: QueryKind | None = None
        self._schemas: list[str] = []
        self._allowed: frozenset[str] = frozenset()
        self._pending: list[ProposedWrite] = []
        self._trace: list[ToolCallRecord] = []
        self.tool_callbacks: dict[str, Callable[[str, dict], str]] = {
            "sumac_find_inventory": self._sumac_find_inventory,
            "sumac_consume_inventory": self._sumac_consume_inventory,
            "sumac_move_inventory": self._sumac_move_inventory,
            "sumac_discover_inventory": self._sumac_discover_inventory,
        }
        self._runner: SendsCompletions = (
            runner
            if runner is not None
            # `Runner.send_chat_completion_request` also returns a chunk
            # iterator when `stream=True`; `_build_request` never sets it, so
            # narrowing to the non-streaming `SendsCompletions` shape here is
            # safe for every request this class actually sends.
            else cast(SendsCompletions, _build_runner())
        )

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
        return json.dumps({"products": list(products.values())})

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
            return json.dumps(
                {
                    "status": "rejected",
                    "reason": "missing_required_argument",
                    "detail": {"missing": missing, "received": sorted(args.keys())},
                }
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
            return json.dumps(
                {
                    "status": "rejected",
                    "reason": "invalid_amount",
                    "detail": {"value": args["amount"]},
                }
            )

        cfg = config.build_config(self._data_dir, self._key)
        inventory = ledger.build_inventory(self._data_dir, self._key)
        try:
            _writes, messages = decide.decide_change(
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
            return json.dumps(
                {
                    "status": "rejected",
                    "reason": e.reason,
                    "detail": {k: str(v) for k, v in e.detail.items()},
                }
            )

        self._pending.append(
            ProposedWrite(
                kind=kind,
                product_id=product_id,
                amount=amount,
                unit=unit,
                from_location=from_location,
                to_location=to_location,
                warnings=tuple(messages),
            )
        )
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

    def _build_request(
        self, messages: list[dict[str, str]], schemas: list[str]
    ) -> mistralrs.ChatCompletionRequest:
        return mistralrs.ChatCompletionRequest(
            messages=messages,
            model=MODEL_ID,
            tool_schemas=schemas,
            tool_choice=mistralrs.ToolChoice.Auto,
            enable_thinking=False,
        )

    def _classify(self, prompt: str) -> QueryKind:
        """One single-purpose call, made before the model sees any domain
        tool: which of find/add/remove this request is, or reject. See the
        design journal for why this is a separate step rather than folded
        into one prompt covering every tool."""
        messages = [
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": prompt},
        ]
        request = self._build_request(messages, [_CLASSIFY_SCHEMA_JSON])
        response = self._runner.send_chat_completion_request(request)
        _print_usage(response, 0)
        message = response.choices[0].message
        if not message.tool_calls:
            self._trace.append(
                ToolCallRecord(name="classify_request", arguments={}, result=message.content or "")
            )
            return QueryKind.REJECT
        call = message.tool_calls[0].function
        args = json.loads(call.arguments)
        self._trace.append(
            ToolCallRecord(name="classify_request", arguments=args, result=json.dumps(args))
        )
        try:
            return QueryKind(args.get("kind"))
        except ValueError:
            return QueryKind.REJECT

    def _run_loop(self) -> AgentPlan:
        """Client-side tool-calling loop — see the module docstring. The
        assistant message for a dispatched call is rendered by
        `_render_tool_call`, per `TOOL_CALL_FORMAT` — hand-built because the
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

            _print_usage(response, round_num)
            message = response.choices[0].message

            if self._debug:
                render.print_agent_message(message)
                render.print_agent_content(message.content)
                render.print_agent_tool_calls(message.tool_calls)

            if not message.tool_calls:
                self._messages.append({"role": "assistant", "content": message.content or ""})
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
                result = json.dumps(
                    {
                        "status": "rejected",
                        "reason": "tool_not_available",
                        "detail": {"name": call.name},
                    }
                )
            self._trace.append(ToolCallRecord(name=call.name, arguments=args, result=result))
            self._messages.append(
                {
                    "role": "assistant",
                    "content": _render_tool_call(call.name, args, call.arguments),
                }
            )
            self._messages.append({"role": "tool", "content": result})
            if self._debug:
                render.print_agent_messages(self._messages, "MESSAGES FOR NEXT ROUND")

        # Round cap reached with no final reply — the accumulated plan (if
        # any) is still returned rather than raised, since the cap is a
        # termination guarantee, not a success condition.
        return AgentPlan(reply_text="", writes=tuple(self._pending))

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
        kind = self._classify(prompt)
        self._kind = kind
        if kind is QueryKind.REJECT:
            self._messages = None
            return AgentPlan(reply_text=_REJECT_REPLY, writes=(), trace=tuple(self._trace))
        self._schemas = _SCHEMAS_BY_KIND[kind]
        self._allowed = _TOOL_NAMES_BY_KIND[kind]
        self._messages = [
            {"role": "system", "content": _PROMPT_BY_KIND[kind]},
            {"role": "user", "content": prompt},
        ]
        plan = self._maybe_self_review(self._run_loop())
        return replace(plan, trace=tuple(self._trace))

    def revise(self, feedback: str) -> AgentPlan:
        if self._messages is None:
            raise RuntimeError(
                "AgentRunner.revise() called before propose(), or propose() rejected the request"
            )
        self._messages.append({"role": "user", "content": feedback})
        self._trace = []
        plan = self._maybe_self_review(self._run_loop())
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
