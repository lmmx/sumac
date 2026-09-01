"""In-process agent for `sumac ask`: mistral.rs `Runner` wrapping the domain
layer's existing `ledger`/`decide` entry points.

Implements docs/journal/2026-09-01-ask-agent-design.md — section numbers in
comments below refer to that document. This is the `sumac/llm.py` §16
describes as not-yet-existing, with one load-bearing deviation from §9/§12's
recommended mechanism — see "Client-side, not server-side" below.

Two phases per invocation, matching §12/§14:

- **Propose/revise** (`AgentRunner.propose`/`.revise`): runs the agentic
  tool-calling loop against the real, current inventory, but every mutating
  tool callback stops short of `store.append` — it calls `decide.decide_change`
  to validate the resolved call and records it in `self._pending`, never
  writing anything. This is what makes dry-run and preview-then-accept the
  same mechanism (§11, §12).
- **Commit** (`AgentRunner.commit`): only reachable after a human has reviewed
  an `AgentPlan` and accepted it. Re-decides each `ProposedWrite` against
  freshly reloaded state (§14's "shelf is authoritative, not the log, even
  between preview and accept" reasoning) and actually appends.

`Rejected` raised inside a tool callback during propose/revise is expected
and is converted to a `{"status": "rejected", ...}` tool result (§18) — it
never escapes `propose`/`revise`. `Rejected` raised during `commit` is not a
modeled outcome (the human is no longer in the loop) and propagates normally,
the same way `cli.py`'s `add` command already lets it reach `cli.main`'s
top-level handler (§23).

**Client-side, not server-side (deviation from §9/§12, found empirically).**
§9/§12 recommended registering `tool_callbacks` on `Runner` and letting
mistral.rs's server-side loop dispatch them (`max_tool_rounds`,
`agent_permission`, per-invocation `session_id`). Running that against a real
model surfaced a bug in the installed 0.9.2 Python SDK: `Runner(tool_callbacks=
{...})` registers each callback through the Rust builder's schema-less
`with_tool_callback(name, callback)` (confirmed against
`EricLBuehler/mistral.rs`'s `mistralrs/src/builder_macros.rs` and
`mistralrs-pyo3/src/lib.rs` on `master`) — every registered tool gets an empty
`Tool` (`parameters: None`, `description: None`, `strict: None`), not the
schema this module defines below. The richer `with_tool_callback_and_tool(name,
callback, tool)` that would attach a real schema exists in Rust but is not
exposed through the Python bindings. Supplying the real schema the only way
left — via `ChatCompletionRequest.tool_schemas` on the request, alongside the
same-named `tool_callbacks` registration — hits a separate check in
`mistralrs-core/src/engine/agentic_loop.rs` that rejects any request tool
whose name already exists in `tool_callbacks`, with exactly the error this
module's tools were seen tripping: `"Tool '<name>' conflicts with a registered
internal tool. Internal tool names cannot be overridden."` ("internal" there
means "already in `tool_callbacks`", not "hardcoded engine builtin" — sumac's
own tools trip it against themselves). This affects the SDK's own
`examples/python/agentic_tools.py`, not just sumac's usage — it registers
`tool_callbacks` and passes matching `tool_schemas` in the same shape this
module used to. Filed upstream as inconsistent with `with_tool_callback`'s
empty-schema behavior; not something a different tool name or a request-field
change on sumac's side can work around while still using the server-side loop.

The fix is the *other* documented loop shape (§7's "client-side loop",
`examples/python/tool_call.py`): `Runner` is built with no `tool_callbacks` at
all, `tool_schemas` carries the real schemas on every request as before, and
this module drives the round-trip itself — inspect
`response.choices[0].message.tool_calls`, dispatch the matching Python
function directly, append the result as a new message, and re-send. This
also simplifies session handling: `self._messages` is sumac's own accumulated
history for one `propose`/`revise` invocation, so there is no need for
mistral.rs's `session_id` continuity mechanism (§13's original reason for
minting one) — §10's open question about whether `remember_for_session` has
anything to scope to is moot along with it, along with `agent_permission` and
`max_tool_rounds` as request fields (the round cap is enforced in this
module's own loop instead, as `MAX_TOOL_ROUNDS` below). Per mistral.rs's own
docs (`agentic-runtime.md`, "the standard OpenAI-compatible flow"), only the
*first* tool call in a multi-call response is executed and the rest are
dropped — matching the server-side loop's own documented one-call-per-round
behavior (§7), so behavior here is the same shape either way.
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
    plain dict (module docstring, "Client-side, not server-side") and
    passes only `content` through, so this module has to hand-render
    exactly what each model's own chat template would have produced from a
    real `tool_calls` field — and that rendering is template-specific, not
    generic. §28 worked this out for Qwen3's `<tool_call>` JSON block by
    reading its GGUF-embedded Jinja template line by line; LFM2.5 uses an
    entirely different, Pythonic `<|tool_call_start|>[name(...)]
    <|tool_call_end|>` syntax (§40) — unconfirmed against a real LFM2.5 run
    in this environment, same as every other real-model claim in this
    module (§21, §24)."""

    QWEN = "qwen"
    LFM = "lfm"


# --- Configuration (§21) ----------------------------------------------------
# Not empirically checked against the tool schemas below (§21, §24) — pick a
# different GGUF repo/file here if this one doesn't call tools reliably.
# id/filename/format travel together as one block (comment/uncomment as a
# unit) so switching models can't leave TOOL_CALL_FORMAT pointing at the
# wrong wire syntax (§40).

QUANTIZED_MODEL_ID = "unsloth/Qwen3-1.7B-GGUF"
QUANTIZED_FILENAME = "Qwen3-1.7B-Q4_K_M.gguf"
TOOL_CALL_FORMAT = ToolCallFormat.QWEN
# QUANTIZED_MODEL_ID = "unsloth/Qwen3-0.6B-GGUF"
# QUANTIZED_FILENAME = "Qwen3-0.6B-Q4_K_M.gguf"
# TOOL_CALL_FORMAT = ToolCallFormat.QWEN
# QUANTIZED_MODEL_ID = "unsloth/LFM2.5-1.2B-Instruct-GGUF"
# QUANTIZED_FILENAME = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
# TOOL_CALL_FORMAT = ToolCallFormat.LFM

# §13: a termination guarantee sized for "one write per round" (every
# sequential search-then-act step a compound request could produce), not a
# cap on how compound a single request may be.
MAX_TOOL_ROUNDS = 20

# §13: one self-check pass by default; 0 disables it.
SELF_REVIEW_ROUNDS = 1

# Passed as ChatCompletionRequest.model — unconfirmed against a worked
# example whether this needs to match the loaded model's own id (§21, §24).
MODEL_ID = QUANTIZED_MODEL_ID

# --- System prompt (§20) ----------------------------------------------------
# A draft, not tuned against a real model's behavior (§20, §24).

SYSTEM_PROMPT = """\
You are the tool-calling layer for a household grocery inventory. You can only
act through the four tools you are given: sumac_find_inventory, sumac_consume_inventory,
sumac_move_inventory, and sumac_discover_inventory. You have no other capability — no
filesystem, no shell, no network, no code execution.

Before consuming, moving, or discovering any product, call sumac_find_inventory to
get its exact product_id, location_id, amount, and unit. Never invent a
product_id, location_id, amount, or unit — use only what a tool result has
just given you. product_id and location_id are internal identifiers, only
for making a further tool call — never mention one in a reply to the
person; describe a location using its location_path in plain language
instead.

sumac_find_inventory groups its results by product, each with every
location that holds it — data for you to reason over, not a script to
read back. "is_exact_match": true is strong evidence for what the person
means, but it does not automatically rule out the other, non-exact
matches. A product name that adds a modifier to the search word (a
flavor, a variant, a preparation) is often still the same basic kind of
product a reasonable person would accept; a product name where the search
word is only part of a different, established product's name usually
isn't, even though the word appears in it. Use your own ordinary
knowledge to judge which situation each non-exact match is, the way a
person reading a shopping list would — don't apply a fixed rule, and
don't assume a partial match is always relevant or always irrelevant.

Decide which product or products in the result — the exact one, several
including some non-exact ones, or just the exact one — actually answer
what the person asked, then answer in your own words, directly and
concisely, naming only the ones you judge relevant. More than one
location for the same product is not ambiguity, it just means it's kept
in more than one place. If more than one genuinely different product is
plausible and the person's wording doesn't distinguish them, either cover
the plausible ones together or ask a concise clarifying question, in
plain text with no further tool call — whichever better fits how they
phrased the request. Don't just list every product the search returned.

Call one tool at a time. After each tool call you will see its result before
deciding the next one — use that result, do not assume what it will be in
advance.

If a product changed identity, container, or unit since it was taken from
inventory (cooked, decanted, repackaged), record the result with
sumac_discover_inventory, not sumac_move_inventory — sumac_move_inventory is only for the same
product and unit changing location.

If a tool result has status "rejected", the action was not valid — explain
why in plain text, or try a corrected call if the correction is obvious from
the rejection reason, rather than repeating the same call unchanged.

When you have made every tool call the request needs, or if the request
needs no tool call at all (a plain question), respond in plain text with no
further tool calls.
"""

_SELF_REVIEW_MESSAGE = (
    "Check the plan above against the original request. If it is correct, "
    "say so with no further tool calls. If not, revise it."
)

# --- Tool schemas (§17) -----------------------------------------------------
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
            "mention one in a reply to the person. Call this before "
            "consuming, moving, or discovering any product."
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
            "location, with no claim about where it came from. Use this for a "
            "product whose identity, container, or unit changed since it was taken "
            "from inventory — a cooked dish, a decanted portion, a repackaged "
            "remainder — where the result cannot be expressed as a move. product_id "
            "may be a new name distinct from any product consumed earlier in this "
            "request."
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

TOOL_SCHEMAS = [
    json.dumps(s)
    for s in (
        _FIND_INVENTORY_SCHEMA,
        _CONSUME_INVENTORY_SCHEMA,
        _MOVE_INVENTORY_SCHEMA,
        _DISCOVER_INVENTORY_SCHEMA,
    )
]

_KIND_BY_TOOL = {
    "sumac_consume_inventory": ChangeKind.CONSUMPTION,
    "sumac_move_inventory": ChangeKind.MOVEMENT,
    "sumac_discover_inventory": ChangeKind.DISCOVERY,
}


# --- Orchestration types (§19) ----------------------------------------------


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
    """One tool call `AgentRunner._run_loop` dispatched and its raw JSON
    result — not part of §19's original `ProposedWrite`/`AgentPlan` pair,
    added once real usage showed a plain final reply (e.g. "the jam is in
    the fridge") hides the exact `sumac_find_inventory` query and match data
    that produced it, with no way for a human to tell a vague answer from a
    genuinely empty result without seeing the underlying call."""

    name: str
    arguments: dict
    result: str


@dataclass(frozen=True, slots=True)
class AgentPlan:
    reply_text: str
    writes: tuple[ProposedWrite, ...]
    trace: tuple[ToolCallRecord, ...] = ()


class SendsCompletions(Protocol):
    """The one `mistralrs.Runner` method `AgentRunner` actually calls (§22) —
    a real `Runner` satisfies this structurally; tests pass a hand-built
    fake instead, with no real model or GGUF download involved."""

    def send_chat_completion_request(
        self, request: mistralrs.ChatCompletionRequest, model_id: str | None = None
    ) -> mistralrs.ChatCompletionResponse: ...


def _lfm_literal(value: object) -> str:
    """A double-quoted, escaped string literal for LFM2.5's Pythonic
    tool-call syntax, e.g. `query="butter"`. Every argument this module's
    tool schemas define is string-typed (§17's comment on why `amount` is
    `"string"`, not `"number"`, applies here too), so this only ever needs
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
    # QWEN (§28): byte-for-byte what the `{%- if message.tool_calls %}`
    # branch renders from a real `tool_calls` field. `raw_arguments` is
    # spliced in verbatim — already what `{{- tool_call.arguments }}`
    # inserts when `arguments is string` — rather than re-serializing
    # `arguments`, which could reorder keys or whitespace differently.
    return f'<tool_call>\n{{"name": "{name}", "arguments": {raw_arguments}}}\n</tool_call>'


def _round_preview(message: mistralrs.ResponseMessage, limit: int = 100) -> str:
    """What the model actually did this round, truncated — a tool call's
    name and arguments, or its plain-text reply. §31: the trace table alone
    didn't say which round was which without cross-referencing by hand."""
    if message.tool_calls:
        call = message.tool_calls[0].function
        text = f"tool call: {call.name}({call.arguments})"
    else:
        text = (message.content or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _print_usage(response: mistralrs.ChatCompletionResponse, round_num: int) -> None:
    """Per-round token/timing numbers from mistral.rs's own `Usage`, labeled
    with `round_num` and a preview of what the round actually produced
    (§30/§31 — unlabeled, content-free rounds made this "impossible to
    trace"). No-op for a fake `SendsCompletions` in tests, which has no
    real `.usage`."""
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
    # No `tool_callbacks` here — see the module docstring's "Client-side, not
    # server-side" section for why: the Python SDK can only register a
    # callback with an empty schema, and supplying the real schema via
    # `tool_schemas` on the request then collides with that registration.
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
    """§19. `data_dir`/`key` are closed over by the tool callbacks below,
    matching `cli.py`'s existing `AgentRunner(data_dir, key)` construction
    (§16). Pass `runner` to substitute a fake `SendsCompletions` in tests
    (§22) instead of building a real `mistralrs.Runner`. `tool_callbacks` is
    dispatched by this class itself, client-side — see the module docstring —
    rather than registered on the `Runner`."""

    def __init__(
        self, data_dir: Path, key: bytes, *, runner: SendsCompletions | None = None
    ) -> None:
        self._data_dir = data_dir
        self._key = key
        self._messages: list[dict[str, str]] | None = None
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

    # -- tool callbacks (§18) ------------------------------------------------

    def _sumac_find_inventory(self, _name: str, args: dict) -> str:
        """§36: grouped by product, not by `ledger.MatchKind` tier (§34) and
        not a flat per-location list (pre-§34) — every location holding a
        given product_id is collected under that product's one entry, so
        "two rows, same product_id, different location_id" is no longer
        bookkeeping the model has to notice itself (§35's real trace: it
        was still doing that bookkeeping wrong). `ledger.search_inventory`'s
        classification is still the only matching logic — this only
        reshapes its already-computed, already-ordered output; nothing
        about what counts as a match changes here. `is_exact_match` is
        `True` only for a product whose match_kind is `MatchKind.EXACT`;
        entries stay in `search_inventory`'s tier order (exact, then
        whole-word, then substring) without naming the tiers themselves —
        §36 also dropped that jargon from the model-facing contract."""
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
        kind = _KIND_BY_TOOL[name]
        product_id = str(args.get("product_id", ""))
        unit = str(args.get("unit", ""))
        from_location = args.get("from_location")
        to_location = args.get("to_location")

        try:
            amount = Decimal(str(args.get("amount", "")))
        except InvalidOperation:
            # Not one of decide.py's own rejection-catalogue reasons (§4) —
            # this is the tool-callback equivalent of cli.py's `_parse_decimal`
            # rejecting before a value ever reaches `decide_change` at all.
            return json.dumps(
                {
                    "status": "rejected",
                    "reason": "invalid_amount",
                    "detail": {"value": args.get("amount")},
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

    def _build_request(self) -> mistralrs.ChatCompletionRequest:
        assert self._messages is not None
        return mistralrs.ChatCompletionRequest(
            messages=self._messages,
            model=MODEL_ID,
            tool_schemas=TOOL_SCHEMAS,
            tool_choice=mistralrs.ToolChoice.Auto,
            enable_thinking=False,
        )

    def _run_loop(self) -> AgentPlan:
        """Client-side tool-calling loop (§25/§26/§27/§28/§40 — see the
        journal for why). The assistant message for a dispatched call is
        rendered by `_render_tool_call`, per `TOOL_CALL_FORMAT` — hand-built
        because the Python bindings drop a real `tool_calls` field but pass
        `content` through unchanged, so this has to be byte-for-byte what
        the loaded model's own chat template would have produced from a
        real `tool_calls` field, and that rendering differs by model family
        (§40). `MAX_TOOL_ROUNDS` is a termination guarantee, not a
        plan-size cap (§13)."""
        assert self._messages is not None
        self._pending = []
        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            request = self._build_request()
            print(f"\n=== REQUEST (round {round_num}) ===")
            print(repr(request))
            print(f"tool_schemas: {getattr(request, 'tool_schemas', None)!r}")
            print(f"tool_choice: {getattr(request, 'tool_choice', None)!r}")

            response = self._runner.send_chat_completion_request(request)

            print(f"\n=== RAW RESPONSE (round {round_num}) ===")
            print(repr(response))

            _print_usage(response, round_num)
            message = response.choices[0].message

            print("\n=== MESSAGE ===")
            print(repr(message))
            print("\n=== CONTENT ===")
            print(repr(message.content))
            print("\n=== TOOL CALLS ===")
            print(repr(message.tool_calls))

            if not message.tool_calls:
                self._messages.append({"role": "assistant", "content": message.content or ""})
                return AgentPlan(reply_text=message.content or "", writes=tuple(self._pending))

            call = message.tool_calls[0].function
            args = json.loads(call.arguments)
            result = self.tool_callbacks[call.name](call.name, args)
            self._trace.append(ToolCallRecord(name=call.name, arguments=args, result=result))
            self._messages.append(
                {
                    "role": "assistant",
                    "content": _render_tool_call(call.name, args, call.arguments),
                }
            )
            self._messages.append({"role": "tool", "content": result})

        # Round cap reached with no final reply — the accumulated plan (if
        # any) is still returned rather than raised, matching §13's framing
        # of this cap as a termination guarantee, not a success condition.
        return AgentPlan(reply_text="", writes=tuple(self._pending))

    def _maybe_self_review(self, plan: AgentPlan) -> AgentPlan:
        """§13/§14 step 3: the model checks its own plan against the original
        request. A round that makes no new tool calls means the model is
        satisfied with `plan` as it stands — stop and keep it. A round that
        does make new tool calls replaces `plan` and, if rounds remain, is
        itself reviewed again."""
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

    # -- public interface (§19) ----------------------------------------------

    def propose(self, prompt: str) -> AgentPlan:
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        self._trace = []
        plan = self._maybe_self_review(self._run_loop())
        return replace(plan, trace=tuple(self._trace))

    def revise(self, feedback: str) -> AgentPlan:
        if self._messages is None:
            raise RuntimeError("AgentRunner.revise() called before propose()")
        self._messages.append({"role": "user", "content": feedback})
        self._trace = []
        plan = self._maybe_self_review(self._run_loop())
        return replace(plan, trace=tuple(self._trace))

    def commit(self, plan: AgentPlan) -> list[str]:
        """§14 step 5 ("Accept"), §23: re-decides each write against freshly
        reloaded state rather than replaying the dry run's resolution — real
        time passed since `propose`/`revise` computed it. Errors here are not
        caught: a `Rejected` at this point is not a modeled outcome the model
        gets to react to, and should propagate the same way `cli.py`'s `add`
        command already lets it (§23)."""
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
