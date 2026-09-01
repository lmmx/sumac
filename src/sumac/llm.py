"""In-process agent for `sumac ask`: mistral.rs `Runner` wrapping the domain
layer's existing `ledger`/`decide` entry points.

Implements docs/journal/2026-09-01-ask-agent-design.md — section numbers in
comments below refer to that document. This is the `sumac/llm.py` §16
describes as not-yet-existing.

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
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import mistralrs

from sumac import config, decide, ledger, paths, render, store
from sumac.errors import Rejected
from sumac.models import ChangeKind

# --- Configuration (§21) ----------------------------------------------------
# Not empirically checked against the tool schemas below (§21, §24) — pick a
# different GGUF repo/file here if this one doesn't call tools reliably.

QUANTIZED_MODEL_ID = "unsloth/Qwen3-1.7B-GGUF"
QUANTIZED_FILENAME = "Qwen3-1.7B-Q4_K_M.gguf"

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
just given you.

If sumac_find_inventory returns more than one plausible match for what the person
asked about, and nothing in their request distinguishes which one they mean,
stop and ask them in plain text which one they mean, with no further tool
call, rather than guessing.

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
            "Search current inventory by product name (partial, case-insensitive "
            "match). Returns every location currently holding a matching product, "
            "with its exact product_id, location_id, amount, and unit as sumac has "
            "them recorded. Call this before consuming, moving, or discovering any "
            "product — never guess a product_id, location_id, amount, or unit that "
            "has not been returned by this tool."
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
class AgentPlan:
    reply_text: str
    writes: tuple[ProposedWrite, ...]


class SendsCompletions(Protocol):
    """The one `mistralrs.Runner` method `AgentRunner` actually calls (§22) —
    a real `Runner` satisfies this structurally; tests pass a hand-built
    fake instead, with no real model or GGUF download involved."""

    def send_chat_completion_request(
        self, request: mistralrs.ChatCompletionRequest, model_id: str | None = None
    ) -> mistralrs.ChatCompletionResponse: ...


def _build_runner(tool_callbacks: Mapping[str, Callable[[str, dict], str]]) -> mistralrs.Runner:
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
    return mistralrs.Runner(
        which=which,  # ty: ignore[invalid-argument-type]
        tool_callbacks=dict(tool_callbacks),
    )


class AgentRunner:
    """§19. `data_dir`/`key` are closed over by the tool callbacks below,
    matching `cli.py`'s existing `AgentRunner(data_dir, key)` construction
    (§16). Pass `runner` to substitute a fake `SendsCompletions` in tests
    (§22) instead of building a real `mistralrs.Runner`."""

    def __init__(
        self, data_dir: Path, key: bytes, *, runner: SendsCompletions | None = None
    ) -> None:
        self._data_dir = data_dir
        self._key = key
        self._session_id: str | None = None
        self._pending: list[ProposedWrite] = []
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
            else cast(SendsCompletions, _build_runner(self.tool_callbacks))
        )

    # -- tool callbacks (§18) ------------------------------------------------

    def _sumac_find_inventory(self, _name: str, args: dict) -> str:
        query = str(args.get("query", ""))
        inventory = ledger.build_inventory(self._data_dir, self._key)
        locations = ledger.load_locations_or_empty(self._data_dir, self._key)
        matches = [
            {
                "product_id": pid,
                "location_id": loc_id,
                "location_path": config.location_path(locations, loc_id),
                "amount": str(qty.amount),
                "unit": qty.unit,
            }
            for loc_id, entries in sorted(inventory.by_location.items())
            for pid, qty in sorted(entries.items())
            if query.lower() in pid.lower()
        ]
        return json.dumps({"matches": matches})

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

    # -- request plumbing -----------------------------------------------------

    def _build_request(self, messages: list[dict[str, str]]) -> mistralrs.ChatCompletionRequest:
        return mistralrs.ChatCompletionRequest(
            messages=messages,
            model=MODEL_ID,
            tool_schemas=TOOL_SCHEMAS,
            tool_choice=mistralrs.ToolChoice.Auto,
            max_tool_rounds=MAX_TOOL_ROUNDS,
            agent_permission=mistralrs.AgentPermission.Auto,
            session_id=self._session_id,
        )

    def _run_request(self, request: mistralrs.ChatCompletionRequest) -> AgentPlan:
        self._pending = []
        response = self._runner.send_chat_completion_request(request)
        reply_text = response.choices[0].message.content or ""
        return AgentPlan(reply_text=reply_text, writes=tuple(self._pending))

    def _maybe_self_review(self, plan: AgentPlan) -> AgentPlan:
        """§13/§14 step 3: the model checks its own plan against the original
        request. A round that makes no new tool calls means the model is
        satisfied with `plan` as it stands — stop and keep it. A round that
        does make new tool calls replaces `plan` and, if rounds remain, is
        itself reviewed again."""
        if not plan.writes:
            return plan
        for _ in range(SELF_REVIEW_ROUNDS):
            reviewed = self._run_request(
                self._build_request([{"role": "user", "content": _SELF_REVIEW_MESSAGE}])
            )
            if not reviewed.writes:
                return plan
            plan = reviewed
        return plan

    # -- public interface (§19) ----------------------------------------------

    def propose(self, prompt: str) -> AgentPlan:
        self._session_id = str(uuid4())
        plan = self._run_request(
            self._build_request(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
        )
        return self._maybe_self_review(plan)

    def revise(self, feedback: str) -> AgentPlan:
        if self._session_id is None:
            raise RuntimeError("AgentRunner.revise() called before propose()")
        plan = self._run_request(self._build_request([{"role": "user", "content": feedback}]))
        return self._maybe_self_review(plan)

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
