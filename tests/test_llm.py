"""§22: orchestration (`AgentRunner`) driven by a hand-built fake in place of
`mistralrs.Runner` — no real model, no GGUF download. The fake still drives
real `decide.decide_change`/`store.append` calls against a real encrypted
`data_dir`/`key`, via `AgentRunner`'s actual tool callbacks.

`AgentRunner` drives a client-side tool-calling loop itself (see the module
docstring in `sumac/llm.py`, "Client-side, not server-side") rather than
registering callbacks on `Runner` — so one scripted `ScriptedResponse` here
is one `send_chat_completion_request` *round*, not a whole multi-tool-call
turn: a tool call the loop must dispatch itself, or a final plain-text reply
with no further tool call."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sumac import config, decide, ledger, llm, store
from sumac.errors import Rejected
from sumac.models import ChangeKind, Location, Product


@dataclass
class ScriptedResponse:
    tool_call: tuple[str, dict] | None = None
    content: str = ""


def _tool_round(name: str, args: dict) -> ScriptedResponse:
    return ScriptedResponse(tool_call=(name, args))


def _final_round(content: str) -> ScriptedResponse:
    return ScriptedResponse(content=content)


class FakeRunner:
    """A `llm.SendsCompletions` returning one scripted response per round.
    Never dispatches a tool itself — `AgentRunner._run_loop` does that,
    against its own `tool_callbacks`, exactly as it would against a real
    `mistralrs.Runner`'s response."""

    def __init__(self, responses: list[ScriptedResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[object] = []

    def send_chat_completion_request(self, request, model_id=None):  # noqa: ANN001, ANN201
        self.requests.append(request)
        scripted = self._responses.pop(0)
        if scripted.tool_call is None:
            message = SimpleNamespace(content=scripted.content, role="assistant", tool_calls=None)
        else:
            name, args = scripted.tool_call
            function = SimpleNamespace(name=name, arguments=json.dumps(args))
            tool_call = SimpleNamespace(function=function)
            message = SimpleNamespace(content=None, role="assistant", tool_calls=[tool_call])
        choice = SimpleNamespace(finish_reason="stop", index=0, message=message)
        return SimpleNamespace(choices=[choice])


def _make_agent(
    responses: list[ScriptedResponse], data_dir: Path, key: bytes
) -> tuple[llm.AgentRunner, FakeRunner]:
    fake = FakeRunner(responses)
    agent = llm.AgentRunner(data_dir, key, runner=fake)
    return agent, fake


def _apply_change(
    data_dir: Path, key: bytes, osuser: str, *, kind: ChangeKind, **kwargs: Any
) -> None:
    writes, _messages = decide.decide_change(
        kind=kind,
        actor=osuser,
        occurred_at=datetime.now(UTC),
        inventory=ledger.build_inventory(data_dir, key),
        cfg=config.build_config(data_dir, key),
        **kwargs,
    )
    for w in writes:
        store.append(data_dir, key, w.stream, w.obj)


def _seed_pantry_with_jam(data_dir: Path, key: bytes, osuser: str) -> None:
    config.add_location(data_dir, key, osuser, Location(id="pantry", name="Pantry"))
    config.add_product(data_dir, key, osuser, Product(id="jam", name="Jam", unit="jar"))
    _apply_change(
        data_dir,
        key,
        osuser,
        kind=ChangeKind.PURCHASE,
        product_id="jam",
        amount=Decimal(3),
        unit="jar",
        from_location=None,
        to_location="pantry",
    )


# --- find_inventory ----------------------------------------------------


def test_find_inventory_returns_matches(data_dir: Path, key: bytes, osuser: str) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    agent, _fake = _make_agent([], data_dir, key)

    result = json.loads(
        agent.tool_callbacks["sumac_find_inventory"]("sumac_find_inventory", {"query": "jam"})
    )

    assert result["products"] == [
        {
            "product_id": "jam",
            "is_exact_match": True,
            "locations": [
                {
                    "location_id": "pantry",
                    "location_path": "Pantry",
                    "amount": "3",
                    "unit": "jar",
                }
            ],
        }
    ]


def test_find_inventory_no_match_returns_empty_list(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    agent, _fake = _make_agent([], data_dir, key)

    result = json.loads(
        agent.tool_callbacks["sumac_find_inventory"](
            "sumac_find_inventory", {"query": "nonexistent"}
        )
    )
    assert result["products"] == []


def _seed_freezer_and_fridge_butters(data_dir: Path, key: bytes, osuser: str) -> None:
    config.add_location(data_dir, key, osuser, Location(id="freezer", name="Freezer"))
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    # Products are unregistered — `decide_change` auto-registers under the
    # exact product_id string passed below, which is all `_sumac_find_
    # inventory` needs (it matches/returns raw product_id, not a `Product`).
    for product_id, unit, loc in (
        ("Butternut Box", "pack", "freezer"),
        ("Salted Butter", "block", "freezer"),
        ("Butter", "packet", "fridge"),
    ):
        _apply_change(
            data_dir,
            key,
            osuser,
            kind=ChangeKind.PURCHASE,
            product_id=product_id,
            amount=Decimal(1),
            unit=unit,
            from_location=None,
            to_location=loc,
        )


def test_find_inventory_includes_every_tier_marked_by_is_exact_match(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """§36: nothing is withheld or excluded here — every matching product
    comes back, grouped, ordered by `ledger.search_inventory`'s tier
    (exact, then whole-word, then substring), each tagged
    `is_exact_match`. Relevance judgment (is "Butternut Box" plausibly
    what someone means by "butter"?) is left entirely to the model — this
    only checks the deterministic data shape it's given to reason over."""
    _seed_freezer_and_fridge_butters(data_dir, key, osuser)
    agent, _fake = _make_agent([], data_dir, key)

    result = json.loads(
        agent.tool_callbacks["sumac_find_inventory"]("sumac_find_inventory", {"query": "butter"})
    )

    products = result["products"]
    assert [p["product_id"] for p in products] == ["Butter", "Salted Butter", "Butternut Box"]
    assert [p["is_exact_match"] for p in products] == [True, False, False]


def test_find_inventory_substring_only_query_still_returns_a_match(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """ "nut" matches no product as a whole word here, only as a substring of
    "Butternut" — that match still comes back, marked as not an exact
    match, rather than an empty result."""
    _seed_freezer_and_fridge_butters(data_dir, key, osuser)
    agent, _fake = _make_agent([], data_dir, key)

    result = json.loads(
        agent.tool_callbacks["sumac_find_inventory"]("sumac_find_inventory", {"query": "nut"})
    )

    assert [p["product_id"] for p in result["products"]] == ["Butternut Box"]
    assert result["products"][0]["is_exact_match"] is False


# --- propose -------------------------------------------------------------


def test_propose_read_only_request_produces_no_writes(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    responses = [_final_round("the jam is in the pantry")]
    agent, fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("where is the jam?")

    assert plan.writes == ()
    assert plan.reply_text == "the jam is in the pantry"
    # No writes proposed -> self-review (§13) has nothing to check, so only
    # the one round fires.
    assert len(fake.requests) == 1
    # No tool call happened either — a read-only reply with no search behind
    # it still surfaces an (empty) trace rather than something to infer.
    assert plan.trace == ()


def test_propose_resolves_a_consume_call_into_a_pending_write(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _tool_round("sumac_find_inventory", {"query": "jam"}),
        _tool_round(
            "sumac_consume_inventory",
            {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
        ),
        _final_round("consumed 1 jar of jam"),
        # self-review round: model is satisfied, no further tool calls.
        _final_round("looks correct"),
    ]
    agent, fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("consume 1 jar of jam")

    assert len(plan.writes) == 1
    w = plan.writes[0]
    assert w.kind is ChangeKind.CONSUMPTION
    assert w.product_id == "jam"
    assert w.amount == Decimal(1)
    assert w.unit == "jar"
    assert w.from_location == "pantry"
    # Nothing committed yet — still a dry run (§11/§12).
    assert ledger.build_inventory(data_dir, key).at("pantry")["jam"].amount == Decimal(3)
    assert len(fake.requests) == 4

    # The trace records both calls, in order, with their raw results — not
    # just the final "consumed 1 jar of jam" the model happened to say.
    assert [t.name for t in plan.trace] == ["sumac_find_inventory", "sumac_consume_inventory"]
    assert plan.trace[0].arguments == {"query": "jam"}
    find_result = json.loads(plan.trace[0].result)
    assert find_result["products"][0]["product_id"] == "jam"
    consume_result = json.loads(plan.trace[1].result)
    assert consume_result["status"] == "proposed"


def test_rejected_tool_call_is_reported_and_not_added_to_pending(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _tool_round(
            "sumac_consume_inventory",
            {
                "product_id": "jam",
                "amount": "1",
                "unit": "jar",
                "from_location": "nonexistent-location",
            },
        ),
        _final_round("that location doesn't exist"),
    ]
    agent, _fake = _make_agent(responses, data_dir, key)

    result = json.loads(
        agent.tool_callbacks["sumac_consume_inventory"](
            "sumac_consume_inventory",
            {
                "product_id": "jam",
                "amount": "1",
                "unit": "jar",
                "from_location": "nonexistent-location",
            },
        )
    )
    assert result["status"] == "rejected"
    assert result["reason"] == "unknown_location"

    plan = agent.propose("consume 1 jar of jam from nowhere")
    assert plan.writes == ()


# --- self-review (§13) ----------------------------------------------------


def test_self_review_replaces_plan_when_model_revises_it(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _tool_round(
            "sumac_consume_inventory",
            {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
        ),
        _final_round("consumed 1 jar"),
        # self-review round: the model reconsiders and makes a new call.
        _tool_round(
            "sumac_consume_inventory",
            {"product_id": "jam", "amount": "2", "unit": "jar", "from_location": "pantry"},
        ),
        _final_round("actually, 2 jars"),
    ]
    agent, _fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("consume some jam")

    assert len(plan.writes) == 1
    assert plan.writes[0].amount == Decimal(2)
    assert plan.reply_text == "actually, 2 jars"


def test_self_review_keeps_original_plan_when_model_confirms(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _tool_round(
            "sumac_consume_inventory",
            {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
        ),
        _final_round("consumed 1 jar"),
        # self-review round: no further tool call, so the original plan stands.
        _final_round("confirmed, no changes"),
    ]
    agent, _fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("consume some jam")

    assert len(plan.writes) == 1
    assert plan.writes[0].amount == Decimal(1)
    assert plan.reply_text == "consumed 1 jar"


# --- revise (§13/§14 step 5) ----------------------------------------------


def test_revise_before_propose_raises(data_dir: Path, key: bytes, osuser: str) -> None:
    agent, _fake = _make_agent([], data_dir, key)
    with pytest.raises(RuntimeError):
        agent.revise("actually make it 2")


def test_revise_continues_after_propose(data_dir: Path, key: bytes, osuser: str) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        # propose
        _tool_round(
            "sumac_consume_inventory",
            {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
        ),
        _final_round("consumed 1 jar"),
        # self-review of propose: confirmed, no change
        _final_round("confirmed"),
        # revise
        _tool_round(
            "sumac_consume_inventory",
            {"product_id": "jam", "amount": "2", "unit": "jar", "from_location": "pantry"},
        ),
        _final_round("updated to 2 jars"),
        # self-review of revise: confirmed, no change
        _final_round("confirmed"),
    ]
    agent, _fake = _make_agent(responses, data_dir, key)

    proposed = agent.propose("consume some jam")
    plan = agent.revise("actually make it 2 jars")

    assert len(plan.writes) == 1
    assert plan.writes[0].amount == Decimal(2)

    # revise()'s trace covers only its own round, not propose()'s — each
    # returned AgentPlan explains only the plan it is currently attached to.
    assert [Decimal(t.arguments["amount"]) for t in proposed.trace] == [Decimal(1)]
    assert [Decimal(t.arguments["amount"]) for t in plan.trace] == [Decimal(2)]


# --- commit (§14 step 5, §23) ---------------------------------------------


def test_commit_appends_writes_and_returns_summaries(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    agent, _fake = _make_agent([], data_dir, key)
    plan = llm.AgentPlan(
        reply_text="",
        writes=(
            llm.ProposedWrite(
                kind=ChangeKind.CONSUMPTION,
                product_id="jam",
                amount=Decimal(1),
                unit="jar",
                from_location="pantry",
                to_location=None,
            ),
        ),
    )

    summaries = agent.commit(plan)

    assert summaries == ["Recorded consumption of 1 jar jam"]
    assert ledger.build_inventory(data_dir, key).at("pantry")["jam"].amount == Decimal(2)


def test_commit_reraises_rejected_uncaught(data_dir: Path, key: bytes, osuser: str) -> None:
    """§23: a `Rejected` at commit time is not a modeled outcome the model
    gets to react to — it propagates, unlike a `Rejected` from a tool
    callback during propose/revise."""
    agent, _fake = _make_agent([], data_dir, key)
    plan = llm.AgentPlan(
        reply_text="",
        writes=(
            llm.ProposedWrite(
                kind=ChangeKind.CONSUMPTION,
                product_id="jam",
                amount=Decimal(1),
                unit="jar",
                from_location="nonexistent-location",
                to_location=None,
            ),
        ),
    )

    with pytest.raises(Rejected):
        agent.commit(plan)


def test_commit_re_decides_against_fresh_state_not_stale_snapshot(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """§14's "shelf is authoritative, not the log, even between preview and
    accept": commit re-runs `decide_change` against current state rather
    than replaying the `Write`s computed during `propose`. Simulated here by
    reducing stock between propose-equivalent plan construction and commit —
    the shortfall reconciliation (§3.5) should still kick in at commit time."""
    _seed_pantry_with_jam(data_dir, key, osuser)
    agent, _fake = _make_agent([], data_dir, key)
    plan = llm.AgentPlan(
        reply_text="",
        writes=(
            llm.ProposedWrite(
                kind=ChangeKind.CONSUMPTION,
                product_id="jam",
                amount=Decimal(3),
                unit="jar",
                from_location="pantry",
                to_location=None,
            ),
        ),
    )

    # Stock drops to 1 jar after the plan was built, before commit.
    _apply_change(
        data_dir,
        key,
        osuser,
        kind=ChangeKind.CONSUMPTION,
        product_id="jam",
        amount=Decimal(2),
        unit="jar",
        from_location="pantry",
        to_location=None,
    )
    assert ledger.build_inventory(data_dir, key).at("pantry")["jam"].amount == Decimal(1)

    summaries = agent.commit(plan)

    assert summaries == ["Recorded consumption of 3 jar jam"]
    # Shortfall reconciliation adjusted the shelf up to 3 before consuming,
    # landing at 0 (removed from the dict) rather than going negative.
    assert "jam" not in ledger.build_inventory(data_dir, key).at("pantry")


# --- prompt/schema content ------------------------------------------------


def test_prompt_and_schema_do_not_worked_example_specific_products() -> None:
    """A worked example naming real products in the prompt or tool
    description (e.g. "Salted Butter is butter, Peanut Butter isn't")
    would hand the model the answer to exactly the semantic judgment it's
    supposed to make on its own — the model should apply category
    reasoning generically, not recognize a memorized pairing. Guards
    against reintroducing one; not exhaustive, just the specific products
    this reasoning was worked out against during development."""
    text = llm.SYSTEM_PROMPT + json.dumps(llm._FIND_INVENTORY_SCHEMA)
    lowered = text.lower()
    for word in ("butter", "peanut", "croissant", "milk", "chocolate"):
        assert word not in lowered, f"{word!r} found in prompt/schema text — worked example leaked"


# --- §40: per-model tool-call rendering ------------------------------------


def test_render_tool_call_qwen_splices_raw_arguments_verbatim() -> None:
    """§28: `raw_arguments` (the model's own emitted JSON string) must be
    spliced in as-is, not re-serialized from `arguments` — re-dumping
    could reorder keys or change whitespace relative to what Qwen3's own
    chat template would have produced from a real `tool_calls` field."""
    result = llm._render_tool_call("sumac_find_inventory", {"query": "jam"}, '{"query":   "jam"}')
    expected = (
        '<tool_call>\n{"name": "sumac_find_inventory", "arguments": '
        '{"query":   "jam"}}\n</tool_call>'
    )
    assert result == expected


def test_render_tool_call_lfm_uses_pythonic_syntax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "TOOL_CALL_FORMAT", llm.ToolCallFormat.LFM)

    result = llm._render_tool_call(
        "sumac_consume_inventory",
        {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
        "unused for LFM",
    )

    assert result == (
        '<|tool_call_start|>[sumac_consume_inventory(product_id="jam", amount="1", '
        'unit="jar", from_location="pantry")]<|tool_call_end|>'
    )


def test_render_tool_call_lfm_escapes_embedded_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "TOOL_CALL_FORMAT", llm.ToolCallFormat.LFM)

    result = llm._render_tool_call("sumac_find_inventory", {"query": 'jam "special"'}, "unused")

    assert (
        result
        == '<|tool_call_start|>[sumac_find_inventory(query="jam \\"special\\"")]<|tool_call_end|>'
    )


def test_run_loop_appends_lfm_formatted_assistant_message_when_configured(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, key: bytes, osuser: str
) -> None:
    """Confirms `TOOL_CALL_FORMAT` actually reaches `_run_loop`'s message
    history, not just that `_render_tool_call` produces the right string
    in isolation — `mistralrs.ChatCompletionRequest` is an opaque Rust
    object with no readable `.messages` attribute, so this checks
    `AgentRunner`'s own accumulated `self._messages` instead."""
    monkeypatch.setattr(llm, "TOOL_CALL_FORMAT", llm.ToolCallFormat.LFM)
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _tool_round("sumac_find_inventory", {"query": "jam"}),
        _final_round("the jam is in the pantry"),
    ]
    agent, _fake = _make_agent(responses, data_dir, key)

    agent.propose("where is the jam?")

    assert agent._messages is not None
    tool_call_messages = [
        m
        for m in agent._messages
        if m["role"] == "assistant" and "<|tool_call_start|>" in m["content"]
    ]
    assert len(tool_call_messages) == 1
    assert "sumac_find_inventory" in tool_call_messages[0]["content"]
    assert "<tool_call>" not in tool_call_messages[0]["content"]
