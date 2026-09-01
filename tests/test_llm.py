"""§22: orchestration (`AgentRunner`) driven by a hand-built fake in place of
`mistralrs.Runner` — no real model, no GGUF download. The fake still drives
real `decide.decide_change`/`store.append` calls against a real encrypted
`data_dir`/`key`, via `AgentRunner`'s actual tool callbacks."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
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
class ScriptedTurn:
    """One `send_chat_completion_request` call's worth of scripted behavior:
    `tool_calls` are dispatched, in order, through `FakeRunner.tool_callbacks`
    (mimicking what mistral.rs's server-side loop does internally, one round
    at a time — see docs/journal/2026-09-01-ask-agent-design.md §7), then the
    turn ends with `final_content` as the assistant's plain-text reply."""

    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    final_content: str = ""


class FakeRunner:
    """A `llm.SendsCompletions`. `tool_callbacks` is wired in after
    construction — `AgentRunner.__init__` builds the callbacks closure before
    it has a runner to hand them to (real `mistralrs.Runner` takes them at
    construction time; this fake takes them after, since dispatch only
    happens once a request actually arrives)."""

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        self.tool_callbacks: dict[str, Callable[[str, dict], str]] = {}
        self._turns = list(turns)
        self.requests: list[object] = []

    def send_chat_completion_request(self, request, model_id=None):  # noqa: ANN001, ANN201
        self.requests.append(request)
        turn = self._turns.pop(0)
        for name, args in turn.tool_calls:
            self.tool_callbacks[name](name, args)
        message = SimpleNamespace(content=turn.final_content, role="assistant", tool_calls=None)
        choice = SimpleNamespace(finish_reason="stop", index=0, message=message)
        return SimpleNamespace(choices=[choice])


def _make_agent(
    turns: list[ScriptedTurn], data_dir: Path, key: bytes
) -> tuple[llm.AgentRunner, FakeRunner]:
    fake = FakeRunner(turns)
    agent = llm.AgentRunner(data_dir, key, runner=fake)
    fake.tool_callbacks = agent.tool_callbacks
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

    assert result["matches"] == [
        {
            "product_id": "jam",
            "location_id": "pantry",
            "location_path": "Pantry",
            "amount": "3",
            "unit": "jar",
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
    assert result["matches"] == []


# --- propose -------------------------------------------------------------


def test_propose_read_only_request_produces_no_writes(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    turns = [ScriptedTurn(tool_calls=[], final_content="the jam is in the pantry")]
    agent, fake = _make_agent(turns, data_dir, key)

    plan = agent.propose("where is the jam?")

    assert plan.writes == ()
    assert plan.reply_text == "the jam is in the pantry"
    # No writes proposed -> self-review (§13) has nothing to check, so only
    # the one request fires.
    assert len(fake.requests) == 1


def test_propose_resolves_a_consume_call_into_a_pending_write(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    turns = [
        ScriptedTurn(
            tool_calls=[
                ("sumac_find_inventory", {"query": "jam"}),
                (
                    "sumac_consume_inventory",
                    {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
                ),
            ],
            final_content="consumed 1 jar of jam",
        ),
        # self-review round: model is satisfied, no further tool calls.
        ScriptedTurn(tool_calls=[], final_content="looks correct"),
    ]
    agent, fake = _make_agent(turns, data_dir, key)

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
    assert len(fake.requests) == 2


def test_rejected_tool_call_is_reported_and_not_added_to_pending(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    turns = [
        ScriptedTurn(
            tool_calls=[
                (
                    "sumac_consume_inventory",
                    {
                        "product_id": "jam",
                        "amount": "1",
                        "unit": "jar",
                        "from_location": "nonexistent-location",
                    },
                )
            ],
            final_content="that location doesn't exist",
        )
    ]
    agent, _fake = _make_agent(turns, data_dir, key)

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
    turns = [
        ScriptedTurn(
            tool_calls=[
                (
                    "sumac_consume_inventory",
                    {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
                )
            ],
            final_content="consumed 1 jar",
        ),
        ScriptedTurn(
            tool_calls=[
                (
                    "sumac_consume_inventory",
                    {"product_id": "jam", "amount": "2", "unit": "jar", "from_location": "pantry"},
                )
            ],
            final_content="actually, 2 jars",
        ),
    ]
    agent, _fake = _make_agent(turns, data_dir, key)

    plan = agent.propose("consume some jam")

    assert len(plan.writes) == 1
    assert plan.writes[0].amount == Decimal(2)
    assert plan.reply_text == "actually, 2 jars"


def test_self_review_keeps_original_plan_when_model_confirms(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    turns = [
        ScriptedTurn(
            tool_calls=[
                (
                    "sumac_consume_inventory",
                    {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
                )
            ],
            final_content="consumed 1 jar",
        ),
        ScriptedTurn(tool_calls=[], final_content="confirmed, no changes"),
    ]
    agent, _fake = _make_agent(turns, data_dir, key)

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
    turns = [
        ScriptedTurn(
            tool_calls=[
                (
                    "sumac_consume_inventory",
                    {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
                )
            ],
            final_content="consumed 1 jar",
        ),
        ScriptedTurn(tool_calls=[], final_content="confirmed"),
        ScriptedTurn(
            tool_calls=[
                (
                    "sumac_consume_inventory",
                    {"product_id": "jam", "amount": "2", "unit": "jar", "from_location": "pantry"},
                )
            ],
            final_content="updated to 2 jars",
        ),
        ScriptedTurn(tool_calls=[], final_content="confirmed"),
    ]
    agent, _fake = _make_agent(turns, data_dir, key)

    agent.propose("consume some jam")
    plan = agent.revise("actually make it 2 jars")

    assert len(plan.writes) == 1
    assert plan.writes[0].amount == Decimal(2)


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
