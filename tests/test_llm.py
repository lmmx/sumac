"""Orchestration (`AgentRunner`) driven by a hand-built fake in place of
`mistralrs.Runner` — no real model, no GGUF download. The fake still drives
real `decide.decide_change`/`store.append` calls against a real encrypted
`data_dir`/`key`, via `AgentRunner`'s actual tool callbacks.

`AgentRunner` drives a client-side tool-calling loop itself (see the module
docstring in `sumac/llm.py`) rather than registering callbacks on `Runner` —
so one scripted `ScriptedResponse` here is one `send_chat_completion_request`
*round*, not a whole multi-tool-call turn: a tool call the loop must
dispatch itself, or a final plain-text reply with no further tool call.

`propose()` always spends its first round classifying the request into a
`QueryKind` before running the domain tool loop — every test driving
`propose()` through the fake must script that round first, via
`_classify_round(kind)`, or the fake has nothing to pop for it."""

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


def _classify_round(kind: str) -> ScriptedResponse:
    return _tool_round("classify_request", {"kind": kind})


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
    responses: list[ScriptedResponse],
    data_dir: Path,
    key: bytes,
    *,
    model: llm.ModelPreset = llm.DEFAULT_MODEL_PRESET,
) -> tuple[llm.AgentRunner, FakeRunner]:
    fake = FakeRunner(responses)
    agent = llm.AgentRunner(data_dir, key, model=model, runner=fake)
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
    """Nothing is withheld or excluded here — every matching product comes
    back, grouped, ordered by `ledger.search_inventory`'s tier (exact, then
    whole-word, then substring), each tagged `is_exact_match`. Relevance
    judgment (is "Butternut Box" plausibly what someone means by "butter"?)
    is left entirely to the model — this only checks the deterministic data
    shape it's given to reason over."""
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


# --- classification --------------------------------------------------------


def test_classify_reject_short_circuits_without_running_domain_loop(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    responses = [_classify_round("reject")]
    agent, fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("what's the capital of France?")

    assert plan.writes == ()
    assert plan.reply_text == llm._REJECT_REPLY
    # Only the classification round fired — no domain tool schema was ever
    # sent, since there is no kind to scope one to.
    assert len(fake.requests) == 1
    assert [t.name for t in plan.trace] == ["classify_request"]


def test_revise_after_reject_raises(data_dir: Path, key: bytes, osuser: str) -> None:
    responses = [_classify_round("reject")]
    agent, _fake = _make_agent(responses, data_dir, key)
    agent.propose("what's the capital of France?")

    with pytest.raises(RuntimeError):
        agent.revise("no really, find something")


def test_find_request_scopes_tool_schemas_to_find_only(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """`mistralrs.ChatCompletionRequest` is an opaque Rust object with no
    readable `tool_schemas` attribute, so this checks the scoping
    `AgentRunner` itself computed and used to build every request in the
    round, rather than reaching into the request object."""
    responses = [_classify_round("find"), _final_round("the jam is in the pantry")]
    agent, _fake = _make_agent(responses, data_dir, key)

    agent.propose("where is the jam?")

    assert agent._allowed == {"sumac_find_inventory"}


def test_add_request_scopes_tool_schemas_to_find_and_discover(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    responses = [
        _classify_round("add"),
        _tool_round(
            "sumac_discover_inventory",
            {
                "product_id": "Moma Pistachio Milk",
                "amount": "6",
                "unit": "carton",
                "to_location": "pantry",
            },
        ),
        _final_round("added 6 cartons"),
        _final_round("confirmed"),
    ]
    config.add_location(data_dir, key, osuser, Location(id="pantry", name="Pantry"))
    agent, _fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("add 6 cartons of Moma pistachio milk to the pantry")

    assert agent._allowed == {"sumac_find_inventory", "sumac_discover_inventory"}
    assert len(plan.writes) == 1
    assert plan.writes[0].kind is ChangeKind.DISCOVERY


def test_remove_request_scopes_tool_schemas_to_find_consume_move(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    responses = [
        _classify_round("remove"),
        _final_round("nothing to remove"),
        # `_maybe_force_action` forces one more round when a mutating
        # request ends with no writes.
        _final_round("still nothing to remove"),
    ]
    agent, _fake = _make_agent(responses, data_dir, key)

    agent.propose("throw away the old jam")

    assert agent._allowed == {
        "sumac_find_inventory",
        "sumac_consume_inventory",
        "sumac_move_inventory",
    }


def test_tool_call_outside_current_kind_is_rejected_not_dispatched(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """A `find`-classified request only ever has `sumac_find_inventory` on
    the request, but a small model can still emit a call for a tool it
    wasn't given — this must not crash `_run_loop` or reach a domain
    callback outside the classified kind's scope."""
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _classify_round("find"),
        _tool_round(
            "sumac_consume_inventory",
            {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
        ),
        _final_round("done"),
    ]
    agent, _fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("where is the jam?")

    assert plan.writes == ()
    rejected = json.loads(plan.trace[-1].result)
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "tool_not_available"
    assert rejected["detail"] == {"name": "sumac_consume_inventory"}
    assert rejected["hint"] == llm._REJECTION_HINT
    # Not committed and not queued — the shelf is untouched.
    assert ledger.build_inventory(data_dir, key).at("pantry")["jam"].amount == Decimal(3)


# --- propose -------------------------------------------------------------


def test_propose_read_only_request_produces_no_writes(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    responses = [_classify_round("find"), _final_round("the jam is in the pantry")]
    agent, fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("where is the jam?")

    assert plan.writes == ()
    assert plan.reply_text == "the jam is in the pantry"
    # No writes proposed -> self-review has nothing to check, so only the
    # classify round and one domain round fire.
    assert len(fake.requests) == 2
    # The classify call still shows up in the trace, even with no domain
    # tool call behind the reply.
    assert [t.name for t in plan.trace] == ["classify_request"]


def test_propose_resolves_a_consume_call_into_a_pending_write(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _classify_round("remove"),
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
    # Nothing committed yet — still a dry run.
    assert ledger.build_inventory(data_dir, key).at("pantry")["jam"].amount == Decimal(3)
    assert len(fake.requests) == 5

    # The trace records the classification and both domain calls, in order,
    # with their raw results — not just the final "consumed 1 jar of jam"
    # the model happened to say.
    assert [t.name for t in plan.trace] == [
        "classify_request",
        "sumac_find_inventory",
        "sumac_consume_inventory",
    ]
    assert plan.trace[1].arguments == {"query": "jam"}
    find_result = json.loads(plan.trace[1].result)
    assert find_result["products"][0]["product_id"] == "jam"
    consume_result = json.loads(plan.trace[2].result)
    assert consume_result["status"] == "proposed"


def test_rejected_tool_call_is_reported_and_not_added_to_pending(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _classify_round("remove"),
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
        # `_maybe_force_action` forces one more round when a mutating
        # request ends with no writes.
        _final_round("still nothing recorded"),
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


def test_add_request_with_no_writes_forces_one_more_round(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """A real LFM2.5 run classified as "add" narrated the change it would
    make in plain text without ever calling `sumac_discover_inventory` —
    the classifier already decided a change was needed, so ending with no
    writes is not the normal "nothing to do" outcome `find` has; it's a
    miss. One forced follow-up round, and this one actually acts."""
    config.add_location(data_dir, key, osuser, Location(id="pantry", name="Pantry"))
    responses = [
        _classify_round("add"),
        _final_round("I would add 1 box of Widgets to the pantry."),
        _tool_round(
            "sumac_discover_inventory",
            {"product_id": "Widgets", "amount": "1", "unit": "box", "to_location": "pantry"},
        ),
        _final_round("Added."),
        # Writes are non-empty now, so self-review runs too.
        _final_round("Looks correct."),
    ]
    agent, fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("add 1 box of widgets to the pantry")

    assert len(plan.writes) == 1
    assert plan.writes[0].product_id == "Widgets"
    # classify + narrated-no-op round + forced round's two rounds + self-review.
    assert len(fake.requests) == 5


def test_add_request_still_producing_no_writes_after_the_forced_round_gives_up(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """The forced round is not unbounded — if the model still doesn't act,
    that becomes the human's call (feedback/regenerate/start over), not
    another automatic retry."""
    responses = [
        _classify_round("add"),
        _final_round("I would add it, but I won't."),
        _final_round("Still not adding it."),
    ]
    agent, fake = _make_agent(responses, data_dir, key)

    plan = agent.propose("add 1 box of widgets to the pantry")

    assert plan.writes == ()
    assert plan.reply_text == "Still not adding it."
    assert len(fake.requests) == 3


def test_missing_required_argument_names_what_was_missing_and_received(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """A real LFM2.5 run produced `_amount` instead of `amount` — this is
    the callback's response to that case: name exactly what's missing and
    what was actually received, rather than a bare `invalid_amount` with no
    hint the key itself was wrong."""
    _seed_pantry_with_jam(data_dir, key, osuser)
    agent, _fake = _make_agent([], data_dir, key)

    result = json.loads(
        agent.tool_callbacks["sumac_consume_inventory"](
            "sumac_consume_inventory",
            {
                "product_id": "jam",
                "_amount": "1",
                "unit": "jar",
                "from_location": "pantry",
            },
        )
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "missing_required_argument"
    assert result["detail"] == {
        "missing": ["amount"],
        "received": ["_amount", "from_location", "product_id", "unit"],
    }
    assert result["hint"] == llm._REJECTION_HINT


def test_repeating_an_identical_successful_write_is_not_queued_twice(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """A real LFM2.5 run repeated an already-successful `sumac_discover_
    inventory` call three more times, byte-for-byte — each repeat silently
    became a second, third, fourth full write, so accepting the resulting
    plan would have recorded four times the requested quantity. The second
    (and third, fourth, ...) identical call must not add another write."""
    config.add_location(data_dir, key, osuser, Location(id="pantry", name="Pantry"))
    agent, _fake = _make_agent([], data_dir, key)
    call_args = {
        "product_id": "Moma Pistachio Milk",
        "amount": "6",
        "unit": "carton",
        "to_location": "pantry",
    }

    first = json.loads(
        agent.tool_callbacks["sumac_discover_inventory"]("sumac_discover_inventory", call_args)
    )
    second = json.loads(
        agent.tool_callbacks["sumac_discover_inventory"]("sumac_discover_inventory", call_args)
    )

    assert first["status"] == "proposed"
    assert second == {
        "status": "already_proposed",
        "product_id": "Moma Pistachio Milk",
        "amount": "6",
        "unit": "carton",
        "from_location": None,
        "to_location": "pantry",
    }
    assert len(agent._pending) == 1


# --- self-review -----------------------------------------------------------


def test_self_review_replaces_plan_when_model_revises_it(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _classify_round("remove"),
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
        _classify_round("remove"),
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


# --- revise ------------------------------------------------------------


def test_revise_before_propose_raises(data_dir: Path, key: bytes, osuser: str) -> None:
    agent, _fake = _make_agent([], data_dir, key)
    with pytest.raises(RuntimeError):
        agent.revise("actually make it 2")


def test_revise_continues_after_propose(data_dir: Path, key: bytes, osuser: str) -> None:
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        # propose (classify once; revise does not reclassify)
        _classify_round("remove"),
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

    # revise()'s trace covers only its own round, not propose()'s (or its
    # classification, which revise() doesn't repeat) — each returned
    # AgentPlan explains only the plan it is currently attached to.
    consume_calls = [t for t in proposed.trace if t.name == "sumac_consume_inventory"]
    assert [Decimal(t.arguments["amount"]) for t in consume_calls] == [Decimal(1)]
    assert [t.name for t in plan.trace] == ["sumac_consume_inventory"]
    assert Decimal(plan.trace[0].arguments["amount"]) == Decimal(2)


# --- commit ------------------------------------------------------------


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
    """A `Rejected` at commit time is not a modeled outcome the model gets
    to react to — it propagates, unlike a `Rejected` from a tool callback
    during propose/revise."""
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
    """ "Shelf is authoritative, not the log, even between preview and
    accept": commit re-runs `decide_change` against current state rather
    than replaying the `Write`s computed during `propose`. Simulated here by
    reducing stock between propose-equivalent plan construction and commit —
    the shortfall reconciliation should still kick in at commit time."""
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


def test_prompts_and_schemas_do_not_worked_example_specific_products() -> None:
    """A worked example naming real products in a prompt or tool
    description (e.g. "Salted Butter is butter, Peanut Butter isn't") would
    hand the model the answer to exactly the semantic judgment it's
    supposed to make on its own — the model should apply category
    reasoning generically, not recognize a memorized pairing. Guards
    against reintroducing one; not exhaustive, just the specific products
    this reasoning was worked out against during development."""
    text = (
        llm.CLASSIFIER_PROMPT
        + "".join(llm._PROMPT_BY_KIND.values())
        + json.dumps(llm._FIND_INVENTORY_SCHEMA)
    )
    lowered = text.lower()
    for word in ("butter", "peanut", "croissant", "milk", "chocolate"):
        assert word not in lowered, f"{word!r} found in prompt/schema text — worked example leaked"


def test_add_prompt_directs_dropping_the_brand_not_the_product_name() -> None:
    """A real LFM2.5 run searched "Moma pistachio milk" then "Moma" —
    narrowing by keeping the one word guaranteed absent from the catalog
    (the invented brand) and dropping the actual product description —
    then invented a new product for something already in inventory under a
    different name ("Pistachio Oat Milk"), fragmenting one product's stock
    across two catalog entries. Guards against losing the fix: the prompt
    must direct dropping the brand and keeping the product name, not
    either instruction alone or the reverse pairing."""
    text = " ".join(llm._ADD_PROMPT.split())
    assert "brand dropped" in text
    assert "product itself kept" in text
    assert "not a new product" in text


def test_add_prompt_directs_searching_for_context_before_guessing_a_location() -> None:
    """Three separate real-model runs of "add 6 Moma pistachio milk cartons
    to the same pantry cupboard as existing stock" (LFM2.5, Qwen3-1.7B,
    Qwen3-4B) all skipped or under-executed searching for the referenced
    "existing stock" and guessed a location string ("pantry cupboard")
    from the person's own wording instead — rejected as unknown, then
    handled reactively rather than avoided. Guards against losing the
    proactive instruction: search first for wording tying the location to
    something already in inventory, never guess in that case."""
    text = " ".join(llm._ADD_PROMPT.split())
    assert "existing stock" in text
    assert "never guess a location" in text


def test_rejected_hint_travels_with_the_rejection_not_the_prompt() -> None:
    """The original single-prompt design told the model how to react to a
    "rejected" tool result unconditionally, on every request — a real
    Qwen3 run showed the cost of dropping that instruction (it narrated an
    intention to retry, "Let me try again", without making one, then
    stopped), but restating it in every static prompt means every request
    pays for it whether or not a rejection ever happens. Instead the hint
    travels inside the rejected result itself (`_rejected`), so it only
    ever reaches the model at the moment it's relevant — the static
    prompts should carry none of this instruction."""
    for prompt in (llm._FIND_PROMPT, llm._ADD_PROMPT, llm._REMOVE_PROMPT):
        assert "rejected" not in prompt.lower()

    result = json.loads(llm._rejected("some_reason", {"detail": "value"}))
    assert result["hint"] == llm._REJECTION_HINT


# --- per-model tool-call rendering ------------------------------------


def test_render_tool_call_qwen_splices_raw_arguments_verbatim() -> None:
    """`raw_arguments` (the model's own emitted JSON string) must be spliced
    in as-is, not re-serialized from `arguments` — re-dumping could reorder
    keys or change whitespace relative to what Qwen3's own chat template
    would have produced from a real `tool_calls` field."""
    result = llm._render_tool_call(
        "sumac_find_inventory", {"query": "jam"}, '{"query":   "jam"}', llm.ToolCallFormat.QWEN
    )
    expected = (
        '<tool_call>\n{"name": "sumac_find_inventory", "arguments": '
        '{"query":   "jam"}}\n</tool_call>'
    )
    assert result == expected


def test_render_tool_call_lfm_uses_pythonic_syntax() -> None:
    result = llm._render_tool_call(
        "sumac_consume_inventory",
        {"product_id": "jam", "amount": "1", "unit": "jar", "from_location": "pantry"},
        "unused for LFM",
        llm.ToolCallFormat.LFM,
    )

    assert result == (
        '<|tool_call_start|>[sumac_consume_inventory(product_id="jam", amount="1", '
        'unit="jar", from_location="pantry")]<|tool_call_end|>'
    )


def test_render_tool_call_lfm_escapes_embedded_quotes() -> None:
    result = llm._render_tool_call(
        "sumac_find_inventory", {"query": 'jam "special"'}, "unused", llm.ToolCallFormat.LFM
    )

    assert (
        result
        == '<|tool_call_start|>[sumac_find_inventory(query="jam \\"special\\"")]<|tool_call_end|>'
    )


def test_render_tool_call_gemma_uses_call_colon_syntax() -> None:
    """Pins down this module's own best-effort reconstruction of Gemma 4's
    tool-call syntax (see `ToolCallFormat.GEMMA`'s docstring — it's not a
    verified byte-exact match against a real chat template the way QWEN and
    LFM are). This test only guards against a regression in what
    `_render_tool_call` itself does, not that Gemma 4 actually expects it."""
    result = llm._render_tool_call(
        "sumac_find_inventory", {"query": "jam"}, "unused for Gemma", llm.ToolCallFormat.GEMMA
    )

    assert result == "<|tool_call>call:sumac_find_inventory{query:jam}<tool_call|>"


def test_run_loop_appends_lfm_formatted_assistant_message_when_configured(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """Confirms the active `ModelPreset`'s `tool_call_format` actually
    reaches `_run_loop`'s message history, not just that `_render_tool_call`
    produces the right string in isolation — `mistralrs.ChatCompletionRequest`
    is an opaque Rust object with no readable `.messages` attribute, so this
    checks `AgentRunner`'s own accumulated `self._messages` instead."""
    # A throwaway preset, not a real registry lookup — this test only needs
    # `tool_call_format=LFM` to reach `_run_loop`; it never touches a real
    # GGUF, and the registry itself (`llm.MODEL_PRESETS`) may not carry an
    # LFM-format entry at all (see docs/journal/2026-09-02-eval-suite.md).
    lfm_preset = llm.ModelPreset("test-lfm", "unused/repo", "unused.gguf", llm.ToolCallFormat.LFM)
    _seed_pantry_with_jam(data_dir, key, osuser)
    responses = [
        _classify_round("find"),
        _tool_round("sumac_find_inventory", {"query": "jam"}),
        _final_round("the jam is in the pantry"),
    ]
    agent, _fake = _make_agent(responses, data_dir, key, model=lfm_preset)

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


# --- sampling configuration (docs/journal/2026-09-02-eval-suite.md) ------


def test_classify_public_alias_delegates_to_private_method(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    agent, _fake = _make_agent([_classify_round("find")], data_dir, key)

    assert agent.classify("where is the jam?") is llm.QueryKind.FIND


def test_build_request_passes_default_sampling_config(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    """`ChatCompletionRequest` is an opaque PyO3 object — none of its fields
    are readable back off a real instance (unlike the `@dataclass` its own
    `.pyi` stub decorates it with, which describes construction, not the
    runtime type). Capturing the kwargs `_build_request` passes, via a
    stand-in swapped in for `mistralrs.ChatCompletionRequest` itself, is the
    only way to confirm what reached it."""
    captured: dict = {}

    class _CapturingRequest:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(llm.mistralrs, "ChatCompletionRequest", _CapturingRequest)
    try:
        agent, _fake = _make_agent([], data_dir, key)
        agent._build_request([{"role": "user", "content": "hi"}], [])
    finally:
        monkeypatch.undo()

    assert captured["temperature"] == llm.DEFAULT_TEMPERATURE
    assert captured["top_p"] == llm.DEFAULT_TOP_P
    assert captured["max_tokens"] == llm.DEFAULT_MAX_TOKENS


def test_build_request_passes_custom_sampling_config(
    data_dir: Path, key: bytes, osuser: str
) -> None:
    captured: dict = {}

    class _CapturingRequest:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(llm.mistralrs, "ChatCompletionRequest", _CapturingRequest)
    try:
        fake = FakeRunner([])
        agent = llm.AgentRunner(
            data_dir, key, runner=fake, temperature=0.7, top_p=0.5, max_tokens=256
        )
        agent._build_request([{"role": "user", "content": "hi"}], [])
    finally:
        monkeypatch.undo()

    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.5
    assert captured["max_tokens"] == 256


def test_build_runner_passes_seed_to_mistralrs_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(llm.mistralrs, "Runner", _CapturingRunner)
    monkeypatch.setattr(
        llm.render.console, "print", lambda *a, **k: None
    )  # silence the loading message

    llm._build_runner(llm.DEFAULT_MODEL_PRESET, seed=12345)

    assert captured["seed"] == 12345


def test_build_runner_defaults_seed_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interactive `sumac ask` use passes no seed — only an eval run pins
    one, so that one epoch reproduces exactly from its seed alone."""
    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(llm.mistralrs, "Runner", _CapturingRunner)
    monkeypatch.setattr(llm.render.console, "print", lambda *a, **k: None)

    llm._build_runner(llm.DEFAULT_MODEL_PRESET)

    assert captured["seed"] is None
