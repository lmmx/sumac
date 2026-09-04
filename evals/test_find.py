"""FIND scenarios — locating existing inventory, or correctly finding
nothing. See docs/journal/2026-09-02-eval-suite.md. Needs a cached GGUF;
see evals/README.md.
"""

from __future__ import annotations

import pytest

from evals.evaluators import (
    evaluate_classification,
    evaluate_no_writes,
    evaluate_only_tools,
    evaluate_reply_mentions,
    evaluate_reply_order,
    evaluate_tools,
)
from sumac.llm import QueryKind

_CATEGORY = "find"
pytestmark = pytest.mark.model


def test_existing_item(agent, cfg, result) -> None:
    plan = agent.propose("where is the strawberry jam?")
    evaluate_classification(result, plan, QueryKind.FIND)
    evaluate_tools(result, plan, called=("sumac_find_inventory",))
    evaluate_no_writes(result, plan)
    evaluate_reply_mentions(result, plan, "white unit r3c1")
    assert result.passed, result.failures


def test_missing_item(agent, cfg, result) -> None:
    plan = agent.propose("where is the caviar?")
    evaluate_classification(result, plan, QueryKind.FIND)
    evaluate_tools(result, plan, called=("sumac_find_inventory",))
    evaluate_no_writes(result, plan)
    assert result.passed, result.failures


def test_quantity(agent, cfg, result) -> None:
    plan = agent.propose("how much basmati rice do we have?")
    evaluate_classification(result, plan, QueryKind.FIND)
    evaluate_tools(result, plan, called=("sumac_find_inventory",))
    evaluate_no_writes(result, plan)
    evaluate_reply_mentions(result, plan, "1")
    evaluate_reply_mentions(result, plan, "jug")
    assert result.passed, result.failures


def test_shared_word_picks_right_product(agent, cfg, result) -> None:
    """`ledger.search_inventory` returns both "Salted Butter" (the right
    answer) and "Butter Beans" (a decoy sharing the word "butter") as
    whole-word matches — this is the wrong-product-from-unranked-results
    failure recorded in docs/journal/2026-09-01-ask-agent-design.md."""
    plan = agent.propose("do we have any butter?")
    evaluate_classification(result, plan, QueryKind.FIND)
    evaluate_no_writes(result, plan)
    evaluate_reply_order(result, plan, first="salted butter", not_before="butter beans")
    assert result.passed, result.failures


def test_uses_only_find_tool(agent, cfg, result) -> None:
    plan = agent.propose("where is the ragu?")
    evaluate_classification(result, plan, QueryKind.FIND)
    evaluate_no_writes(result, plan)
    evaluate_only_tools(result, plan, {"sumac_find_inventory"})
    assert result.passed, result.failures
