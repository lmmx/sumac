"""REMOVE scenarios — consumption and movement, both classified `REMOVE`.
See docs/journal/2026-09-02-eval-suite.md. Needs a cached GGUF; see
evals/README.md.
"""

from __future__ import annotations

import pytest

from evals.evaluators import evaluate_ask_or_act, evaluate_classification, evaluate_write
from sumac.llm import QueryKind
from sumac.models import ChangeKind

_CATEGORY = "remove"
pytestmark = pytest.mark.model


def test_partial(agent, cfg, result) -> None:
    plan = agent.propose("I used 1 tub of Ragu")
    evaluate_classification(result, plan, QueryKind.REMOVE)
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.CONSUMPTION, product_id="Ragu",
        amount="1", unit="tub", from_location="freezer-drawer-3",
    )  # fmt: skip
    assert result.passed, result.failures


def test_all(agent, cfg, result) -> None:
    plan = agent.propose("we finished the strawberry jam")
    evaluate_classification(result, plan, QueryKind.REMOVE)
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.CONSUMPTION, product_id="Strawberry Jam",
        amount="1", unit="jar", from_location="pantry-white-unit-r3c1",
    )  # fmt: skip
    assert result.passed, result.failures


def test_move_explicit(agent, cfg, result) -> None:
    plan = agent.propose(
        "move 1 tub of Ragu from the third drawer of the big freezer to the fridge door"
    )
    evaluate_classification(result, plan, QueryKind.REMOVE)
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.MOVEMENT, product_id="Ragu",
        amount="1", unit="tub", from_location="freezer-drawer-3", to_location="fridge-door",
    )  # fmt: skip
    assert result.passed, result.failures


def test_move_vague_asks_or_acts(agent, cfg, result) -> None:
    """ "the ragu" with no amount is defensibly the full 2 tubs already
    stocked — unlike a missing-amount `add`, no field needs inventing to
    act correctly, so acting is an acceptable outcome alongside asking."""
    plan = agent.propose("move the ragu to the fridge")
    evaluate_classification(result, plan, QueryKind.REMOVE)
    evaluate_ask_or_act(result, plan)
    assert result.passed, result.failures
