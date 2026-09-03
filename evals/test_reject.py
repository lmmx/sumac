"""REJECT scenarios — out-of-domain and gibberish requests. See
docs/journal/2026-09-02-eval-suite.md. Needs a cached GGUF; see
evals/README.md.
"""

from __future__ import annotations

import pytest

from evals.evaluators import evaluate_classification, evaluate_no_writes
from sumac.llm import QueryKind

_CATEGORY = "reject"
pytestmark = pytest.mark.model


@pytest.fixture
def agent(agent_runner_factory):
    return agent_runner_factory()


def test_out_of_domain_weather(agent, cfg, result) -> None:
    plan = agent.propose("What's the weather in Edinburgh?")
    evaluate_classification(result, plan, QueryKind.REJECT)
    evaluate_no_writes(result, plan)
    assert result.passed, result.failures


def test_gibberish(agent, cfg, result) -> None:
    plan = agent.propose("asdf")
    evaluate_classification(result, plan, QueryKind.REJECT)
    evaluate_no_writes(result, plan)
    assert result.passed, result.failures


def test_joke_with_inventory_word(agent, cfg, result) -> None:
    """A word that also names a seeded product ("tomatoes") must not pull
    an out-of-domain request into the inventory-shaped kinds."""
    plan = agent.propose("Tell me a joke about tomatoes")
    evaluate_classification(result, plan, QueryKind.REJECT)
    evaluate_no_writes(result, plan)
    assert result.passed, result.failures
