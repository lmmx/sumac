"""Full `propose()`, scored against expected writes and traces — see
docs/journal/2026-09-02-eval-suite.md, "Scoring". Runs at
`--eval-temperature` (default 0.7): unlike `test_routing.py`, this module
exists to measure the agent's reliability under real sampling variance,
which `run.py` repeats across independently-seeded sessions to get pass^k
— see "Epochs are separate pytest sessions" in the eval spec.

Model-gated (`pytest.mark.model`). Unverified against a real model as of
this suite's initial implementation.
"""

from __future__ import annotations

import pytest

from evals import scoring
from evals.cases import AskOrAct, EvalCase, NoWrites, Writes

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def proposal_cases(family_fixtures: dict[str, tuple]):
    from evals.cases import all_cases
    from evals.vocab import FAMILIES_BY_ID

    families = tuple(FAMILIES_BY_ID[fid] for fid in family_fixtures)
    return all_cases(families=families)


def _actual_kind(plan):
    from sumac.llm import QueryKind

    if not plan.trace or plan.trace[0].name != "classify_request":
        return None
    kind_str = plan.trace[0].arguments.get("kind")
    if kind_str is None:
        return QueryKind.REJECT
    try:
        return QueryKind(kind_str)
    except ValueError:
        return QueryKind.REJECT


def _score_outcome(case: EvalCase, plan, cfg) -> tuple[bool, float | None, str | None]:
    """Returns `(outcome_ok, write_f1, ask_or_act_branch)` — `write_f1` is
    `None` for an `AskOrAct` case (there's no write set to score), and
    `ask_or_act_branch` is `None` for every other case."""
    if isinstance(case.expect, NoWrites):
        result = scoring.score_no_writes(plan.writes)
        return result.exact_match, result.f1, None
    if isinstance(case.expect, Writes):
        result = scoring.score_writes(plan.writes, case.expect.specs, cfg)
        return result.exact_match, result.f1, None
    if isinstance(case.expect, AskOrAct):
        branch = scoring.classify_ask_or_act(plan)
        return branch in ("ask", "act"), None, branch
    raise TypeError(f"unhandled expectation type: {type(case.expect)}")  # pragma: no cover


@pytest.fixture(scope="module")
def proposal_results(
    proposal_cases,
    family_fixtures: dict[str, tuple],
    eval_runner,
    eval_results_collector: list[dict],
    request: pytest.FixtureRequest,
):
    from sumac import config as sumac_config
    from sumac import llm

    base_runner, model = eval_runner
    temperature = request.config.getoption("--eval-temperature")
    rows = []
    for case in proposal_cases:
        data_dir, key = family_fixtures[case.family_id]
        agent = llm.AgentRunner(
            data_dir, key, model=model, runner=base_runner, temperature=temperature
        )
        plan = agent.propose(case.prompt)
        cfg = sumac_config.build_config(data_dir, key)

        actual_kind = _actual_kind(plan)
        kind_ok = actual_kind == case.kind
        outcome_ok, write_f1, ask_or_act_branch = _score_outcome(case, plan, cfg)
        trace_result = scoring.check_trace(plan.trace, plan.reply_text, case.trace)
        passed = kind_ok and outcome_ok and trace_result.ok

        row = {
            "type": "proposal",
            "case_id": case.id,
            "family_id": case.family_id,
            "template": case.template,
            "tags": sorted(case.tags),
            "blocked": "blocked" in case.tags,
            "expected_kind": case.kind.value,
            "actual_kind": actual_kind.value if actual_kind else None,
            "kind_correct": kind_ok,
            "expect_type": type(case.expect).__name__,
            "outcome_ok": outcome_ok,
            "write_f1": write_f1,
            "ask_or_act_branch": ask_or_act_branch,
            "trace_ok": trace_result.ok,
            "trace_failures": list(trace_result.failures),
            "passed": passed,
            "reply_text": plan.reply_text,
        }
        rows.append(row)
        eval_results_collector.append(row)
    return rows


def test_proposal_summary_report(proposal_results) -> None:
    """Reporting, not a gate — no established pass-rate threshold exists
    yet for any preset. `report.py` aggregates `--eval-json` across epochs
    for pass^k; this only confirms every case produced a scoreable row."""
    assert len(proposal_results) > 0
    headline = [r for r in proposal_results if not r["blocked"]]
    blocked = [r for r in proposal_results if r["blocked"]]
    passed = sum(1 for r in headline if r["passed"])
    print(f"\nheadline: {passed}/{len(headline)} ({passed / len(headline):.1%})")
    if blocked:
        blocked_passed = sum(1 for r in blocked if r["passed"])
        print(
            f"blocked (excluded from headline): {blocked_passed}/{len(blocked)} "
            f"({blocked_passed / len(blocked):.1%}) — see docs/journal/2026-09-02-eval-suite.md, "
            "location-reference taxonomy"
        )

    ask_or_act_rows = [r for r in proposal_results if r["expect_type"] == "AskOrAct"]
    if ask_or_act_rows:
        from collections import Counter

        branches = Counter(r["ask_or_act_branch"] for r in ask_or_act_rows)
        print(f"ask-vs-act branches: {dict(branches)}")

    hard_rows = [r for r in proposal_results if "hard" in r["tags"]]
    if hard_rows:
        failed_hard = [r["case_id"] for r in hard_rows if not r["passed"]]
        print(f"hard cases failing: {failed_hard or 'none'}")
