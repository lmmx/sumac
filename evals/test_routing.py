"""Classification only — one model call per case via `AgentRunner.classify()`
(`src/sumac/llm.py`'s public alias for `_classify`), not a full `propose()`,
so the confusion matrix is cheap enough to run on its own. Always at
temperature 0: the classifier is a four-way decision whose boundary is
what's being measured, not its sampling variance — see
docs/journal/2026-09-02-eval-suite.md, "Sampling configuration".

Model-gated (`pytest.mark.model`) — never runs without `eval_runner`
successfully loading a real GGUF. Unverified against a real model as of
this suite's initial implementation: see docs/journal/2026-09-02-eval-suite.md.
"""

from __future__ import annotations

from collections import Counter

import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def routing_cases(family_fixtures: dict[str, tuple]):
    from evals.cases import all_cases
    from evals.vocab import FAMILIES_BY_ID

    families = tuple(FAMILIES_BY_ID[fid] for fid in family_fixtures)
    return all_cases(families=families)


@pytest.fixture(scope="module")
def routing_results(
    routing_cases,
    family_fixtures: dict[str, tuple],
    eval_runner,
    eval_results_collector: list[dict],
):
    from sumac import llm

    base_runner, model = eval_runner
    rows = []
    for case in routing_cases:
        data_dir, key = family_fixtures[case.family_id]
        agent = llm.AgentRunner(data_dir, key, model=model, runner=base_runner, temperature=0.0)
        actual = agent.classify(case.prompt)
        rows.append((case, actual))
        eval_results_collector.append(
            {
                "type": "routing",
                "case_id": case.id,
                "family_id": case.family_id,
                "template": case.template,
                "tags": sorted(case.tags),
                "expected_kind": case.kind.value,
                "actual_kind": actual.value,
                "correct": actual == case.kind,
            }
        )
    return rows


def test_classification_confusion_matrix(routing_results) -> None:
    """Reporting, not a gate — there is no established accuracy threshold
    yet for any preset. `report.py` aggregates this across epochs; this
    assertion only confirms every case actually classified to something,
    which fails loudly if `AgentRunner.classify()` raised instead."""
    assert len(routing_results) > 0
    confusion = Counter((c.kind.value, a.value) for c, a in routing_results)
    correct = sum(1 for c, a in routing_results if a == c.kind)
    total = len(routing_results)
    print(f"\nclassification accuracy: {correct}/{total} ({correct / total:.1%})")
    for (expected, actual), n in sorted(confusion.items()):
        marker = "" if expected == actual else "  <-- MISCLASSIFIED"
        print(f"  expected={expected:8s} actual={actual:8s}  n={n:3d}{marker}")


def test_reject_class_abstention(routing_results) -> None:
    """BFCL's split, tracked separately per the eval spec: irrelevance
    detection (does the model correctly abstain on a no-call case) and
    relevance detection (does it emit a call on a tool-requiring one). A
    single accuracy number hides this asymmetry — a model that abstains
    on everything and a model that never abstains can land on the same
    overall accuracy while failing in opposite, and very differently
    costly, ways."""
    from sumac import llm

    reject_rows = [(c, a) for c, a in routing_results if c.kind == llm.QueryKind.REJECT]
    non_reject_rows = [(c, a) for c, a in routing_results if c.kind != llm.QueryKind.REJECT]

    if reject_rows:
        abstention = sum(1 for _c, a in reject_rows if a == llm.QueryKind.REJECT) / len(reject_rows)
        print(
            f"\nabstention rate (reject cases correctly rejected): {abstention:.1%} "
            f"over {len(reject_rows)} cases"
        )
    if non_reject_rows:
        false_reject = sum(1 for _c, a in non_reject_rows if a == llm.QueryKind.REJECT) / len(
            non_reject_rows
        )
        print(
            f"false-reject rate (tool-requiring requests rejected): {false_reject:.1%} "
            f"over {len(non_reject_rows)} cases"
        )
