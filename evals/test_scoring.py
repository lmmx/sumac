"""Scorer self-tests, null baselines, and vocab hygiene — the only module
in this suite that runs without a GPU or a downloaded GGUF (see
docs/journal/2026-09-02-eval-suite.md, "Layout"). `test_null_baselines_*`
is the load-bearing one: it runs the full generated case table through
`AgentRunner.propose()` against each null-baseline stub, over real seeded
families, and prints each baseline's score — the check that a future
change to `cases.py`/`generate.py` can't quietly regress the case table
back to something a do-nothing agent would pass."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal

import pytest

from evals import scoring, seed
from evals.baselines import (
    BASELINES,
    BaselineResult,
    DoNothingRunner,
    RejectEverythingRunner,
    run_baseline,
)
from evals.cases import Writes, all_cases
from evals.vocab import FAMILIES, FAMILIES_BY_ID
from sumac import config as sumac_config
from sumac import llm


def _seeded_families(family_fixtures: dict[str, tuple]) -> tuple:
    """The `FamilyVocab`s actually seeded by the `family_fixtures` fixture
    — respects `--families N`, so a baseline run never looks up a family
    `--families` didn't build (`KeyError`)."""
    return tuple(FAMILIES_BY_ID[fid] for fid in family_fixtures)


# --- vocab hygiene ---------------------------------------------------------


def test_no_vocab_name_leaks_into_prompt_constants() -> None:
    """A product/brand/absent-product name that also appears in one of
    `sumac.llm`'s prompt constants would let a case measure recall of the
    prompt's own wording rather than the behaviour under test —
    `_ADD_PROMPT`'s worked example names "Heinz" and "Baked Beans"
    explicitly. See `vocab.py`'s module docstring."""
    prompt_text = " ".join(
        [llm.CLASSIFIER_PROMPT, llm._FIND_PROMPT, llm._ADD_PROMPT, llm._REMOVE_PROMPT]
    ).casefold()

    names: list[str] = []
    for family in FAMILIES:
        names += [
            family.unit_collision.id,
            family.near_miss_brand.id,
            family.discriminator_a.id,
            family.discriminator_b.id,
            family.shared_word_decoy.id,
            family.rice.id,
            family.consumption_target.id,
            family.movement_source.id,
            family.category_stocked.id,
            family.category_new_name,
            family.absent_product,
            family.shared_word,
            family.category_word,
            family.rice_new_unit,
            family.rice_size,
        ]

    leaked = [name for name in names if name.casefold() in prompt_text]
    assert leaked == [], f"vocab names leak into a prompt constant: {leaked}"


def test_case_ids_are_unique() -> None:
    cases = all_cases()
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    assert len(cases) > 0


def test_reject_prompts_are_distinct_per_family_not_replicated() -> None:
    """The first draft drew reject prompts from one family-independent
    list, replicated across every family — thirty rows over about three
    distinct items, inflating `n` for no real gain. Each family's own list
    must be genuinely different from every other family's."""
    all_prompt_sets = [frozenset(f.reject_prompts) for f in FAMILIES]
    assert len(set(all_prompt_sets)) == len(all_prompt_sets)


# --- location_path sanity ---------------------------------------------------


def test_location_path_matches_seeded_config(
    family_fixtures: dict[str, tuple],
) -> None:
    data_dir, key = family_fixtures[FAMILIES[0].id]
    cfg = sumac_config.build_config(data_dir, key)
    for location_id in ("pantry-white-unit-r2c3", "fridge-main-shelf-2", "freezer-drawer-1"):
        assert seed.location_path(location_id) == sumac_config.location_path(
            cfg.known_locations, location_id
        )


def test_seeding_writes_stay_inside_eval_root(family_fixtures: dict[str, tuple], eval_root) -> None:
    """The guard in `conftest._guard_store_append` (rail 3) is exercised
    for real by every family's seeding — this just confirms that legitimate
    usage doesn't trip it, i.e. it isn't a guard that happens to never
    fire."""
    for data_dir, _key in family_fixtures.values():
        assert data_dir.resolve().is_relative_to(eval_root.resolve())


# --- scoring: canonicalisation -----------------------------------------------


def test_canonical_unit_folds_plurals_via_explicit_table_not_stemming() -> None:
    assert scoring.canonical_unit("boxes") == "box"
    assert scoring.canonical_unit("box") == "box"
    assert scoring.canonical_unit("cans") == "can"
    assert scoring.canonical_unit("jars") == "jar"
    # A unit outside the table compares verbatim rather than being stemmed.
    assert scoring.canonical_unit("fillets") == "fillets"


def test_canonical_location_resolves_id_and_display_path(
    family_fixtures: dict[str, tuple],
) -> None:
    data_dir, key = family_fixtures[FAMILIES[0].id]
    cfg = sumac_config.build_config(data_dir, key)

    assert scoring.canonical_location(cfg, "pantry-white-unit-r2c3") == "pantry-white-unit-r2c3"
    assert scoring.canonical_location(cfg, "Pantry > White Unit R2C3") == "pantry-white-unit-r2c3"
    # Unresolvable — matches `ProposedWrite`'s raw, unresolved value; scores
    # as a mismatch rather than raising.
    assert scoring.canonical_location(cfg, "Nowhere In Particular") == "Nowhere In Particular"
    assert scoring.canonical_location(cfg, None) is None


def test_score_writes_exact_match_and_f1(family_fixtures: dict[str, tuple]) -> None:
    data_dir, key = family_fixtures[FAMILIES[0].id]
    cfg = sumac_config.build_config(data_dir, key)
    from evals.cases import WriteSpec
    from sumac.models import ChangeKind

    expected = (WriteSpec(ChangeKind.DISCOVERY, "Widget", "1", "box", to_location="storage"),)
    actual_correct = (
        llm.ProposedWrite(
            kind=ChangeKind.DISCOVERY,
            product_id="Widget",
            amount=Decimal("1"),
            unit="boxes",
            from_location=None,
            to_location="storage",
        ),
    )
    result = scoring.score_writes(actual_correct, expected, cfg)
    assert result.exact_match is True
    assert result.f1 == 1.0

    actual_wrong = (
        llm.ProposedWrite(
            kind=ChangeKind.DISCOVERY,
            product_id="Different Widget",
            amount=Decimal("1"),
            unit="boxes",
            from_location=None,
            to_location="storage",
        ),
    )
    result_wrong = scoring.score_writes(actual_wrong, expected, cfg)
    assert result_wrong.exact_match is False
    assert result_wrong.f1 == 0.0

    result_empty = scoring.score_writes((), expected, cfg)
    assert result_empty.exact_match is False
    assert result_empty.f1 == 0.0

    result_no_expectation = scoring.score_no_writes(())
    assert result_no_expectation.exact_match is True


def test_check_trace_reply_before_allows_naming_decoy_to_dismiss_it() -> None:
    """`reply_excludes` (the first draft) would fail a *correct* reply that
    names the decoy in order to rule it out ("you have Salted Butter; the
    Butter Beans aren't butter"). `reply_before` only fails a reply that
    names the decoy *ahead of* the intended product."""
    from evals.cases import TraceExpectation

    expectation = TraceExpectation(reply_before=(("Salted Butter", "Butter Beans"),))

    @dataclass
    class _FakeToolCall:
        name: str

    good_reply = "You have Salted Butter; the Butter Beans aren't butter."
    result = scoring.check_trace((), good_reply, expectation)
    assert result.ok, result.failures

    bad_reply = "The Butter Beans are the only thing matching; Salted Butter is also there."
    result_bad = scoring.check_trace((), bad_reply, expectation)
    assert not result_bad.ok


def test_check_trace_reply_amount_requires_adjacency_not_bare_substring() -> None:
    from evals.cases import TraceExpectation

    expectation = TraceExpectation(reply_amount=("2", "jars"))
    vacuous_reply = "There are 2 items and separately R2C3 holds something else."
    assert not scoring.check_trace((), vacuous_reply, expectation).ok

    adjacent_reply = "You have 2 jars of it in the pantry."
    assert scoring.check_trace((), adjacent_reply, expectation).ok


def test_classify_ask_or_act_distinguishes_asking_from_inaction() -> None:
    @dataclass
    class _Call:
        name: str

    @dataclass
    class _Plan:
        writes: tuple
        reply_text: str
        trace: tuple

    acted = _Plan(writes=("something",), reply_text="Done.", trace=())
    assert scoring.classify_ask_or_act(acted) == "act"

    asked = _Plan(
        writes=(),
        reply_text="How many would you like, and which unit?",
        trace=(_Call("classify_request"), _Call("sumac_find_inventory")),
    )
    assert scoring.classify_ask_or_act(asked) == "ask"

    rambled = _Plan(
        writes=(),
        reply_text="I looked into this and it seems complicated to resolve here.",
        trace=(_Call("classify_request"),),
    )
    assert scoring.classify_ask_or_act(rambled) == "inaction"

    acted_then_described = _Plan(
        writes=(),
        reply_text="Should I add this? Let me check the discovery tool first.",
        trace=(_Call("classify_request"), _Call("sumac_discover_inventory")),
    )
    assert scoring.classify_ask_or_act(acted_then_described) == "inaction"


# --- null baselines, run through the real pipeline --------------------------
# `baselines.run_baseline`/`scoring.score_case` are the same functions
# `report.py` uses for a real model's score — one definition of "passed"
# shared across the null-floor check here and the real report there.


@pytest.fixture(scope="module")
def baseline_results(family_fixtures: dict[str, tuple]) -> dict[str, BaselineResult]:
    cases = all_cases(families=_seeded_families(family_fixtures))
    return {name: run_baseline(name, family_fixtures, cases) for name in BASELINES}


def test_null_baselines_never_pass_the_full_suite(
    baseline_results: dict[str, BaselineResult],
) -> None:
    for result in baseline_results.values():
        assert result.pass_rate < 0.5, (
            f"{result.name} scored {result.passed}/{result.total} "
            f"({result.pass_rate:.1%}) — the case table is too easy"
        )


def test_do_nothing_fails_every_writes_case(family_fixtures: dict[str, tuple]) -> None:
    cases = all_cases(families=_seeded_families(family_fixtures))
    writes_cases = [c for c in cases if isinstance(c.expect, Writes)]
    assert len(writes_cases) > 0
    for case in writes_cases:
        data_dir, key = family_fixtures[case.family_id]
        agent = llm.AgentRunner(data_dir, key, runner=DoNothingRunner())
        plan = agent.propose(case.prompt)
        assert plan.writes == ()


def test_reject_everything_fails_find_cases_via_trace_assertion(
    family_fixtures: dict[str, tuple],
) -> None:
    """The first draft's baseline check only asserted `RejectEverythingRunner`
    failed *somewhere* — which hid a 44% null floor, since it silently
    passed every `find` case by never writing anything. The trace
    assertion (`called=("sumac_find_inventory",)`) is what makes this
    baseline fail those cases now: it never calls the tool at all."""
    cases = [
        c
        for c in all_cases(families=_seeded_families(family_fixtures))
        if c.template in ("find.where", "find.quantity", "find.shared_word")
    ]
    assert len(cases) > 0
    for case in cases:
        data_dir, key = family_fixtures[case.family_id]
        agent = llm.AgentRunner(data_dir, key, runner=RejectEverythingRunner())
        plan = agent.propose(case.prompt)
        result = scoring.check_trace(plan.trace, plan.reply_text, case.trace)
        assert not result.ok, f"{case.id} unexpectedly passed against reject-everything"


def test_baseline_summary_report(
    baseline_results: dict[str, BaselineResult], family_fixtures: dict[str, tuple]
) -> None:
    """Not an assertion beyond `test_null_baselines_never_pass_the_full_suite`
    — prints the floor every baseline sets, per case kind, so it shows up
    in `pytest -s` output the same way `report.py` will show it beside a
    real model's score."""
    cases = all_cases(families=_seeded_families(family_fixtures))
    by_kind = Counter(c.kind.value for c in cases)
    lines = [f"\n{len(cases)} generated+hard cases, by expected kind: {dict(by_kind)}"]
    for result in baseline_results.values():
        lines.append(
            f"  {result.name:20s} writes/trace {result.passed:3d}/{result.total} "
            f"({result.pass_rate:5.1%})   kind-correct {result.kind_correct:3d}/{result.total} "
            f"({result.kind_correct / result.total:5.1%})"
        )
    print("\n".join(lines))
