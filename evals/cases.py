"""Expectation types, the hand-written adversarial case table, and
`EvalCase` assembly — see docs/journal/2026-09-02-eval-suite.md, "Case
generation". Generated cases come from `generate.py`; this module also
holds the seven `hard`-tagged cases, which run once, against `fam-01`
only, rather than once per family (see "counted once" in the spec)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from evals.vocab import FAMILIES
from sumac.llm import QueryKind
from sumac.models import ChangeKind

# --- expectation types ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class WriteSpec:
    kind: ChangeKind
    product_id: str
    amount: str
    unit: str
    from_location: str | None = None
    to_location: str | None = None


@dataclass(frozen=True, slots=True)
class NoWrites:
    """The correct outcome is an empty `AgentPlan.writes` — either because
    nothing should change (a `find`) or because a change was attempted and
    the domain layer correctly rejected it (`hard-unit-conflict` and
    `add.unit_collision`, both classified `add` with zero writes)."""


@dataclass(frozen=True, slots=True)
class Writes:
    specs: tuple[WriteSpec, ...]


@dataclass(frozen=True, slots=True)
class AskOrAct:
    """Both a clarifying question and a correct action are acceptable —
    see `scoring.classify_ask_or_act`. Scored as a reported branch, not a
    pass/fail, except that `"inaction"` counts as a failure."""


Expectation = NoWrites | Writes | AskOrAct


@dataclass(frozen=True, slots=True)
class TraceExpectation:
    called: tuple[str, ...] = ()
    not_called: tuple[str, ...] = ()
    max_calls: Mapping[str, int] = field(default_factory=dict)
    reply_mentions: tuple[str, ...] = ()
    reply_before: tuple[tuple[str, str], ...] = ()
    reply_amount: tuple[str, str] | None = None


_NO_TRACE_EXPECTATION = TraceExpectation()


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    family_id: str
    prompt: str
    kind: QueryKind
    expect: Expectation
    trace: TraceExpectation = _NO_TRACE_EXPECTATION
    tags: frozenset[str] = frozenset()
    template: str | None = None


# --- hand-written adversarial cases, fam-01 only --------------------------

_FAM1 = FAMILIES[0]


def _hard_cases() -> tuple[EvalCase, ...]:
    from evals import seed

    tomatoes_path = seed.location_path(seed.ROLE_LOCATIONS["unit_collision"])

    return (
        EvalCase(
            id="hard-unit-conflict",
            family_id=_FAM1.id,
            prompt=f"Add 1 can of {_FAM1.unit_collision.id} to {tomatoes_path}",
            kind=QueryKind.ADD,
            expect=NoWrites(),
            tags=frozenset({"hard"}),
        ),
        EvalCase(
            id="hard-ambiguous-product",
            family_id=_FAM1.id,
            prompt=(
                f"Add 1 can of {_FAM1.unit_collision.id.lower()}, along with the existing 3 cans"
            ),
            kind=QueryKind.ADD,
            expect=AskOrAct(),
            tags=frozenset({"hard", "ambiguous"}),
        ),
        EvalCase(
            id="hard-vague-move",
            family_id=_FAM1.id,
            prompt=f"move the {_FAM1.movement_source.id.lower()} to the fridge",
            kind=QueryKind.REMOVE,
            expect=AskOrAct(),
            tags=frozenset({"hard", "ambiguous"}),
        ),
        EvalCase(
            id="hard-joke-inventory-word",
            family_id=_FAM1.id,
            prompt=f"Tell me a joke about {_FAM1.unit_collision.id.lower()}",
            kind=QueryKind.REJECT,
            expect=NoWrites(),
            tags=frozenset({"hard"}),
        ),
        EvalCase(
            id="hard-gibberish",
            family_id=_FAM1.id,
            prompt="asdf",
            kind=QueryKind.REJECT,
            expect=NoWrites(),
            tags=frozenset({"hard"}),
        ),
        EvalCase(
            id="hard-duplicate-search-bait",
            family_id=_FAM1.id,
            prompt=(
                f"Add 6 {_FAM1.near_miss_brand.id} to the same pantry cupboard as existing stock"
            ),
            kind=QueryKind.ADD,
            expect=Writes(
                (
                    WriteSpec(
                        kind=ChangeKind.DISCOVERY,
                        product_id=_FAM1.near_miss_brand.id,
                        amount="6",
                        unit=_FAM1.near_miss_brand.unit,
                        to_location=seed.ROLE_LOCATIONS["near_miss_brand"],
                    ),
                )
            ),
            trace=TraceExpectation(max_calls={"sumac_find_inventory": 2}),
            tags=frozenset({"hard"}),
        ),
        EvalCase(
            id="hard-odd-destination",
            family_id=_FAM1.id,
            prompt="Add 1 box of Pizza Express Margherita Pizza to the fridge bottle rack",
            kind=QueryKind.ADD,
            expect=Writes(
                (
                    WriteSpec(
                        kind=ChangeKind.DISCOVERY,
                        product_id="Pizza Express Margherita Pizza",
                        amount="1",
                        unit="box",
                        to_location="fridge-bottle-rack",
                    ),
                )
            ),
            tags=frozenset({"hard"}),
        ),
    )


HARD_CASES: tuple[EvalCase, ...] = _hard_cases()


def all_cases(*, families: tuple = FAMILIES) -> tuple[EvalCase, ...]:
    """Generated cases over `families`, plus the hand-written `hard` cases
    (always against `fam-01`, regardless of `families` — dropping `fam-01`
    from a `--families N` development-loop subset would otherwise silently
    drop the adversarial table too)."""
    from evals.generate import generate_cases

    generated = tuple(case for family in families for case in generate_cases(family))
    return generated + HARD_CASES
