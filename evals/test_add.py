"""ADD scenarios. See docs/journal/2026-09-02-eval-suite.md, in particular
the "FIND can be broad, ADD needs a concrete destination or enough
existing-stock context to infer one" asymmetry and the location-language
audit that followed it. Needs a cached GGUF; see evals/README.md.
"""

from __future__ import annotations

import pytest

from evals.evaluators import (
    evaluate_ask_or_act,
    evaluate_classification,
    evaluate_tools,
    evaluate_write,
)
from sumac.llm import QueryKind
from sumac.models import ChangeKind

_CATEGORY = "add"
pytestmark = pytest.mark.model


def test_existing_item_explicit_location(agent, cfg, result) -> None:
    """An explicit, non-inferred destination for an existing product. The
    fridge's main-shelf array is numbered "Shelf 1".."Shelf 4" — a real
    person would say "the second shelf", so this is honestly resolvable
    without the agent needing to have seen any hidden naming convention
    (unlike the pantry grid's "White Unit R2C3" ids, which have no natural
    spoken form at all — see the location-language audit in the journal
    for why a grid-cell version of this test was replaced)."""
    plan = agent.propose("Add 1 jar of Chopped Tomatoes to the second shelf of the fridge")
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Chopped Tomatoes",
        amount="1", unit="jar", to_location="fridge-main-shelf-2",
    )  # fmt: skip
    assert result.passed, result.failures


def test_existing_item_indirect_location(agent, cfg, result) -> None:
    plan = agent.propose(
        "Add 1 can of Ocado Italian Chopped Tomatoes to the pantry, same spot as existing stock"
    )
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_tools(result, plan, called=("sumac_find_inventory",))
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Ocado Italian Chopped Tomatoes",
        amount="1", unit="cans", to_location="pantry-white-unit-r2c3",
    )  # fmt: skip
    assert result.passed, result.failures


def test_missing_item_discovers_new_product(agent, cfg, result) -> None:
    """A genuinely new product has no existing stock to infer a location
    from, so this must name a concrete, naturally-nameable destination —
    "the fridge door" is the location's actual name, not an invented
    convention. "The pantry" alone would be underspecified: it's a
    grouping node in this fixture (nothing is ever stocked on it
    directly), not itself a storage location any product lives on."""
    plan = agent.propose("Add 2 bottles of Irn-Bru Zero to the fridge door")
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_tools(result, plan, called=("sumac_find_inventory",))
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Irn-Bru Zero",
        amount="2", unit="bottles", to_location="fridge-door",
    )  # fmt: skip
    assert result.passed, result.failures


def test_discriminator_variant_not_confused(agent, cfg, result) -> None:
    """Salted Butter and Unsalted Butter are seeded at different locations
    — no destination is named here at all, only "the existing stock",
    which forces the agent to search and resolve Unsalted Butter's own
    location rather than being handed it (a request naming the drawer
    outright wouldn't test whether it can tell the two products' locations
    apart in the first place)."""
    plan = agent.propose("Add 2 more packs of Unsalted Butter, with the existing stock")
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Unsalted Butter",
        amount="2", unit="packs", to_location="freezer-drawer-2",
    )  # fmt: skip
    assert result.passed, result.failures


def test_basmati_rice_in_different_unit(agent, cfg, result) -> None:
    """Product identity stays independent of unit — the same "Basmati
    Rice" can legitimately have both a jug and a bag registered. This
    fails today: `decide._resolve_product` has no registered bag-to-jug
    conversion, so `decide` rejects it — a real `decide.py`/`llm.py` gap
    (accept-with-confirmation would be preferable to a flat reject), left
    failing deliberately rather than "fixed" by weakening this assertion
    or inventing a new product identity."""
    plan = agent.propose("Add 1 bag of Basmati Rice (1kg) next to the existing jug of Basmati Rice")
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Basmati Rice", amount="1", unit="bag",
    )  # fmt: skip
    assert result.passed, result.failures


def test_odd_destination_respected(agent, cfg, result) -> None:
    """The location a person names, even an unusual one, is recorded as
    given rather than silently retargeted to a more sensible-sounding
    one."""
    plan = agent.propose("Add 1 box of Pizza Express Margherita Pizza to the fridge bottle rack")
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Pizza Express Margherita Pizza",
        amount="1", unit="box", to_location="fridge-bottle-rack",
    )  # fmt: skip
    assert result.passed, result.failures


def test_duplicate_search_bounded(agent, cfg, result) -> None:
    """The real Moma-pistachio-milk transcript
    (docs/journal/2026-09-01-ask-agent-design.md) made four
    `sumac_find_inventory` calls for one request before giving up."""
    plan = agent.propose(
        "Add 6 cans of Ocado Italian Chopped Tomatoes to the same pantry cupboard as existing stock"
    )
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_tools(result, plan, at_most={"sumac_find_inventory": 2})
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Ocado Italian Chopped Tomatoes",
        amount="6", unit="cans", to_location="pantry-white-unit-r2c3",
    )  # fmt: skip
    assert result.passed, result.failures


def test_product_with_omitted_amount(agent, cfg, result) -> None:
    """No amount or unit given — the agent is expected to infer a
    plausible default (e.g. "1 box") rather than ask, so this checks the
    right product lands in the right place, not which exact quantity it
    picked. The request names "the pantry" *and* "the other pasta" —
    `_ADD_PROMPT` instructs searching for the latter and using its
    location rather than guessing from the person's own wording in this
    case, so only Fusilli Pasta's own location is accepted, not the
    literally-named "pantry". This is the exact real query from
    docs/journal/2026-09-01-ask-agent-design.md, kept verbatim."""
    plan = agent.propose("Add Barilla Rigatoni to the pantry, with the other pasta")
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Barilla Rigatoni",
        to_location="pantry-white-unit-r2c1",
    )  # fmt: skip
    assert result.passed, result.failures


def test_multiple_products_with_omitted_amounts(agent, cfg, result) -> None:
    """Two distinct products, no amount for either — same inference
    expectation as `test_product_with_omitted_amount`, checked for both at
    once. "Butter" and "Butter Beans" are distinct products: neither
    "Salted Butter" nor "Unsalted Butter" is an exact match for "butter",
    so registering a new "Butter" is a fine outcome — what's checked is
    that the butter write isn't "Butter Beans", not which of the three
    plausible butter identities the model landed on."""
    plan = agent.propose("Add butter and jam to the pantry")
    evaluate_classification(result, plan, QueryKind.ADD)
    if len(plan.writes) != 2:
        result.check("writes", False, f"expected two writes (butter, jam), got {plan.writes!r}")
    else:
        result.check("writes", True)
        for w in plan.writes:
            result.check(f"kind:{w.product_id}", w.kind == ChangeKind.DISCOVERY)
            result.check(f"amount:{w.product_id}", w.amount > 0)
            result.check(f"unit:{w.product_id}", bool(w.unit.strip()))

        product_ids = [w.product_id.strip().lower() for w in plan.writes]
        butter = [p for p in product_ids if "butter" in p]
        jam = [p for p in product_ids if "jam" in p]
        result.check(
            "product:butter",
            len(butter) == 1,
            f"expected exactly one butter-related write, got {product_ids}",
        )
        if butter:
            result.check(
                "product:butter_not_beans",
                "beans" not in butter[0],
                f"registered against Butter Beans instead of butter: {butter[0]!r}",
            )
        result.check(
            "product:jam",
            len(jam) == 1,
            f"expected exactly one jam-related write, got {product_ids}",
        )
    assert result.passed, result.failures


def test_ambiguous_product_asks_or_acts(agent, cfg, result) -> None:
    """ "chopped tomatoes" (no brand) matches both the jar-registered
    "Chopped Tomatoes" and the near-miss "Ocado Italian Chopped Tomatoes"
    — genuinely ambiguous which the person means."""
    plan = agent.propose("Add 1 can of chopped tomatoes, along with the existing 3 cans")
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_ask_or_act(result, plan)
    assert result.passed, result.failures
