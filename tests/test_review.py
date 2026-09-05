"""`review`'s deterministic checks over a proposed plan."""

from __future__ import annotations

from decimal import Decimal

from sumac import llm, models, review
from sumac.config import Config
from sumac.models import ChangeKind

LOCATIONS = {
    "pantry": models.Location(id="pantry", name="Pantry"),
    "fridge": models.Location(id="fridge", name="Fridge"),
}
PRODUCTS = {
    "jam": models.Product(id="jam", name="Jam", unit="jar"),
    "Basmati Rice": models.Product(id="Basmati Rice", name="Basmati Rice", unit="jug"),
}
CFG = Config(
    known_locations=LOCATIONS,
    active_locations=LOCATIONS,
    known_products=PRODUCTS,
    active_products=PRODUCTS,
    anomalies=(),
)


def _write(product_id: str, *, unit: str = "jar", to: str | None = "pantry") -> llm.ProposedWrite:
    return llm.ProposedWrite(
        kind=ChangeKind.DISCOVERY,
        product_id=product_id,
        amount=Decimal(1),
        unit=unit,
        from_location=None,
        to_location=to,
    )


def _plan(*writes: llm.ProposedWrite, searched: str = "") -> llm.AgentPlan:
    trace = (
        (llm.ToolCallRecord("sumac_find_inventory", {"query": "q"}, searched),) if searched else ()
    )
    return llm.AgentPlan(reply_text="", writes=writes, trace=trace)


def _codes(plan: llm.AgentPlan) -> list[list[str]]:
    return [[f.code for f in per_write] for per_write in review.review_plan(plan, CFG)]


def test_a_registered_product_in_its_own_unit_raises_nothing() -> None:
    assert _codes(_plan(_write("jam"))) == [[]]


def test_an_unregistered_product_nothing_looked_up_is_ungrounded() -> None:
    """The `docs/journal/2026-09-04-basmati-rice-unit-mismatch.md` failure: a
    `product_id` the vault never held and no search returned."""
    plan = _plan(_write("Basmati Rice Bag", unit="bag"), searched='{"products": ["Basmati Rice"]}')

    assert _codes(plan) == [["ungrounded", "new-product", "near-match"]]


def test_a_product_a_search_returned_is_grounded_even_though_it_is_new() -> None:
    """A product read from a search result is new to the registry but has a
    source: `new-product` applies, `ungrounded` does not."""
    plan = _plan(_write("chutney"), searched='{"products": [{"product_id": "chutney"}]}')

    assert _codes(plan) == [["new-product"]]


def test_grounding_ignores_a_write_tool_echoing_its_own_argument() -> None:
    """`_propose_write`'s result JSON repeats the `product_id` it was called
    with. Counting that as a search result would treat every invented id as
    grounded in the record of the call that invented it."""
    plan = llm.AgentPlan(
        reply_text="",
        writes=(_write("Basmati Rice Bag", unit="bag"),),
        trace=(
            llm.ToolCallRecord(
                "sumac_discover_inventory",
                {"product_id": "Basmati Rice Bag"},
                '{"status": "proposed", "product_id": "Basmati Rice Bag"}',
            ),
        ),
    )

    assert "ungrounded" in _codes(plan)[0]


def test_a_registered_product_in_an_unconvertible_unit_is_a_new_unit() -> None:
    plan = _plan(_write("Basmati Rice", unit="bag"))

    assert _codes(plan) == [["new-unit"]]


def test_an_unconfigured_location_is_flagged() -> None:
    assert _codes(_plan(_write("jam", to="shed"))) == [["unknown-location"]]


def test_only_ungrounded_and_unknown_location_explain_themselves() -> None:
    """The other findings duplicate a `decide` warning already attached to the
    write, so only these two print their own line."""
    plan = _plan(_write("Basmati Rice Bag", unit="bag"), _write("jam", to="shed"))
    explained = {
        f.code for per_write in review.review_plan(plan, CFG) for f in per_write if f.explain
    }

    assert explained == {"ungrounded", "unknown-location"}


def test_headline_counts_writes_not_findings() -> None:
    plan = _plan(_write("Basmati Rice Bag", unit="bag"), _write("jam"))
    findings = review.review_plan(plan, CFG)

    assert review.headline(findings) == (
        "2 changes · 1 names a product nothing looked up · 1 creates a new product"
    )


def test_a_single_clean_change_has_no_headline_at_all() -> None:
    """The write's own row states everything a "1 change" line would."""
    findings = review.review_plan(_plan(_write("jam")), CFG)

    assert review.headline(findings) == ""


def test_a_single_flagged_change_still_says_what_is_flagged() -> None:
    findings = review.review_plan(_plan(_write("jam", to="shed")), CFG)

    assert review.headline(findings) == "1 change · 1 names a new location"


def test_a_hand_typed_product_is_not_ungrounded_but_is_still_new() -> None:
    write = llm.ProposedWrite(
        kind=ChangeKind.DISCOVERY,
        product_id="Billy Bear Ham",
        amount=Decimal(1),
        unit="packet",
        from_location=None,
        to_location="pantry",
        edited_fields=frozenset({"product_id"}),
    )

    assert [f.code for f in review.review_write(write, CFG, "")] == ["new-product"]


def test_editing_another_field_leaves_the_grounding_check_alone() -> None:
    """Correcting an amount says nothing about where the product name came
    from, so the ungrounded name is still reported."""
    write = llm.ProposedWrite(
        kind=ChangeKind.DISCOVERY,
        product_id="Basmati Rice Bag",
        amount=Decimal(1),
        unit="bag",
        from_location=None,
        to_location="pantry",
        edited_fields=frozenset({"amount"}),
    )

    assert "ungrounded" in [f.code for f in review.review_write(write, CFG, "")]
