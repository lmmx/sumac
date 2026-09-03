"""Behavioural cases for `sumac ask` against a real local model — see
docs/journal/2026-09-02-eval-suite.md. Every test is one named scenario:
run `uv run pytest evals/test_agent.py -v` and read the names top to
bottom to see exactly which behaviour broke.

Needs a cached GGUF (`agent_runner_factory` skips cleanly, with no network
attempt, if the configured preset isn't already downloaded — run
`sumac ask` once against it first). No epochs, no seeds beyond a single
optional `--eval-seed` for reproducing one run, no statistical apparatus:
one model, one seed, one pass — see the eval spec for why that trade was
made deliberately at this stage.

Waste (`ChangeKind.WASTE`) and purchase (`ChangeKind.PURCHASE`) are not
covered: `AgentRunner.tool_callbacks` only ever emits `DISCOVERY`
(`sumac_discover_inventory`), `CONSUMPTION` (`sumac_consume_inventory`), or
`MOVEMENT` (`sumac_move_inventory`) — those two `ChangeKind`s have no route
through the agent at all, only through `sumac add` directly
(`src/sumac/llm.py:311-315`). Nothing here tests a distinction the agent
cannot make.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from evals.conftest import (
    assert_classified,
    assert_no_writes,
    assert_tool_called,
    assert_write,
    is_ask_or_act,
)
from sumac import config as sumac_config
from sumac.llm import QueryKind
from sumac.models import ChangeKind

pytestmark = pytest.mark.model


@pytest.fixture
def agent(agent_runner_factory):
    return agent_runner_factory()


# --- find --------------------------------------------------------------


def test_find_existing_item(agent, cfg) -> None:
    plan = agent.propose("where is the strawberry jam?")
    assert_classified(plan, QueryKind.FIND)
    assert_tool_called(plan, "sumac_find_inventory")
    assert_no_writes(plan)
    assert (
        "white unit r3c1" in plan.reply_text.lower()
    ), f"reply doesn't name the jam's location: {plan.reply_text!r}"


def test_find_missing_item(agent, cfg) -> None:
    plan = agent.propose("where is the caviar?")
    assert_classified(plan, QueryKind.FIND)
    assert_tool_called(plan, "sumac_find_inventory")
    assert_no_writes(plan)


def test_find_quantity(agent, cfg) -> None:
    plan = agent.propose("how much basmati rice do we have?")
    assert_classified(plan, QueryKind.FIND)
    assert_tool_called(plan, "sumac_find_inventory")
    assert_no_writes(plan)
    reply = plan.reply_text.lower()
    assert "1" in reply and "jug" in reply, f"reply doesn't name the quantity: {plan.reply_text!r}"


def test_find_shared_word_picks_right_product(agent, cfg) -> None:
    """`ledger.search_inventory` returns both "Salted Butter" (the right
    answer) and "Butter Beans" (a decoy sharing the word "butter") as
    whole-word matches — this is the wrong-product-from-unranked-results
    failure recorded in docs/journal/2026-09-01-ask-agent-design.md."""
    plan = agent.propose("do we have any butter?")
    assert_classified(plan, QueryKind.FIND)
    assert_no_writes(plan)
    reply = plan.reply_text.lower()
    assert "salted butter" in reply, f"reply doesn't name Salted Butter: {plan.reply_text!r}"
    decoy_idx = reply.find("butter beans")
    intended_idx = reply.find("salted butter")
    if decoy_idx != -1:
        assert (
            decoy_idx > intended_idx
        ), f"decoy named before the intended product: {plan.reply_text!r}"


def test_find_uses_only_find_tool(agent, cfg) -> None:
    plan = agent.propose("where is the ragu?")
    assert_classified(plan, QueryKind.FIND)
    assert_no_writes(plan)
    tool_names = {t.name for t in plan.trace if t.name != "classify_request"}
    assert tool_names <= {"sumac_find_inventory"}, f"a find request called: {tool_names}"


# --- add -----------------------------------------------------------------


def test_add_existing_item_full_path(agent, cfg) -> None:
    plan = agent.propose("Add 1 can of Ocado Italian Chopped Tomatoes to Pantry > White Unit R2C3")
    assert_classified(plan, QueryKind.ADD)
    assert_write(
        plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Ocado Italian Chopped Tomatoes",
        amount="1", unit="cans", to_location="pantry-white-unit-r2c3",
    )  # fmt: skip


def test_add_existing_item_indirect_location(agent, cfg) -> None:
    plan = agent.propose(
        "Add 1 can of Ocado Italian Chopped Tomatoes to the pantry, same spot as existing stock"
    )
    assert_classified(plan, QueryKind.ADD)
    assert_tool_called(plan, "sumac_find_inventory")
    assert_write(
        plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Ocado Italian Chopped Tomatoes",
        amount="1", unit="cans", to_location="pantry-white-unit-r2c3",
    )  # fmt: skip


def test_add_missing_item_discovers_new_product(agent, cfg) -> None:
    plan = agent.propose("Add 2 bottles of Irn-Bru Zero to the pantry")
    assert_classified(plan, QueryKind.ADD)
    assert_tool_called(plan, "sumac_find_inventory")
    assert_write(
        plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Irn-Bru Zero",
        amount="2", unit="bottles", to_location="pantry",
    )  # fmt: skip


def test_add_discriminator_variant_not_confused(agent, cfg) -> None:
    """Salted Butter and Unsalted Butter are seeded at different locations
    — a correct discovery must land on Unsalted Butter's own location, not
    the one-word-different product's."""
    plan = agent.propose(
        "Add 2 packs of Unsalted Butter to Big Freezer > Drawer 2, with the existing stock"
    )
    assert_classified(plan, QueryKind.ADD)
    assert_write(
        plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Unsalted Butter",
        amount="2", unit="packs", to_location="freezer-drawer-2",
    )  # fmt: skip


def test_add_basmati_rice_in_different_unit(agent, cfg) -> None:
    """The same product can be added in a different unit."""
    plan = agent.propose("Add 1 bag of Basmati Rice (1kg) next to the existing jug of Basmati Rice")
    assert_classified(plan, QueryKind.ADD)
    assert len(plan.writes) == 1, f"expected one write, got {plan.writes!r}"

    w = plan.writes[0]
    assert w.kind == ChangeKind.DISCOVERY
    assert w.product_id.strip().lower() == "basmati rice"
    assert w.amount == Decimal("1")
    assert w.unit.strip().lower() == "bag"


def test_odd_destination_respected(agent, cfg) -> None:
    """The location a person names, even an unusual one, is recorded as
    given rather than silently retargeted to a more sensible-sounding
    one."""
    plan = agent.propose("Add 1 box of Pizza Express Margherita Pizza to the fridge bottle rack")
    assert_classified(plan, QueryKind.ADD)
    assert_write(
        plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Pizza Express Margherita Pizza",
        amount="1", unit="box", to_location="fridge-bottle-rack",
    )  # fmt: skip


def test_duplicate_search_bounded(agent, cfg) -> None:
    """The real Moma-pistachio-milk transcript
    (docs/journal/2026-09-01-ask-agent-design.md) made four
    `sumac_find_inventory` calls for one request before giving up."""
    plan = agent.propose(
        "Add 6 cans of Ocado Italian Chopped Tomatoes to the same pantry cupboard as existing stock"
    )
    assert_classified(plan, QueryKind.ADD)
    assert_tool_called(plan, "sumac_find_inventory", at_most=2)
    assert_write(
        plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Ocado Italian Chopped Tomatoes",
        amount="6", unit="cans", to_location="pantry-white-unit-r2c3",
    )  # fmt: skip


def test_add_product_with_omitted_amount(agent, cfg) -> None:
    """`sumac_discover_inventory` requires `amount` and `unit`
    (`src/sumac/llm.py`), and the request gives neither — the agent is
    expected to infer a plausible default (e.g. "1 box") rather than ask,
    so this checks the right product lands somewhere sensible, not which
    exact quantity it picked. `_ADD_PROMPT` instructs searching for "the
    other pasta" and using its location rather than guessing a location
    string — Fusilli Pasta's own location and the literally-named "pantry"
    are both accepted, since the request names both."""
    plan = agent.propose("Add Barilla Rigatoni to the pantry, with the other pasta")
    assert_classified(plan, QueryKind.ADD)
    assert len(plan.writes) == 1, f"expected exactly one write, got {plan.writes!r}"
    w = plan.writes[0]
    assert w.kind == ChangeKind.DISCOVERY, f"expected a discovery, got {w.kind}"
    assert (
        w.product_id.strip().lower() == "barilla rigatoni"
    ), f"expected Barilla Rigatoni, got {w.product_id!r}"
    assert w.amount > 0, f"expected a positive amount, got {w.amount!r}"
    assert w.unit.strip(), "expected a non-empty unit"

    resolved = w.to_location
    if resolved not in cfg.known_locations:
        for loc_id in cfg.known_locations:
            if sumac_config.location_path(cfg.known_locations, loc_id) == w.to_location:
                resolved = loc_id
                break
    assert resolved in ("pantry", "pantry-white-unit-r2c1"), (
        f"expected the pantry or Fusilli Pasta's own location, got {w.to_location!r} "
        f"(resolved: {resolved!r})"
    )


def test_add_multiple_products_with_omitted_amounts(agent, cfg) -> None:
    """Two distinct products, no amount for either — same inference
    expectation as `test_add_product_with_omitted_amount`, checked for
    both products at once. "Butter" and "Butter Beans" are distinct
    products: neither "Salted Butter" nor "Unsalted Butter" is an exact
    match for "butter", so registering a new "Butter" is a fine outcome —
    what's checked is that the butter write isn't "Butter Beans", not
    which of the three plausible butter identities the model landed on."""
    plan = agent.propose("Add butter and jam to the pantry")
    assert_classified(plan, QueryKind.ADD)
    assert len(plan.writes) == 2, f"expected two writes (butter, jam), got {plan.writes!r}"
    for w in plan.writes:
        assert w.kind == ChangeKind.DISCOVERY, f"expected a discovery, got {w.kind}"
        assert w.amount > 0, f"expected a positive amount, got {w.amount!r}"
        assert w.unit.strip(), "expected a non-empty unit"

    product_ids = [w.product_id.strip().lower() for w in plan.writes]
    butter_writes = [p for p in product_ids if "butter" in p]
    jam_writes = [p for p in product_ids if "jam" in p]
    assert len(butter_writes) == 1, f"expected exactly one butter-related write, got {product_ids}"
    assert (
        "beans" not in butter_writes[0]
    ), f"registered against Butter Beans instead of butter: {butter_writes[0]!r}"
    assert len(jam_writes) == 1, f"expected exactly one jam-related write, got {product_ids}"


# --- remove ----------------------------------------------------------------


def test_remove_partial(agent, cfg) -> None:
    plan = agent.propose("I used 1 tub of Ragu")
    assert_classified(plan, QueryKind.REMOVE)
    assert_tool_called(plan, "sumac_find_inventory")
    assert_write(
        plan, cfg,
        kind=ChangeKind.CONSUMPTION, product_id="Ragu",
        amount="1", unit="tub", from_location="freezer-drawer-3",
    )  # fmt: skip


def test_remove_all(agent, cfg) -> None:
    plan = agent.propose("we finished the strawberry jam")
    assert_classified(plan, QueryKind.REMOVE)
    assert_write(
        plan, cfg,
        kind=ChangeKind.CONSUMPTION, product_id="Strawberry Jam",
        amount="1", unit="jar", from_location="pantry-white-unit-r3c1",
    )  # fmt: skip


def test_move_explicit(agent, cfg) -> None:
    plan = agent.propose("move 1 tub of Ragu from Big Freezer > Drawer 3 to Fridge > Door")
    assert_classified(plan, QueryKind.REMOVE)
    assert_write(
        plan, cfg,
        kind=ChangeKind.MOVEMENT, product_id="Ragu",
        amount="1", unit="tub", from_location="freezer-drawer-3", to_location="fridge-door",
    )  # fmt: skip


def test_move_vague_asks_or_acts(agent, cfg) -> None:
    """ "the ragu" with no amount is defensibly the full 2 tubs already
    stocked — unlike a missing-amount `add`, no field needs inventing to
    act correctly, so acting is an acceptable outcome alongside asking."""
    plan = agent.propose("move the ragu to the fridge")
    assert_classified(plan, QueryKind.REMOVE)
    branch = is_ask_or_act(plan)
    assert branch in ("ask", "act"), f"neither asked nor acted: branch={branch!r}, plan={plan!r}"


def test_ambiguous_product_asks_or_acts(agent, cfg) -> None:
    """ "chopped tomatoes" (no brand) matches both the jar-registered
    "Chopped Tomatoes" and the near-miss "Ocado Italian Chopped Tomatoes"
    — genuinely ambiguous which the person means."""
    plan = agent.propose("Add 1 can of chopped tomatoes, along with the existing 3 cans")
    assert_classified(plan, QueryKind.ADD)
    branch = is_ask_or_act(plan)
    assert branch in ("ask", "act"), f"neither asked nor acted: branch={branch!r}, plan={plan!r}"


# --- reject ------------------------------------------------------------


def test_reject_out_of_domain_weather(agent, cfg) -> None:
    plan = agent.propose("What's the weather in Edinburgh?")
    assert_classified(plan, QueryKind.REJECT)
    assert_no_writes(plan)


def test_reject_gibberish(agent, cfg) -> None:
    plan = agent.propose("asdf")
    assert_classified(plan, QueryKind.REJECT)
    assert_no_writes(plan)


def test_reject_joke_with_inventory_word(agent, cfg) -> None:
    """A word that also names a seeded product ("tomatoes") must not pull
    an out-of-domain request into the inventory-shaped kinds."""
    plan = agent.propose("Tell me a joke about tomatoes")
    assert_classified(plan, QueryKind.REJECT)
    assert_no_writes(plan)
