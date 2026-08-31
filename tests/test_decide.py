from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

import pytest

from sumac import config, decide, ledger
from sumac.errors import Rejected
from sumac.models import ChangeKind, Location, Product, Quantity

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _cfg(
    locations: dict[str, Location] | None = None,
    products: dict[str, Product] | None = None,
) -> config.Config:
    known_locations = locations or {}
    known_products = products or {}
    return config.Config(
        known_locations=known_locations,
        active_locations={i: loc for i, loc in known_locations.items() if not loc.retired},
        known_products=known_products,
        active_products={i: p for i, p in known_products.items() if not p.retired},
        anomalies=(),
    )


_EMPTY_INVENTORY = ledger.Inventory(by_location={})


def _decide(
    *,
    kind: ChangeKind = ChangeKind.PURCHASE,
    product_id: str = "milk",
    amount: Decimal = Decimal("1"),
    unit: str = "l",
    from_location: str | None = None,
    to_location: str | None = "pantry",
    actor: str = "alice",
    occurred_at: datetime = T0,
    cfg: config.Config | None = None,
    inventory: ledger.Inventory = _EMPTY_INVENTORY,
) -> tuple[list[decide.Write], list[str]]:
    if cfg is None:
        cfg = _cfg(
            locations={"pantry": Location(id="pantry", name="Pantry")},
            products={"milk": Product(id="milk", name="Milk", unit="l")},
        )
    return decide.decide_change(
        kind=kind,
        product_id=product_id,
        amount=amount,
        unit=unit,
        from_location=from_location,
        to_location=to_location,
        actor=actor,
        occurred_at=occurred_at,
        inventory=inventory,
        cfg=cfg,
    )


def test_near_matches_finds_close_typo() -> None:
    assert decide.near_matches("pantr", ["pantry", "fridge"]) == ["pantry"]


def test_near_matches_empty_when_nothing_close() -> None:
    assert decide.near_matches("xyz123", ["pantry", "fridge"]) == []


def test_purchase_missing_to_location_is_rejected_not_a_bare_valueerror() -> None:
    """Found by the gate soundness property test: InventoryChange's own
    __post_init__ raises ValueError for a missing endpoint, which isn't a
    SumacError — uncaught, it would have produced a raw traceback instead of
    a clean rejection cli.main() can render."""
    with pytest.raises(Rejected) as exc_info:
        _decide(kind=ChangeKind.PURCHASE, to_location=None, from_location=None)
    assert exc_info.value.reason == "missing_endpoint"


def test_valid_purchase_produces_one_log_write() -> None:
    writes, messages = _decide()
    assert messages == []
    assert len(writes) == 1
    assert writes[0].stream == "log:alice"
    assert writes[0].obj["schema_version"] == 2
    assert writes[0].obj["type"] == "acquired"
    assert writes[0].obj["payload"]["product_id"] == "milk"
    assert writes[0].obj["payload"]["to"] == "pantry"
    assert writes[0].obj["payload"]["amount"] == "1"
    assert writes[0].obj["payload"]["unit"] == "l"
    assert writes[0].obj["payload"]["reason"] is None


def test_unknown_location_rejected_with_suggestions() -> None:
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={"milk": Product(id="milk", name="Milk", unit="l")},
    )
    with pytest.raises(Rejected) as exc_info:
        _decide(to_location="pantr", cfg=cfg)
    assert exc_info.value.reason == "unknown_location"
    assert exc_info.value.detail["field"] == "to"
    suggestions = cast("list[str]", exc_info.value.detail["suggestions"])
    assert "pantry" in suggestions


def test_retired_location_rejected_distinctly_from_unknown() -> None:
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry", retired=True)},
        products={"milk": Product(id="milk", name="Milk", unit="l")},
    )
    with pytest.raises(Rejected) as exc_info:
        _decide(to_location="pantry", cfg=cfg)
    assert exc_info.value.reason == "retired_location"


def test_display_path_resolves_to_real_id_not_rejected() -> None:
    locations = {
        "pantry": Location(id="pantry", name="Pantry"),
        "pantry-shelf": Location(id="pantry-shelf", name="Shelf", parent_id="pantry"),
    }
    cfg = _cfg(locations=locations, products={"milk": Product(id="milk", name="Milk", unit="l")})
    writes, _messages = _decide(to_location="Pantry > Shelf", cfg=cfg)
    assert writes[0].obj["payload"]["to"] == "pantry-shelf"


def test_display_path_to_retired_location_still_rejected() -> None:
    locations = {
        "pantry": Location(id="pantry", name="Pantry", retired=True),
    }
    cfg = _cfg(locations=locations, products={"milk": Product(id="milk", name="Milk", unit="l")})
    with pytest.raises(Rejected) as exc_info:
        _decide(to_location="Pantry", cfg=cfg)
    assert exc_info.value.reason == "retired_location"


def test_noop_move_rejected() -> None:
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={"milk": Product(id="milk", name="Milk", unit="l")},
    )
    with pytest.raises(Rejected) as exc_info:
        _decide(kind=ChangeKind.MOVEMENT, from_location="pantry", to_location="pantry", cfg=cfg)
    assert exc_info.value.reason == "noop_move"


def test_non_positive_amount_rejected() -> None:
    with pytest.raises(Rejected) as exc_info:
        _decide(amount=Decimal("0"))
    assert exc_info.value.reason == "non_positive_amount"

    with pytest.raises(Rejected) as exc_info:
        _decide(amount=Decimal("-1"))
    assert exc_info.value.reason == "non_positive_amount"


def test_retired_product_rejected() -> None:
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={"milk": Product(id="milk", name="Milk", unit="l", retired=True)},
    )
    with pytest.raises(Rejected) as exc_info:
        _decide(cfg=cfg)
    assert exc_info.value.reason == "retired_product"


def test_unit_unconvertible_rejected() -> None:
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={"flour": Product(id="flour", name="Flour", unit="kg")},
    )
    with pytest.raises(Rejected) as exc_info:
        _decide(product_id="flour", unit="lb", cfg=cfg)
    assert exc_info.value.reason == "unit_unconvertible"


def test_registered_product_applies_conversion() -> None:
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={
            "rice-pudding": Product(
                id="rice-pudding",
                name="Rice Pudding",
                unit="g",
                conversions={"jar": Decimal("340")},
            )
        },
    )
    writes, messages = _decide(product_id="rice-pudding", amount=Decimal("2"), unit="jar", cfg=cfg)
    assert messages == []
    assert writes[0].obj["payload"]["amount"] == "680"
    assert writes[0].obj["payload"]["unit"] == "g"


def test_unknown_product_auto_registers_before_the_change() -> None:
    cfg = _cfg(locations={"pantry": Location(id="pantry", name="Pantry")}, products={})
    writes, messages = _decide(product_id="kimchi", unit="jar", cfg=cfg)
    assert len(writes) == 2
    assert writes[0].stream == "config"
    assert writes[0].obj["product"]["id"] == "kimchi"
    assert writes[0].obj["product"]["unit"] == "jar"
    assert writes[0].obj["product"]["metadata"] == {"auto": True}
    assert writes[1].stream == "log:alice"
    assert writes[1].obj["payload"]["product_id"] == "kimchi"
    assert writes[1].obj["payload"]["amount"] == "1"
    assert writes[1].obj["payload"]["unit"] == "jar"
    assert messages == []


def test_unknown_product_no_warning_when_no_near_match() -> None:
    cfg = _cfg(locations={"pantry": Location(id="pantry", name="Pantry")}, products={})
    _writes, messages = _decide(product_id="kimchi", cfg=cfg)
    assert messages == []


def test_unknown_product_warns_on_near_match() -> None:
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={"milk": Product(id="milk", name="Milk", unit="l")},
    )
    _writes, messages = _decide(product_id="milc", cfg=cfg)
    assert len(messages) == 1
    assert "milk" in messages[0]
    assert "milc" in messages[0]


def test_retired_product_does_not_trigger_reregistration() -> None:
    """A retired product must reject, not fall through to auto-register."""
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={"milk": Product(id="milk", name="Milk", unit="l", retired=True)},
    )
    with pytest.raises(Rejected):
        _decide(product_id="milk", cfg=cfg)


def test_auto_register_tripwire_fires_if_active_known_invariant_breaks() -> None:
    """Regression guard for the auto-register-can't-clobber argument in
    decide.py: `config.build_config` always derives `active_products` as
    `known_products` filtered to `not retired`, so a known, non-retired
    product can never be missing from `active_products` in practice — but
    nothing in the `Config` dataclass itself enforces that. If some future
    caller ever hand-builds one that breaks the invariant, the assertion in
    `_resolve_product` must fire rather than silently auto-registering over
    an existing product."""
    product = Product(id="milk", name="Milk", unit="l", retired=False)
    cfg = config.Config(
        known_locations={"pantry": Location(id="pantry", name="Pantry")},
        active_locations={"pantry": Location(id="pantry", name="Pantry")},
        known_products={"milk": product},
        active_products={},  # broken: milk is known and not retired, but omitted here
        anomalies=(),
    )
    with pytest.raises(AssertionError):
        _decide(product_id="milk", cfg=cfg)


def test_purchase_with_spurious_from_location_rejected() -> None:
    """Latent gap found while rewriting for v2 (see _check_endpoint_shape's
    docstring): a purchase with a from_location too used to silently store
    it, and the fold would subtract from that location for no real reason."""
    cfg = _cfg(
        locations={
            "pantry": Location(id="pantry", name="Pantry"),
            "fridge": Location(id="fridge", name="Fridge"),
        },
        products={"milk": Product(id="milk", name="Milk", unit="l")},
    )
    with pytest.raises(Rejected) as exc_info:
        _decide(kind=ChangeKind.PURCHASE, from_location="fridge", to_location="pantry", cfg=cfg)
    assert exc_info.value.reason == "missing_endpoint"


def test_consumption_with_spurious_to_location_rejected() -> None:
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={"milk": Product(id="milk", name="Milk", unit="l")},
    )
    with pytest.raises(Rejected) as exc_info:
        _decide(kind=ChangeKind.CONSUMPTION, from_location="pantry", to_location="pantry", cfg=cfg)
    assert exc_info.value.reason == "missing_endpoint"


def test_correction_with_both_endpoints_rejected_before_resolution() -> None:
    cfg = _cfg(
        locations={
            "pantry": Location(id="pantry", name="Pantry"),
            "fridge": Location(id="fridge", name="Fridge"),
        },
        products={"milk": Product(id="milk", name="Milk", unit="l")},
    )
    with pytest.raises(Rejected) as exc_info:
        _decide(kind=ChangeKind.CORRECTION, from_location="pantry", to_location="fridge", cfg=cfg)
    assert exc_info.value.reason == "missing_endpoint"


# --- §3.5 insufficient stock: "the shelf is authoritative, not the log" ---


def _cfg_with_pantry_and_fridge() -> config.Config:
    return _cfg(
        locations={
            "pantry": Location(id="pantry", name="Pantry"),
            "fridge": Location(id="fridge", name="Fridge"),
        },
        products={"milk": Product(id="milk", name="Milk", unit="l")},
    )


def test_insufficient_stock_emits_counted_before_consumption() -> None:
    inventory = ledger.Inventory(by_location={"pantry": {"milk": Quantity(Decimal("1"), "l")}})
    writes, messages = _decide(
        kind=ChangeKind.CONSUMPTION,
        from_location="pantry",
        to_location=None,
        amount=Decimal("3"),
        cfg=_cfg_with_pantry_and_fridge(),
        inventory=inventory,
    )
    assert len(writes) == 2
    assert writes[0].obj["type"] == "counted"
    assert writes[0].obj["payload"]["at"] == "pantry"
    assert writes[0].obj["payload"]["amount"] == "3"
    assert writes[0].obj["payload"]["reason"] == "implied_by_movement"
    assert writes[1].obj["type"] == "consumed"
    assert any("adjusted" in m for m in messages)


def test_sufficient_stock_does_not_emit_counted() -> None:
    inventory = ledger.Inventory(by_location={"pantry": {"milk": Quantity(Decimal("10"), "l")}})
    writes, messages = _decide(
        kind=ChangeKind.CONSUMPTION,
        from_location="pantry",
        to_location=None,
        amount=Decimal("3"),
        cfg=_cfg_with_pantry_and_fridge(),
        inventory=inventory,
    )
    assert len(writes) == 1
    assert writes[0].obj["type"] == "consumed"
    assert not any("adjusted" in m for m in messages)


def test_nothing_recorded_counts_as_insufficient() -> None:
    """held is None (nothing ever recorded there) — must still be treated as
    insufficient, not skipped as "no data to compare"."""
    writes, _messages = _decide(
        kind=ChangeKind.CONSUMPTION,
        from_location="pantry",
        to_location=None,
        amount=Decimal("2"),
        cfg=_cfg_with_pantry_and_fridge(),
        inventory=ledger.Inventory(by_location={}),
    )
    assert len(writes) == 2
    assert writes[0].obj["type"] == "counted"
    assert writes[0].obj["payload"]["amount"] == "2"


def test_insufficient_stock_skipped_on_preexisting_unit_mismatch() -> None:
    """A unit mismatch already sitting at this location is a pre-existing
    fold-level anomaly this check must not try to paper over — leave it to
    the normal unit_mismatch handling, unchanged."""
    inventory = ledger.Inventory(by_location={"pantry": {"milk": Quantity(Decimal("1"), "gal")}})
    writes, messages = _decide(
        kind=ChangeKind.CONSUMPTION,
        from_location="pantry",
        to_location=None,
        amount=Decimal("3"),
        cfg=_cfg_with_pantry_and_fridge(),
        inventory=inventory,
    )
    assert len(writes) == 1
    assert writes[0].obj["type"] == "consumed"
    assert not any("adjusted" in m for m in messages)


def test_insufficient_stock_for_movement_only_touches_from_side() -> None:
    inventory = ledger.Inventory(by_location={"pantry": {"milk": Quantity(Decimal("1"), "l")}})
    writes, messages = _decide(
        kind=ChangeKind.MOVEMENT,
        from_location="pantry",
        to_location="fridge",
        amount=Decimal("3"),
        cfg=_cfg_with_pantry_and_fridge(),
        inventory=inventory,
    )
    assert len(writes) == 2
    assert writes[0].obj["type"] == "counted"
    assert writes[0].obj["payload"]["at"] == "pantry"
    assert writes[1].obj["type"] == "moved"
    assert any("pantry" in m for m in messages)


def test_acquired_events_never_get_a_counted_correction() -> None:
    """Acquired has no `frm` side — there's nothing for it to fall short of."""
    writes, messages = _decide(kind=ChangeKind.PURCHASE, to_location="pantry")
    assert len(writes) == 1
    assert writes[0].obj["type"] == "acquired"
    assert messages == []
