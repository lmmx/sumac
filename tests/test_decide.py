from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

import pytest

from sumac import config, decide
from sumac.errors import Rejected
from sumac.models import ChangeKind, Location, Product

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
) -> tuple[list[decide.Write], str | None]:
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
    writes, warning = _decide()
    assert warning is None
    assert len(writes) == 1
    assert writes[0].stream == "log:alice"
    assert writes[0].obj["payload"]["kind"] == "purchase"
    assert writes[0].obj["payload"]["quantity"] == {"amount": "1", "unit": "l"}


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
    writes, _warning = _decide(to_location="Pantry > Shelf", cfg=cfg)
    assert writes[0].obj["payload"]["to_location"] == "pantry-shelf"


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
    writes, warning = _decide(product_id="rice-pudding", amount=Decimal("2"), unit="jar", cfg=cfg)
    assert warning is None
    assert writes[0].obj["payload"]["quantity"] == {"amount": "680", "unit": "g"}


def test_unknown_product_auto_registers_before_the_change() -> None:
    cfg = _cfg(locations={"pantry": Location(id="pantry", name="Pantry")}, products={})
    writes, warning = _decide(product_id="kimchi", unit="jar", cfg=cfg)
    assert len(writes) == 2
    assert writes[0].stream == "config"
    assert writes[0].obj["product"]["id"] == "kimchi"
    assert writes[0].obj["product"]["unit"] == "jar"
    assert writes[0].obj["product"]["metadata"] == {"auto": True}
    assert writes[1].stream == "log:alice"
    assert writes[1].obj["payload"]["product_id"] == "kimchi"
    assert writes[1].obj["payload"]["quantity"] == {"amount": "1", "unit": "jar"}


def test_unknown_product_no_warning_when_no_near_match() -> None:
    cfg = _cfg(locations={"pantry": Location(id="pantry", name="Pantry")}, products={})
    _writes, warning = _decide(product_id="kimchi", cfg=cfg)
    assert warning is None


def test_unknown_product_warns_on_near_match() -> None:
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={"milk": Product(id="milk", name="Milk", unit="l")},
    )
    _writes, warning = _decide(product_id="milc", cfg=cfg)
    assert warning is not None
    assert "milk" in warning
    assert "milc" in warning


def test_retired_product_does_not_trigger_reregistration() -> None:
    """A retired product must reject, not fall through to auto-register."""
    cfg = _cfg(
        locations={"pantry": Location(id="pantry", name="Pantry")},
        products={"milk": Product(id="milk", name="Milk", unit="l", retired=True)},
    )
    with pytest.raises(Rejected):
        _decide(product_id="milk", cfg=cfg)
