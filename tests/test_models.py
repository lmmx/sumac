from __future__ import annotations

from decimal import Decimal

import pytest

from sumac.models import ChangeKind, InventoryChange, Quantity


def test_quantity_add_same_unit() -> None:
    a = Quantity(Decimal("1"), "kg")
    b = Quantity(Decimal("2"), "kg")
    assert (a + b) == Quantity(Decimal("3"), "kg")


def test_quantity_add_mismatched_unit_raises() -> None:
    with pytest.raises(ValueError, match="unit mismatch"):
        Quantity(Decimal("1"), "kg") + Quantity(Decimal("1"), "lb")


def test_quantity_neg() -> None:
    assert -Quantity(Decimal("2"), "kg") == Quantity(Decimal("-2"), "kg")


def test_movement_requires_both_locations() -> None:
    with pytest.raises(ValueError, match="movement requires"):
        InventoryChange(
            kind=ChangeKind.MOVEMENT,
            product_id="milk",
            quantity=Quantity(Decimal("1"), "l"),
            from_location="fridge",
        )


def test_purchase_requires_to_location() -> None:
    with pytest.raises(ValueError, match="purchase requires"):
        InventoryChange(
            kind=ChangeKind.PURCHASE, product_id="milk", quantity=Quantity(Decimal("1"), "l")
        )


def test_consumption_requires_from_location() -> None:
    with pytest.raises(ValueError, match="consumption requires"):
        InventoryChange(
            kind=ChangeKind.CONSUMPTION, product_id="milk", quantity=Quantity(Decimal("1"), "l")
        )


def test_valid_purchase() -> None:
    change = InventoryChange(
        kind=ChangeKind.PURCHASE,
        product_id="milk",
        quantity=Quantity(Decimal("1"), "l"),
        to_location="fridge",
    )
    assert change.to_location == "fridge"
