from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from sumac import events, upcast
from sumac.models import (
    ChangeKind,
    InventoryChange,
    InventorySnapshot,
    Quantity,
    Record,
    SnapshotEntry,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _record(payload: InventoryChange | InventorySnapshot) -> Record:
    return Record(
        schema_version=1,
        type="change" if isinstance(payload, InventoryChange) else "snapshot",
        id="r1",
        ts=T0,
        actor="alice",
        supersedes=None,
        payload=payload,
    )


def _change(
    kind: ChangeKind,
    *,
    from_location: str | None = None,
    to_location: str | None = None,
) -> InventoryChange:
    return InventoryChange(
        kind=kind,
        product_id="milk",
        quantity=Quantity(Decimal("1"), "l"),
        from_location=from_location,
        to_location=to_location,
    )


def test_purchase_upcasts_to_acquired() -> None:
    ev = upcast.upcast(_record(_change(ChangeKind.PURCHASE, to_location="pantry")))
    assert ev == events.Acquired(product_id="milk", to="pantry", amount=Decimal("1"), unit="l")


def test_discovery_upcasts_to_acquired_with_reason() -> None:
    ev = upcast.upcast(_record(_change(ChangeKind.DISCOVERY, to_location="pantry")))
    assert isinstance(ev, events.Acquired)
    assert ev.reason == "discovery"
    assert ev.to == "pantry"


def test_consumption_upcasts_to_consumed() -> None:
    ev = upcast.upcast(_record(_change(ChangeKind.CONSUMPTION, from_location="pantry")))
    assert ev == events.Consumed(product_id="milk", frm="pantry", amount=Decimal("1"), unit="l")


def test_waste_upcasts_to_discarded() -> None:
    ev = upcast.upcast(_record(_change(ChangeKind.WASTE, from_location="pantry")))
    assert ev == events.Discarded(product_id="milk", frm="pantry", amount=Decimal("1"), unit="l")


def test_movement_upcasts_to_moved() -> None:
    ev = upcast.upcast(
        _record(_change(ChangeKind.MOVEMENT, from_location="pantry", to_location="fridge"))
    )
    assert ev == events.Moved(
        product_id="milk", frm="pantry", to="fridge", amount=Decimal("1"), unit="l"
    )


def test_correction_to_only_upcasts_to_acquired_with_reason() -> None:
    ev = upcast.upcast(_record(_change(ChangeKind.CORRECTION, to_location="pantry")))
    assert isinstance(ev, events.Acquired)
    assert ev.reason == "correction"
    assert ev.to == "pantry"


def test_correction_from_only_upcasts_to_consumed_with_reason() -> None:
    """Decided in §3.3a: Consumed, not Discarded — Discarded asserts the food
    was binned, a claim the record doesn't support."""
    ev = upcast.upcast(_record(_change(ChangeKind.CORRECTION, from_location="pantry")))
    assert isinstance(ev, events.Consumed)
    assert ev.reason == "correction"
    assert ev.frm == "pantry"


def test_correction_with_both_endpoints_raises_upcast_error() -> None:
    obj = _change(ChangeKind.CORRECTION, from_location="pantry", to_location="fridge")
    with pytest.raises(upcast.UpcastError):
        upcast.upcast(_record(obj))


def test_correction_with_neither_endpoint_raises_upcast_error() -> None:
    obj = _change(ChangeKind.CORRECTION)
    with pytest.raises(upcast.UpcastError):
        upcast.upcast(_record(obj))


def test_snapshot_upcasts_1to1_preserving_entries() -> None:
    snap = InventorySnapshot(
        location_id="pantry",
        entries=(
            SnapshotEntry(product_id="milk", quantity=Quantity(Decimal("2"), "l")),
            SnapshotEntry(product_id="eggs", quantity=Quantity(Decimal("6"), "ct")),
        ),
    )
    ev = upcast.upcast(_record(snap))
    assert ev == events.Snapshot(
        location_id="pantry",
        entries=(
            events.SnapshotEntry(product_id="milk", amount=Decimal("2"), unit="l"),
            events.SnapshotEntry(product_id="eggs", amount=Decimal("6"), unit="ct"),
        ),
    )


def test_empty_snapshot_upcasts_to_snapshot_with_zero_entries_not_nothing() -> None:
    """The finding that drove §3.3a's design: this must not upcast to zero
    events, or "this location is empty" silently disappears."""
    snap = InventorySnapshot(location_id="pantry", entries=())
    ev = upcast.upcast(_record(snap))
    assert ev == events.Snapshot(location_id="pantry", entries=())
