from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from sumac import events
from sumac.models import ChangeKind, InventoryChange, InventorySnapshot
from sumac.schemas import RecordSchema


def _change_obj() -> dict:
    return {
        "schema_version": 1,
        "type": "change",
        "id": "r1",
        "ts": datetime.now(UTC).isoformat(),
        "actor": "alice",
        "supersedes": None,
        "payload": {
            "kind": "purchase",
            "product_id": "milk",
            "quantity": {"amount": "2", "unit": "l"},
            "from_location": None,
            "to_location": "fridge",
            "metadata": {"brand": "acme"},
        },
    }


def test_record_schema_change_round_trip() -> None:
    record = RecordSchema.model_validate(_change_obj()).to_domain()
    assert isinstance(record.payload, InventoryChange)
    assert record.payload.kind is ChangeKind.PURCHASE
    assert record.payload.quantity.amount == Decimal("2")
    assert record.payload.metadata == {"brand": "acme"}


def test_record_schema_snapshot_round_trip() -> None:
    obj = {
        "schema_version": 1,
        "type": "snapshot",
        "id": "s1",
        "ts": datetime.now(UTC).isoformat(),
        "actor": "alice",
        "supersedes": None,
        "payload": {
            "location_id": "fridge",
            "entries": [
                {"product_id": "milk", "quantity": {"amount": "1", "unit": "l"}, "metadata": {}}
            ],
        },
    }
    record = RecordSchema.model_validate(obj).to_domain()
    assert isinstance(record.payload, InventorySnapshot)
    assert record.payload.location_id == "fridge"
    assert len(record.payload.entries) == 1


def test_record_schema_rejects_extra_fields() -> None:
    obj = _change_obj()
    obj["surprise"] = "field"
    with pytest.raises(ValidationError):
        RecordSchema.model_validate(obj)


def test_record_schema_rejects_non_serializable_metadata() -> None:
    obj = _change_obj()
    obj["payload"]["metadata"] = {"bad": {1, 2, 3}}
    with pytest.raises(ValidationError):
        RecordSchema.model_validate(obj)


def test_record_schema_type_payload_mismatch() -> None:
    obj = _change_obj()
    obj["type"] = "snapshot"
    with pytest.raises(ValidationError):
        RecordSchema.model_validate(obj)


def _v2_obj(type_name: str, payload: dict) -> dict:
    return {
        "schema_version": 2,
        "type": type_name,
        "id": "r1",
        "ts": datetime.now(UTC).isoformat(),
        "actor": "alice",
        "supersedes": None,
        "payload": payload,
    }


def test_v2_acquired_round_trip() -> None:
    obj = _v2_obj(
        "acquired",
        {
            "product_id": "milk",
            "to": "fridge",
            "amount": "2",
            "unit": "l",
            "reason": "discovery",
            "nominal_basis": None,
        },
    )
    record = RecordSchema.model_validate(obj).to_domain()
    assert isinstance(record.payload, events.Acquired)
    assert record.payload.amount == Decimal("2")
    assert record.payload.reason == "discovery"


def test_v2_moved_round_trip() -> None:
    obj = _v2_obj(
        "moved",
        {
            "product_id": "milk",
            "frm": "pantry",
            "to": "fridge",
            "amount": "1",
            "unit": "l",
            "nominal_basis": {"raw_amount": "1", "raw_unit": "jug", "ratio": "1"},
        },
    )
    record = RecordSchema.model_validate(obj).to_domain()
    assert isinstance(record.payload, events.Moved)
    assert record.payload.frm == "pantry"
    assert record.payload.to == "fridge"
    assert record.payload.nominal_basis == {"raw_amount": "1", "raw_unit": "jug", "ratio": "1"}


def test_v2_counted_round_trip() -> None:
    obj = _v2_obj(
        "counted",
        {
            "product_id": "milk",
            "at": "pantry",
            "amount": "3",
            "unit": "l",
            "reason": "implied_by_movement",
            "nominal_basis": None,
        },
    )
    record = RecordSchema.model_validate(obj).to_domain()
    assert isinstance(record.payload, events.Counted)
    assert record.payload.reason == "implied_by_movement"


def test_v2_snapshot_round_trip() -> None:
    obj = _v2_obj(
        "snapshot",
        {
            "location_id": "pantry",
            "entries": [
                {"product_id": "milk", "amount": "2", "unit": "l", "nominal_basis": None},
            ],
        },
    )
    record = RecordSchema.model_validate(obj).to_domain()
    assert isinstance(record.payload, events.Snapshot)
    assert len(record.payload.entries) == 1
    assert record.payload.entries[0].product_id == "milk"


def test_v2_snapshot_zero_entries_round_trips() -> None:
    """The empty-snapshot finding from §3.3a: must validate cleanly, not be
    treated as a malformed/empty record."""
    obj = _v2_obj("snapshot", {"location_id": "pantry", "entries": []})
    record = RecordSchema.model_validate(obj).to_domain()
    assert isinstance(record.payload, events.Snapshot)
    assert record.payload.entries == ()


def test_v1_shaped_payload_rejected_under_schema_version_2() -> None:
    """A v1-shaped "change" payload tagged schema_version=2 must fail, not
    silently validate as something it isn't — the version and the shape have
    to agree."""
    obj = _change_obj()
    obj["schema_version"] = 2
    with pytest.raises(ValidationError):
        RecordSchema.model_validate(obj)


def test_v2_shaped_payload_rejected_under_schema_version_1() -> None:
    obj = _v2_obj("acquired", {"product_id": "milk", "to": "pantry", "amount": "1", "unit": "l"})
    obj["schema_version"] = 1
    with pytest.raises(ValidationError):
        RecordSchema.model_validate(obj)
