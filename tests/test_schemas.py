from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

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
