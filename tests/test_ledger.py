from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from sumac import SCHEMA_VERSION, ledger, store
from sumac.errors import SchemaVersionError

T0 = datetime(2026, 1, 1, tzinfo=None).astimezone()


def _change_obj(
    record_id: str,
    ts: datetime,
    actor: str,
    kind: str,
    product_id: str,
    amount: str,
    unit: str,
    *,
    from_location: str | None = None,
    to_location: str | None = None,
    supersedes: str | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "change",
        "id": record_id,
        "ts": ts.isoformat(),
        "actor": actor,
        "supersedes": supersedes,
        "payload": {
            "kind": kind,
            "product_id": product_id,
            "quantity": {"amount": amount, "unit": unit},
            "from_location": from_location,
            "to_location": to_location,
            "metadata": {},
        },
    }


def _snapshot_obj(
    record_id: str, ts: datetime, actor: str, location_id: str, entries: list[tuple[str, str, str]]
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "snapshot",
        "id": record_id,
        "ts": ts.isoformat(),
        "actor": actor,
        "supersedes": None,
        "payload": {
            "location_id": location_id,
            "entries": [
                {"product_id": p, "quantity": {"amount": a, "unit": u}, "metadata": {}}
                for p, a, u in entries
            ],
        },
    }


def test_movement_between_locations(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "5", "l", to_location="pantry"),
    )
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c2",
            T0 + timedelta(minutes=1),
            osuser,
            "movement",
            "milk",
            "2",
            "l",
            from_location="pantry",
            to_location="fridge",
        ),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry")["milk"].amount == Decimal("3")
    assert inventory.at("fridge")["milk"].amount == Decimal("2")


def test_snapshot_resets_and_later_changes_apply_on_top(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "5", "l", to_location="fridge"),
    )
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _snapshot_obj("s1", T0 + timedelta(minutes=1), osuser, "fridge", [("milk", "2", "l")]),
    )
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c2",
            T0 + timedelta(minutes=2),
            osuser,
            "consumption",
            "milk",
            "1",
            "l",
            from_location="fridge",
        ),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("fridge")["milk"].amount == Decimal("1")


def test_supersede_drops_original(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "5", "l", to_location="fridge"),
    )
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c2",
            T0 + timedelta(minutes=1),
            osuser,
            "correction",
            "milk",
            "3",
            "l",
            to_location="fridge",
            supersedes="c1",
        ),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("fridge")["milk"].amount == Decimal("3")


def test_unit_mismatch_raises(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "flour", "1", "kg", to_location="pantry"),
    )
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c2",
            T0 + timedelta(minutes=1),
            osuser,
            "purchase",
            "flour",
            "1",
            "lb",
            to_location="pantry",
        ),
    )
    with pytest.raises(ValueError, match="unit mismatch"):
        ledger.build_inventory(data_dir, key)


def test_zero_quantity_drops_entry(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge"),
    )
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c2",
            T0 + timedelta(minutes=1),
            osuser,
            "consumption",
            "milk",
            "1",
            "l",
            from_location="fridge",
        ),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert "milk" not in inventory.at("fridge")


def test_schema_version_too_new_raises(data_dir: Path, osuser: str, key: bytes) -> None:
    obj = _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge")
    obj["schema_version"] = SCHEMA_VERSION + 1
    store.append(data_dir, key, f"log:{osuser}", obj)
    with pytest.raises(SchemaVersionError):
        ledger.load_records(data_dir, key)


def test_verify_all_detects_actor_mismatch(data_dir: Path, osuser: str, key: bytes) -> None:
    obj = _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge")
    obj["actor"] = "someone-else"
    store.append(data_dir, key, f"log:{osuser}", obj)
    result = ledger.verify_all(data_dir, key)
    assert not result.ok
    assert len(result.actor_mismatches) == 1


def test_verify_all_ok_on_clean_data(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge"),
    )
    result = ledger.verify_all(data_dir, key)
    assert result.ok
