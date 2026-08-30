from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sumac import SCHEMA_VERSION, config, ledger, models, paths, store

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
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
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
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
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
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
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


def test_unit_mismatch_becomes_anomaly(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
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
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry")["flour"].amount == Decimal("1")
    assert inventory.at("pantry")["flour"].unit == "kg"
    assert any(a.reason == "unit_mismatch" for a in inventory.anomalies)


def test_zero_quantity_drops_entry(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
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


def test_schema_too_new_becomes_anomaly(data_dir: Path, osuser: str, key: bytes) -> None:
    """A too-new record (e.g. from a household member who's upgraded) must not brick
    every command for everyone else until they upgrade too — it's quarantined like any
    other unfoldable record, and the rest of the log still folds."""
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge"),
    )
    obj = _change_obj(
        "c2",
        T0 + timedelta(minutes=1),
        osuser,
        "purchase",
        "eggs",
        "6",
        "ct",
        to_location="fridge",
    )
    obj["schema_version"] = SCHEMA_VERSION + 1
    store.append(data_dir, key, f"log:{osuser}", obj)

    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("fridge")["milk"].amount == Decimal("1")
    assert "eggs" not in inventory.at("fridge")
    assert any(a.reason == "schema_too_new" for a in inventory.anomalies)


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


def test_diagnose_clean_log_has_no_findings(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge"),
    )
    report = ledger.diagnose(data_dir, key)
    assert report.anomalies == ()
    assert report.total_lines == 1


def test_diagnose_flags_unknown_location(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c1",
            T0,
            osuser,
            "movement",
            "milk",
            "1",
            "l",
            from_location="pantry",
            to_location="hob-right-below-bottom",
        ),
    )
    report = ledger.diagnose(data_dir, key)
    reasons = {(f.reason, f.detail) for f in report.anomalies}
    assert ("unknown_location", "pantry") in reasons
    assert ("unknown_location", "hob-right-below-bottom") in reasons


def test_diagnose_does_not_raise_on_unit_mismatch(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
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
    report = ledger.diagnose(data_dir, key)
    assert any(f.reason == "unit_mismatch" for f in report.anomalies)


def test_diagnose_does_not_raise_on_malformed_movement(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    obj = _change_obj("c1", T0, osuser, "movement", "milk", "1", "l", to_location="pantry")
    obj["payload"]["from_location"] = None  # movement missing an endpoint
    store.append(data_dir, key, f"log:{osuser}", obj)
    report = ledger.diagnose(data_dir, key)
    assert any(f.reason == "invalid_record" for f in report.anomalies)


def test_diagnose_reports_line_failures(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge"),
    )
    log_path = paths.log_path(data_dir, osuser)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("not-valid-base64!!!\n")
    report = ledger.diagnose(data_dir, key)
    assert any(f.reason == "line_failure" for f in report.anomalies)


def test_diagnose_does_not_raise_on_unreadable_config(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """`config.load_locations` raises when a config line fails to decrypt (it uses
    `SealedLog.__iter__`, which stops at the first bad line), so it's a genuine current
    crash path for `diagnose` — unlike a circular parent, which `load_locations` doesn't
    detect at all yet (that's Phase 2a's `build_config`, not implemented here)."""
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge"),
    )
    config_path = paths.config_path(data_dir)
    with config_path.open("a", encoding="utf-8") as f:
        f.write("not-valid-base64!!!\n")

    report = ledger.diagnose(data_dir, key)
    assert any(f.reason == "config_unreadable" for f in report.anomalies)
    assert any(f.reason == "unknown_location" and f.detail == "fridge" for f in report.anomalies)


def test_build_inventory_flags_unknown_location_without_applying(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c1",
            T0,
            osuser,
            "movement",
            "milk",
            "1",
            "l",
            from_location="pantry",
            to_location="hob-right-below-bottom",
        ),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert "milk" not in inventory.at("pantry")
    assert "milk" not in inventory.at("hob-right-below-bottom")
    assert any(a.reason == "unknown_location" for a in inventory.anomalies)


def test_build_inventory_does_not_raise_on_malformed_movement(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    obj = _change_obj("c1", T0, osuser, "movement", "milk", "1", "l", to_location="pantry")
    obj["payload"]["from_location"] = None  # movement missing an endpoint
    store.append(data_dir, key, f"log:{osuser}", obj)
    inventory = ledger.build_inventory(data_dir, key)
    assert any(a.reason == "invalid_record" for a in inventory.anomalies)


def test_build_inventory_does_not_raise_on_decrypt_failure(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge"),
    )
    log_path = paths.log_path(data_dir, osuser)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("not-valid-base64!!!\n")

    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("fridge")["milk"].amount == Decimal("1")
    assert any(a.reason == "line_failure" for a in inventory.anomalies)


def test_build_inventory_does_not_raise_on_unreadable_config(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config_path = paths.config_path(data_dir)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("not-valid-base64!!!\n")
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="fridge"),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert any(a.reason == "config_unreadable" for a in inventory.anomalies)
    assert any(a.reason == "unknown_location" for a in inventory.anomalies)
