from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from sumac import SCHEMA_VERSION, config, decide, events, ledger, models, paths, store
from sumac.errors import Rejected

T0 = datetime(2026, 1, 1, tzinfo=None).astimezone()

# These build v1-shaped wire dicts specifically (kind/quantity/from_location/
# to_location) — schema_version is hardcoded to 1, not SCHEMA_VERSION (which
# now tracks the *current* build's max, 2), since a v1 shape under a v2 tag
# would (correctly) fail validation. Most of this file exists to prove the
# upcaster still handles exactly these shapes.


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
        "schema_version": 1,
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
        "schema_version": 1,
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


def test_diagnose_survives_one_bad_config_line_alongside_a_good_one(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """A corrupted config line becomes a line_failure anomaly and is skipped — it
    must not take a valid location registered earlier down with it. (Config reads
    go through `store.verify_stream`, which processes every line rather than
    stopping at the first bad one, so this is a config_unreadable-free path.)"""
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
    assert any(f.reason == "line_failure" for f in report.anomalies)
    assert not any(f.reason == "unknown_location" for f in report.anomalies)
    assert not any(f.reason == "config_unreadable" for f in report.anomalies)


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
    """No location was ever validly registered here (the whole config file is
    one corrupted line), so `fridge` is legitimately unknown_location — this
    differs from test_diagnose_survives_one_bad_config_line_alongside_a_good_one,
    where a valid registration coexists with a bad line."""
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
    assert any(a.reason == "line_failure" for a in inventory.anomalies)
    assert any(a.reason == "unknown_location" for a in inventory.anomalies)


def test_build_inventory_surfaces_circular_parent_without_crashing(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="a", name="A", parent_id="b"))
    config.add_location(data_dir, key, osuser, models.Location(id="b", name="B", parent_id="a"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="a"),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("a")["milk"].amount == Decimal("1")
    assert any(a.reason == "circular_parent" for a in inventory.anomalies)


def test_build_inventory_folds_movement_to_retired_location(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """Referential integrity is monotone (§3.4): retiring a location stops new
    writes (Phase 3's job) but must not un-resolve historical ones."""
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    config.retire_location(data_dir, key, osuser, "pantry")
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "1", "l", to_location="pantry"),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry")["milk"].amount == Decimal("1")
    assert inventory.anomalies == ()


def test_observed_product_units_tallies_changes_and_snapshots(
    data_dir: Path, osuser: str, key: bytes
) -> None:
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
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _snapshot_obj("s1", T0 + timedelta(minutes=2), osuser, "pantry", [("flour", "2", "kg")]),
    )
    observed = ledger.observed_product_units(data_dir, key)
    assert observed["flour"] == {"kg": 2, "lb": 1}


def test_observed_product_units_includes_records_that_cannot_fold(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """Backfill needs what was actually written, not just what currently folds —
    a movement to an unregistered location still recorded a real unit."""
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c1", T0, osuser, "purchase", "milk", "1", "l", to_location="hob-right-below-bottom"
        ),
    )
    observed = ledger.observed_product_units(data_dir, key)
    assert observed["milk"] == {"l": 1}


def test_empty_snapshot_clears_prior_holdings(data_dir: Path, osuser: str, key: bytes) -> None:
    """The finding that drove Phase 4a's design (§3.3a): a 0-entry snapshot
    means "this location is empty" and must reset it, not be a no-op."""
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
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
        _snapshot_obj("s1", T0 + timedelta(minutes=1), osuser, "pantry", []),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry") == {}
    assert inventory.anomalies == ()


def test_correction_to_only_adds_stock(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "correction", "flour", "2", "kg", to_location="pantry"),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry")["flour"].amount == Decimal("2")
    assert inventory.anomalies == ()


def test_correction_from_only_removes_stock(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "flour", "5", "kg", to_location="pantry"),
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
            "flour",
            "2",
            "kg",
            from_location="pantry",
        ),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry")["flour"].amount == Decimal("3")


def test_correction_with_both_endpoints_becomes_upcast_failed_anomaly(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """A structurally-possible-but-unmapped correction shape (both from and
    to set — InventoryChange.__post_init__ doesn't constrain correction at
    all) must quarantine, not silently misapply as a two-sided movement."""
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c1",
            T0,
            osuser,
            "correction",
            "eggs",
            "1",
            "ct",
            from_location="pantry",
            to_location="fridge",
        ),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry") == {}
    assert inventory.at("fridge") == {}
    assert any(a.reason == "upcast_failed" for a in inventory.anomalies)


def test_discovery_folds_like_purchase(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "discovery", "jam", "1", "jar", to_location="pantry"),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry")["jam"].amount == Decimal("1")
    assert inventory.anomalies == ()


def test_movement_neither_side_commits_when_only_destination_mismatches(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """_apply_sides must be atomic w.r.t. failures: a Moved whose source side
    would resolve cleanly but whose destination side hits a unit mismatch
    must leave *both* sides untouched, not just skip the bad one."""
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "flour", "5", "kg", to_location="pantry"),
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
            to_location="fridge",
        ),
    )
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj(
            "c3",
            T0 + timedelta(minutes=2),
            osuser,
            "movement",
            "flour",
            "1",
            "kg",
            from_location="pantry",
            to_location="fridge",
        ),
    )
    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry")["flour"].amount == Decimal("5")  # unchanged, not 4
    assert inventory.at("fridge")["flour"].amount == Decimal("1")
    assert inventory.at("fridge")["flour"].unit == "lb"  # unchanged, not overwritten
    assert any(a.reason == "unit_mismatch" for a in inventory.anomalies)


def test_mixed_v1_and_v2_records_fold_together(data_dir: Path, osuser: str, key: bytes) -> None:
    """Phase 4b's acceptance criterion (§3.3a/§5): a log containing both v1
    and v2 records folds correctly. v1 keeps working through the upcaster
    forever; v2 is read natively. Both touch the same product/location so a
    mistake in either path, or in how they compose, shows up as a wrong total."""
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "2", "l", to_location="pantry"),
    )
    v2_obj = decide.serialize_event(
        events.Acquired(product_id="milk", to="pantry", amount=Decimal("3"), unit="l"),
        actor=osuser,
        occurred_at=T0 + timedelta(minutes=1),
    )
    store.append(data_dir, key, f"log:{osuser}", v2_obj)

    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry")["milk"].amount == Decimal("5")
    assert inventory.anomalies == ()


def test_mixed_v1_and_v2_snapshot_reset_interacts_correctly(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """A v2 snapshot must reset a location exactly like a v1 one does — the
    baseline-gating logic in build_inventory doesn't get to know or care
    which version produced it."""
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("c1", T0, osuser, "purchase", "milk", "5", "l", to_location="pantry"),
    )
    v2_snapshot = decide.serialize_event(
        events.Snapshot(location_id="pantry", entries=()),
        actor=osuser,
        occurred_at=T0 + timedelta(minutes=1),
    )
    store.append(data_dir, key, f"log:{osuser}", v2_snapshot)

    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry") == {}
    assert inventory.anomalies == ()


def test_insufficient_stock_counted_actually_precedes_the_movement_after_reload(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """Regression test for a real bug: decide_change's `writes` list has the
    Counted correction before the movement, but that ordering only matters if
    it survives storage and refold. It didn't — both writes shared the same
    `occurred_at`, so the fold's (ts, actor, id) sort tie-broke on a random
    uuid4, and the movement committed before the Counted about half the time,
    silently overwriting the correction's effect (caught by manual end-to-end
    smoke testing, not by decide.py's own unit tests, which only ever checked
    list order). Exercises the actual write -> store -> reload -> fold path,
    not just decide_change's return value, since that's what the bug needed."""
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    config.add_location(data_dir, key, osuser, models.Location(id="fridge", name="Fridge"))
    config.add_product(data_dir, key, osuser, models.Product(id="milk", name="Milk", unit="l"))

    cfg = config.build_config(data_dir, key)
    writes, _messages = decide.decide_change(
        kind=models.ChangeKind.PURCHASE,
        product_id="milk",
        amount=Decimal("1"),
        unit="l",
        from_location=None,
        to_location="pantry",
        actor=osuser,
        occurred_at=T0,
        inventory=ledger.build_inventory(data_dir, key),
        cfg=cfg,
    )
    for w in writes:
        store.append(data_dir, key, w.stream, w.obj)

    cfg = config.build_config(data_dir, key)
    writes, messages = decide.decide_change(
        kind=models.ChangeKind.MOVEMENT,
        product_id="milk",
        amount=Decimal("3"),
        unit="l",
        from_location="pantry",
        to_location="fridge",
        actor=osuser,
        occurred_at=T0 + timedelta(minutes=1),
        inventory=ledger.build_inventory(data_dir, key),
        cfg=cfg,
    )
    assert any("adjusted" in m for m in messages)
    for w in writes:
        store.append(data_dir, key, w.stream, w.obj)

    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry") == {}
    assert inventory.at("fridge")["milk"].amount == Decimal("3")


def test_correction_cancels_target_record_from_fold(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """§3.6: supersedes means cancel, not replace — the targeted purchase
    must vanish from the fold entirely once corrected, and the Correction
    record itself contributes nothing on its own merits."""
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("bad-1", T0, osuser, "purchase", "milk", "2", "l", to_location="pantry"),
    )
    records = ledger.load_all_records(data_dir, key)
    write = decide.decide_correct(
        target_id="bad-1",
        reason="typo, wrong product",
        actor=osuser,
        occurred_at=T0 + timedelta(minutes=1),
        records=records,
    )
    store.append(data_dir, key, write.stream, write.obj)

    inventory = ledger.build_inventory(data_dir, key)
    assert inventory.at("pantry") == {}
    assert inventory.anomalies == ()


def test_load_all_records_keeps_superseded_load_records_drops_them(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("bad-1", T0, osuser, "purchase", "milk", "2", "l", to_location="pantry"),
    )
    records = ledger.load_all_records(data_dir, key)
    write = decide.decide_correct(
        target_id="bad-1",
        reason="typo",
        actor=osuser,
        occurred_at=T0 + timedelta(minutes=1),
        records=records,
    )
    store.append(data_dir, key, write.stream, write.obj)

    assert "bad-1" not in {r.id for r in ledger.load_records(data_dir, key)}
    assert "bad-1" in {r.id for r in ledger.load_all_records(data_dir, key)}


def test_correcting_an_already_superseded_record_is_rejected(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """`decide_correct` needs the unfiltered view (`load_all_records`) to
    tell this apart from `supersede_target_missing` — a filtered view would
    make an already-corrected record look like it never existed."""
    config.add_location(data_dir, key, osuser, models.Location(id="pantry", name="Pantry"))
    store.append(
        data_dir,
        key,
        f"log:{osuser}",
        _change_obj("bad-1", T0, osuser, "purchase", "milk", "2", "l", to_location="pantry"),
    )
    records = ledger.load_all_records(data_dir, key)
    write = decide.decide_correct(
        target_id="bad-1",
        reason="typo",
        actor=osuser,
        occurred_at=T0 + timedelta(minutes=1),
        records=records,
    )
    store.append(data_dir, key, write.stream, write.obj)

    records = ledger.load_all_records(data_dir, key)
    with pytest.raises(Rejected) as exc_info:
        decide.decide_correct(
            target_id="bad-1",
            reason="again",
            actor=osuser,
            occurred_at=T0 + timedelta(minutes=2),
            records=records,
        )
    assert exc_info.value.reason == "supersede_already_applied"
