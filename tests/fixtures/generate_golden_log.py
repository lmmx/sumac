"""Regenerates tests/fixtures/golden_log/ — run manually (`uv run python
tests/fixtures/generate_golden_log.py`), not part of the test suite.

Produces a small, fully synthetic log covering every v1 change kind + both
correction shapes + snapshot shapes, and every v2 event type (including a
Correction that supersedes a v1 record), written under a fixed test-only key
via the real writer paths (decide.decide_change / decide.serialize_event /
decide.decide_correct / config.add_location / config.add_product) wherever
those exist, and by hand for v1 shapes the current writer no longer emits.

Re-running this changes the corpus's ciphertext bytes (fresh AEAD nonces
per line) but not its plaintext content or fold result — only regenerate
when the corpus itself needs to change, and re-verify
test_model_properties.py's expected holdings/anomalies afterward.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from sumac import config, decide, events, ledger, models, store

GOLDEN_KEY = bytes(range(32))
ACTOR = "alice"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _v1_change(
    record_id: str,
    ts: datetime,
    kind: str,
    product_id: str,
    amount: str,
    unit: str,
    *,
    frm: str | None = None,
    to: str | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "type": "change",
        "id": record_id,
        "ts": ts.isoformat(),
        "actor": ACTOR,
        "supersedes": None,
        "payload": {
            "kind": kind,
            "product_id": product_id,
            "quantity": {"amount": amount, "unit": unit},
            "from_location": frm,
            "to_location": to,
            "metadata": {},
        },
    }


def _v1_snapshot(
    record_id: str, ts: datetime, location_id: str, entries: list[tuple[str, str, str]]
) -> dict:
    return {
        "schema_version": 1,
        "type": "snapshot",
        "id": record_id,
        "ts": ts.isoformat(),
        "actor": ACTOR,
        "supersedes": None,
        "payload": {
            "location_id": location_id,
            "entries": [
                {"product_id": pid, "quantity": {"amount": amt, "unit": unit}, "metadata": {}}
                for pid, amt, unit in entries
            ],
        },
    }


def main() -> None:
    data_dir = Path(__file__).parent / "golden_log"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    with patch("getpass.getuser", return_value=ACTOR):
        config.add_location(
            data_dir, GOLDEN_KEY, ACTOR, models.Location(id="pantry", name="Pantry")
        )
        config.add_location(
            data_dir, GOLDEN_KEY, ACTOR, models.Location(id="fridge", name="Fridge")
        )
        config.add_product(
            data_dir, GOLDEN_KEY, ACTOR, models.Product(id="milk", name="Milk", unit="l")
        )
        config.add_product(
            data_dir, GOLDEN_KEY, ACTOR, models.Product(id="flour", name="Flour", unit="kg")
        )

        # v1: every ChangeKind (both correction shapes) + a non-empty and an
        # empty snapshot. Distinct microseconds so ordering is deterministic.
        v1_records = [
            _v1_change("v1-purchase", T0, "purchase", "milk", "2", "l", to="pantry"),
            _v1_change("v1-consumption", T0, "consumption", "milk", "1", "l", frm="pantry"),
            _v1_change("v1-waste", T0, "waste", "milk", "1", "l", frm="pantry"),
            _v1_change("v1-discovery", T0, "discovery", "flour", "1", "kg", to="pantry"),
            _v1_change("v1-correction-to", T0, "correction", "flour", "1", "kg", to="pantry"),
            _v1_change("v1-correction-from", T0, "correction", "flour", "1", "kg", frm="pantry"),
            _v1_change(
                "v1-movement", T0, "movement", "flour", "1", "kg", frm="pantry", to="fridge"
            ),
            _v1_snapshot("v1-snapshot", T0, "fridge", [("milk", "3", "l")]),
            _v1_snapshot("v1-snapshot-empty", T0, "pantry", []),
        ]
        for i, obj in enumerate(v1_records):
            obj["ts"] = T0.replace(microsecond=i).isoformat()
            store.append(data_dir, GOLDEN_KEY, f"log:{ACTOR}", obj)

        cfg = config.build_config(data_dir, GOLDEN_KEY)

        # v2: every event type via the real writer.
        inventory = ledger.build_inventory(data_dir, GOLDEN_KEY)
        writes, _messages = decide.decide_change(
            kind=models.ChangeKind.PURCHASE,
            product_id="milk",
            amount=Decimal("5"),
            unit="l",
            from_location=None,
            to_location="fridge",
            actor=ACTOR,
            occurred_at=T0.replace(hour=1),
            inventory=inventory,
            cfg=cfg,
        )
        for w in writes:
            store.append(data_dir, GOLDEN_KEY, w.stream, w.obj)

        inventory = ledger.build_inventory(data_dir, GOLDEN_KEY)
        writes, _messages = decide.decide_change(
            kind=models.ChangeKind.MOVEMENT,
            product_id="milk",
            amount=Decimal("2"),
            unit="l",
            from_location="fridge",
            to_location="pantry",
            actor=ACTOR,
            occurred_at=T0.replace(hour=2),
            inventory=inventory,
            cfg=cfg,
        )
        for w in writes:
            store.append(data_dir, GOLDEN_KEY, w.stream, w.obj)

        # Insufficient-stock consumption: forces a Counted write alongside
        # the Consumed write, covering Counted in the same pass.
        inventory = ledger.build_inventory(data_dir, GOLDEN_KEY)
        writes, _messages = decide.decide_change(
            kind=models.ChangeKind.CONSUMPTION,
            product_id="flour",
            amount=Decimal("99"),
            unit="kg",
            from_location="fridge",
            to_location=None,
            actor=ACTOR,
            occurred_at=T0.replace(hour=3),
            inventory=inventory,
            cfg=cfg,
        )
        for w in writes:
            store.append(data_dir, GOLDEN_KEY, w.stream, w.obj)

        # v2 snapshot, including the empty-entries case.
        snap_obj = decide.serialize_event(
            events.Snapshot(location_id="pantry", entries=()),
            actor=ACTOR,
            occurred_at=T0.replace(hour=4),
            cmd_id=str(uuid4()),
        )
        store.append(data_dir, GOLDEN_KEY, f"log:{ACTOR}", snap_obj)

        # Correction: cancels the v1 waste record above (cross-schema-version
        # supersede — v1 and v2 records share the same id namespace, no
        # special-casing needed).
        records = ledger.load_all_records(data_dir, GOLDEN_KEY)
        write = decide.decide_correct(
            target_id="v1-waste",
            reason="golden corpus: exercise Correction/supersedes",
            actor=ACTOR,
            occurred_at=T0.replace(hour=5),
            records=records,
        )
        store.append(data_dir, GOLDEN_KEY, write.stream, write.obj)

    print("generated", data_dir)


if __name__ == "__main__":
    main()
