from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from unittest.mock import patch

import pytest

from sumac import config, decide, events, ledger
from sumac.errors import Rejected
from sumac.models import ChangeKind, Location, Product, Quantity, Record
from sumac.schemas import RecordSchema

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
    # §5.3: input unit ("l") already equals milk's canonical unit — nothing
    # was converted, so there is deliberately nothing to record here.
    assert writes[0].obj["payload"]["nominal_basis"] is None


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
    # §5.3: the one existing test that exercises a real conversion — the
    # producer must populate nominal_basis here, not leave it null forever.
    assert writes[0].obj["payload"]["nominal_basis"] == {
        "raw_amount": "2",
        "raw_unit": "jar",
        "ratio": "340",
    }


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
    # §5.3: an auto-registered product's canonical unit *is* the unit just
    # used, so nothing was converted — nominal_basis stays None.
    assert writes[1].obj["payload"]["nominal_basis"] is None
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


def test_nominal_basis_round_trips_through_record_schema() -> None:
    """No fixture currently exercises a real conversion (golden_log's
    decide_change calls all use each product's own canonical unit — checked
    directly against tests/fixtures/generate_golden_log.py), so this is the
    only coverage of a populated nominal_basis surviving the real ingest
    path: `RecordSchema.model_validate(...).to_domain()` must not reject a
    payload this producer writes (the ingest type is `dict[str, str]` —
    every value here is already a `str`, not a `Decimal`)."""
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
    writes, _messages = _decide(product_id="rice-pudding", amount=Decimal("2"), unit="jar", cfg=cfg)
    record = RecordSchema.model_validate(writes[0].obj).to_domain()
    assert isinstance(record.payload, events.Acquired)
    assert record.payload.nominal_basis == {"raw_amount": "2", "raw_unit": "jar", "ratio": "340"}


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


# --- §5.2: _reconcile_shortfall, pinned directly (not just via decide_change) ---


def test_reconcile_shortfall_acquired_short_circuits_before_the_match() -> None:
    """An event type with no `frm` side at all (not just an absent one)
    returns `([], [])` immediately."""
    event = events.Acquired(product_id="milk", to="pantry", amount=Decimal("1"), unit="l")
    writes, messages = decide._reconcile_shortfall(
        event, _EMPTY_INVENTORY, actor="alice", occurred_at=T0, cmd_id="cmd-1"
    )
    assert writes == []
    assert messages == []


def test_reconcile_shortfall_emits_counted_reading_amount_off_the_event() -> None:
    """Amount/unit come from `event`, not a separately-threaded `canon` —
    this pins that by never passing anything but the event itself."""
    event = events.Consumed(product_id="milk", frm="pantry", amount=Decimal("3"), unit="l")
    inventory = ledger.Inventory(by_location={"pantry": {"milk": Quantity(Decimal("1"), "l")}})
    writes, messages = decide._reconcile_shortfall(
        event, inventory, actor="alice", occurred_at=T0, cmd_id="cmd-1"
    )
    assert len(writes) == 1
    assert writes[0].obj["type"] == "counted"
    assert writes[0].obj["payload"]["amount"] == "3"
    assert writes[0].obj["payload"]["at"] == "pantry"
    assert any("adjusted" in m for m in messages)


def _record(record_id: str, *, supersedes: str | None = None) -> Record:
    return Record(
        schema_version=2,
        type="acquired",
        id=record_id,
        ts=T0,
        actor="alice",
        supersedes=supersedes,
        payload=events.Acquired(product_id="milk", to="pantry", amount=Decimal("1"), unit="l"),
    )


def test_correct_produces_a_correction_write_superseding_the_target() -> None:
    write = decide.decide_correct(
        target_id="bad-1",
        reason="typo, location does not exist",
        actor="alice",
        occurred_at=T0,
        records=[_record("bad-1")],
    )
    assert write.stream == "log:alice"
    assert write.obj["schema_version"] == 2
    assert write.obj["type"] == "correction"
    assert write.obj["supersedes"] == "bad-1"
    assert write.obj["payload"] == {"reason": "typo, location does not exist"}


def test_correct_missing_target_is_rejected() -> None:
    with pytest.raises(Rejected) as exc_info:
        decide.decide_correct(
            target_id="nope", reason="typo", actor="alice", occurred_at=T0, records=[_record("r1")]
        )
    assert exc_info.value.reason == "supersede_target_missing"


def test_correct_already_superseded_target_is_rejected() -> None:
    """`records` must be the unfiltered view: a target that's already been
    superseded once is still `in the log` — it must be told apart from one
    that never existed, not conflated with `supersede_target_missing`."""
    records = [_record("bad-1"), _record("fix-1", supersedes="bad-1")]
    with pytest.raises(Rejected) as exc_info:
        decide.decide_correct(
            target_id="bad-1", reason="again", actor="alice", occurred_at=T0, records=records
        )
    assert exc_info.value.reason == "supersede_already_applied"


def test_correct_blank_reason_is_rejected() -> None:
    with pytest.raises(Rejected) as exc_info:
        decide.decide_correct(
            target_id="bad-1",
            reason="   ",
            actor="alice",
            occurred_at=T0,
            records=[_record("bad-1")],
        )
    assert exc_info.value.reason == "missing_reason"


def test_correct_self_supersede_is_rejected() -> None:
    """Unreachable via the CLI in practice (the new record's id is a fresh
    uuid4, never the target's), but `decide_correct` guards it anyway per
    §4's catalogue — verified here by forcing the generated id to collide."""
    with patch("sumac.decide.uuid4", return_value="bad-1"):
        with pytest.raises(Rejected) as exc_info:
            decide.decide_correct(
                target_id="bad-1",
                reason="typo",
                actor="alice",
                occurred_at=T0,
                records=[_record("bad-1")],
            )
    assert exc_info.value.reason == "supersede_self"


# --- §5.1: decide_correct routed through serialize_event ---


def test_correct_write_has_exactly_the_envelope_keys() -> None:
    """Not byte-identity (RecordSchema doesn't care about key order — see
    docs/journal/2026-08-31-decide-simplification-review.md §5.1) — the key
    *set* is what an `extra="forbid"` schema would refuse to read back if
    it drifted."""
    write = decide.decide_correct(
        target_id="bad-1",
        reason="typo, location does not exist",
        actor="alice",
        occurred_at=T0,
        records=[_record("bad-1")],
    )
    assert set(write.obj) == {
        "schema_version",
        "type",
        "id",
        "ts",
        "actor",
        "supersedes",
        "cmd_id",
        "payload",
    }
    assert write.obj["actor"] == "alice"
    assert write.obj["supersedes"] == "bad-1"
    assert write.obj["payload"] == {"reason": "typo, location does not exist"}


def test_correct_write_round_trips_through_record_schema() -> None:
    """The actual regression guard for "still readable" — stronger than
    eyeballing key order: parses back through the real ingest path used by
    every reader (`RecordSchema.model_validate(...).to_domain()`)."""
    write = decide.decide_correct(
        target_id="bad-1",
        reason="typo, location does not exist",
        actor="alice",
        occurred_at=T0,
        records=[_record("bad-1")],
    )
    record = RecordSchema.model_validate(write.obj).to_domain()
    assert record.type == "correction"
    assert record.actor == "alice"
    assert record.supersedes == "bad-1"
    assert record.payload == events.Correction(reason="typo, location does not exist")
