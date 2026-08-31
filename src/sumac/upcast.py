"""v1 → v2 upcaster — see docs/journal/2026-08-30 §3.3a for the mapping
table and the reasoning behind each mapping.

Pure domain-to-domain transform: takes an already-validated v1
`models.Record` and returns the one v2 event it represents. Every real
mapping is 1:1 (§3.3a's "keep Snapshot whole" call), so this returns a
single event, never a list.

Raises `UpcastError` for a shape the mapping table doesn't cover. Only
`ChangeKind.CORRECTION` can actually produce one in practice — every other
`ChangeKind` has its from/to shape enforced by `InventoryChange.__post_init__`
before a record ever reaches here, so those checks below are unreachable
tripwires, not live validation.
"""

from __future__ import annotations

from sumac import events
from sumac.models import ChangeKind, InventoryChange, InventorySnapshot, Record


class UpcastError(ValueError):
    """A v1 record's shape doesn't match anything in the v1→v2 mapping
    table — structurally possible under v1's loose per-kind validation
    (only `CORRECTION` is actually unconstrained), never observed in the
    real log as of the Phase 4a design pass."""


def upcast(record: Record) -> events.Event:
    payload = record.payload

    if isinstance(payload, InventorySnapshot):
        return events.Snapshot(
            location_id=payload.location_id,
            entries=tuple(
                events.SnapshotEntry(
                    product_id=e.product_id,
                    amount=e.quantity.amount,
                    unit=e.quantity.unit,
                )
                for e in payload.entries
            ),
        )

    assert isinstance(payload, InventoryChange)
    kind = payload.kind
    frm, to = payload.from_location, payload.to_location
    amount, unit = payload.quantity.amount, payload.quantity.unit
    product_id = payload.product_id

    if kind is ChangeKind.MOVEMENT:
        assert frm is not None and to is not None  # __post_init__ guarantees this
        return events.Moved(product_id=product_id, frm=frm, to=to, amount=amount, unit=unit)

    if kind is ChangeKind.PURCHASE:
        assert to is not None
        return events.Acquired(product_id=product_id, to=to, amount=amount, unit=unit)

    if kind is ChangeKind.DISCOVERY:
        assert to is not None
        return events.Acquired(
            product_id=product_id, to=to, amount=amount, unit=unit, reason="discovery"
        )

    if kind is ChangeKind.CONSUMPTION:
        assert frm is not None
        return events.Consumed(product_id=product_id, frm=frm, amount=amount, unit=unit)

    if kind is ChangeKind.WASTE:
        assert frm is not None
        return events.Discarded(product_id=product_id, frm=frm, amount=amount, unit=unit)

    if kind is ChangeKind.CORRECTION:
        # Unlike every other kind, InventoryChange.__post_init__ doesn't
        # constrain correction's shape at all — this is the one live check.
        if to is not None and frm is None:
            return events.Acquired(
                product_id=product_id, to=to, amount=amount, unit=unit, reason="correction"
            )
        if frm is not None and to is None:
            return events.Consumed(
                product_id=product_id, frm=frm, amount=amount, unit=unit, reason="correction"
            )
        raise UpcastError(
            f"correction with ambiguous shape (from={frm!r}, to={to!r}) on record {record.id}"
        )

    raise UpcastError(f"unrecognized ChangeKind {kind!r} on record {record.id}")  # unreachable
