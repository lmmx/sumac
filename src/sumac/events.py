"""v2 domain events — see docs/journal/2026-08-30 §3.3/§3.3a for the design,
the field-by-field schemas, and the v1→v2 mapping table.

In-memory only as of Phase 4a: the writer still emits v1
(`sumac.models.InventoryChange`/`InventorySnapshot`), upcast at read time by
`sumac.upcast`. Phase 4b is what starts writing these directly.

`Retired(entity_type, entity_id)` from §3.3's original sketch is deliberately
not here. Retirement already shipped in Phase 2a/2b as a `retired: bool`
field on `Location`/`Product` in the *config* stream (latest-revision-wins
re-append) — a log-stream `Retired` event would be a second, conflicting
mechanism for the same fact. §3.3's sketch predates that implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

# Audit-only, never read by the fold: what was actually typed and the
# conversion ratio applied, e.g. {"raw_amount": "2", "raw_unit": "jar", "ratio": "340"}.
NominalBasis = dict[str, str] | None


@dataclass(frozen=True, slots=True)
class Acquired:
    product_id: str
    to: str
    amount: Decimal
    unit: str
    reason: str | None = None  # "correction" | "discovery" | None (ordinary purchase)
    nominal_basis: NominalBasis = None


@dataclass(frozen=True, slots=True)
class Consumed:
    product_id: str
    frm: str
    amount: Decimal
    unit: str
    reason: str | None = None  # "correction" | None (ordinary consumption)
    nominal_basis: NominalBasis = None


@dataclass(frozen=True, slots=True)
class Discarded:
    product_id: str
    frm: str
    amount: Decimal
    unit: str
    nominal_basis: NominalBasis = None


@dataclass(frozen=True, slots=True)
class Moved:
    product_id: str
    frm: str
    to: str
    amount: Decimal
    unit: str
    nominal_basis: NominalBasis = None


@dataclass(frozen=True, slots=True)
class Counted:
    """Observed truth, absolute — sets the holding rather than adding to it.
    New in v2; never produced by the v1 upcaster (v1 had no single-product
    absolute count). Phase 4b's `decide` is the only producer, for §3.5's
    insufficient-stock behavior."""

    product_id: str
    at: str
    amount: Decimal
    unit: str
    reason: str | None = None  # e.g. "implied_by_movement"
    nominal_basis: NominalBasis = None


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    product_id: str
    amount: Decimal
    unit: str
    nominal_basis: NominalBasis = None


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Resets every product at `location_id` to exactly `entries` — an empty
    tuple is valid and means the location is empty, not "nothing happened."
    Kept as its own event rather than decomposed into per-product `Counted`;
    see §3.3a for why (the empty-snapshot data-loss finding)."""

    location_id: str
    entries: tuple[SnapshotEntry, ...]


@dataclass(frozen=True, slots=True)
class Correction:
    """Cancel-only payload for `Record.supersedes` (§3.6): the targeted record
    is excluded from the fold and this carries no change of its own — `reason`
    is why, `actor` is deliberately not here since `Record.actor` already has
    it. A record that both cancels and asserts a real replacement event
    ("replace" in §3.6's terms) uses one of the event types above with
    `reason="correction"` and `supersedes` set, not this."""

    reason: str


Event = Acquired | Consumed | Discarded | Moved | Counted | Snapshot | Correction
