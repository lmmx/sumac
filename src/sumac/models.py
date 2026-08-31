"""Standalone data model: frozen dataclasses, no I/O, no crypto, no pydantic.

Edit this module to change what sumac stores; nothing else in the package
should need to change in step with it.

`Record` is the envelope written to JSONL (id/ts/actor/supersedes live there,
once). Its payload is either a v1 shape (`InventoryChange`/`InventorySnapshot`
— schema_version 1, the only thing ever written before Phase 4b) or a v2
event (`sumac.events.Event` — schema_version 2, what the writer emits now).
Both live in the same log forever; nothing gets rewritten. See
docs/journal/2026-08-30 §3.3a/§3.3 for the full v1/v2 design.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from sumac import events

JsonValue = None | bool | int | float | str | list["JsonValue"] | Mapping[str, "JsonValue"]


class ChangeKind(StrEnum):
    PURCHASE = "purchase"
    CONSUMPTION = "consumption"
    WASTE = "waste"
    DISCOVERY = "discovery"
    CORRECTION = "correction"
    MOVEMENT = "movement"


@dataclass(frozen=True, slots=True)
class Location:
    id: str
    name: str
    parent_id: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    retired: bool = False


@dataclass(frozen=True, slots=True)
class Product:
    id: str
    name: str
    unit: str  # canonical unit; nominal, not authoritative — see Config.convert
    category: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    retired: bool = False
    # alt unit -> how many canonical units one of it equals, e.g. {"jar": 340}
    # for a canonical unit of "g" meaning 1 jar = 340 g. Nominal, not exact —
    # see Config.convert and §3.4(c) of the design journal.
    conversions: Mapping[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Quantity:
    amount: Decimal
    unit: str

    def __add__(self, other: Quantity) -> Quantity:
        if self.unit != other.unit:
            raise ValueError(f"unit mismatch: {self.unit!r} vs {other.unit!r}")
        return Quantity(self.amount + other.amount, self.unit)

    def __neg__(self) -> Quantity:
        return Quantity(-self.amount, self.unit)


@dataclass(frozen=True, slots=True)
class InventoryChange:
    """A delta (purchase/consumption/waste/discovery/correction) or a transfer (movement)."""

    kind: ChangeKind
    product_id: str
    quantity: Quantity
    from_location: str | None = None
    to_location: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind is ChangeKind.MOVEMENT:
            if self.from_location is None or self.to_location is None:
                raise ValueError("movement requires both from_location and to_location")
        elif self.kind in (ChangeKind.PURCHASE, ChangeKind.DISCOVERY):
            if self.to_location is None:
                raise ValueError(f"{self.kind} requires to_location")
        elif self.kind in (ChangeKind.CONSUMPTION, ChangeKind.WASTE):
            if self.from_location is None:
                raise ValueError(f"{self.kind} requires from_location")


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    product_id: str
    quantity: Quantity
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    """The full observed state of one location at one time; resets rather than merges."""

    location_id: str
    entries: tuple[SnapshotEntry, ...]


@dataclass(frozen=True, slots=True)
class Record:
    """Envelope for every JSONL line: a v1 change/snapshot, or a v2 event."""

    schema_version: int
    # v1: "snapshot" | "change"
    # v2: "snapshot" | "acquired" | "consumed" | "discarded" | "moved" | "counted"
    type: str
    id: str
    ts: datetime
    actor: str
    supersedes: str | None
    payload: InventorySnapshot | InventoryChange | events.Event


@dataclass(frozen=True, slots=True)
class Anomaly:
    """A record, line, or config entry the fold could not apply or resolve.
    Never raised — only recorded. Shared between `ledger` (data-level anomalies)
    and `config` (e.g. `circular_parent`) so both surface through one channel."""

    record_id: str | None
    reason: str  # "line_failure" | "invalid_record" | "unknown_location" | "unit_mismatch" | ...
    detail: str
