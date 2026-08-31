"""Pydantic v2 models at ingest boundaries: parsing untrusted JSON into `models.py` types.

Every sumac entry point that accepts external data (JSONL lines, CLI input) should
construct one of these, call `.to_domain()`, and never hand raw dicts further downstream.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from sumac import events, models
from sumac.models import ChangeKind

MetadataDict = Annotated[dict[str, JsonValue], Field(default_factory=dict)]


def _parse_decimal(v: object) -> Decimal:
    try:
        return Decimal(str(v))
    except InvalidOperation as e:
        raise ValueError(f"invalid decimal amount: {v!r}") from e


DecimalAmount = Annotated[Decimal, BeforeValidator(_parse_decimal)]
NominalBasisDict = dict[str, str] | None


class LocationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parent_id: str | None = None
    metadata: MetadataDict
    retired: bool = False

    def to_domain(self) -> models.Location:
        return models.Location(
            id=self.id,
            name=self.name,
            parent_id=self.parent_id,
            metadata=self.metadata,
            retired=self.retired,
        )


class ProductSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    category: str | None = None
    metadata: MetadataDict
    retired: bool = False
    conversions: dict[str, Decimal] = Field(default_factory=dict)

    @field_validator("conversions", mode="before")
    @classmethod
    def _parse_conversions(cls, v: object) -> dict[str, Decimal]:
        if not isinstance(v, dict):
            raise ValueError(f"conversions must be a mapping, got {v!r}")
        try:
            return {str(unit): Decimal(str(ratio)) for unit, ratio in v.items()}
        except InvalidOperation as e:
            raise ValueError(f"invalid decimal in conversions: {v!r}") from e

    def to_domain(self) -> models.Product:
        return models.Product(
            id=self.id,
            name=self.name,
            unit=self.unit,
            category=self.category,
            metadata=self.metadata,
            retired=self.retired,
            conversions=self.conversions,
        )


class QuantitySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: Decimal
    unit: str = Field(min_length=1)

    @field_validator("amount", mode="before")
    @classmethod
    def _parse_amount(cls, v: object) -> Decimal:
        try:
            return Decimal(str(v))
        except InvalidOperation as e:
            raise ValueError(f"invalid decimal amount: {v!r}") from e

    def to_domain(self) -> models.Quantity:
        return models.Quantity(amount=self.amount, unit=self.unit)


class InventoryChangeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ChangeKind
    product_id: str = Field(min_length=1)
    quantity: QuantitySchema
    from_location: str | None = None
    to_location: str | None = None
    metadata: MetadataDict

    def to_domain(self) -> models.InventoryChange:
        return models.InventoryChange(
            kind=self.kind,
            product_id=self.product_id,
            quantity=self.quantity.to_domain(),
            from_location=self.from_location,
            to_location=self.to_location,
            metadata=self.metadata,
        )


class SnapshotEntrySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    quantity: QuantitySchema
    metadata: MetadataDict

    def to_domain(self) -> models.SnapshotEntry:
        return models.SnapshotEntry(
            product_id=self.product_id,
            quantity=self.quantity.to_domain(),
            metadata=self.metadata,
        )


class InventorySnapshotSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location_id: str = Field(min_length=1)
    entries: tuple[SnapshotEntrySchema, ...]

    def to_domain(self) -> models.InventorySnapshot:
        return models.InventorySnapshot(
            location_id=self.location_id,
            entries=tuple(e.to_domain() for e in self.entries),
        )


# --- v2 (schema_version 2) — see docs/journal/2026-08-30 §3.3/§3.3a ---
# Field names deliberately differ from the v1 schemas above (frm/to/amount/unit
# vs from_location/to_location/quantity) so the two never structurally collide
# under RecordSchema.payload's union — see the note there.


class AcquiredSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    to: str = Field(min_length=1)
    amount: DecimalAmount
    unit: str = Field(min_length=1)
    reason: str | None = None
    nominal_basis: NominalBasisDict = None

    def to_domain(self) -> events.Acquired:
        return events.Acquired(
            product_id=self.product_id,
            to=self.to,
            amount=self.amount,
            unit=self.unit,
            reason=self.reason,
            nominal_basis=self.nominal_basis,
        )


class ConsumedSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    frm: str = Field(min_length=1)
    amount: DecimalAmount
    unit: str = Field(min_length=1)
    reason: str | None = None
    nominal_basis: NominalBasisDict = None

    def to_domain(self) -> events.Consumed:
        return events.Consumed(
            product_id=self.product_id,
            frm=self.frm,
            amount=self.amount,
            unit=self.unit,
            reason=self.reason,
            nominal_basis=self.nominal_basis,
        )


class DiscardedSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    frm: str = Field(min_length=1)
    amount: DecimalAmount
    unit: str = Field(min_length=1)
    nominal_basis: NominalBasisDict = None

    def to_domain(self) -> events.Discarded:
        return events.Discarded(
            product_id=self.product_id,
            frm=self.frm,
            amount=self.amount,
            unit=self.unit,
            nominal_basis=self.nominal_basis,
        )


class MovedSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    frm: str = Field(min_length=1)
    to: str = Field(min_length=1)
    amount: DecimalAmount
    unit: str = Field(min_length=1)
    nominal_basis: NominalBasisDict = None

    def to_domain(self) -> events.Moved:
        return events.Moved(
            product_id=self.product_id,
            frm=self.frm,
            to=self.to,
            amount=self.amount,
            unit=self.unit,
            nominal_basis=self.nominal_basis,
        )


class CountedSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    at: str = Field(min_length=1)
    amount: DecimalAmount
    unit: str = Field(min_length=1)
    reason: str | None = None
    nominal_basis: NominalBasisDict = None

    def to_domain(self) -> events.Counted:
        return events.Counted(
            product_id=self.product_id,
            at=self.at,
            amount=self.amount,
            unit=self.unit,
            reason=self.reason,
            nominal_basis=self.nominal_basis,
        )


class SnapshotEntrySchemaV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    amount: DecimalAmount
    unit: str = Field(min_length=1)
    nominal_basis: NominalBasisDict = None

    def to_domain(self) -> events.SnapshotEntry:
        return events.SnapshotEntry(
            product_id=self.product_id,
            amount=self.amount,
            unit=self.unit,
            nominal_basis=self.nominal_basis,
        )


class SnapshotSchemaV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location_id: str = Field(min_length=1)
    entries: tuple[SnapshotEntrySchemaV2, ...]

    def to_domain(self) -> events.Snapshot:
        return events.Snapshot(
            location_id=self.location_id,
            entries=tuple(e.to_domain() for e in self.entries),
        )


class ConfigRecordSchema(BaseModel):
    """A config line is a location record or a product record, never both. Both
    fields are optional (rather than a `type` + payload union, as `RecordSchema`
    uses) so that lines written before products existed — which have only
    `location` and no `product` key at all — keep validating unchanged."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    ts: datetime
    actor: str = Field(min_length=1)
    location: LocationSchema | None = None
    product: ProductSchema | None = None

    @model_validator(mode="after")
    def _check_exactly_one(self) -> ConfigRecordSchema:
        if (self.location is None) == (self.product is None):
            raise ValueError("config record must set exactly one of location or product")
        return self


_V1_PAYLOAD_BY_TYPE: dict[str, type[BaseModel]] = {
    "change": InventoryChangeSchema,
    "snapshot": InventorySnapshotSchema,
}
_V2_PAYLOAD_BY_TYPE: dict[str, type[BaseModel]] = {
    "acquired": AcquiredSchema,
    "consumed": ConsumedSchema,
    "discarded": DiscardedSchema,
    "moved": MovedSchema,
    "counted": CountedSchema,
    "snapshot": SnapshotSchemaV2,
}


class RecordSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    type: Literal["change", "snapshot", "acquired", "consumed", "discarded", "moved", "counted"]
    id: str = Field(min_length=1)
    ts: datetime
    actor: str = Field(min_length=1)
    supersedes: str | None = None
    payload: (
        InventoryChangeSchema
        | InventorySnapshotSchema
        | AcquiredSchema
        | ConsumedSchema
        | DiscardedSchema
        | MovedSchema
        | CountedSchema
        | SnapshotSchemaV2
    )

    @model_validator(mode="before")
    @classmethod
    def _route_payload_by_version(cls, data: object) -> object:
        """Pydantic's union resolution alone can't be trusted to pick the
        right payload schema: v1's `InventorySnapshotSchema` and v2's
        `SnapshotSchemaV2` are structurally identical for a *zero-entry*
        snapshot (both are just `location_id` + an empty `entries` tuple),
        so with nothing to disambiguate on, it always matches whichever is
        listed first — v1 — regardless of `schema_version`. That's exactly
        the empty-snapshot case §3.3a's whole design is about, so silently
        misreading one as the other isn't acceptable. Pre-validate the
        payload against the one schema `(schema_version, type)` actually
        names, before the union ever has to guess."""
        if not isinstance(data, dict):
            return data
        table = _V1_PAYLOAD_BY_TYPE if data.get("schema_version") == 1 else _V2_PAYLOAD_BY_TYPE
        expected = table.get(data.get("type"))
        payload = data.get("payload")
        if expected is not None and isinstance(payload, dict):
            data = dict(data)
            data["payload"] = expected.model_validate(payload)
        return data

    @model_validator(mode="after")
    def _check_payload_matches_type(self) -> RecordSchema:
        # Backstop for a (schema_version, type) pair with no entry in either
        # table at all (e.g. type="counted" under schema_version=1) — the
        # before-validator above leaves payload unrouted in that case, so
        # this is what actually produces the rejection for it.
        table = _V1_PAYLOAD_BY_TYPE if self.schema_version == 1 else _V2_PAYLOAD_BY_TYPE
        expected = table.get(self.type)
        if expected is None or not isinstance(self.payload, expected):
            raise ValueError(
                f"payload does not match type={self.type!r} "
                f"for schema_version={self.schema_version}"
            )
        return self

    def to_domain(self) -> models.Record:
        return models.Record(
            schema_version=self.schema_version,
            type=self.type,
            id=self.id,
            ts=self.ts,
            actor=self.actor,
            supersedes=self.supersedes,
            payload=self.payload.to_domain(),
        )
