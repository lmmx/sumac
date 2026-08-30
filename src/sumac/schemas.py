"""Pydantic v2 models at ingest boundaries: parsing untrusted JSON into `models.py` types.

Every sumac entry point that accepts external data (JSONL lines, CLI input) should
construct one of these, call `.to_domain()`, and never hand raw dicts further downstream.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from sumac import models
from sumac.models import ChangeKind

MetadataDict = Annotated[dict[str, JsonValue], Field(default_factory=dict)]


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


class RecordSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    type: Literal["change", "snapshot"]
    id: str = Field(min_length=1)
    ts: datetime
    actor: str = Field(min_length=1)
    supersedes: str | None = None
    payload: InventoryChangeSchema | InventorySnapshotSchema

    @model_validator(mode="after")
    def _check_payload_matches_type(self) -> RecordSchema:
        expected = InventoryChangeSchema if self.type == "change" else InventorySnapshotSchema
        if not isinstance(self.payload, expected):
            raise ValueError(f"payload does not match type={self.type!r}")
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
