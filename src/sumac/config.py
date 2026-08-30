"""Location and product config on top of `store`: an append-only, latest-revision-wins
registry for each, sharing one stream (`store.CONFIG_STREAM_ID`) with a record being
one or the other, never both — see `ConfigRecordSchema`.

Nothing is ever deleted — `retire_location`/`retire_product` are the only removal
path, so a historical record that names a retired entity still resolves forever
(`known_locations`/`known_products`); only new writes are expected to check the
`active_*` views.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from sumac import SCHEMA_VERSION, models, store
from sumac.errors import SchemaVersionError, UnknownLocationError, UnknownProductError
from sumac.schemas import ConfigRecordSchema


def add_location(data_dir: Path, key: bytes, actor: str, location: models.Location) -> None:
    obj = {
        "schema_version": SCHEMA_VERSION,
        "ts": datetime.now(UTC).isoformat(),
        "actor": actor,
        "location": {
            "id": location.id,
            "name": location.name,
            "parent_id": location.parent_id,
            "metadata": dict(location.metadata),
            "retired": location.retired,
        },
    }
    store.append(data_dir, key, store.CONFIG_STREAM_ID, obj)


def retire_location(data_dir: Path, key: bytes, actor: str, location_id: str) -> None:
    """Re-appends the location with `retired=True`. Latest-ts-wins means this is
    the only removal path there is — there is no delete to alias."""
    current = load_locations(data_dir, key).get(location_id)
    if current is None:
        raise UnknownLocationError(f"no such location: {location_id!r}")
    add_location(data_dir, key, actor, replace(current, retired=True))


def load_locations(data_dir: Path, key: bytes) -> dict[str, models.Location]:
    """Latest-ts-wins per location id. Every location ever defined, retired or
    not — this is `known_locations`; see `build_config` for `active_locations`."""
    latest: dict[str, tuple[datetime, models.Location]] = {}
    for obj in store.iter_stream(data_dir, key, store.CONFIG_STREAM_ID):
        if obj["schema_version"] > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"config record schema_version {obj['schema_version']} is newer than "
                f"supported ({SCHEMA_VERSION}); upgrade sumac"
            )
        record = ConfigRecordSchema.model_validate(obj)
        if record.location is None:
            continue
        prior = latest.get(record.location.id)
        if prior is None or record.ts >= prior[0]:
            latest[record.location.id] = (record.ts, record.location.to_domain())
    return {loc_id: loc for loc_id, (_, loc) in latest.items()}


def add_product(data_dir: Path, key: bytes, actor: str, product: models.Product) -> None:
    obj = {
        "schema_version": SCHEMA_VERSION,
        "ts": datetime.now(UTC).isoformat(),
        "actor": actor,
        "product": {
            "id": product.id,
            "name": product.name,
            "unit": product.unit,
            "category": product.category,
            "metadata": dict(product.metadata),
            "retired": product.retired,
        },
    }
    store.append(data_dir, key, store.CONFIG_STREAM_ID, obj)


def retire_product(data_dir: Path, key: bytes, actor: str, product_id: str) -> None:
    """Re-appends the product with `retired=True`. Unlike a location, a product may
    be retired at any time — there is no stock check, since inventory quantities are
    keyed by (product_id, location_id) and stay resolvable via `known_products`
    regardless of whether the product is still active."""
    current = load_products(data_dir, key).get(product_id)
    if current is None:
        raise UnknownProductError(f"no such product: {product_id!r}")
    add_product(data_dir, key, actor, replace(current, retired=True))


def load_products(data_dir: Path, key: bytes) -> dict[str, models.Product]:
    """Latest-ts-wins per product id. Every product ever defined, retired or
    not — this is `known_products`; see `build_config` for `active_products`."""
    latest: dict[str, tuple[datetime, models.Product]] = {}
    for obj in store.iter_stream(data_dir, key, store.CONFIG_STREAM_ID):
        if obj["schema_version"] > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"config record schema_version {obj['schema_version']} is newer than "
                f"supported ({SCHEMA_VERSION}); upgrade sumac"
            )
        record = ConfigRecordSchema.model_validate(obj)
        if record.product is None:
            continue
        prior = latest.get(record.product.id)
        if prior is None or record.ts >= prior[0]:
            latest[record.product.id] = (record.ts, record.product.to_domain())
    return {prod_id: p for prod_id, (_, p) in latest.items()}


def _detect_location_cycles(known: dict[str, models.Location]) -> tuple[models.Anomaly, ...]:
    """Iterative with a visited set per starting node — never recursive, never
    unbounded. A cycle produces a diagnostic naming the chain, not a hang."""
    anomalies: list[models.Anomaly] = []
    for start in known:
        chain: list[str] = []
        seen: set[str] = set()
        node: str | None = start
        while node is not None:
            if node in seen:
                anomalies.append(
                    models.Anomaly(start, "circular_parent", " -> ".join([*chain, node]))
                )
                break
            seen.add(node)
            chain.append(node)
            node = known[node].parent_id if node in known else None
    return tuple(anomalies)


@dataclass(frozen=True, slots=True)
class Config:
    known_locations: dict[str, models.Location]
    active_locations: dict[str, models.Location]
    known_products: dict[str, models.Product]
    active_products: dict[str, models.Product]
    anomalies: tuple[models.Anomaly, ...]


def build_config(data_dir: Path, key: bytes) -> Config:
    known_locations = load_locations(data_dir, key)
    active_locations = {i: loc for i, loc in known_locations.items() if not loc.retired}
    known_products = load_products(data_dir, key)
    active_products = {i: p for i, p in known_products.items() if not p.retired}
    return Config(
        known_locations=known_locations,
        active_locations=active_locations,
        known_products=known_products,
        active_products=active_products,
        anomalies=_detect_location_cycles(known_locations),
    )


def descendants(locations: dict[str, models.Location], root_id: str) -> set[str]:
    """`root_id` plus every location nested under it, however deep."""
    children_of: dict[str | None, list[str]] = {}
    for loc in locations.values():
        children_of.setdefault(loc.parent_id, []).append(loc.id)

    result: set[str] = set()
    stack = [root_id]
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(children_of.get(current, []))
    return result


def location_path(locations: dict[str, models.Location], location_id: str) -> str:
    """Root-to-leaf display path, e.g. 'Fridge > Door'. Falls back to the raw id
    for an unknown location."""
    names: list[str] = []
    seen: set[str] = set()
    current = locations.get(location_id)
    while current is not None and current.id not in seen:
        seen.add(current.id)
        names.append(current.name)
        current = locations.get(current.parent_id) if current.parent_id else None
    return " > ".join(reversed(names)) if names else location_id
