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
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from sumac import SCHEMA_VERSION, models, paths, store
from sumac.errors import UnknownLocationError, UnknownProductError
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
    return _load_config_records(data_dir, key).locations


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
            "conversions": {u: str(r) for u, r in product.conversions.items()},
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
    return _load_config_records(data_dir, key).products


@dataclass(frozen=True, slots=True)
class _ConfigLoadResult:
    locations: dict[str, models.Location]
    products: dict[str, models.Product]
    anomalies: list[models.Anomaly]


def _load_config_records(data_dir: Path, key: bytes) -> _ConfigLoadResult:
    """One pass over the config stream, building both dicts — `load_locations`
    and `load_products` are thin wrappers over this rather than each doing
    their own pass.

    Total, like `ledger._load`: a bad line (decrypt failure), a record naming
    neither or both of location/product, or a too-new `schema_version` each
    become an anomaly and are skipped, never a raise. One bad config line must
    not make the rest of config unreadable — same blast-radius concern §3.1
    raises for the main log, just one layer up the stack."""
    latest_locations: dict[str, tuple[datetime, models.Location]] = {}
    latest_products: dict[str, tuple[datetime, models.Product]] = {}
    anomalies: list[models.Anomaly] = []

    objs, failures = store.verify_stream(paths.config_path(data_dir), key, store.CONFIG_STREAM_ID)
    for f in failures:
        anomalies.append(models.Anomaly(None, "line_failure", f"{f.path}:{f.lineno}: {f.error}"))

    for obj in objs:
        version = obj.get("schema_version") if isinstance(obj, dict) else None
        if isinstance(version, int) and version > SCHEMA_VERSION:
            anomalies.append(models.Anomaly(None, "schema_too_new", f"schema_version={version}"))
            continue
        try:
            record = ConfigRecordSchema.model_validate(obj)
        except ValidationError as e:
            anomalies.append(models.Anomaly(None, "invalid_config_record", str(e)))
            continue

        if record.location is not None:
            prior = latest_locations.get(record.location.id)
            if prior is None or record.ts >= prior[0]:
                latest_locations[record.location.id] = (record.ts, record.location.to_domain())
        else:
            assert record.product is not None
            prior = latest_products.get(record.product.id)
            if prior is None or record.ts >= prior[0]:
                latest_products[record.product.id] = (record.ts, record.product.to_domain())

    return _ConfigLoadResult(
        locations={loc_id: loc for loc_id, (_, loc) in latest_locations.items()},
        products={prod_id: p for prod_id, (_, p) in latest_products.items()},
        anomalies=anomalies,
    )


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

    def convert_with_basis(
        self, product_id: str, amount: Decimal, unit: str
    ) -> tuple[models.Quantity, dict[str, str] | None] | None:
        """`amount` of `unit` expressed in `product_id`'s canonical unit, paired
        with the audit record of how it got there, or `None` if `product_id`
        isn't known or `unit` has no conversion path to it. Nominal, per §3.4(c):
        resolved once at decide-time and frozen into the event — never called
        from `evolve`, which never converts.

        Resolves against `known_products`, not `active_products` — a deliberate
        divergence from §3.4's "decide validates against active_*" rule. Resolution
        (what a unit *means*) and permission (whether a write to it is currently
        *allowed*) are different questions: retirement should stop new writes
        (`retired_product`, decide's job, not this method's), but must not stop the
        arithmetic from resolving — `check-units` needs to interpret a retired
        product's historical units too, and a `decide` call that's already past its
        own `active_products`/`retired_product` check shouldn't have this method
        re-reject on the same grounds a second time under a different name.

        The single lookup this and `convert` both need — `convert` delegates
        here rather than duplicating it, so the two can never resolve a
        product or a ratio differently. The second element of the returned
        pair is `None` when `unit` already *is* the canonical unit: nothing
        was converted, and the event's own `amount`/`unit` fields already
        reproduce the input exactly, so there is nothing a basis record would
        add (see docs/journal/2026-08-31-decide-simplification-review.md §5.3,
        Decision 1)."""
        product = self.known_products.get(product_id)
        if product is None:
            return None
        if unit == product.unit:
            return models.Quantity(amount, product.unit), None
        ratio = product.conversions.get(unit)
        if ratio is None:
            return None
        basis = {"raw_amount": str(amount), "raw_unit": unit, "ratio": str(ratio)}
        return models.Quantity(amount * ratio, product.unit), basis

    def convert(self, product_id: str, amount: Decimal, unit: str) -> models.Quantity | None:
        """`amount` of `unit` expressed in `product_id`'s canonical unit, or
        `None` on the same terms as `convert_with_basis` — see there for the
        resolution rules. A thin wrapper: callers that don't need the audit
        trail (`can_convert`, and through it `render.py:98`'s
        `print_unit_check`) keep this narrower contract unchanged."""
        result = self.convert_with_basis(product_id, amount, unit)
        return result[0] if result is not None else None

    def can_convert(self, product_id: str, unit: str) -> bool:
        return self.convert(product_id, Decimal(0), unit) is not None


def build_config(data_dir: Path, key: bytes) -> Config:
    result = _load_config_records(data_dir, key)
    known_locations = result.locations
    active_locations = {i: loc for i, loc in known_locations.items() if not loc.retired}
    known_products = result.products
    active_products = {i: p for i, p in known_products.items() if not p.retired}
    anomalies = (*result.anomalies, *_detect_location_cycles(known_locations))
    return Config(
        known_locations=known_locations,
        active_locations=active_locations,
        known_products=known_products,
        active_products=active_products,
        anomalies=anomalies,
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


def search_locations(locations: dict[str, models.Location], query: str) -> list[models.Location]:
    """Every active location whose id, name, or display path contains
    `query`, case-insensitively, ordered by display path.

    The counterpart to `ledger.search_inventory`, which matches products
    only: an agent asked to put something on "the top shelf of the fridge"
    must resolve that phrase to a location id, and searching the inventory for
    "fridge" correctly returns nothing, since no product has that name.
    Matching the path as well as the name lets a query for a container return
    the locations nested inside it."""
    lowered = query.strip().lower()
    if not lowered:
        return []
    matches = [
        loc
        for loc_id, loc in locations.items()
        if not loc.retired
        and (
            lowered in loc_id.lower()
            or lowered in loc.name.lower()
            or lowered in location_path(locations, loc_id).lower()
        )
    ]
    return sorted(matches, key=lambda loc: location_path(locations, loc.id))


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
