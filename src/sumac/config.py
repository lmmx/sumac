"""Location layout on top of `store`: an append-only, latest-revision-wins registry.

Nothing is ever deleted — `retire_location` is the only removal path, so a
historical record that names a retired location still resolves forever
(`known_locations`); only new writes are expected to check `active_locations`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from sumac import SCHEMA_VERSION, models, store
from sumac.errors import SchemaVersionError, UnknownLocationError
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
        prior = latest.get(record.location.id)
        if prior is None or record.ts >= prior[0]:
            latest[record.location.id] = (record.ts, record.location.to_domain())
    return {loc_id: loc for loc_id, (_, loc) in latest.items()}


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
    anomalies: tuple[models.Anomaly, ...]


def build_config(data_dir: Path, key: bytes) -> Config:
    known = load_locations(data_dir, key)
    active = {loc_id: loc for loc_id, loc in known.items() if not loc.retired}
    return Config(
        known_locations=known,
        active_locations=active,
        anomalies=_detect_location_cycles(known),
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
