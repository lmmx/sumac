"""Location layout on top of `store`: an append-only, latest-revision-wins registry."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sumac import SCHEMA_VERSION, models, store
from sumac.errors import SchemaVersionError
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
        },
    }
    store.append(data_dir, key, store.CONFIG_STREAM_ID, obj)


def load_locations(data_dir: Path, key: bytes) -> dict[str, models.Location]:
    """Latest-ts-wins per location id."""
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
