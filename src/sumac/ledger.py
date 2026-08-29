"""Fold snapshots + changes into current inventory.

Ordering is `(ts, actor, id)` for determinism across machines. A location's
baseline is its most recent snapshot at or before the query time — snapshots
reset a location's products rather than merging into them — and only changes
strictly after that baseline apply on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sumac import SCHEMA_VERSION, paths, store
from sumac.errors import SchemaVersionError
from sumac.models import InventoryChange, InventorySnapshot, Quantity, Record
from sumac.schemas import RecordSchema


def load_records(data_dir: Path, key: bytes) -> list[Record]:
    """All live records (superseded ones dropped), ordered for deterministic folding."""
    records: list[Record] = []
    for _osuser, obj in store.iter_all_logs(data_dir, key):
        if obj["schema_version"] > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"record schema_version {obj['schema_version']} is newer than supported "
                f"({SCHEMA_VERSION}); upgrade sumac"
            )
        records.append(RecordSchema.model_validate(obj).to_domain())

    superseded = {r.supersedes for r in records if r.supersedes is not None}
    live = [r for r in records if r.id not in superseded]
    live.sort(key=lambda r: (r.ts, r.actor, r.id))
    return live


@dataclass(frozen=True, slots=True)
class Inventory:
    by_location: dict[str, dict[str, Quantity]]

    def at(self, location_id: str) -> dict[str, Quantity]:
        return dict(self.by_location.get(location_id, {}))


def _apply_delta(
    state: dict[str, dict[str, Quantity]], location_id: str, product_id: str, delta: Quantity
) -> None:
    loc = state.setdefault(location_id, {})
    current = loc.get(product_id)
    new = delta if current is None else current + delta
    if new.amount == 0:
        loc.pop(product_id, None)
    else:
        loc[product_id] = new


def build_inventory(data_dir: Path, key: bytes, as_of: datetime | None = None) -> Inventory:
    records = load_records(data_dir, key)
    if as_of is not None:
        records = [r for r in records if r.ts <= as_of]

    state: dict[str, dict[str, Quantity]] = {}
    baseline_ts: dict[str, datetime] = {}

    for r in records:
        if isinstance(r.payload, InventorySnapshot):
            loc_id = r.payload.location_id
            if loc_id not in baseline_ts or r.ts >= baseline_ts[loc_id]:
                baseline_ts[loc_id] = r.ts
                state[loc_id] = {e.product_id: e.quantity for e in r.payload.entries}

    for r in records:
        if not isinstance(r.payload, InventoryChange):
            continue
        change = r.payload
        if change.from_location is not None:
            base = baseline_ts.get(change.from_location)
            if base is None or r.ts > base:
                _apply_delta(state, change.from_location, change.product_id, -change.quantity)
        if change.to_location is not None:
            base = baseline_ts.get(change.to_location)
            if base is None or r.ts > base:
                _apply_delta(state, change.to_location, change.product_id, change.quantity)

    return Inventory(by_location=state)


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    line_failures: list[store.LineFailure]
    actor_mismatches: list[tuple[Path, str, str]]


def verify_all(data_dir: Path, key: bytes) -> VerifyResult:
    line_failures: list[store.LineFailure] = []
    actor_mismatches: list[tuple[Path, str, str]] = []

    _, cfg_failures = store.verify_stream(paths.config_path(data_dir), key, store.CONFIG_STREAM_ID)
    line_failures.extend(cfg_failures)

    for log_path in paths.all_log_paths(data_dir):
        osuser = log_path.stem
        stream_id = f"log:{osuser}"
        objs, failures = store.verify_stream(log_path, key, stream_id)
        line_failures.extend(failures)
        for obj in objs:
            actor = obj.get("actor")
            if actor is not None and actor != osuser:
                actor_mismatches.append((log_path, actor, osuser))

    ok = not line_failures and not actor_mismatches
    return VerifyResult(ok=ok, line_failures=line_failures, actor_mismatches=actor_mismatches)
