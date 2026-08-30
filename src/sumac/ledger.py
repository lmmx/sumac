"""Fold snapshots + changes into current inventory.

Ordering is `(ts, actor, id)` for determinism across machines. A location's
baseline is its most recent snapshot at or before the query time — snapshots
reset a location's products rather than merging into them — and only changes
strictly after that baseline apply on top of it.

The fold is total: it never raises on a record it can't interpret or apply.
`decide` (not yet split out — see docs/journal/2026-08-30) is where semantic
validity should be rejected before append; this module can only quarantine
what already made it into the log, via `Anomaly`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError
from sealedlog.errors import SealError

from sumac import SCHEMA_VERSION, config, paths, store
from sumac.errors import SumacError
from sumac.models import Anomaly, InventoryChange, InventorySnapshot, Location, Quantity, Record
from sumac.schemas import RecordSchema

_READ_TIME_ERRORS = (SumacError, SealError, ValidationError)


@dataclass(frozen=True, slots=True)
class _LoadResult:
    records: list[Record]
    anomalies: list[Anomaly]


def _load(data_dir: Path, key: bytes) -> _LoadResult:
    """Tolerant read of every log: decrypt/schema/domain failures become
    anomalies instead of propagating. A `schema_version` newer than this build
    supports is just another reason a record can't be interpreted — quarantined
    like any other, not raised — since one too-new record from another writer
    must not brick every command until everyone upgrades."""
    anomalies: list[Anomaly] = []
    parsed: list[Record] = []

    for log_path in paths.all_log_paths(data_dir):
        stream_id = f"log:{log_path.stem}"
        objs, failures = store.verify_stream(log_path, key, stream_id)
        for f in failures:
            anomalies.append(Anomaly(None, "line_failure", f"{f.path}:{f.lineno}: {f.error}"))
        for obj in objs:
            record_id = obj.get("id") if isinstance(obj, dict) else None
            version = obj.get("schema_version") if isinstance(obj, dict) else None
            if isinstance(version, int) and version > SCHEMA_VERSION:
                anomalies.append(Anomaly(record_id, "schema_too_new", f"schema_version={version}"))
                continue
            try:
                parsed.append(RecordSchema.model_validate(obj).to_domain())
            except (ValidationError, ValueError) as e:
                anomalies.append(Anomaly(record_id, "invalid_record", str(e)))

    superseded = {r.supersedes for r in parsed if r.supersedes is not None}
    live = [r for r in parsed if r.id not in superseded]
    live.sort(key=lambda r: (r.ts, r.actor, r.id))
    return _LoadResult(records=live, anomalies=anomalies)


def load_records(data_dir: Path, key: bytes) -> list[Record]:
    """All live records (superseded ones dropped, unfoldable ones dropped as
    anomalies), ordered for deterministic folding."""
    return _load(data_dir, key).records


def load_locations_or_empty(data_dir: Path, key: bytes) -> dict[str, Location]:
    """Never raises. A config the fold can't read shows up as a `config_unreadable`
    anomaly via `build_inventory`; this just gives other callers (rendering) a
    dict to work with instead of propagating the failure a second time."""
    try:
        return config.load_locations(data_dir, key)
    except _READ_TIME_ERRORS:
        return {}


@dataclass(frozen=True, slots=True)
class Inventory:
    by_location: dict[str, dict[str, Quantity]]
    anomalies: tuple[Anomaly, ...] = ()

    def at(self, location_id: str) -> dict[str, Quantity]:
        return dict(self.by_location.get(location_id, {}))


def _next_quantity(
    state: dict[str, dict[str, Quantity]], location_id: str, product_id: str, delta: Quantity
) -> tuple[Quantity | None, str | None]:
    """The quantity `location_id`/`product_id` would have after `delta`, without
    mutating `state`. Returns `(None, error)` on a unit mismatch instead of
    raising, so the caller can flag an anomaly and leave both sides of a
    movement uncommitted rather than apply only one."""
    current = state.get(location_id, {}).get(product_id)
    if current is None:
        return delta, None
    try:
        return current + delta, None
    except ValueError as e:
        return None, str(e)


def _commit(
    state: dict[str, dict[str, Quantity]], location_id: str, product_id: str, q: Quantity
) -> None:
    loc = state.setdefault(location_id, {})
    if q.amount == 0:
        loc.pop(product_id, None)
    else:
        loc[product_id] = q


def build_inventory(data_dir: Path, key: bytes, as_of: datetime | None = None) -> Inventory:
    load_result = _load(data_dir, key)
    anomalies = list(load_result.anomalies)
    records = load_result.records
    if as_of is not None:
        records = [r for r in records if r.ts <= as_of]

    try:
        cfg = config.build_config(data_dir, key)
        locations = cfg.known_locations
        anomalies.extend(cfg.anomalies)
    except _READ_TIME_ERRORS as e:
        anomalies.append(Anomaly(None, "config_unreadable", str(e)))
        locations = {}

    state: dict[str, dict[str, Quantity]] = {}
    baseline_ts: dict[str, datetime] = {}

    for r in records:
        if not isinstance(r.payload, InventorySnapshot):
            continue
        loc_id = r.payload.location_id
        if loc_id not in locations:
            anomalies.append(Anomaly(r.id, "unknown_location", loc_id))
            continue
        if loc_id not in baseline_ts or r.ts >= baseline_ts[loc_id]:
            baseline_ts[loc_id] = r.ts
            state[loc_id] = {e.product_id: e.quantity for e in r.payload.entries}

    for r in records:
        if not isinstance(r.payload, InventoryChange):
            continue
        change = r.payload

        bad_location = False
        for loc_id in (change.from_location, change.to_location):
            if loc_id is not None and loc_id not in locations:
                anomalies.append(Anomaly(r.id, "unknown_location", loc_id))
                bad_location = True
        if bad_location:
            continue

        pending: list[tuple[str, str, Quantity]] = []
        mismatch = False

        if change.from_location is not None:
            base = baseline_ts.get(change.from_location)
            if base is None or r.ts > base:
                new_q, err = _next_quantity(
                    state, change.from_location, change.product_id, -change.quantity
                )
                if err is not None:
                    anomalies.append(Anomaly(r.id, "unit_mismatch", err))
                    mismatch = True
                else:
                    assert new_q is not None
                    pending.append((change.from_location, change.product_id, new_q))

        if not mismatch and change.to_location is not None:
            base = baseline_ts.get(change.to_location)
            if base is None or r.ts > base:
                new_q, err = _next_quantity(
                    state, change.to_location, change.product_id, change.quantity
                )
                if err is not None:
                    anomalies.append(Anomaly(r.id, "unit_mismatch", err))
                    mismatch = True
                else:
                    assert new_q is not None
                    pending.append((change.to_location, change.product_id, new_q))

        if mismatch:
            continue
        for loc_id, product_id, q in pending:
            _commit(state, loc_id, product_id, q)

    return Inventory(by_location=state, anomalies=tuple(anomalies))


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


@dataclass(frozen=True, slots=True)
class DoctorReport:
    anomalies: tuple[Anomaly, ...]
    total_lines: int


def diagnose(data_dir: Path, key: bytes) -> DoctorReport:
    """`sumac doctor`'s view of `build_inventory`'s anomaly channel, plus a raw
    line count for context ("N anomalies out of M lines")."""
    inventory = build_inventory(data_dir, key)
    total_lines = 0
    for log_path in paths.all_log_paths(data_dir):
        objs, failures = store.verify_stream(log_path, key, f"log:{log_path.stem}")
        total_lines += len(objs) + len(failures)
    return DoctorReport(anomalies=inventory.anomalies, total_lines=total_lines)
