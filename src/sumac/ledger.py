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

from pydantic import ValidationError
from sealedlog.errors import SealError

from sumac import SCHEMA_VERSION, config, paths, store
from sumac.errors import SchemaVersionError, SumacError
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


@dataclass(frozen=True, slots=True)
class DoctorFinding:
    """One record (or line) that a tolerant fold could not apply.

    Phase 0 of docs/journal/2026-08-30_decide-pattern-data-integrity-upgrade.md — diagnostic
    only, superseded by the real `Anomaly` channel in Phase 1."""

    path: Path
    record_id: str | None
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class DoctorReport:
    findings: tuple[DoctorFinding, ...]
    total_lines: int


def diagnose(data_dir: Path, key: bytes) -> DoctorReport:
    """Tolerant fold: never raises, reports every line/record it could not apply.

    Line-level decrypt/auth failures use `store.verify_stream` (already tolerant).
    Record-level schema and domain-construction failures, and fold-time failures
    (unit mismatch, unknown location), are caught here since `RecordSchema.to_domain()`
    and `_apply_delta` both raise today."""
    findings: list[DoctorFinding] = []
    total_lines = 0
    raw: list[tuple[Path, dict]] = []

    for log_path in paths.all_log_paths(data_dir):
        stream_id = f"log:{log_path.stem}"
        objs, failures = store.verify_stream(log_path, key, stream_id)
        total_lines += len(objs) + len(failures)
        for f in failures:
            findings.append(DoctorFinding(f.path, None, "line_failure", f.error))
        raw.extend((log_path, obj) for obj in objs)

    parsed: list[tuple[Path, Record]] = []
    for path, obj in raw:
        record_id = obj.get("id") if isinstance(obj, dict) else None
        version = obj.get("schema_version") if isinstance(obj, dict) else None
        if isinstance(version, int) and version > SCHEMA_VERSION:
            findings.append(
                DoctorFinding(path, record_id, "schema_too_new", f"schema_version={version}")
            )
            continue
        try:
            parsed.append((path, RecordSchema.model_validate(obj).to_domain()))
        except (ValidationError, ValueError) as e:
            findings.append(DoctorFinding(path, record_id, "invalid_record", str(e)))

    superseded = {r.supersedes for _, r in parsed if r.supersedes is not None}
    live = [(path, r) for path, r in parsed if r.id not in superseded]
    live.sort(key=lambda item: (item[1].ts, item[1].actor, item[1].id))

    try:
        locations = config.load_locations(data_dir, key)
    except (SumacError, SealError, ValidationError) as e:
        findings.append(
            DoctorFinding(paths.config_path(data_dir), None, "config_unreadable", str(e))
        )
        locations = {}
    state: dict[str, dict[str, Quantity]] = {}

    for path, r in live:
        if isinstance(r.payload, InventorySnapshot):
            loc_id = r.payload.location_id
            if loc_id not in locations:
                findings.append(DoctorFinding(path, r.id, "unknown_location", loc_id))
            state[loc_id] = {e.product_id: e.quantity for e in r.payload.entries}
            continue

        change = r.payload
        for loc_id in (change.from_location, change.to_location):
            if loc_id is not None and loc_id not in locations:
                findings.append(DoctorFinding(path, r.id, "unknown_location", loc_id))
        try:
            if change.from_location is not None:
                _apply_delta(state, change.from_location, change.product_id, -change.quantity)
            if change.to_location is not None:
                _apply_delta(state, change.to_location, change.product_id, change.quantity)
        except ValueError as e:
            findings.append(DoctorFinding(path, r.id, "unit_mismatch", str(e)))

    return DoctorReport(findings=tuple(findings), total_lines=total_lines)
