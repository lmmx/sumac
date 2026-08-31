"""Fold v2 events into current inventory.

Ordering is `(ts, actor, id)` for determinism across machines. A location's
baseline is its most recent snapshot at or before the query time — snapshots
reset a location's products rather than merging into them — and only events
strictly after that baseline apply on top of it.

Every stored record is v1 as of Phase 4a (the writer hasn't changed yet) and
is upcast to a v2 event (`sumac.events`) at read time — see `_load_v2` and
docs/journal/2026-08-30 §3.3a. The fold itself only ever sees v2 events.

The fold is total: it never raises on a record it can't interpret or apply.
`decide` (`sumac.decide`) is where semantic validity is rejected before
append; this module can only quarantine what already made it into the log,
via `Anomaly`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError
from sealedlog.errors import SealError

from sumac import SCHEMA_VERSION, config, events, paths, store, upcast
from sumac.errors import SumacError
from sumac.models import Anomaly, InventoryChange, InventorySnapshot, Location, Quantity, Record
from sumac.schemas import RecordSchema

_READ_TIME_ERRORS = (SumacError, SealError, ValidationError)


@dataclass(frozen=True, slots=True)
class _LoadResult:
    records: list[Record]
    all_records: list[Record]
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
    return _LoadResult(records=live, all_records=parsed, anomalies=anomalies)


def load_records(data_dir: Path, key: bytes) -> list[Record]:
    """All live records (superseded ones dropped, unfoldable ones dropped
    as anomalies), ordered for deterministic folding. Used by `sumac log`,
    which displays the stored shape — not upcast, unlike `build_inventory`."""
    return _load(data_dir, key).records


def load_all_records(data_dir: Path, key: bytes) -> list[Record]:
    """Every parsed record, live *and* superseded alike — unlike `load_records`,
    which drops superseded records entirely. `decide.decide_correct` needs
    this to tell "never existed" (`supersede_target_missing`) apart from
    "already superseded" (`supersede_already_applied`), a distinction
    `load_records`'s filtered view has already erased."""
    return _load(data_dir, key).all_records


@dataclass(frozen=True, slots=True)
class _EventRecord:
    """One upcast record: the v2 event plus the envelope fields the fold
    needs (`id` for anomaly attribution, `ts`/`actor` for ordering)."""

    id: str
    ts: datetime
    actor: str
    event: events.Event


@dataclass(frozen=True, slots=True)
class _V2LoadResult:
    records: list[_EventRecord]
    anomalies: list[Anomaly]


def _load_v2(data_dir: Path, key: bytes) -> _V2LoadResult:
    """`_load` plus the upcast pass for whatever's still v1. Since Phase 4b,
    `_load`'s records are a mix: a v2 record's `payload` is already an
    `events.Event` (schemas.py routes schema_version 2 straight to the v2
    ingest schemas), so only v1 payloads need upcasting. A v1 record that
    upcasts to nothing the mapping table covers becomes an `upcast_failed`
    anomaly rather than a raise — same totality guarantee as everything
    else in `_load`."""
    v1 = _load(data_dir, key)
    anomalies = list(v1.anomalies)
    records: list[_EventRecord] = []
    for r in v1.records:
        if isinstance(r.payload, (InventoryChange, InventorySnapshot)):
            try:
                event = upcast.upcast(r)
            except upcast.UpcastError as e:
                anomalies.append(Anomaly(r.id, "upcast_failed", str(e)))
                continue
        else:
            event = r.payload
        records.append(_EventRecord(id=r.id, ts=r.ts, actor=r.actor, event=event))
    return _V2LoadResult(records=records, anomalies=anomalies)


def observed_product_units(data_dir: Path, key: bytes) -> dict[str, Counter[str]]:
    """For every product_id recorded in a change or snapshot entry, how many
    times each unit was used with it — including records the fold can't yet
    apply for unrelated reasons (unknown location, etc.), since the point is
    what units were actually written, not what currently folds.

    Deliberately reads v1 records via `_load`, not the upcast v2 events via
    `_load_v2`: a record that fails to upcast (an `upcast_failed` anomaly —
    currently only an ambiguous `correction`) was still *written*, and this
    function's whole job is "what was written," not "what currently folds
    or upcasts." Going through `_load_v2` would silently drop exactly the
    records this is meant to surface.

    Feeds Phase 2c's canonical-unit backfill: a product's most-observed unit is
    a reasonable canonical-unit default, and `sumac config check-units` uses
    this to find (product, unit) pairs `Config.convert` can't yet resolve."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in _load(data_dir, key).records:
        if isinstance(r.payload, InventoryChange):
            counts[r.payload.product_id][r.payload.quantity.unit] += 1
        elif isinstance(r.payload, InventorySnapshot):
            for entry in r.payload.entries:
                counts[entry.product_id][entry.quantity.unit] += 1
    return dict(counts)


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
    raising, so the caller can flag an anomaly and leave every side of a
    multi-location event uncommitted rather than apply only some of it."""
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


def _apply_sides(
    state: dict[str, dict[str, Quantity]],
    locations: dict[str, Location],
    baseline_ts: dict[str, datetime],
    anomalies: list[Anomaly],
    r: _EventRecord,
    product_id: str,
    sides: list[tuple[str, Quantity]],
) -> None:
    """Applies a delta-style event (`Acquired`/`Consumed`/`Discarded`: one
    side; `Moved`: two).

    Atomic with respect to *failures*: every location must be known and
    every side actually attempted must convert cleanly, or nothing commits.
    Not atomic with respect to baseline gating — a side whose location has
    since been reset by a later snapshot is skipped on its own, independent
    of the event's other side(s), per this module's own invariant that a
    snapshot's reset is scoped to the one location it names, not to every
    event that ever touched it. So a `Moved` whose source is behind its
    location's baseline but whose destination isn't will commit the
    destination side only — that's not a partial-application bug, it's the
    baseline rule applied per-location the way it's supposed to be; it's
    also exactly the pre-Phase-4a fold's shape for movement, unchanged here
    and verified against it on the real log (§3.3a's Phase 4a proof)."""
    bad_location = False
    for loc_id, _delta in sides:
        if loc_id not in locations:
            anomalies.append(Anomaly(r.id, "unknown_location", loc_id))
            bad_location = True
    if bad_location:
        return

    pending: list[tuple[str, str, Quantity]] = []
    for loc_id, delta in sides:
        base = baseline_ts.get(loc_id)
        if base is not None and r.ts <= base:
            continue  # a later snapshot already reset this location
        new_q, err = _next_quantity(state, loc_id, product_id, delta)
        if err is not None:
            anomalies.append(Anomaly(r.id, "unit_mismatch", err))
            return
        assert new_q is not None
        pending.append((loc_id, product_id, new_q))

    for loc_id, pid, q in pending:
        _commit(state, loc_id, pid, q)


def _fold(
    records: list[_EventRecord], locations: dict[str, Location]
) -> tuple[dict[str, dict[str, Quantity]], list[Anomaly]]:
    """The pure event-folding core of `build_inventory`, split out so Phase 6's
    in-memory property tests (model agreement, fold determinism, upcaster
    round-trip) can drive it directly with hand-built events and a locations
    dict — no files, no crypto. Behaves identically to what `build_inventory`
    always did; this only moves where the I/O boundary sits. Sorts its own
    input by `(ts, actor, id)` rather than trusting the caller to have done
    it — `_load` already sorts for its own purposes (`sumac log`'s display
    order), but a property test handing this function events straight out of
    a Hypothesis strategy shouldn't have to replicate that sort to get a
    meaningful result; sorting an already-sorted list is a no-op, so this
    changes nothing for `build_inventory`'s existing callers."""
    records = sorted(records, key=lambda r: (r.ts, r.actor, r.id))
    anomalies: list[Anomaly] = []
    state: dict[str, dict[str, Quantity]] = {}
    baseline_ts: dict[str, datetime] = {}

    for r in records:
        if not isinstance(r.event, events.Snapshot):
            continue
        loc_id = r.event.location_id
        if loc_id not in locations:
            anomalies.append(Anomaly(r.id, "unknown_location", loc_id))
            continue
        if loc_id not in baseline_ts or r.ts >= baseline_ts[loc_id]:
            baseline_ts[loc_id] = r.ts
            # Found by Phase 6's model-agreement property test: a zero-amount
            # entry ("0 of X here") must drop out entirely, same as `_commit`
            # already does for a delta that lands exactly on zero — a snapshot
            # baseline built without this filter left a phantom
            # Quantity(amount=0) key, inconsistent with "zero means absent"
            # everywhere else in the fold.
            state[loc_id] = {
                e.product_id: Quantity(e.amount, e.unit) for e in r.event.entries if e.amount != 0
            }

    for r in records:
        match r.event:
            case events.Snapshot():
                continue

            case events.Correction():
                # Cancel-only (§3.6): the record it targets is already
                # excluded from `records` by `_load`'s supersedes filtering.
                # This one carries no change of its own.
                continue

            case events.Counted(product_id=p, at=loc_id, amount=amount, unit=unit):
                if loc_id not in locations:
                    anomalies.append(Anomaly(r.id, "unknown_location", loc_id))
                    continue
                base = baseline_ts.get(loc_id)
                if base is not None and r.ts <= base:
                    continue
                _commit(state, loc_id, p, Quantity(amount, unit))

            case events.Acquired(product_id=p, to=loc_id, amount=amount, unit=unit):
                _apply_sides(
                    state,
                    locations,
                    baseline_ts,
                    anomalies,
                    r,
                    p,
                    [(loc_id, Quantity(amount, unit))],
                )

            case (
                events.Consumed(product_id=p, frm=loc_id, amount=amount, unit=unit)
                | events.Discarded(product_id=p, frm=loc_id, amount=amount, unit=unit)
            ):
                _apply_sides(
                    state,
                    locations,
                    baseline_ts,
                    anomalies,
                    r,
                    p,
                    [(loc_id, -Quantity(amount, unit))],
                )

            case events.Moved(product_id=p, frm=frm, to=to, amount=amount, unit=unit):
                q = Quantity(amount, unit)
                _apply_sides(state, locations, baseline_ts, anomalies, r, p, [(frm, -q), (to, q)])

    return state, anomalies


def build_inventory(data_dir: Path, key: bytes, as_of: datetime | None = None) -> Inventory:
    load_result = _load_v2(data_dir, key)
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

    state, fold_anomalies = _fold(records, locations)
    anomalies.extend(fold_anomalies)
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
