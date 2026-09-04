"""Write-time validation gate: rejects before append.

`evolve` (`ledger.build_inventory`) never validates — see docs/journal
2026-08-30 §3.1. This module is the other half: `decide` resolves a command
into the writes it should produce, or raises `Rejected`.

Phase 4b: constructs v2 events (`sumac.events`) directly and serializes them
to v2-shaped wire dicts — see §3.3a. The v1 upcaster stays; it's what makes
old records still fold.

Scope note: this covers `sumac add` (all `ChangeKind` variants) against
docs/journal §4's rejection catalogue, minus `retire_nonempty` (already
shipped in Phase 2a, in `cli.py`) and minus the config-command rejections
(`duplicate_id`, `unknown_parent`, `circular_parent`-on-write) for
`add-location`/`add-product`, which are lower-stakes and left for a
follow-up.

Pure except for `uuid4()` record ids, which carry no domain meaning any of
decide's own logic depends on. `actor`, `occurred_at`, and `inventory` (for
§3.5's insufficient-stock check) come from the caller rather than being
fetched here, so the actual *decision* — reject or not, and what gets
written — is fully deterministic and testable without mocking time, I/O,
or the fold.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from sumac import SCHEMA_VERSION, config, events
from sumac.errors import Rejected
from sumac.ledger import Inventory
from sumac.models import ChangeKind, Quantity, Record

# (needs_from, needs_to) per kind — CORRECTION is handled separately since it
# needs exactly one of the two, either shape valid, rather than a fixed one.
_ENDPOINT_SHAPE: dict[ChangeKind, tuple[bool, bool]] = {
    ChangeKind.MOVEMENT: (True, True),
    ChangeKind.PURCHASE: (False, True),
    ChangeKind.DISCOVERY: (False, True),
    ChangeKind.CONSUMPTION: (True, False),
    ChangeKind.WASTE: (True, False),
}


@dataclass(frozen=True, slots=True)
class Write:
    """A single write `store.append` can take verbatim: `stream` is a
    stream_id (`"config"` or `"log:<actor>"`), `obj` the record body."""

    stream: str
    obj: dict


def near_matches(value: str, candidates: Iterable[str], n: int = 1) -> list[str]:
    return difflib.get_close_matches(value, list(candidates), n=n, cutoff=0.6)


def _check_endpoint_shape(kind: ChangeKind, frm: str | None, to: str | None) -> None:
    """Rejects a command whose from/to shape doesn't match what `kind` takes —
    both required-missing *and* unexpected-but-present. `events.py`'s
    dataclasses (unlike v1's `InventoryChange`) have no `__post_init__` of
    their own to catch this for free, so it has to be explicit here.

    Also closes a latent gap Phase 3 shipped with: `InventoryChange.__post_init__`
    only ever checked that a *required* endpoint was present, never that an
    *unexpected* one was absent — `sumac add purchase milk 1 l --from pantry
    --to fridge` would have silently stored `from_location="pantry"` on a
    purchase, which the fold would then have subtracted from, unrelated to
    the movement having ever been intended. Never reachable through §4's
    existing test suite because no test happened to pass both; found while
    rewriting this function for v2 event construction, not by a failing test."""
    if kind is ChangeKind.CORRECTION:
        if (frm is None) == (to is None):
            raise Rejected(
                "missing_endpoint",
                kind=kind.value,
                value="correction requires exactly one of from/to",
            )
        return

    needs_from, needs_to = _ENDPOINT_SHAPE[kind]
    if needs_from and frm is None:
        raise Rejected("missing_endpoint", kind=kind.value, value=f"{kind.value} requires from")
    if needs_to and to is None:
        raise Rejected("missing_endpoint", kind=kind.value, value=f"{kind.value} requires to")
    if not needs_from and frm is not None:
        raise Rejected(
            "missing_endpoint", kind=kind.value, value=f"{kind.value} does not take from"
        )
    if not needs_to and to is not None:
        raise Rejected("missing_endpoint", kind=kind.value, value=f"{kind.value} does not take to")


def _resolve_location(value: str | None, field: str, cfg: config.Config) -> str | None:
    """`None` passes through unchanged — not every change has both endpoints.
    Checks an exact `location_path` match before `near_matches`: a display
    string pasted into `--to` (§3.5) is exact, a typo is only ever fuzzy."""
    if value is None:
        return None
    if value in cfg.active_locations:
        return value

    for loc_id, loc in cfg.known_locations.items():
        if config.location_path(cfg.known_locations, loc_id) == value:
            if loc.retired:
                raise Rejected("retired_location", field=field, value=loc_id)
            return loc_id

    if value in cfg.known_locations:
        raise Rejected("retired_location", field=field, value=value)

    raise Rejected(
        "unknown_location",
        field=field,
        value=value,
        suggestions=near_matches(value, cfg.active_locations),
    )


@dataclass(frozen=True, slots=True)
class _ResolvedProduct:
    """`writes` is a `tuple`, not a `list` — every other frozen dataclass in
    this codebase uses an immutable field type for a collection
    (`events.Snapshot.entries`, `models.Product.conversions`'s `Mapping`); a
    `list` field on a frozen dataclass is cosmetic immutability only,
    since `resolved.writes.append(...)` still mutates it in place."""

    canon: Quantity
    basis: dict[str, str] | None
    writes: tuple[Write, ...]
    warning: str | None


def _resolve_product(
    product_id: str,
    amount: Decimal,
    unit: str,
    actor: str,
    occurred_at: datetime,
    cfg: config.Config,
) -> _ResolvedProduct:
    """Unknown products auto-register rather than reject — §3.5a: the real
    log is ~469 distinct products, 75% bought only once, so a fixed registry
    maintained by hand doesn't fit this household's actual usage.

    A known product's write in a unit with no canonical match or
    `conversions` entry is accepted too, not rejected — the same
    accept-with-confirmation shape as auto-registration above. A bag and a
    jug of rice aren't a fixed ratio the way jar-to-grams is, so there's no
    conversion to demand before recording what was actually said; product
    identity stays `product_id` in both units. See
    docs/journal/2026-09-04-basmati-rice-unit-mismatch.md — this used to be
    `Rejected("unit_unconvertible", ...)`, which just pushed the model into
    fabricating a second product identity to route around the rejection."""
    if product_id in cfg.active_products:
        result = cfg.convert_with_basis(product_id, amount, unit)
        if result is not None:
            canon, basis = result
            return _ResolvedProduct(canon, basis, (), None)
        product = cfg.active_products[product_id]
        warning = (
            f"{unit!r} has no registered conversion to {product_id!r}'s canonical "
            f"unit {product.unit!r} — recording {amount} {unit} as its own "
            f"tracked quantity rather than converting it."
        )
        return _ResolvedProduct(Quantity(amount, unit), None, (), warning)

    known = cfg.known_products.get(product_id)
    if known is not None and known.retired:
        raise Rejected("retired_product", value=product_id)

    # Confirmed, not assumed: `Config.active_products` is always exactly
    # `known_products` filtered to `not retired` (see config.build_config
    # and every hand-built Config in the test suite) — there is no way to be
    # known but neither active nor retired. So having fallen through both
    # checks above, `product_id` cannot be in `known_products` at all; the
    # auto-registration below can never collide with — and latest-revision-
    # wins can never silently clobber — an existing registration under this
    # id, deliberate or otherwise.
    assert known is None, (
        "known_products entry reached auto-register: active/retired invariant broke"
    )

    suggestion = near_matches(product_id, cfg.active_products)
    warning = None
    if suggestion:
        warning = (
            f"{product_id!r} is not a registered product — did you mean "
            f"{suggestion[0]!r}? Registering {product_id!r} instead; "
            f"run `sumac correct` if it was a typo."
        )
    # Canonical unit = whatever this command used (the first-use unit trap,
    # §3.5a) — not worth solving now, add-product/Counted correct it later.
    # An auto-registered product's canonical unit *is* the unit just used,
    # so nothing was converted — `basis` stays `None`, same as the
    # unit-already-canonical case in `Config.convert_with_basis`.
    registration = Write(
        "config",
        {
            "schema_version": SCHEMA_VERSION,
            "ts": occurred_at.isoformat(),
            "actor": actor,
            "product": {
                "id": product_id,
                "name": product_id,
                "unit": unit,
                "category": None,
                "metadata": {"auto": True},
                "retired": False,
                "conversions": {},
            },
        },
    )
    return _ResolvedProduct(Quantity(amount, unit), None, (registration,), warning)


def _build_event(
    kind: ChangeKind,
    product_id: str,
    canon: Quantity,
    frm: str | None,
    to: str | None,
    nominal_basis: dict[str, str] | None,
) -> events.Event:
    """`_check_endpoint_shape` has already run, so `frm`/`to` are exactly
    what this `kind` needs — the asserts below document that, not guard it.
    `nominal_basis` is `_resolve_product`'s audit trail (§5.3), forwarded
    unchanged into whichever event type this builds — see
    `Config.convert_with_basis` for what populates it and when it's `None`."""
    if kind is ChangeKind.MOVEMENT:
        assert frm is not None and to is not None
        return events.Moved(
            product_id=product_id,
            frm=frm,
            to=to,
            amount=canon.amount,
            unit=canon.unit,
            nominal_basis=nominal_basis,
        )
    if kind is ChangeKind.PURCHASE:
        assert to is not None
        return events.Acquired(
            product_id=product_id,
            to=to,
            amount=canon.amount,
            unit=canon.unit,
            nominal_basis=nominal_basis,
        )
    if kind is ChangeKind.DISCOVERY:
        assert to is not None
        return events.Acquired(
            product_id=product_id,
            to=to,
            amount=canon.amount,
            unit=canon.unit,
            reason="discovery",
            nominal_basis=nominal_basis,
        )
    if kind is ChangeKind.CONSUMPTION:
        assert frm is not None
        return events.Consumed(
            product_id=product_id,
            frm=frm,
            amount=canon.amount,
            unit=canon.unit,
            nominal_basis=nominal_basis,
        )
    if kind is ChangeKind.WASTE:
        assert frm is not None
        return events.Discarded(
            product_id=product_id,
            frm=frm,
            amount=canon.amount,
            unit=canon.unit,
            nominal_basis=nominal_basis,
        )

    assert kind is ChangeKind.CORRECTION
    if to is not None:
        assert frm is None
        return events.Acquired(
            product_id=product_id,
            to=to,
            amount=canon.amount,
            unit=canon.unit,
            reason="correction",
            nominal_basis=nominal_basis,
        )
    assert frm is not None
    return events.Consumed(
        product_id=product_id,
        frm=frm,
        amount=canon.amount,
        unit=canon.unit,
        reason="correction",
        nominal_basis=nominal_basis,
    )


def serialize_event(
    event: events.Event,
    *,
    actor: str,
    occurred_at: datetime,
    cmd_id: str,
    supersedes: str | None = None,
    record_id: str | None = None,
) -> dict:
    """One v2 event -> its wire dict. Public: `cli.py`'s `snapshot` command
    uses this too, not just `decide_change` — it's a serializer, not part of
    the validation gate. `type` names the event kind directly
    (§3.3a) rather than a `kind` field inside a generic "change" payload.

    `cmd_id` identifies the *command* this event came from, not the record —
    a single command can produce more than one record (§3.5's Counted+the
    event it precedes), and both share the same `cmd_id` so a future reader
    can tell they're one causal unit (docs/journal §3.7). Required, not
    generated here, since the caller is what knows whether several calls to
    this function belong to the same command.

    `supersedes` and `record_id` default to `None`/generated-here, matching
    every call site before these params existed (`decide_change`'s two
    calls, `cli.py`'s `snapshot`). `decide_correct` passes both explicitly —
    `record_id` because it must know the id before this call, to run its
    own self-supersede check (see `decide_correct`)."""
    payload: dict[str, object]
    type_name: str
    match event:
        case events.Acquired(
            product_id=p, to=to, amount=amount, unit=unit, reason=reason, nominal_basis=nb
        ):
            type_name = "acquired"
            payload = {
                "product_id": p,
                "to": to,
                "amount": str(amount),
                "unit": unit,
                "reason": reason,
                "nominal_basis": nb,
            }
        case events.Consumed(
            product_id=p, frm=frm, amount=amount, unit=unit, reason=reason, nominal_basis=nb
        ):
            type_name = "consumed"
            payload = {
                "product_id": p,
                "frm": frm,
                "amount": str(amount),
                "unit": unit,
                "reason": reason,
                "nominal_basis": nb,
            }
        case events.Discarded(product_id=p, frm=frm, amount=amount, unit=unit, nominal_basis=nb):
            type_name = "discarded"
            payload = {
                "product_id": p,
                "frm": frm,
                "amount": str(amount),
                "unit": unit,
                "nominal_basis": nb,
            }
        case events.Moved(product_id=p, frm=frm, to=to, amount=amount, unit=unit, nominal_basis=nb):
            type_name = "moved"
            payload = {
                "product_id": p,
                "frm": frm,
                "to": to,
                "amount": str(amount),
                "unit": unit,
                "nominal_basis": nb,
            }
        case events.Counted(
            product_id=p, at=at, amount=amount, unit=unit, reason=reason, nominal_basis=nb
        ):
            type_name = "counted"
            payload = {
                "product_id": p,
                "at": at,
                "amount": str(amount),
                "unit": unit,
                "reason": reason,
                "nominal_basis": nb,
            }
        case events.Snapshot(location_id=loc, entries=entries):
            type_name = "snapshot"
            payload = {
                "location_id": loc,
                "entries": [
                    {
                        "product_id": e.product_id,
                        "amount": str(e.amount),
                        "unit": e.unit,
                        "nominal_basis": e.nominal_basis,
                    }
                    for e in entries
                ],
            }
        case events.Correction(reason=r):
            type_name = "correction"
            payload = {"reason": r}
        case _:  # pragma: no cover - defensive; every events.Event member is handled above
            raise TypeError(f"unrecognized event type: {type(event).__name__}")

    return {
        "schema_version": SCHEMA_VERSION,
        "type": type_name,
        "id": record_id or str(uuid4()),
        "ts": occurred_at.isoformat(),
        "actor": actor,
        "supersedes": supersedes,
        "cmd_id": cmd_id,
        "payload": payload,
    }


def _reconcile_shortfall(
    event: events.Event,
    inventory: Inventory,
    *,
    actor: str,
    occurred_at: datetime,
    cmd_id: str,
) -> tuple[list[Write], list[str]]:
    """§3.5: "insufficient stock is not a rejection" — the shelf is
    authoritative, not the log. If removing `event`'s amount would take the
    recorded holding below zero, the log must already be behind reality
    (you can't remove what was never there), so a `Counted` asserting the
    holding was at least that amount is emitted first — no flag, no
    --force. Returns `([], [])` when `event` has no `frm` side (`Acquired`
    has nothing to fall short of) or the recorded holding already covers
    it. `amount`/`unit` are read straight off `event` in the match below
    rather than taking a separate `canon: Quantity` parameter, so there is
    no second value that could ever drift from what the event itself
    asserts."""
    match event:
        case (
            events.Consumed(product_id=p, frm=frm_side, amount=amount, unit=unit)
            | events.Discarded(product_id=p, frm=frm_side, amount=amount, unit=unit)
            | events.Moved(product_id=p, frm=frm_side, amount=amount, unit=unit)
        ):
            pass
        case _:
            return [], []

    held = inventory.at(frm_side).get(p)
    if held is not None and held.unit != unit:
        # A unit mismatch already sitting at this location is a pre-
        # existing anomaly this check shouldn't try to resolve on its
        # own — skip the correction and let the normal fold-time
        # unit_mismatch handling (unchanged) deal with it as it already does.
        return [], []
    if held is not None and held.amount >= amount:
        return [], []

    held_amount = held.amount if held is not None else Decimal(0)
    counted = events.Counted(
        product_id=p,
        at=frm_side,
        amount=amount,
        unit=unit,
        reason="implied_by_movement",
    )
    # "Then the movement" (§3.5) is an ordering guarantee the fold's
    # (ts, actor, id) sort has to actually deliver, not just a list-
    # append order that's discarded on the way to disk. Sharing
    # `occurred_at` with the main event would make the sort's
    # tie-break fall to `id` — a random uuid4 — so this event
    # commits before that one only about half the time, silently
    # corrupting the correction it's supposed to make (found by
    # smoke-testing this end to end: pantry ended up holding the
    # Counted amount undisturbed, meaning the movement's subtraction
    # had already happened and been overwritten). One microsecond
    # earlier is enough to make the ordering deterministic without
    # inventing new envelope machinery for it.
    counted_at = occurred_at - timedelta(microseconds=1)
    write = Write(
        f"log:{actor}",
        serialize_event(counted, actor=actor, occurred_at=counted_at, cmd_id=cmd_id),
    )
    message = f"note: {frm_side} held {held_amount} {unit}, recorded {amount} {unit} — adjusted"
    return [write], [message]


def decide_change(
    *,
    kind: ChangeKind,
    product_id: str,
    amount: Decimal,
    unit: str,
    from_location: str | None,
    to_location: str | None,
    actor: str,
    occurred_at: datetime,
    inventory: Inventory,
    cfg: config.Config,
) -> tuple[list[Write], list[str]]:
    """Validates and resolves one `sumac add` command into the writes it
    should produce, or raises `Rejected`. Config writes (a product
    auto-registration, if any) are ordered before the log write, so the
    registration lands before the change that depends on it. Returns
    `(writes, messages)` — `messages` are informational, printed but never
    blocking: a near-match warning on auto-register, an insufficient-stock
    adjustment note, both, or neither."""
    if amount <= 0:
        raise Rejected("non_positive_amount", value=amount)

    _check_endpoint_shape(kind, from_location, to_location)

    from_id = _resolve_location(from_location, "from", cfg)
    to_id = _resolve_location(to_location, "to", cfg)
    if kind is ChangeKind.MOVEMENT and from_id == to_id:
        raise Rejected("noop_move", value=from_id)

    resolved = _resolve_product(product_id, amount, unit, actor, occurred_at, cfg)
    writes = list(resolved.writes)
    messages = [resolved.warning] if resolved.warning else []

    # One cmd_id for every log write this call produces — the Counted below
    # and the main event, when both happen, are one causal command (§3.7).
    cmd_id = str(uuid4())

    event = _build_event(kind, product_id, resolved.canon, from_id, to_id, resolved.basis)

    shortfall_writes, shortfall_messages = _reconcile_shortfall(
        event, inventory, actor=actor, occurred_at=occurred_at, cmd_id=cmd_id
    )
    writes.extend(shortfall_writes)
    messages.extend(shortfall_messages)

    writes.append(
        Write(
            f"log:{actor}",
            serialize_event(event, actor=actor, occurred_at=occurred_at, cmd_id=cmd_id),
        )
    )
    return writes, messages


def decide_correct(
    *,
    target_id: str,
    reason: str,
    actor: str,
    occurred_at: datetime,
    records: Iterable[Record],
) -> Write:
    """Validates and resolves `sumac correct` into the single write it should
    produce, or raises `Rejected`. §3.6: cancel not replace — this appends a
    `Correction(reason)` record with `supersedes=target_id`; it never rewrites
    or removes anything. `records` must be `ledger.load_all_records`'s
    unfiltered view (live and already-superseded alike) — `load_records`
    already drops superseded records, which would make a record that was
    validly superseded once indistinguishable from one that never existed."""
    if not reason.strip():
        raise Rejected("missing_reason", value=reason)

    ids = {r.id for r in records}
    already_superseded = {r.supersedes for r in records if r.supersedes is not None}
    if target_id not in ids:
        raise Rejected("supersede_target_missing", value=target_id)
    if target_id in already_superseded:
        raise Rejected("supersede_already_applied", value=target_id)

    record_id = str(uuid4())
    if record_id == target_id:  # pragma: no cover - uuid4 collision, not reachable in practice
        raise Rejected("supersede_self", value=target_id)

    return Write(
        f"log:{actor}",
        serialize_event(
            events.Correction(reason=reason),
            actor=actor,
            occurred_at=occurred_at,
            cmd_id=str(uuid4()),
            supersedes=target_id,
            record_id=record_id,
        ),
    )
