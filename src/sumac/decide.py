"""Write-time validation gate: rejects before append.

`evolve` (`ledger.build_inventory`) never validates — see docs/journal
2026-08-30 §3.1. This module is the other half: `decide` resolves a command
into the writes it should produce, or raises `Rejected`.

Scope note: this covers `sumac add` (all `ChangeKind` variants) against
docs/journal §4's rejection catalogue, minus `retire_nonempty` (already
shipped in Phase 2a, in `cli.py`) and minus the config-command rejections
(`duplicate_id`, `unknown_parent`, `circular_parent`-on-write) for
`add-location`/`add-product`, which are lower-stakes and left for a
follow-up. It also does not implement §3.5's "insufficient stock is not a
rejection" auto-`Counted` behavior — that needs the fine-grained per-product
`Counted` event, which doesn't exist until Phase 4b's schema change; today's
`InventorySnapshot` only resets a whole location, not one product in it.

Pure except for `uuid4()` record ids, which carry no domain meaning any of
decide's own logic depends on. `actor` and `occurred_at` come from the
caller rather than reading the clock, so the actual *decision* — reject or
not, and what gets written — is fully deterministic and testable without
mocking time.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from sumac import SCHEMA_VERSION, config
from sumac.errors import Rejected
from sumac.models import ChangeKind, InventoryChange, Quantity


@dataclass(frozen=True, slots=True)
class Write:
    """A single write `store.append` can take verbatim: `stream` is a
    stream_id (`"config"` or `"log:<actor>"`), `obj` the record body."""

    stream: str
    obj: dict


def near_matches(value: str, candidates: Iterable[str], n: int = 1) -> list[str]:
    return difflib.get_close_matches(value, list(candidates), n=n, cutoff=0.6)


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


def _resolve_product(
    product_id: str,
    amount: Decimal,
    unit: str,
    actor: str,
    occurred_at: datetime,
    cfg: config.Config,
) -> tuple[Quantity, list[Write], str | None]:
    """Returns (canonical quantity, extra config writes, a warning to print
    or `None`). Unknown products auto-register rather than reject — §3.5a:
    the real log is ~469 distinct products, 75% bought only once, so a fixed
    registry maintained by hand doesn't fit this household's actual usage."""
    if product_id in cfg.active_products:
        canon = cfg.convert(product_id, amount, unit)
        if canon is None:
            raise Rejected(
                "unit_unconvertible", value=unit, expected=cfg.active_products[product_id].unit
            )
        return canon, [], None

    known = cfg.known_products.get(product_id)
    if known is not None and known.retired:
        raise Rejected("retired_product", value=product_id)

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
    return Quantity(amount, unit), [registration], warning


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
    cfg: config.Config,
) -> tuple[list[Write], str | None]:
    """Validates and resolves one `sumac add` command into the writes it
    should produce, or raises `Rejected`. Config writes (a product
    auto-registration, if any) are ordered before the log write, so the
    registration lands before the change that depends on it."""
    if amount <= 0:
        raise Rejected("non_positive_amount", value=amount)

    from_id = _resolve_location(from_location, "from", cfg)
    to_id = _resolve_location(to_location, "to", cfg)
    if kind is ChangeKind.MOVEMENT and from_id == to_id:
        raise Rejected("noop_move", value=from_id)

    canon, writes, warning = _resolve_product(product_id, amount, unit, actor, occurred_at, cfg)

    # InventoryChange's own __post_init__ enforces the from/to shape for
    # `kind` — reuse it rather than duplicating that check here. It's a
    # ValueError, not a Rejected, so it must be converted: found by the gate
    # soundness property test (a bare ValueError isn't a SumacError, so
    # cli.main()'s handler wouldn't catch it — a `sumac add purchase` with no
    # `--to` would have crashed with a raw traceback instead of a clean
    # rejection).
    try:
        change = InventoryChange(
            kind=kind,
            product_id=product_id,
            quantity=canon,
            from_location=from_id,
            to_location=to_id,
        )
    except ValueError as e:
        raise Rejected("missing_endpoint", kind=kind.value, value=str(e)) from e
    obj = {
        "schema_version": SCHEMA_VERSION,
        "type": "change",
        "id": str(uuid4()),
        "ts": occurred_at.isoformat(),
        "actor": actor,
        "supersedes": None,
        "payload": {
            "kind": change.kind.value,
            "product_id": change.product_id,
            "quantity": {"amount": str(change.quantity.amount), "unit": change.quantity.unit},
            "from_location": change.from_location,
            "to_location": change.to_location,
            "metadata": {},
        },
    }
    writes.append(Write(f"log:{actor}", obj))
    return writes, warning
