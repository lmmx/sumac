"""Gate soundness (docs/journal/2026-08-30 §5, Phase 3): no sequence of
commands accepted by `decide` should produce an anomaly — the class of bug
that produced `hob-right-below-bottom` (an accepted write referencing
something the fold can't resolve). Pulled forward from Phase 6 so it's
exercised while `decide` and `evolve` are both freshly written, per the
Phase 3 review that asked for this rather than deferring it four phases.

`decide_change` is pure, so this runs entirely in memory — no files, no
crypto, no function-scoped-fixture health check to suppress.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from sumac import config, decide, ledger
from sumac.errors import Rejected
from sumac.models import ChangeKind, Location, Product

_LOCATION_KEYS_BY_TYPE = {
    "acquired": ["to"],
    "consumed": ["frm"],
    "discarded": ["frm"],
    "moved": ["frm", "to"],
    "counted": ["at"],
}
_EMPTY_INVENTORY = ledger.Inventory(by_location={})

_LOCATIONS = {
    "pantry": Location(id="pantry", name="Pantry"),
    "pantry-shelf": Location(id="pantry-shelf", name="Shelf", parent_id="pantry"),
    "fridge": Location(id="fridge", name="Fridge"),
    "hob-right-shelf-bottom": Location(id="hob-right-shelf-bottom", name="Bottom"),
}
_LOCATION_IDS = list(_LOCATIONS)
_DISPLAY_PATHS = [config.location_path(_LOCATIONS, lid) for lid in _LOCATION_IDS]
_PRODUCT_IDS = ["milk", "flour", "eggs"]


def _base_cfg() -> config.Config:
    return config.Config(
        known_locations=_LOCATIONS,
        active_locations=_LOCATIONS,
        known_products={pid: Product(id=pid, name=pid, unit="unit") for pid in _PRODUCT_IDS},
        active_products={pid: Product(id=pid, name=pid, unit="unit") for pid in _PRODUCT_IDS},
        anomalies=(),
    )


_kind = st.sampled_from(list(ChangeKind))
_maybe_location = st.one_of(
    st.none(),
    st.sampled_from([*_LOCATION_IDS, *_DISPLAY_PATHS, "bogus-location"]),
)
_product_id = st.sampled_from([*_PRODUCT_IDS, "new-product", "milc"])
_amount = st.decimals(
    min_value=Decimal("-5"),
    max_value=Decimal("10"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)
_unit = st.sampled_from(["l", "kg", "unit", "jar"])


@settings(max_examples=200, deadline=None)
@given(
    kind=_kind,
    product_id=_product_id,
    amount=_amount,
    unit=_unit,
    from_loc=_maybe_location,
    to_loc=_maybe_location,
)
def test_gate_soundness_accepted_writes_reference_only_known_entities(
    kind: ChangeKind,
    product_id: str,
    amount: Decimal,
    unit: str,
    from_loc: str | None,
    to_loc: str | None,
) -> None:
    cfg = _base_cfg()
    try:
        writes, _messages = decide.decide_change(
            kind=kind,
            product_id=product_id,
            amount=amount,
            unit=unit,
            from_location=from_loc,
            to_location=to_loc,
            actor="alice",
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
            inventory=_EMPTY_INVENTORY,
            cfg=cfg,
        )
    except Rejected:
        return  # rejection is a legal outcome

    config_writes = [w for w in writes if w.stream == "config"]
    for w in writes:
        if w.stream == "config":
            continue
        payload = w.obj["payload"]
        for key in _LOCATION_KEYS_BY_TYPE[w.obj["type"]]:
            loc = payload[key]
            assert loc is None or loc in cfg.known_locations
        pid = payload["product_id"]
        if pid not in cfg.known_products:
            # Not just "some config write happened" — the registration must
            # be for *this* product, or an unknown product could accept
            # silently under the wrong id's cover.
            assert any(cw.obj["product"]["id"] == pid for cw in config_writes)
