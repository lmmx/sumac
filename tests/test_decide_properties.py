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

from sumac import config, decide
from sumac.errors import Rejected
from sumac.models import ChangeKind, Location, Product

_LOCATION_IDS = ["pantry", "fridge", "hob-right-shelf-bottom"]
_PRODUCT_IDS = ["milk", "flour", "eggs"]


def _base_cfg() -> config.Config:
    locations = {lid: Location(id=lid, name=lid) for lid in _LOCATION_IDS}
    products = {pid: Product(id=pid, name=pid, unit="unit") for pid in _PRODUCT_IDS}
    return config.Config(
        known_locations=locations,
        active_locations=locations,
        known_products=products,
        active_products=products,
        anomalies=(),
    )


_kind = st.sampled_from(list(ChangeKind))
_maybe_location = st.one_of(
    st.none(), st.sampled_from([*_LOCATION_IDS, "bogus-location", "Pantry"])
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
        writes, _warning = decide.decide_change(
            kind=kind,
            product_id=product_id,
            amount=amount,
            unit=unit,
            from_location=from_loc,
            to_location=to_loc,
            actor="alice",
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
            cfg=cfg,
        )
    except Rejected:
        return  # rejection is a legal outcome

    auto_registered = any(w.stream == "config" for w in writes)
    for w in writes:
        if w.stream == "config":
            continue
        payload = w.obj["payload"]
        for loc in (payload["from_location"], payload["to_location"]):
            assert loc is None or loc in cfg.known_locations
        assert payload["product_id"] in cfg.known_products or auto_registered
