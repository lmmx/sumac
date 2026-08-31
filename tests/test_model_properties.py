"""Phase 6 (docs/journal/2026-08-30 §5): the three properties left after gate
soundness (test_decide_properties.py, pulled forward to Phase 3) and totality
(test_ledger_properties.py, Phase 1) — model agreement, fold determinism, and
upcaster round-trip. All in-memory: no files, no crypto.

Model agreement drives `decide.decide_change` through a Hypothesis stateful
machine and checks the result against an independently, naively coded dict
model — the point being that `decide` and `evolve` sharing a mistake (e.g.
both getting a sign wrong) is exactly what gate soundness alone can't catch,
since gate soundness only checks that accepted writes *reference* known
entities, never that the *arithmetic* the fold does with them is right.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from sumac import config, decide, events, ledger, upcast
from sumac.errors import Rejected
from sumac.models import ChangeKind, Location, Product, Quantity
from sumac.schemas import RecordSchema

T0 = datetime(2026, 1, 1, tzinfo=UTC)

# --- fixtures for golden_log's checked-in corpus ------------------------

GOLDEN_DATA_DIR = Path(__file__).parent / "fixtures" / "golden_log"
GOLDEN_KEY = bytes(range(32))


def test_golden_log_folds_to_expected_state() -> None:
    """Regenerating tests/fixtures/golden_log/ (generate_golden_log.py) changes
    its ciphertext bytes but should never change this — if it does, either
    the generator or the fold changed behavior; check which was intended."""
    inv = ledger.build_inventory(GOLDEN_DATA_DIR, GOLDEN_KEY)
    assert inv.anomalies == ()
    assert inv.by_location == {
        "fridge": {"milk": Quantity(Decimal("6"), "l")},
        "pantry": {},
    }
    assert len(ledger.load_records(GOLDEN_DATA_DIR, GOLDEN_KEY)) == 14
    assert len(ledger.load_all_records(GOLDEN_DATA_DIR, GOLDEN_KEY)) == 15


# --- fold determinism -----------------------------------------------------

_DET_LOCATIONS = {
    "pantry": Location(id="pantry", name="Pantry"),
    "fridge": Location(id="fridge", name="Fridge"),
}
_DET_LOCATION_IDS = list(_DET_LOCATIONS)
_DET_PRODUCT_IDS = ["milk", "flour"]
_DET_UNIT = "l"
_det_amount = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("20"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)
_det_loc = st.sampled_from(_DET_LOCATION_IDS)
_det_pid = st.sampled_from(_DET_PRODUCT_IDS)
_det_ts = st.integers(min_value=0, max_value=5).map(lambda s: T0 + timedelta(seconds=s))
_det_actor = st.sampled_from(["alice", "bob"])

_det_event = st.one_of(
    st.builds(
        events.Acquired,
        product_id=_det_pid,
        to=_det_loc,
        amount=_det_amount,
        unit=st.just(_DET_UNIT),
    ),
    st.builds(
        events.Consumed,
        product_id=_det_pid,
        frm=_det_loc,
        amount=_det_amount,
        unit=st.just(_DET_UNIT),
    ),
    st.builds(
        events.Discarded,
        product_id=_det_pid,
        frm=_det_loc,
        amount=_det_amount,
        unit=st.just(_DET_UNIT),
    ),
    st.builds(
        events.Moved,
        product_id=_det_pid,
        frm=_det_loc,
        to=_det_loc,
        amount=_det_amount,
        unit=st.just(_DET_UNIT),
    ),
    st.builds(
        events.Counted,
        product_id=_det_pid,
        at=_det_loc,
        amount=_det_amount,
        unit=st.just(_DET_UNIT),
    ),
    st.builds(
        events.Snapshot,
        location_id=_det_loc,
        entries=st.lists(
            st.builds(
                events.SnapshotEntry,
                product_id=_det_pid,
                amount=_det_amount,
                unit=st.just(_DET_UNIT),
            ),
            max_size=2,
        ).map(tuple),
    ),
)


@settings(max_examples=150, deadline=None)
@given(
    events_list=st.lists(_det_event, max_size=6),
    data=st.data(),
)
def test_fold_determinism(events_list: list[events.Event], data: st.DataObject) -> None:
    n = len(events_list)
    ts_choices = data.draw(st.lists(_det_ts, min_size=n, max_size=n))
    actor_choices = data.draw(st.lists(_det_actor, min_size=n, max_size=n))
    records = [
        ledger._EventRecord(id=f"r{i}", ts=ts_choices[i], actor=actor_choices[i], event=ev)
        for i, ev in enumerate(events_list)
    ]

    state1, anomalies1 = ledger._fold(records, _DET_LOCATIONS)
    state2, anomalies2 = ledger._fold(records, _DET_LOCATIONS)
    assert state1 == state2
    assert anomalies1 == anomalies2

    permutation = data.draw(st.permutations(range(n)))
    shuffled = [records[i] for i in permutation]
    state3, anomalies3 = ledger._fold(shuffled, _DET_LOCATIONS)
    assert state3 == state1
    assert anomalies3 == anomalies1


# --- model agreement --------------------------------------------------

_MA_LOCATIONS = {
    "pantry": Location(id="pantry", name="Pantry"),
    "fridge": Location(id="fridge", name="Fridge"),
}
_MA_PRODUCTS = {
    "milk": Product(id="milk", name="Milk", unit="l"),
    "flour": Product(id="flour", name="Flour", unit="l"),  # same unit: no conversion path involved
}
_MA_CFG = config.Config(
    known_locations=_MA_LOCATIONS,
    active_locations=_MA_LOCATIONS,
    known_products=_MA_PRODUCTS,
    active_products=_MA_PRODUCTS,
    anomalies=(),
)
_MA_DELTA_KINDS = [
    ChangeKind.PURCHASE,
    ChangeKind.CONSUMPTION,
    ChangeKind.WASTE,
    ChangeKind.DISCOVERY,
    ChangeKind.MOVEMENT,
]


def _apply_to_naive_model(model: dict[tuple[str, str], Decimal], event: events.Event) -> None:
    """Independently reimplements what the fold *should* do, from
    docs/journal §3.1/§3.5 directly — not by calling `ledger._fold` or
    reusing any of its helpers. The point of model agreement is catching a
    mistake `decide` and `_fold` could share; an independent second
    implementation is what makes that possible."""
    match event:
        case events.Acquired(product_id=p, to=loc, amount=amount):
            key = (loc, p)
            model[key] = model.get(key, Decimal(0)) + amount
        case (
            events.Consumed(product_id=p, frm=loc, amount=amount)
            | events.Discarded(product_id=p, frm=loc, amount=amount)
        ):
            key = (loc, p)
            model[key] = model.get(key, Decimal(0)) - amount
        case events.Counted(product_id=p, at=loc, amount=amount):
            model[(loc, p)] = amount
        case events.Moved(product_id=p, frm=frm, to=to, amount=amount):
            model[(frm, p)] = model.get((frm, p), Decimal(0)) - amount
            model[(to, p)] = model.get((to, p), Decimal(0)) + amount
        case events.Snapshot(location_id=loc, entries=entries):
            for key in [k for k in model if k[0] == loc]:
                del model[key]
            for e in entries:
                model[(loc, e.product_id)] = e.amount
        case events.Correction():
            pass
        case _:  # pragma: no cover - exhaustive given events.Event
            raise TypeError(f"unhandled event type: {type(event).__name__}")


class PantryMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[ledger._EventRecord] = []
        self.model: dict[tuple[str, str], Decimal] = {}
        self.now = T0

    def _inventory(self) -> ledger.Inventory:
        state, anomalies = ledger._fold(self.records, _MA_LOCATIONS)
        return ledger.Inventory(by_location=state, anomalies=tuple(anomalies))

    @rule(
        kind=st.sampled_from(_MA_DELTA_KINDS),
        product_id=st.sampled_from(list(_MA_PRODUCTS)),
        amount=st.decimals(
            min_value=Decimal("0.01"),
            max_value=Decimal("10"),
            places=2,
            allow_nan=False,
            allow_infinity=False,
        ),
        frm=st.sampled_from(list(_MA_LOCATIONS)),
        to=st.sampled_from(list(_MA_LOCATIONS)),
    )
    def apply_delta(
        self, kind: ChangeKind, product_id: str, amount: Decimal, frm: str, to: str
    ) -> None:
        needs_from = kind in (ChangeKind.CONSUMPTION, ChangeKind.WASTE, ChangeKind.MOVEMENT)
        needs_to = kind in (ChangeKind.PURCHASE, ChangeKind.DISCOVERY, ChangeKind.MOVEMENT)
        self.now += timedelta(seconds=1)
        try:
            writes, _messages = decide.decide_change(
                kind=kind,
                product_id=product_id,
                amount=amount,
                unit="l",
                from_location=frm if needs_from else None,
                to_location=to if needs_to else None,
                actor="alice",
                occurred_at=self.now,
                inventory=self._inventory(),
                cfg=_MA_CFG,
            )
        except Rejected:
            return  # rejection is a legal outcome (e.g. noop_move)

        for w in writes:
            if w.stream == "config":  # not expected here (both products pre-registered)
                continue
            record = RecordSchema.model_validate(w.obj).to_domain()
            payload = record.payload
            assert isinstance(
                payload,
                events.Acquired
                | events.Consumed
                | events.Discarded
                | events.Moved
                | events.Counted,
            )
            self.records.append(
                ledger._EventRecord(id=record.id, ts=record.ts, actor=record.actor, event=payload)
            )
            _apply_to_naive_model(self.model, payload)

    @rule(
        location_id=st.sampled_from(list(_MA_LOCATIONS)),
        entries=st.lists(
            st.tuples(
                st.sampled_from(list(_MA_PRODUCTS)),
                st.decimals(
                    min_value=Decimal("0"),
                    max_value=Decimal("10"),
                    places=2,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            max_size=2,
            unique_by=lambda t: t[0],
        ),
    )
    def apply_snapshot(self, location_id: str, entries: list[tuple[str, Decimal]]) -> None:
        self.now += timedelta(seconds=1)
        event = events.Snapshot(
            location_id=location_id,
            entries=tuple(
                events.SnapshotEntry(product_id=pid, amount=amount, unit="l")
                for pid, amount in entries
            ),
        )
        obj = decide.serialize_event(event, actor="alice", occurred_at=self.now)
        record = RecordSchema.model_validate(obj).to_domain()
        payload = record.payload
        assert isinstance(payload, events.Snapshot)
        self.records.append(
            ledger._EventRecord(id=record.id, ts=record.ts, actor=record.actor, event=payload)
        )
        _apply_to_naive_model(self.model, payload)

    @invariant()
    def fold_matches_model(self) -> None:
        inv = self._inventory()
        assert inv.anomalies == ()
        folded = {
            (loc, pid): q.amount
            for loc, entries in inv.by_location.items()
            for pid, q in entries.items()
        }
        expected = {k: v for k, v in self.model.items() if v != 0}
        assert folded == expected


TestModelAgreement = PantryMachine.TestCase
TestModelAgreement.settings = settings(max_examples=100, stateful_step_count=30, deadline=None)


# --- upcaster round-trip ------------------------------------------------

# (kind, needs_from, needs_to) for every shape upcast.py's mapping table
# covers — the one shape it doesn't (correction with both or neither
# endpoint) is deliberately excluded; that's UpcastError's job, already
# covered by test_upcast.py's example-based tests.
_VALID_V1_SHAPES = [
    ("movement", True, True),
    ("purchase", False, True),
    ("discovery", False, True),
    ("consumption", True, False),
    ("waste", True, False),
    ("correction", False, True),  # to-only
    ("correction", True, False),  # from-only
]


def _v1_change_wire(
    kind: str, product_id: str, amount: str, unit: str, frm: str | None, to: str | None
) -> dict:
    return {
        "schema_version": 1,
        "type": "change",
        "id": "r1",
        "ts": T0.isoformat(),
        "actor": "alice",
        "supersedes": None,
        "payload": {
            "kind": kind,
            "product_id": product_id,
            "quantity": {"amount": amount, "unit": unit},
            "from_location": frm,
            "to_location": to,
            "metadata": {},
        },
    }


def _expected_v2_wire(
    kind: str, product_id: str, amount: str, unit: str, frm: str | None, to: str | None
) -> dict:
    """Independently reimplements docs/journal §3.3a's v1->v2 mapping table
    by hand, rather than calling `upcast.upcast` or `decide._build_event` —
    the whole point is a second, separately-written source of truth to check
    the real upcaster against."""
    if kind == "movement":
        type_name, payload = "moved", {"frm": frm, "to": to, "nominal_basis": None}
    elif kind == "purchase":
        type_name, payload = "acquired", {"to": to, "reason": None, "nominal_basis": None}
    elif kind == "discovery":
        type_name, payload = "acquired", {"to": to, "reason": "discovery", "nominal_basis": None}
    elif kind == "consumption":
        type_name, payload = "consumed", {"frm": frm, "reason": None, "nominal_basis": None}
    elif kind == "waste":
        type_name, payload = "discarded", {"frm": frm, "nominal_basis": None}
    elif kind == "correction" and to is not None:
        type_name, payload = "acquired", {"to": to, "reason": "correction", "nominal_basis": None}
    elif kind == "correction" and frm is not None:
        type_name, payload = "consumed", {"frm": frm, "reason": "correction", "nominal_basis": None}
    else:  # pragma: no cover - not in _VALID_V1_SHAPES
        raise ValueError(kind)

    return {
        "schema_version": 2,
        "type": type_name,
        "id": "r2",
        "ts": T0.isoformat(),
        "actor": "alice",
        "supersedes": None,
        "payload": {"product_id": product_id, "amount": amount, "unit": unit, **payload},
    }


@settings(max_examples=200, deadline=None)
@given(
    shape=st.sampled_from(_VALID_V1_SHAPES),
    product_id=st.sampled_from(["milk", "flour", "eggs"]),
    amount=st.decimals(
        min_value=Decimal("0.01"),
        max_value=Decimal("1000"),
        places=2,
        allow_nan=False,
        allow_infinity=False,
    ),
    unit=st.sampled_from(["l", "kg", "unit"]),
    frm_loc=st.sampled_from(["pantry", "fridge"]),
    to_loc=st.sampled_from(["pantry", "fridge"]),
)
def test_upcast_matches_independently_coded_v2_mapping(
    shape: tuple[str, bool, bool],
    product_id: str,
    amount: Decimal,
    unit: str,
    frm_loc: str,
    to_loc: str,
) -> None:
    kind, needs_from, needs_to = shape
    frm = frm_loc if needs_from else None
    to = to_loc if needs_to else None
    amount_str = str(amount)

    v1_record = RecordSchema.model_validate(
        _v1_change_wire(kind, product_id, amount_str, unit, frm, to)
    ).to_domain()
    got = upcast.upcast(v1_record)

    expected_record = RecordSchema.model_validate(
        _expected_v2_wire(kind, product_id, amount_str, unit, frm, to)
    ).to_domain()
    assert got == expected_record.payload


@settings(max_examples=100, deadline=None)
@given(
    location_id=st.sampled_from(["pantry", "fridge"]),
    entries=st.lists(
        st.tuples(
            st.sampled_from(["milk", "flour"]),
            st.decimals(
                min_value=Decimal("0"),
                max_value=Decimal("10"),
                places=2,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.sampled_from(["l", "kg"]),
        ),
        max_size=3,
    ),
)
def test_upcast_snapshot_matches_independently_coded_v2_mapping(
    location_id: str, entries: list[tuple[str, Decimal, str]]
) -> None:
    v1_wire = {
        "schema_version": 1,
        "type": "snapshot",
        "id": "r1",
        "ts": T0.isoformat(),
        "actor": "alice",
        "supersedes": None,
        "payload": {
            "location_id": location_id,
            "entries": [
                {"product_id": pid, "quantity": {"amount": str(amt), "unit": unit}, "metadata": {}}
                for pid, amt, unit in entries
            ],
        },
    }
    v2_wire = {
        "schema_version": 2,
        "type": "snapshot",
        "id": "r2",
        "ts": T0.isoformat(),
        "actor": "alice",
        "supersedes": None,
        "payload": {
            "location_id": location_id,
            "entries": [
                {"product_id": pid, "amount": str(amt), "unit": unit, "nominal_basis": None}
                for pid, amt, unit in entries
            ],
        },
    }
    v1_record = RecordSchema.model_validate(v1_wire).to_domain()
    got = upcast.upcast(v1_record)
    expected_record = RecordSchema.model_validate(v2_wire).to_domain()
    assert got == expected_record.payload
