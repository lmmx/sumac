"""Phase 1 acceptance criterion (docs/journal/2026-08-30_decide-pattern-data-integrity-upgrade.md):
`evolve` (here, `ledger.build_inventory`) must never raise, for any event a permissive
generator can produce — including malformed shapes no CLI command would ever write.

Runs against real encrypted files rather than an in-memory model, so each example gets
its own throwaway data dir instead of a shared pytest fixture (Hypothesis warns about
reusing function-scoped fixtures across examples). Phase 6 adds the pure in-memory
property suite (model agreement, determinism, upcaster round-trip) once `decide` exists.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sumac import ledger, store, writer
from sumac import vault as sumac_vault

_WRITER_ID = "alice-mac"

_id = st.text(min_size=0, max_size=8)
_maybe_id = st.one_of(st.none(), _id)
_json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=8),
)
_quantity = st.fixed_dictionaries(
    {
        "amount": st.one_of(_id, st.integers(min_value=-1000, max_value=1000)),
        "unit": _id,
    }
)
_change_payload = st.fixed_dictionaries(
    {
        "kind": st.sampled_from(
            ["purchase", "consumption", "waste", "discovery", "correction", "movement", "bogus"]
        ),
        "product_id": _id,
        "quantity": _quantity,
        "from_location": _maybe_id,
        "to_location": _maybe_id,
        "metadata": st.dictionaries(_id, _json_scalar, max_size=3),
    }
)
_snapshot_entry = st.fixed_dictionaries(
    {
        "product_id": _id,
        "quantity": _quantity,
        "metadata": st.dictionaries(_id, _json_scalar, max_size=2),
    }
)
_snapshot_payload = st.fixed_dictionaries(
    {"location_id": _id, "entries": st.lists(_snapshot_entry, max_size=3)}
)
_record = st.fixed_dictionaries(
    {
        "schema_version": st.integers(min_value=-1, max_value=4),  # SCHEMA_VERSION is 2
        "type": st.sampled_from(["change", "snapshot", "bogus"]),
        "id": _id,
        "ts": st.one_of(st.just("2026-01-01T00:00:00+00:00"), _id),
        "actor": _id,
        "supersedes": _maybe_id,
        "payload": st.one_of(_change_payload, _snapshot_payload),
    }
)


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(records=st.lists(_record, max_size=6))
def test_build_inventory_never_raises(records: list[dict]) -> None:
    tmp = Path(tempfile.mkdtemp())
    prior_env = os.environ.get(writer.WRITER_ID_ENV)
    os.environ[writer.WRITER_ID_ENV] = _WRITER_ID
    try:
        data_dir = tmp / "data"
        vault = sumac_vault.create("pw")
        key = sumac_vault.unlock(vault, "pw")
        for obj in records:
            store.append(data_dir, key, writer.log_stream_id(_WRITER_ID), obj)
        ledger.build_inventory(data_dir, key)  # must not raise
    finally:
        if prior_env is None:
            os.environ.pop(writer.WRITER_ID_ENV, None)
        else:
            os.environ[writer.WRITER_ID_ENV] = prior_env
        shutil.rmtree(tmp, ignore_errors=True)
