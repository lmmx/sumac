"""Phase 6 (docs/journal/2026-08-30 §5): "a local-only test folds the actual
log and asserts the resulting inventory matches a checked-in hash — gated on
the passphrase being present, skipped in CI. Real-data regression detection
without committing real data."

Only ever runs on a machine with SUMAC_DATA_DIR/SUMAC_PASSPHRASE pointing at
a real vault (never true in CI). The assertion compares two hex digests, so
a failure reveals nothing about the underlying holdings/anomalies — only
that they changed. This is a regression guard on the *fold/decode logic*,
using the real vault's current state as a frozen baseline, not a source of
truth for what that state *should* be: it will start failing the next time
the real vault is used normally (a new `sumac add`, a new correction), and
that's expected — re-run generate_expected_hash() below and check in the new
digest the same way you'd update any other golden fixture, once you've
confirmed the change in state was intentional.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from sealedlog import Vault

from sumac import ledger, paths
from sumac import vault as sumac_vault

_DATA_DIR = os.environ.get("SUMAC_DATA_DIR")
_PASSPHRASE = os.environ.get("SUMAC_PASSPHRASE")
_EXPECTED_HASH_PATH = Path(__file__).parent / "real_log_inventory.sha256"

pytestmark = pytest.mark.skipif(
    not (_DATA_DIR and _PASSPHRASE),
    reason="local-only: set SUMAC_DATA_DIR and SUMAC_PASSPHRASE to a real vault to run this",
)


def _canonical_hash(inventory: ledger.Inventory) -> str:
    """Deterministic regardless of dict insertion order or Decimal string
    formatting quirks — sorts every level explicitly and normalizes amounts
    via `str(Decimal)` rather than trusting the fold's own iteration order."""
    holdings = {
        loc: {pid: [str(q.amount), q.unit] for pid, q in sorted(entries.items())}
        for loc, entries in sorted(inventory.by_location.items())
    }
    anomaly_rows = sorted((a.record_id or "", a.reason, a.detail) for a in inventory.anomalies)
    canonical = json.dumps(
        {"holdings": holdings, "anomalies": anomaly_rows}, sort_keys=True, ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_real_inventory() -> ledger.Inventory:
    assert _DATA_DIR is not None and _PASSPHRASE is not None  # narrows for the type checker
    data_dir = Path(_DATA_DIR)
    vault = Vault.from_dict(json.loads(paths.vault_path(data_dir).read_text(encoding="utf-8")))
    key = sumac_vault.unlock(vault, _PASSPHRASE)
    return ledger.build_inventory(data_dir, key)


def generate_expected_hash() -> None:
    """Not a test — a one-off maintenance helper. Run manually after a
    deliberate change to the real vault's data:
    `uv run python -c "from tests.test_real_log_regression import generate_expected_hash as g; g()"`
    """
    digest = _canonical_hash(_load_real_inventory())
    _EXPECTED_HASH_PATH.write_text(digest + "\n", encoding="utf-8")


def test_real_log_matches_checked_in_hash() -> None:
    actual = _canonical_hash(_load_real_inventory())
    expected = _EXPECTED_HASH_PATH.read_text(encoding="utf-8").strip()
    assert actual == expected
