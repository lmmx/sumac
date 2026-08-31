from __future__ import annotations

from pathlib import Path

import pytest
from sealedlog import aead
from sealedlog._aad import build_aad

from sumac import NAMESPACE, paths, store
from sumac.errors import ForeignStreamError


def test_append_and_iter_round_trip(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(data_dir, key, f"log:{osuser}", {"n": 1})
    store.append(data_dir, key, f"log:{osuser}", {"n": 2})
    assert list(store.iter_stream(data_dir, key, f"log:{osuser}")) == [
        {"n": 1, "seq": 0},
        {"n": 2, "seq": 1},
    ]


def test_append_assigns_monotone_seq_per_stream(data_dir: Path, osuser: str, key: bytes) -> None:
    """docs/journal §3.7: seq is append-time envelope data, assigned by
    `store.append` itself (not the caller), monotone starting at 0, and
    independent per log stream."""
    for i in range(3):
        store.append(data_dir, key, f"log:{osuser}", {"n": i})
    seqs = [obj["seq"] for obj in store.iter_stream(data_dir, key, f"log:{osuser}")]
    assert seqs == [0, 1, 2]


def test_append_ignores_a_caller_supplied_seq(data_dir: Path, osuser: str, key: bytes) -> None:
    """seq is append's job, not the caller's — a stray `seq` key on the
    object passed in must never survive; `store.append` is the sole source
    of truth for what gets written."""
    store.append(data_dir, key, f"log:{osuser}", {"n": 1, "seq": 999})
    [obj] = list(store.iter_stream(data_dir, key, f"log:{osuser}"))
    assert obj["seq"] == 0


def test_config_stream_never_gets_a_seq(data_dir: Path, osuser: str, key: bytes) -> None:
    """Config is latest-revision-wins, not an append-sequential segment —
    seq assignment is scoped to `log:`-prefixed streams only."""
    store.append(data_dir, key, store.CONFIG_STREAM_ID, {"loc": "fridge"})
    [obj] = list(store.iter_stream(data_dir, key, store.CONFIG_STREAM_ID))
    assert "seq" not in obj


def test_append_rejects_foreign_stream_id(data_dir: Path, osuser: str, key: bytes) -> None:
    with pytest.raises(ForeignStreamError):
        store.append(data_dir, key, "log:bob", {"n": 1})


def test_append_is_byte_append(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(data_dir, key, f"log:{osuser}", {"n": 1})
    path = paths.log_path(data_dir, osuser)
    before = path.read_bytes()
    store.append(data_dir, key, f"log:{osuser}", {"n": 2})
    after = path.read_bytes()
    assert after.startswith(before)


def test_config_stream_unrestricted(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(data_dir, key, store.CONFIG_STREAM_ID, {"loc": "fridge"})
    assert list(store.iter_stream(data_dir, key, store.CONFIG_STREAM_ID)) == [{"loc": "fridge"}]


def test_line_moved_between_streams_fails_to_open(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(data_dir, key, f"log:{osuser}", {"n": 1})
    alice_path = paths.log_path(data_dir, osuser)
    sealed_line = alice_path.read_text().strip()

    bob_path = paths.log_path(data_dir, "bob")
    bob_path.parent.mkdir(parents=True, exist_ok=True)
    bob_path.write_text(sealed_line + "\n")

    _ok, failures = store.verify_stream(bob_path, key, "log:bob")
    assert len(failures) == 1


def test_verify_stream_reports_line_and_position(data_dir: Path, osuser: str, key: bytes) -> None:
    path = paths.log_path(data_dir, osuser)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-valid-base64!!!\n")

    ok, failures = store.verify_stream(path, key, f"log:{osuser}")
    assert ok == []
    assert len(failures) == 1
    assert failures[0].lineno == 1


def test_verify_stream_reports_authenticated_non_json_line(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """A line that authenticates but whose plaintext isn't JSON must be a LineFailure,
    not an uncaught JSONDecodeError — this is what makes verify_stream (and doctor,
    which relies on it) genuinely tolerant rather than tolerant-except-for-this."""
    path = paths.log_path(data_dir, osuser)
    path.parent.mkdir(parents=True, exist_ok=True)
    aad = build_aad(NAMESPACE, f"log:{osuser}")
    sealed = aead.seal(key, aad, b"not json{")
    path.write_text(sealed + "\n")

    ok, failures = store.verify_stream(path, key, f"log:{osuser}")
    assert ok == []
    assert len(failures) == 1
    assert "JSON" in failures[0].error


def test_iter_all_logs_spans_users(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, key: bytes
) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    store.append(data_dir, key, "log:alice", {"n": 1})
    monkeypatch.setattr("getpass.getuser", lambda: "bob")
    store.append(data_dir, key, "log:bob", {"n": 2})

    seen = sorted(store.iter_all_logs(data_dir, key))
    assert seen == [("alice", {"n": 1, "seq": 0}), ("bob", {"n": 2, "seq": 0})]


def test_assigned_seqs_backfills_position_for_records_without_a_stored_seq() -> None:
    """docs/journal §3.7: a record with no stored `seq` (everything written
    before Phase 7) is treated as having whatever position it holds among
    the decoded objects — the same backfill the v1 upcaster already relies
    on for old data."""
    objs = [{"n": 1}, {"n": 2, "seq": 5}, {"n": 3}]
    assert store.assigned_seqs(objs) == [0, 5, 2]
