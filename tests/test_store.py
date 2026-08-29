from __future__ import annotations

from pathlib import Path

import pytest

from sumac import paths, store
from sumac.errors import ForeignStreamError


def test_append_and_iter_round_trip(data_dir: Path, osuser: str, key: bytes) -> None:
    store.append(data_dir, key, f"log:{osuser}", {"n": 1})
    store.append(data_dir, key, f"log:{osuser}", {"n": 2})
    assert list(store.iter_stream(data_dir, key, f"log:{osuser}")) == [{"n": 1}, {"n": 2}]


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


def test_iter_all_logs_spans_users(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, key: bytes
) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    store.append(data_dir, key, "log:alice", {"n": 1})
    monkeypatch.setattr("getpass.getuser", lambda: "bob")
    store.append(data_dir, key, "log:bob", {"n": 2})

    seen = sorted(store.iter_all_logs(data_dir, key))
    assert seen == [("alice", {"n": 1}), ("bob", {"n": 2})]
