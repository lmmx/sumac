from __future__ import annotations

import pytest

from sumac import crypto
from sumac.errors import DecryptionError, WrongPassphraseError


def _key() -> bytes:
    return crypto.derive_key("pw", b"0" * crypto.SALT_SIZE, 1, 8192)


def test_seal_open_round_trip() -> None:
    key = _key()
    sealed = crypto.seal_line(key, "log:alice", b"hello")
    assert crypto.open_line(key, "log:alice", sealed) == b"hello"


def test_seal_produces_fresh_nonce_each_time() -> None:
    key = _key()
    a = crypto.seal_line(key, "log:alice", b"hello")
    b = crypto.seal_line(key, "log:alice", b"hello")
    assert a != b


def test_open_line_wrong_stream_fails() -> None:
    key = _key()
    sealed = crypto.seal_line(key, "log:alice", b"hello")
    with pytest.raises(DecryptionError):
        crypto.open_line(key, "log:bob", sealed)


def test_open_line_wrong_key_fails() -> None:
    sealed = crypto.seal_line(_key(), "log:alice", b"hello")
    other_key = crypto.derive_key("different", b"1" * crypto.SALT_SIZE, 1, 8192)
    with pytest.raises(DecryptionError):
        crypto.open_line(other_key, "log:alice", sealed)


def test_open_line_rejects_garbage() -> None:
    with pytest.raises(DecryptionError):
        crypto.open_line(_key(), "log:alice", "not-valid-base64!!!")


def test_new_header_and_check_passphrase() -> None:
    header = crypto.new_header("correct horse", format_version=1)
    key = crypto.check_passphrase(header, "correct horse")
    assert len(key) == crypto.KEY_SIZE


def test_check_passphrase_wrong_raises() -> None:
    header = crypto.new_header("correct horse", format_version=1)
    with pytest.raises(WrongPassphraseError):
        crypto.check_passphrase(header, "wrong horse")


def test_header_round_trips_through_dict() -> None:
    header = crypto.new_header("pw", format_version=1)
    restored = crypto.VaultHeader.from_dict(header.to_dict())
    assert restored == header
