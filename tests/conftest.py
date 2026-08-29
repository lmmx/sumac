from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import nacl.pwhash.argon2id as argon2id
import pytest

from sumac import crypto, passphrase


@pytest.fixture(autouse=True)
def _fast_kdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Argon2id at MIN cost so tests don't pay real KDF latency."""
    original = crypto.new_header

    def fast_new_header(
        pw: str,
        *,
        format_version: int,
        opslimit: int = argon2id.OPSLIMIT_MIN,
        memlimit: int = argon2id.MEMLIMIT_MIN,
    ) -> crypto.VaultHeader:
        return original(pw, format_version=format_version, opslimit=opslimit, memlimit=memlimit)

    monkeypatch.setattr(crypto, "new_header", fast_new_header)


@pytest.fixture(autouse=True)
def _reset_key_cache() -> Iterator[None]:
    passphrase.reset_cache()
    yield
    passphrase.reset_cache()


@pytest.fixture
def osuser(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    return "alice"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"
