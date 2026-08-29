from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import nacl.pwhash.argon2id as argon2id
import pytest
from sealedlog import Vault

from sumac import passphrase
from sumac import vault as sumac_vault


@pytest.fixture(autouse=True)
def _fast_kdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Argon2id at MIN cost so tests don't pay real KDF latency."""
    original = sumac_vault.create

    def fast_create(
        pw: str,
        *,
        opslimit: int = argon2id.OPSLIMIT_MIN,
        memlimit: int = argon2id.MEMLIMIT_MIN,
    ) -> Vault:
        return original(pw, opslimit=opslimit, memlimit=memlimit)

    monkeypatch.setattr(sumac_vault, "create", fast_create)


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


@pytest.fixture
def key(_fast_kdf: None) -> bytes:
    vault = sumac_vault.create("pw")
    return sumac_vault.unlock(vault, "pw")
