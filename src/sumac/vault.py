"""Wraps `sealedlog.Vault` with sumac's namespace baked in, so call sites
never pass `namespace=` themselves and can't typo it into a `WrongPassphraseError`
that looks like data corruption.
"""

from __future__ import annotations

from sealedlog import Vault
from sealedlog.vault import MEMLIMIT_DEFAULT, OPSLIMIT_DEFAULT

from sumac import NAMESPACE


def create(
    passphrase: str, *, opslimit: int = OPSLIMIT_DEFAULT, memlimit: int = MEMLIMIT_DEFAULT
) -> Vault:
    return Vault.create(passphrase, namespace=NAMESPACE, opslimit=opslimit, memlimit=memlimit)


def unlock(vault: Vault, passphrase: str) -> bytes:
    return vault.unlock(passphrase, namespace=NAMESPACE)
