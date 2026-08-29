"""Env-then-prompt passphrase resolution, with key caching within a process."""

from __future__ import annotations

import getpass
import os

from sumac import crypto

ENV_VAR = "SUMAC_PASSPHRASE"

_key_cache: bytes | None = None


def resolve_passphrase() -> str:
    env = os.environ.get(ENV_VAR)
    if env:
        return env
    return getpass.getpass("sumac passphrase: ")


def get_key(header: crypto.VaultHeader) -> bytes:
    """Resolve the passphrase and derive the key, caching within this process."""
    global _key_cache
    if _key_cache is None:
        _key_cache = crypto.check_passphrase(header, resolve_passphrase())
    return _key_cache


def reset_cache() -> None:
    global _key_cache
    _key_cache = None
