"""Per-line AEAD: XChaCha20-Poly1305 with a fresh nonce per line, keyed by an
Argon2id-derived key. AAD binds each line to the stream it belongs to, so a
ciphertext line copied between streams fails to authenticate.
"""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass

import nacl.bindings as sodium
import nacl.exceptions
import nacl.pwhash

from sumac.errors import DecryptionError, WrongPassphraseError

AAD_PREFIX = b"sumac/v1|"
VERIFIER_STREAM_ID = "verifier"
VERIFIER_PLAINTEXT = b"sumac-verify"

KEY_SIZE = sodium.crypto_aead_xchacha20poly1305_ietf_KEYBYTES
NONCE_SIZE = sodium.crypto_aead_xchacha20poly1305_ietf_NPUBBYTES
SALT_SIZE = nacl.pwhash.argon2id.SALTBYTES

OPSLIMIT_DEFAULT = nacl.pwhash.argon2id.OPSLIMIT_MODERATE
MEMLIMIT_DEFAULT = nacl.pwhash.argon2id.MEMLIMIT_MODERATE


def _aad(stream_id: str) -> bytes:
    return AAD_PREFIX + stream_id.encode("utf-8")


def derive_key(passphrase: str, salt: bytes, opslimit: int, memlimit: int) -> bytes:
    return nacl.pwhash.argon2id.kdf(
        KEY_SIZE, passphrase.encode("utf-8"), salt, opslimit=opslimit, memlimit=memlimit
    )


def seal_line(key: bytes, stream_id: str, plaintext: bytes) -> str:
    """Encrypt `plaintext` for `stream_id`; returns base64(nonce‖ciphertext‖tag)."""
    nonce = os.urandom(NONCE_SIZE)
    ct = sodium.crypto_aead_xchacha20poly1305_ietf_encrypt(plaintext, _aad(stream_id), nonce, key)
    return base64.b64encode(nonce + ct).decode("ascii")


def open_line(key: bytes, stream_id: str, line: str) -> bytes:
    """Decrypt a sealed line, verifying it was sealed for `stream_id`."""
    try:
        raw = base64.b64decode(line, validate=True)
    except (ValueError, binascii.Error) as e:
        raise DecryptionError(f"not valid base64: {e}") from e
    if len(raw) < NONCE_SIZE:
        raise DecryptionError("sealed line too short")
    nonce, ct = raw[:NONCE_SIZE], raw[NONCE_SIZE:]
    try:
        return sodium.crypto_aead_xchacha20poly1305_ietf_decrypt(ct, _aad(stream_id), nonce, key)
    except nacl.exceptions.CryptoError as e:
        raise DecryptionError(f"failed to open line for stream {stream_id!r}: {e}") from e


@dataclass(frozen=True, slots=True)
class VaultHeader:
    """Plaintext `vault.json` content: KDF params, salt, and a sealed verifier."""

    format_version: int
    salt: bytes
    opslimit: int
    memlimit: int
    verifier: str

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "kdf": "argon2id",
            "salt": base64.b64encode(self.salt).decode("ascii"),
            "opslimit": self.opslimit,
            "memlimit": self.memlimit,
            "verifier": self.verifier,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VaultHeader:
        return cls(
            format_version=d["format_version"],
            salt=base64.b64decode(d["salt"]),
            opslimit=d["opslimit"],
            memlimit=d["memlimit"],
            verifier=d["verifier"],
        )


def new_header(
    passphrase: str,
    *,
    format_version: int,
    opslimit: int = OPSLIMIT_DEFAULT,
    memlimit: int = MEMLIMIT_DEFAULT,
) -> VaultHeader:
    salt = os.urandom(SALT_SIZE)
    key = derive_key(passphrase, salt, opslimit, memlimit)
    verifier = seal_line(key, VERIFIER_STREAM_ID, VERIFIER_PLAINTEXT)
    return VaultHeader(
        format_version=format_version,
        salt=salt,
        opslimit=opslimit,
        memlimit=memlimit,
        verifier=verifier,
    )


def check_passphrase(header: VaultHeader, passphrase: str) -> bytes:
    """Derive the key and verify it against the header's verifier; returns the key."""
    key = derive_key(passphrase, header.salt, header.opslimit, header.memlimit)
    try:
        opened = open_line(key, VERIFIER_STREAM_ID, header.verifier)
    except DecryptionError as e:
        raise WrongPassphraseError("passphrase does not match this vault") from e
    if opened != VERIFIER_PLAINTEXT:
        raise WrongPassphraseError("passphrase does not match this vault")
    return key
