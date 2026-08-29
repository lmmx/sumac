# sealedlog: build spec

## Context

You're being handed a read-only repo called `sumac` — a small Python app for encrypted home
grocery inventory. Its `src/sumac/crypto.py` and `src/sumac/store.py`, plus their tests
(`tests/test_crypto.py`, `tests/test_store.py`) and `docs/FORMAT.md`, implement a pattern worth
extracting: an encrypted, append-only, line-oriented log where each line is independently
authenticated and git packs it well. That pattern is the entire scope of this task. Everything
else in `sumac` (the grocery domain model, the ledger, the CLI) is out of scope — read it only
to understand how the crypto/store layer gets consumed, not to carry any of it over.

Build a new, standalone package called `sealedlog`. Treat `sumac`'s implementation as a working
reference, not a template to copy verbatim — it hardcoded some things (a fixed `"sumac/v1|"` AAD
prefix, a fixed data-dir layout, an ownership check tied to `getpass.getuser()`) that were
correct for one app and need to become either configurable or excluded entirely, per the
non-goals below.

## What this library is

A primitive for storing a sequence of JSON records, encrypted at rest, one independently
authenticated line per record, such that:

- Appending a record is a byte-append to the file — no rewriting, no reordering. Git diffs and
  merges cleanly on files built this way.
- Each line decrypts independently. A corrupted or truncated line doesn't block reading the
  lines before or after it.
- Lines are bound to the logical stream they were written for. A line copied from one stream
  into another fails to authenticate, even under the correct key.
- The key comes from a passphrase via a memory-hard KDF, and a wrong passphrase is detected
  immediately and unambiguously — not by silently producing garbage.

Think "SQLite for encrypted append-only JSON logs, minus the SQL" — small, embeddable, no
daemon, no external services, just a file format and the code to read/write it correctly.

## Non-goals — leave these to the calling application

- **Ownership/ACL enforcement.** `sumac`'s `store.append()` refused to write to another user's
  stream by checking `getpass.getuser()`. That policy is app-specific. `sealedlog` should make
  policies like this *easy to bolt on* (e.g., stream IDs are just strings; the AAD binding
  gives the app a way to make violations detectable) but should not itself know what a "user"
  or "owner" is.
- **Schema versioning / record shape validation.** `sumac` rejected records from a newer
  `schema_version`. That's application-level protocol, not storage-level. `sealedlog` stores
  whatever JSON-serializable dict it's given.
- **Merge semantics, ledger folding, "latest revision wins," supersede/correction chains.** All
  application logic on top of a plain sequence of records.
- **Directory layout conventions.** `sumac` decided path components must never leak location
  names. `sealedlog` operates on a file handle or path the caller provides; it has no opinion
  on where that path lives or what it's named.
- **Concurrent multi-process writers to the same file.** Assume single-writer-per-file (as
  `sumac` does: one log file per OS user). Document this assumption; don't build locking.
- **Algorithm agility / pluggable ciphers.** Pick one good AEAD (see below) and ship it. Don't
  build a registry of cipher suites for a v1.

## Security requirements

- **AEAD: XChaCha20-Poly1305**, via PyNaCl (`nacl.bindings`). Not `cryptography`'s
  `ChaCha20Poly1305`, which only supports the 12-byte-nonce IETF variant — too small a nonce
  space to generate randomly per line at scale without a counter. XChaCha20's 24-byte nonce
  makes "just call `os.urandom()` per line, no coordination needed" safe.
- **Fresh random nonce per line.** Never derive or reuse a nonce.
- **AAD binds each line to a logical stream.** A stream identifier (opaque string, app-supplied)
  goes into the AAD for every seal/open call. Design the AAD so two different applications using
  `sealedlog` with the same library version don't accidentally produce cross-compatible
  ciphertexts — i.e., don't hardcode a bare version string like `sumac` did; take a
  caller-supplied namespace (e.g. the app's own name) and fold it into the AAD alongside the
  stream ID and the library's own format version. Exact scheme is your call; document it.
- **KDF: Argon2id** (via `nacl.pwhash.argon2id`), with the salt and cost parameters stored
  alongside the ciphertext, not derived implicitly. Expose the cost parameters as constructor
  arguments with sane defaults (`sumac` used `OPSLIMIT_MODERATE`/`MEMLIMIT_MODERATE`); tests
  should use `OPSLIMIT_MIN`/`MEMLIMIT_MIN` for speed, same as `sumac`'s test suite does — check
  `tests/conftest.py` there for the pattern (it monkeypatches the header-creation function to
  default to MIN cost, so callers never have to thread cost params through every call site).
- **Passphrase verification without decrypting real data.** A wrong passphrase must fail loudly
  and immediately, distinguishable from "this line is corrupted." `sumac`'s approach — seal a
  known plaintext under a reserved `"verifier"` stream ID at vault-creation time, and check it
  decrypts to the expected value before trusting the derived key for anything else — is worth
  keeping essentially as-is.
- **Line format:** `base64(nonce ‖ ciphertext ‖ tag)`, one line per record, newline-terminated.
  Base64 keeps the file text-safe for git and diff tools; a malformed base64 line should raise
  the same "this line failed" error as a failed decryption, not a different exception type the
  caller has to special-case.

## Proposed API

Two layers: a small stateless byte-level primitive, and a thin stateful log wrapper around it.
Keep both — the byte-level functions are useful standalone for anyone who wants their own framing
(e.g., not JSONL), and the log wrapper is what most callers actually want.

```python
# sealedlog/aead.py — stateless, no file I/O

def seal(key: bytes, aad: bytes, plaintext: bytes) -> str:
    """Returns base64(nonce‖ciphertext‖tag)."""

def open_(key: bytes, aad: bytes, sealed: str) -> bytes:
    """Raises SealError on any authentication failure or malformed input."""
```

```python
# sealedlog/vault.py — passphrase -> key, with a verifier

@dataclass(frozen=True)
class Vault:
    salt: bytes
    opslimit: int
    memlimit: int
    verifier: str  # a sealed line under a reserved stream id

    @classmethod
    def create(cls, passphrase: str, *, namespace: str, opslimit=..., memlimit=...) -> Vault: ...

    def unlock(self, passphrase: str, *, namespace: str) -> bytes:
        """Returns the derived key, or raises WrongPassphrase."""

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> Vault: ...
```

```python
# sealedlog/log.py — the append-only file wrapper

class SealedLog:
    def __init__(self, path: Path, key: bytes, stream_id: str, *, namespace: str): ...

    def append(self, record: dict) -> None:
        """JSON-encodes and appends one sealed line. Byte-append only."""

    def __iter__(self) -> Iterator[dict]:
        """Lazily decrypts and yields records in file order. Raises SealError on
        the first bad line — use verify() if you want to survive corruption."""

    def verify(self) -> list[LineFailure]:
        """Never raises. Attempts every line; returns failures instead of stopping."""

@dataclass(frozen=True)
class LineFailure:
    lineno: int
    error: str
```

Adjust names/signatures as you see fit — this is a starting shape, not a contract. Things worth
holding onto from `sumac`'s version regardless of exact naming:
- `verify()` returning a list rather than raising, so a caller can report every bad line in one
  pass instead of stopping at the first.
- Keeping `append`/`__iter__` ignorant of *why* a stream ID is what it is — no ownership checks,
  no filename derivation. That belongs one layer up.
- The plaintext record being a plain `dict` (via `json.dumps(..., separators=(",", ":"))` /
  `json.loads`), not a schema — validation is the caller's job.

## Error taxonomy

At minimum, distinguish:
- Wrong passphrase (verifier mismatch) from
- A corrupted/tampered/foreign-stream line (auth failure on a non-verifier line) from
- Malformed input (not valid base64, too short to contain a nonce)

`sumac`'s hierarchy (`sumac/errors.py`) is a reasonable model: a single `SealError` base, with
`WrongPassphrase` and a generic decryption/auth failure as subtypes. Whether malformed-base64
gets its own subtype or folds into the generic auth-failure type is your call — just be
consistent, and make sure `verify()` can report *which* kind happened per line if that's useful
for diagnostics.

## Testing requirements

Port and generalize the properties covered in `tests/test_crypto.py` and `tests/test_store.py`
in the `sumac` repo. At minimum:
- Seal/open round-trips; fresh nonce each call (two seals of identical plaintext produce
  different output).
- Open fails under the wrong key.
- Open fails when a line sealed for stream A is presented as stream B (this is the core
  guarantee of the whole library — test it explicitly and don't let it regress).
- Open fails on garbage/malformed base64, with the same error type as a real auth failure.
- Vault: correct passphrase unlocks; wrong passphrase raises before any real data is touched;
  `Vault.to_dict()`/`from_dict()` round-trips.
- `SealedLog.append` is a strict byte-append (assert file contents before are a prefix of file
  contents after).
- `SealedLog.verify()` on a file with one tampered line in the middle: returns exactly one
  failure, at the correct line number, and doesn't lose the surrounding valid records if you
  separately iterate the untampered lines.
- A property test or explicit case for "many records round-trip through append+iterate in
  order" — order preservation matters since this is meant to back an event log.

Use fast Argon2id params in tests (`opslimit=1`, low `memlimit`) — see `sumac/tests/conftest.py`
for the pattern of wrapping the header/vault-creation function so individual tests don't need to
thread cost params through.

## Packaging

Mirror `sumac`'s tooling since it's a reasonable, already-working baseline: `uv` for
dependencies with a committed `uv.lock`, `ruff format` + `ruff check` for style/lint, `ty` for
type checking, `pytest` for tests, a GitHub Actions CI running all four. `pynacl` is the only
runtime dependency. No CLI, no Typer, no rich — this is a library, not an app.

## Open decisions left to you

- Exact AAD construction (namespace + stream ID + library format version — pick a delimiter and
  document it; the only hard requirement is that it can't collide across stream IDs or
  namespaces in a way that lets a line from one context authenticate in another).
- Whether `SealedLog` owns the file handle/lifecycle or takes an already-open file-like object.
- Whether to support non-dict JSON values as records (`sumac` only ever wrote objects) — bare
  minimum is objects; decide if lists/scalars are worth supporting too.
- Package/module naming below the top level — `sealedlog` as the distribution name is a
  placeholder; keep it if you like it.
