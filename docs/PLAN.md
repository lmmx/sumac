# sumac — encrypted home grocery inventory

## Context

`/workspace` is a greenfield repo (`README.md`, `prompt.md`, one commit). `prompt.md` specifies a Python
app that logs a household's grocery inventory across locations, with everything sensitive encrypted at
rest so the repo can be pushed to a normal git remote.

The threat model is narrow and worth stating, because it drives the whole design: **anyone holding the
repo but not the passphrase must learn nothing about the home's layout or its contents.** That means the
location config is encrypted too, and no filename, directory name, or git metadata may be derived from a
location or product name. What we knowingly accept as leakage: record count, approximate record size,
timestamps of commits, and OS usernames.

Households share one passphrase. Users `git pull` the app, append their own records, and push. Ownership
is a convention (`getpass.getuser()`), not file permissions — so the design makes violations *detectable*
rather than impossible.

Decisions taken with the user: passphrase from `SUMAC_PASSPHRASE` else interactive prompt; Typer + rich
for the CLI; one JSONL log per user.

## On-disk layout

```
data/                       # mutable — users append here
  vault.json                # plaintext header: format version, KDF params, salt, verifier
  config.jsonl.enc          # encrypted JSONL, append-only; latest revision wins
  log/<osuser>.jsonl        # encrypted JSONL, one per user, append-only
src/sumac/                  # read-only for users
```

Every path component is a fixed literal or an OS username. Nothing is derived from user data.

**Sealed line** — `base64(nonce‖ciphertext‖tag)` + `\n`, XChaCha20-Poly1305, fresh 24-byte nonce per line.
Appending is a byte-append; git packs it well.

**AAD binds each line to its stream**: `b"sumac/v1|" + stream_id`, where `stream_id` is `"config"` or
`"log:<osuser>"`. A ciphertext line copied out of alice's log into bob's fails to open. This is what makes
the ownership convention auditable — it can't stop alice from truncating her own file, but no one can
launder a record into someone else's history.

**`vault.json`** holds `format_version`, Argon2id params (`salt`, `opslimit`, `memlimit`), and a `verifier`
line sealed under AAD `sumac/v1|verifier`. A wrong passphrase fails there with a clear message instead of
producing garbage downstream.

## Modules

One scope each; `models.py` is standalone so the data model can be edited without touching anything else.

| Module | Responsibility |
| --- | --- |
| `sumac/__init__.py` | `__version__`, `SCHEMA_VERSION`, `FORMAT_VERSION` |
| `sumac/models.py` | Frozen dataclasses. No I/O, no crypto, no pydantic imports. |
| `sumac/schemas.py` | Pydantic v2 models at ingest boundaries + `to_domain()` converters |
| `sumac/crypto.py` | `derive_key`, `seal_line`, `open_line`, `new_header`, `check_passphrase` |
| `sumac/passphrase.py` | env-then-prompt resolution, key caching within a process |
| `sumac/paths.py` | Data-dir layout; the single place path names are constructed |
| `sumac/store.py` | Encrypted JSONL append/iterate over a `stream_id` |
| `sumac/config.py` | Location layout on top of `store` |
| `sumac/ledger.py` | Fold snapshots + changes → current inventory |
| `sumac/render.py` | rich tables/panels — kept out of `cli.py` so command logic stays testable |
| `sumac/cli.py` | Typer app |
| `sumac/errors.py` | Exception hierarchy the CLI maps to exit codes |

## Data model (`models.py`)

- `Location(id, name, parent_id, metadata)` — flat with optional parent, so sublocations nest arbitrarily.
- `Product(id, name, unit, category, metadata)`
- `Quantity(amount: Decimal, unit: str)` — mismatched units raise rather than silently coerce.
- `ChangeKind` enum: `purchase | consumption | waste | discovery | correction | movement`.
- `InventoryChange(..., product_id, quantity, from_location, to_location, ...)` — a delta or a transfer.
- `SnapshotEntry(product_id, quantity, metadata)`; `InventorySnapshot(location_id, entries, ...)` — the
  full observed state of one location at one time.
- `Record(schema_version, type, id, ts, actor, supersedes, payload)` — the envelope on every JSONL line.

`metadata: Mapping[str, JsonValue]` on products, changes, snapshots and snapshot entries carries seller- or
user-supplied extras beyond the core model; validated as JSON-serialisable at ingest, otherwise untouched.

Corrections never rewrite: a new record carries `supersedes: <record-id>`.

## Ledger semantics (`ledger.py`)

1. Read all logs, drop any record id named by a `supersedes` field.
2. Order by `(ts, actor, id)` for determinism across machines.
3. Baseline per location = its most recent snapshot at or before the query time; a snapshot **resets** that
   location's products rather than merging into them.
4. Apply changes after that snapshot; `movement` applies a negative delta at `from_location` and a positive
   one at `to_location`.

## Ownership and versioning

- `store.append()` refuses any `stream_id` other than the current user's — with the AAD binding above, that
  is the mechanism, and `docs/LAYOUT.md` documents which paths are read-only by convention.
- `sumac verify` re-opens every line of every log under its own AAD and reports lines that fail plus records
  whose `actor` disagrees with the owning file.
- `SCHEMA_VERSION` on every record; readers reject records from a newer schema with an "upgrade sumac" error.
- `.gitattributes`: `merge=union` on `data/**/*.jsonl` — correct for append-only streams and it keeps
  concurrent pushes from conflicting.

## CLI

`init`, `config show|add-location`, `add` (a change), `snapshot`, `status [location]`, `find <product>`,
`log`, `verify`. Rendering lives in `render.py`.

## Tooling

- `pyproject.toml` (uv, `uv.lock` committed): `pynacl`, `pydantic>=2`, `rich`, `typer`; dev group `ruff`,
  `ty`, `pytest`. PyNaCl for XChaCha20-Poly1305 — `cryptography` only ships the 12-byte-nonce variant.
- `.claude/settings.json`: `PostToolUse` hook on `Edit|Write` running `uv run ruff format` on edited `.py`.
- `.github/workflows/ci.yml`: `uv sync` → `ruff format --check` → `ruff check` → `ty check` → `pytest`.
- Docs: `README.md` (usage), `docs/FORMAT.md` (on-disk format + threat model), `docs/LAYOUT.md`
  (read-only vs mutable). Concise; no narration.

## Build order

1. Scaffolding: `pyproject.toml`, ruff/ty config, CI, Claude hook, doc skeletons.
2. `models.py` + `schemas.py` + tests.
3. `crypto.py`, `passphrase.py`, `paths.py`, `store.py` + tests.
4. `config.py` + `ledger.py` + tests.
5. `cli.py` + `render.py`, docs filled in.

## Verification

Tests: crypto round-trip; wrong passphrase fails at the verifier; a line moved between streams fails to
open; `append` rejects a foreign `stream_id`; ledger cases (snapshot reset, movement, supersede, unit
mismatch); config latest-revision-wins; and a leak test walking `data/` asserting every path component is a
fixed literal or a known username.

End-to-end, against a scratch data dir:

```
SUMAC_PASSPHRASE=test uv run sumac init
… add-location, add, snapshot, status, verify
grep -r "pantry\|fridge" data/     # must find nothing
uv run ruff format --check . && uv run ruff check . && uv run ty check && uv run pytest
```
