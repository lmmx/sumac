# Read-only vs mutable

Read-only-ness is a convention, not enforced by file permissions. `sumac verify` detects
violations after the fact; nothing prevents them at write time except `store.append`'s check
that a `log:<osuser>` stream matches the current OS user.

| Path | Mutability | Notes |
| --- | --- | --- |
| `src/sumac/` | Read-only for users | Pulled with the app; users don't edit it. |
| `pyproject.toml`, `uv.lock`, CI, hooks | Read-only for users | App-level tooling. |
| `data/vault.json` | Written once, by `sumac init` | KDF params and verifier; not append-only. |
| `data/config.jsonl.enc` | Mutable by any user | Shared location registry; append-only. |
| `data/log/<osuser>.jsonl` | Mutable only by `<osuser>` | Everyone else: read-only. |

A user appends their own changes and snapshots to `data/log/<their-username>.jsonl` and never
writes into another user's log. Corrections are new records carrying `supersedes: <record-id>`;
no one ever rewrites or deletes a line in a log that isn't theirs — or, for that matter, in
their own, since the format is append-only end to end.

Run `sumac verify` after pulling to confirm every line in every log still authenticates under
its own stream, and that no record's `actor` field disagrees with the file it lives in.
