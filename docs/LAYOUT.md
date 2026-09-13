# Read-only vs mutable

Read-only-ness is a convention, not enforced by file permissions. `sumac verify` detects
violations after the fact; nothing prevents them at write time — you are on your own
`writer/<id>` branch, and nothing stops you writing on someone else's (docs/journal
2026-09-11-branch-per-user-design.md §5). A stray write there is a misattributed record, not a
corrupt one: it decrypts, it folds, and `sumac correct` cancels it from either branch, because
`supersedes` resolves across the aggregate rather than within one stream. `sumac verify` is what
surfaces it after the fact.

| Path | Mutability | Notes |
| --- | --- | --- |
| `src/sumac/` | Read-only for users | Pulled with the app; users don't edit it. |
| `pyproject.toml`, `uv.lock`, CI, hooks | Read-only for users | App-level tooling. |
| `data/vault.json` | Written once, by `sumac init` | KDF params and verifier; shared via the root commit every `writer/<id>` branch descends from — never rewritten after. |
| `data/config.jsonl.enc` | Mutable by its own writer, append-only | One per `writer/<id>` branch: that writer's location/product definitions. |
| `data/log.jsonl.enc` | Mutable by its own writer, append-only | One per `writer/<id>` branch: that writer's changes and snapshots. |

A writer appends their own changes, snapshots, and config to their own branch's
`log.jsonl.enc`/`config.jsonl.enc`, and never checks out or commits to another writer's branch
in ordinary use. Corrections are new records carrying `supersedes: <record-id>`; no one ever
rewrites or deletes a line in any log — theirs or anyone else's, since the format is append-only
end to end.

Run `sumac verify` after `sumac sync` to confirm every line in every writer's branch still
authenticates under its own stream, that no record's `actor` field disagrees with the branch it
lives on, and that each branch's history is append-only — every commit's decoded records are a
prefix of the next commit's, re-derived from git history rather than a stored checkpoint
(docs/journal 2026-09-11-branch-per-user-design.md §4).
