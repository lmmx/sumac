# 2026-09-11: Branch-Per-Writer Design — Draft Plan

**Status:** design discussion, no code written
**Author:** drafted with Claude, 2026-09-11
**Scope:** `docs/FORMAT.md`, `docs/LAYOUT.md`, `src/sumac/store.py`, git remote/branch layout

---

## 1. Problem

The current design (`docs/FORMAT.md`, `docs/LAYOUT.md`) puts every user's append-only log on
one shared branch: `data/log/<osuser>.jsonl` per user, `data/config.jsonl.enc` shared, both
declared `merge=union` in `.gitattributes` so concurrent appends never produce merge conflict
markers. That solves *content* conflicts. It does not solve *coordination*: pushing to a shared
branch means a non-fast-forward rejection whenever two people's local HEADs have diverged, even
though the underlying union-merge would be trivial. On a household shared server with multiple
residents this shows up immediately — one person's push, once it lands, forces everyone else to
fetch-and-merge before they can push, even though their changes touch a disjoint file.

The idea, arrived at from using `git worktree` for the first time: give each writer its own
branch, so a push can only ever be a fast-forward of that branch's own prior tip.

This entry keeps three concepts that the first draft blurred apart:

1. **authoritative storage** — the `writer/<id>` branches (§2, §3)
2. **aggregation/viewing** — combining writers' data for reading (§2)
3. **git history mechanics** — whether any synthetic ref (a `trunk`, a merge commit) exists at
   all (§2 — it doesn't)

## 2. Shape of the design

**Root commit.** `sumac init` creates one commit containing only `data/vault.json`. Every
`writer/<id>` branch descends from it, so Argon2id params and the verifier are reachable from
anywhere via git ancestry, never duplicated. `vault.json` is treated as immutable repository
metadata for the life of the repo — key rotation, if it's ever needed, is a separate event with
its own migration, not something this design accommodates by letting the root commit change.
A `writer/<id>` branch created long after the repo's first use still descends from this same root
commit, not from some later commit — nothing else is shared ancestry all writers need.

**One branch per writer instance, `writer/<id>`.** Not one branch per person. §3 spells out why:
a person with two machines gets two branches, e.g. `writer/louis-laptop` and
`writer/louis-desktop`, each independently fast-forward-only and each with exactly one thing that
ever pushes to it. This is what actually makes the "no non-fast-forward rejection" property hold
structurally — a per-*person* branch that two of that person's own machines both write to
reintroduces the original two-writers-one-branch problem, just with a smaller blast radius.
"Person" becomes a grouping applied at read time (§2's aggregation, or a UI label), never a git
concept.

**Two unsuffixed files per branch.** Today's paths, `data/log/<osuser>.jsonl` and a hypothetical
per-user `data/config/<id>.jsonl.enc`, would encode identity twice once branches exist — once in
the branch name, once in the filename. Each `writer/<id>` branch carries exactly:

- `data/log.jsonl.enc` — `stream_id = "log:<id>"` — changes and snapshots, as today
- `data/config.jsonl.enc` — `stream_id = "config:<id>"` — location definitions, as today

Config moves from one shared file to one per writer for the same reason the log already is one
per writer, not because it's conceptually writer-private data (§9 below addresses that directly).
A location definition is still visible and usable household-wide — that's an aggregation-layer
property, not a storage-layer one. Renaming a poorly-named location is a new record carrying
`supersedes: <record-id>`, exactly like a correction to the log today (`docs/LAYOUT.md`) — nobody
edits or deletes an existing config line, including the writer who created it.

**No trunk ref, no merge commit — a read-only aggregate view.** The branches are the authoritative
data; nothing needs to combine them into a single ref for that to be true. `sumac sync` fetches
the remote `writer/*` refs and updates local remote-tracking refs. It does not merge branches,
create merge commits, update a `trunk`, or modify any writer's branch — git synchronization stays
git synchronization. A separate concern — call it discovery/validation/aggregation, not
"sync" — is what `sumac verify`, reports, and queries actually do with the fetched refs:

1. **Discovery** — enumerate `writer/*` refs. Assumes nothing about their validity.
2. **Validation** — `sumac verify` reads and checks each branch independently (§4).
3. **Aggregation** — only verified branches' decrypted records feed a query or report, combined
   **in memory**, recomputed from scratch each time. An unverifiable branch is surfaced as an
   error, not silently dropped from the aggregate.

There's no ref for two writers to race over because no shared ref exists. Git's tree-level
"conflict-free by construction" claim from the first draft is true and worth keeping — writer
branches touch disjoint paths, so an octopus merge (if anyone ever wanted one) needs no
`merge=union` heuristic — but it's a claim about trees, not about data: two writers' config
branches can each define a location record that means the same real-world place under a different
name, or the same name for two different places. Git sees no conflict because the files don't
overlap; the aggregation layer is where that has to be noticed, if it's noticed at all. This
design does not solve that — it only removes the git-level coordination problem.

**`merge=union` is retired.** Once branches don't share files, `.gitattributes`' union-merge
declaration for `data/**/*.jsonl` has nothing left to apply to and should be removed, not left in
place as a mechanism nobody remembers the purpose of.

**Worktrees stay per-branch.** `git worktree add ../me writer/<id>` for writing. Reading/reporting
doesn't need a worktree at all — aggregation happens in memory across the fetched `writer/*` refs
directly. If a literal combined-filesystem view turns out to be convenient for some other tool
later, that's an explicitly disposable `git read-tree`-built scratch directory, never a ref
anything else depends on.

## 3. Identity: an identifier, not an authenticator

`writer/<id>` names a writer instance, and needs to be read as exactly that: a namespace label,
not a credential. `writer/louis-laptop` doesn't prove the pushing party is Louis's laptop —
whether that's true depends on what actually authenticates and authorizes pushes, which is a
separate concern this design has to name explicitly rather than let the branch name imply:

- **Logical ownership** — sumac's own convention: "this branch belongs to this writer instance."
  This is what `<id>` expresses.
- **Authentication** — whatever mechanism the remote uses to know who's pushing (SSH key, a
  hosted service's account, OS user on a bare repo over a shared filesystem).
- **Ref protection** — whether anything actually stops a push to `writer/bob` from someone other
  than Bob. This is deliberately *not* enforced by OS file permissions or git server-side hooks —
  the household's threat model (`docs/FORMAT.md` §"Threat model") already accepts that anyone
  holding the shared passphrase can write anywhere; ownership of a stream is a convention sumac
  makes *detectable* (`sumac verify`), not one it makes impossible to violate. Branch-per-writer
  doesn't change that stance, it just moves the same convention from "append to the right file"
  to "push to the right branch." Editing someone else's branch stays possible and occasionally
  legitimate (fixing another resident's mistake while they're away) — it's just not the ordinary
  path, and `sumac verify` is what surfaces it after the fact, same as today.

`<id>` is primarily a branch-name component, not a general-purpose identity string, and gets
slugged accordingly: lowercased, non-alphanumeric runs collapsed to a single hyphen, leading and
trailing hyphens trimmed (`Louis's Laptop` → `louis-s-laptop`). The `stream_id` (`log:<id>`,
`config:<id>`) and the AAD byte layout `sealedlog` builds from it (`docs/FORMAT.md` §"AAD
binding") are both derived from this same slug, not validated separately against their own rules
— one slugging step feeds all three, so there's a single place "is this identity valid" is
decided rather than three that could silently disagree.

**Choosing `<id>` at `sumac init-writer` time**, not read off the OS at every invocation. A sane
default assembled from environment facts, with a free-form override — the right formula depends
on the household, and that's a judgement call for the person setting it up, not a fixed
convention:

- `{getpass.getuser()}` — single-resident machine, one writer per person is unambiguous
- `{socket.gethostname()}` — fine when the hostname alone is already unique on the shared remote
- `{getpass.getuser()}-{socket.gethostname()}` — shared machine with multiple OS accounts, or the
  same OS username recurring across machines
- free text — a chosen nickname, anything the above don't cover

**Uniqueness is enforced by the push, not the preflight check.** `init-writer` checks the
candidate against the fetched remote branch list before creating anything locally, but that check
alone races: two machines can both see `writer/louis` absent, both create it, and only one push
can land. The authoritative uniqueness check is the remote refusing a non-fast-forward/non-empty
push to a `writer/<id>` ref that already has commits from a different init — the preflight check
is a courtesy that avoids the race in the common case, not the mechanism the design depends on.
On a rejected push, `init-writer` picks a different candidate and retries rather than assuming its
local branch is authoritative.

**Retirement, not deletion.** A `writer/<id>` branch that stops being actively written to (a
resident moves out, retires a machine) keeps its history — append-only applies to the branch's
existence, not just its file contents. `<id>` is never reused for a new writer once claimed, even
after the branch is inactive, so recreating the identifier can't graft new, unrelated history onto
an old identity.

## 4. What append-only still has to mean

**Fast-forward-only git history is necessary, not sufficient.** A fast-forward commit can still
*replace the content* of `data/log.jsonl.enc` wholesale — nothing about "B is a fast-forward of A"
implies B's file is A's file plus new lines appended. Owning `writer/<id>` outright means nothing
external stops the owner's own tooling from doing exactly that, the way a shared branch's other
pushers implicitly prevented it today by holding a divergent history the rewriter would have to
force past. Two mechanisms are both needed, doing different jobs:

- **Ref protection** (§3): reject non-fast-forward pushes to `writer/*` at the remote, so no
  identity's history can be rewritten by anyone, authenticated as that identity or not.
- **`sumac verify`**: establishes append-only at the *data* level, which git's ref rule doesn't.

**The append-only invariant, stated precisely:** for consecutive verified states of a branch, the
older state's decoded record sequence is an exact, order-preserving prefix of the newer state's —
every previously-verified record still decrypts to the same plaintext, in the same position, and
new records may only be added at the end. This is a claim about the *decoded* sequence, not raw
ciphertext bytes (re-sealing during migration, per §5, changes ciphertext without violating the
invariant) and not file-prefix-as-bytes (the sealed-line format is one line per record, so record-
sequence-prefix and line-sequence-prefix coincide here, but the invariant being verified is the
record sequence, stated that way so it doesn't quietly break if the line format ever changes).

**Where "last verified state" lives:** derived from git history, not persisted local state. A
persisted local checkpoint can be erased by a reinitialized or malicious local client, silently
resetting what "last verified" means. Instead, `sumac verify` walks a branch's commit history and
checks, for each consecutive pair of commits touching `data/log.jsonl.enc` or
`data/config.jsonl.enc`, that the invariant above holds between them — the property is re-derived
from git objects every run, the same way `sumac verify` already re-derives correctness from the
files on disk today rather than trusting a cached prior answer.

## 5. Migration from the current layout

No new repo, and no permanent `sumac migrate` subcommand — this runs once per household and never
again, so it doesn't belong in the CLI surface. It's a one-off script, run by someone holding the
passphrase against their existing data repo, thrown away afterward:

1. Decrypt everything the script needs to read: every `data/log/<id>.jsonl` and the shared
   `data/config.jsonl.enc`.
2. For each `<id>` found in `data/log/`, branch `writer/<id>` off the root commit (just
   `data/vault.json`, unchanged content — same blob, no re-encryption), and write that writer's
   decrypted log records to a fresh `data/log.jsonl.enc`. Partition the shared config file's
   records the same way, by whichever field identifies the writer that created each record — this
   assumes the legacy config records carry writer-level attribution at all; if some only carry an
   OS username that maps to more than one legacy writer instance, those records can't be
   partitioned automatically and need a manual assignment pass before the script runs, same
   ambiguity §3 already names for the log data itself. Both files re-sealed under the new
   `stream_id`s (`log:<id>`, `config:<id>`).
3. Verify by **decrypted record-set equality, not record counts and not ciphertext equality.**
   Re-sealing under a new `stream_id` deliberately changes ciphertext, so ciphertext comparison
   would always fail; a bare count match would miss a record silently relocated to the wrong
   writer or corrupted in transit. The check is: the legacy branch's full decrypted record
   sequence, partitioned by writer the same way the script partitioned it, equals each new
   branch's decrypted record sequence — same records, same order, same fields, only the container
   changed. Then run `sumac verify` (§4) on each new branch as its own independent check.
4. Keep the old branch around (don't delete it) until everyone's confirmed the new branches are
   good, then it's just history.

## 6. Open questions carried into implementation

- Legacy config records that lack per-writer attribution fine enough to partition automatically
  (§5, step 2) — whether that's rare enough to handle by hand per household, or common enough to
  need a documented manual-assignment step in the migration script itself.
- Whether `sumac verify`'s commit-by-commit history walk (§4) needs any bound on history length
  for a very long-lived branch, or whether re-deriving the invariant from full ancestry every run
  stays cheap enough not to matter.

---

## Current State

- Single shared branch, per-user append-only files, union-merge on `.jsonl` (`.gitattributes`,
  `docs/FORMAT.md` §"Versioning")
- Ownership enforced only at `store.append` by comparing `stream_id` against
  `getpass.getuser()` (`docs/LAYOUT.md`) — detectable-after-the-fact, not enforced at write time
- One shared mutable file, `data/config.jsonl.enc`, writable by any user (`docs/LAYOUT.md`)
- No branch-per-writer mechanism, no `sumac init-writer`, no `sumac sync` command anywhere in
  `src/sumac/`

## Missing

- `sumac init-writer` command: candidate identity generation, free-form override, preflight
  collision check against fetched `writer/*`, retry-on-rejected-push for the race case (§3)
- Identity slugging (lowercase, hyphenated alphanumeric) feeding the git ref, `stream_id`, and AAD
  byte layout from one canonical value, replacing ad hoc string validation in three places (§3)
- Branch-per-writer repo layout: immutable root commit, `writer/<id>` branches, fast-forward-only
  push enforcement (§2, §4) — who is *permitted* to push where remains the same convention-based,
  detectable-not-prevented posture the current design already takes (§3)
- `sumac sync` as pure git fetch/remote-tracking-ref update, with discovery/validation/aggregation
  as a separate layer built on top — no `trunk` ref, no merge commit, ever (§2)
- Unsuffixed per-branch `data/log.jsonl.enc` and `data/config.jsonl.enc`, replacing
  `data/log/<osuser>.jsonl` and the single shared `data/config.jsonl.enc` (§2)
- Removal of the `merge=union` `.gitattributes` declaration (§2)
- `sumac verify`'s commit-history-walk implementation of the precise append-only invariant in §4,
  replacing the current single-state check
- One-off migration script (not a shipped CLI command): per-writer branch seeding, config
  repartition by writer attribution, re-seal under new `stream_id`s, decrypted-record-set
  verification (§5)

## Divergence

- None yet — no code has changed to diverge from `docs/FORMAT.md`/`docs/LAYOUT.md`. This entry
  is the plan those two docs will need to be rewritten against once implemented.
