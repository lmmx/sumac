# 2026-09-11: Branch-Per-Writer Design — Draft Plan

**Status:** design discussion, no code written — implemented 2026-09-12, see
`docs/journal/2026-09-12-branch-per-writer-implemented.md` and `docs/plans/branch-per-writer.md`
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

**Root commit.** `sumac init` creates one commit containing only `data/vault.json` and a
`.gitignore` for `ask`'s per-machine retry cache (`data/ask_queue.json`, §6's closing note). Every
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

**Writes commit, and commit messages say nothing.** An append left uncommitted in the worktree is
invisible to every other writer, is what a later `git switch` refuses to step over (§5), and is
not a state §4's commit-to-commit invariant can check — so a command that appends commits what it
appended. One commit per command, never a half-applied intermediate state: a multi-record command
(a `cmd_id`'s `Counted` plus the event it precedes, an accepted multi-change `ask` plan) is one
commit. The message is a fixed literal plus a record count, because the threat model
(`docs/FORMAT.md`) accepts leaking record count and commit timestamps and nothing more — "consumed
1 jar of jam" as a subject line would leak the lot. Commit authorship is whatever git identity the
machine already has, an OS-username-level leak the threat model already accepts.

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
ciphertext bytes (re-sealing during migration, per §6, changes ciphertext without violating the
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

## 5. The branch is the identity

**Decided: `<id>` comes from the checked-out branch and from nowhere else.** No command compares a
stream, a record, or a branch against `getpass.getuser()`. `store.append`'s existing comparison
(`store.py:53-57`) is deleted rather than relocated — it is what mints a second log stream when one
checkout is used from a container whose OS user is `node`, and once a branch has one writer there
is nothing left for it to protect. `actor` comes from the same source, so a record's attribution,
the stream it is sealed under, and the branch it lives on are one value with one origin, identical
whether the command ran on the host or in a container.

**Writing on another writer's branch is trusted not to happen, and stays benign if it does.**
Standing on `writer/bob` and running `sumac add` produces a well-formed record, sealed under
`log:bob` and attributed to `bob` — misattributed, not corrupt. It decrypts, it folds, and
`sumac correct` cancels it like any other mistaken record, from either writer's branch, because
`supersedes` resolves across the aggregate rather than within one stream (`ledger.py:96-97`,
`decide.py:570-575`). Nothing becomes unopenable and no history needs rewriting, which is what
makes trust an adequate mechanism here.

**Nothing in ordinary use puts a writer on another writer's branch.** `sumac init` leaves the repo
checked out on the writer's own branch, no command switches, creates, or checks out a branch
afterwards, and reading other writers goes through their refs rather than the worktree (§2's
aggregation) — no operation sumac offers is a reason to stand anywhere else. A fresh clone is the
one exception: `git clone` follows `origin/HEAD` and lands on whichever writer's branch the remote
names, so `sumac init-writer` is the first command a new machine runs, and a write before it
appends to that writer's stream. Git covers the remaining gap incidentally: `git switch` refuses while `data/log.jsonl.enc` carries uncommitted
changes, since the checkout would overwrite them, so a pending append blocks the switch and a
committed one leaves nothing to carry across.

**Not pursued**, recorded so it isn't re-derived: an untracked per-worktree identity file checked
against HEAD before each write; any comparison of the branch against the OS user; fetching other
writers into a ref namespace outside `refs/heads` to make them un-checkout-able; `checkout.guess =
false` and a single-ref push refspec installed as defaults. Each defends against a mistake whose
cost is one correction record.

## 6. Migrating the household's existing data repo

What there is to migrate: one branch carrying `data/vault.json`, one shared `data/config.jsonl.enc`
sealed under `"config"`, and one `data/log/<osuser>.jsonl` per OS username that ever appended —
including the duplicate `node` stream a container run created, the case §5 removes. A one-off
script run once by someone holding the passphrase, not a shipped `sumac migrate`:

1. **Name the writers, and map legacy streams onto them.** The mapping is many-to-one and written
   by hand for the household: `{alice: alice-mac, node: alice-mac, bob: bob-linux}`. It is the one
   input no script can derive, and it is what folds the accidental container stream back into the
   writer it always belonged to.
2. **Build the shared root.** An orphan commit (`git switch --orphan`) containing only
   `data/vault.json`, byte-identical to the legacy branch's copy — no re-encryption, since every
   writer already unlocks with those KDF params and that verifier. Every `writer/<id>` branches
   from this commit (§2).
3. **Per writer, seal one log stream and one config stream.** Decrypt every legacy log the mapping
   assigns to this writer, interleave the records by `(ts, id)`, renumber `seq` contiguously from 0
   across the merged sequence, rewrite `actor` to the writer id, and re-seal each record under
   `log:<id>`. Partition the shared config file by each record's `actor` (`config.py:29`,
   `config.py:60`) through the same mapping and re-seal those under `config:<id>`. Commit both
   files to `writer/<id>`.
4. **Two fields change deliberately, so equality is checked modulo those two.** Merging two legacy
   streams collides their per-stream `seq` values (each starts at 0), and an OS username is not a
   writer id. The check is: the legacy repo's full decrypted record sequence, partitioned by the
   mapping, equals each new branch's decrypted record sequence field-for-field except `seq` and
   `actor`, with `seq` contiguous from 0 on each branch. Ciphertext equality can't be used at all —
   re-sealing under a new `stream_id` changes every byte — and a count match would miss a record
   sent to the wrong writer.
5. **Fold equality is the acceptance test.** `build_inventory` over the legacy repo and over the
   aggregate of the new branches produce identical holdings, and identical anomalies except the
   `seq_*` ones — those legitimately differ, since step 3 renumbers `seq` and merging two streams
   that each start at 0 is exactly the duplicate the renumbering removes. Records sort by
   `(ts, actor, id)` (`ledger.py:98`), so rewriting `actor` can reorder records sharing an exact
   `ts` — the holdings comparison is what catches that if it happened. Then `sumac verify` (§4) on
   each new branch as its own independent check.
6. **Keep the legacy branch** until every writer has confirmed their own branch reads correctly.
   After that it is history: reachable, in no writer's ancestry, never written again.

The repo also loses `.gitattributes`' `merge=union` line (§2) and gains an ignore for
`data/ask_queue.json` — `ask`'s retry cache is per-machine convenience state (`queue.py`), not
vault data, and one copy per writer branch is one copy too many.

## 7. Open questions carried into implementation

- Legacy config records attributed to an OS username that two different people used (§6, step 1) —
  whether that's rare enough to resolve by hand per household, or needs a per-record assignment
  pass in the migration script.
- Whether `sumac verify`'s commit-by-commit history walk (§4) needs a bound on history length for
  a long-lived branch, or whether re-deriving the invariant from full ancestry every run stays
  cheap enough not to matter.
- Whether the branch is `writer/<id>` or a bare `<id>` — the prefix is what makes discovery a
  single refspec glob and keeps writer branches from colliding with `master`/`main`, at the cost
  of every branch name carrying five extra characters.
- Whether a writing command pushes, or pushing stays manual alongside `sumac sync`'s fetch (§2) —
  a command that commits but doesn't push leaves the household's data one step from shared.
- Whether the OS username is worth recording as descriptive record metadata now that it is not an
  identity (§5) — "who physically typed this" is a different question from "whose stream is this",
  and only the second has a home in the record today.
- Whether a data dir outside a git repo stays supported at all: every test, every eval fixture and
  `tests/fixtures/golden_log/` is a plain directory today. Resolved for implementation as a
  single-writer fallback whose id comes from `SUMAC_WRITER_ID` — the multi-writer aggregate is
  reachable only through git refs.

---

## Current State

- `sumac init` writes `data/vault.json` and creates `data/log/` (`cli.py:131-142`) — no git repo,
  branch, or commit is created by any sumac command
- No module in `src/sumac/` invokes git: no `subprocess` call and no git library import anywhere
  under `src/`
- `paths.current_user()` returns `getpass.getuser()` (`paths.py:21-22`) and is the only identity
  source in the codebase, called at `cli.py:176,194,210,222,258,277,303,335,356,955` and
  `llm.py:1282,1720` to fill each record's `actor`
- `paths.log_path(data_dir, osuser)` returns `data/log/<osuser>.jsonl` after
  `validate_osuser` checks the name against `_SAFE_OSUSER_RE` (`paths.py:20,25-29,44-45`), and
  `paths.all_log_paths` globs that directory for `*.jsonl` (`paths.py:48-51`)
- The per-user log files carry no `.enc` suffix while `data/config.jsonl.enc` does
  (`paths.py:15-17`), a spelling `docs/FORMAT.md` §"Layout" documents as-is
- `store.append` refuses a `log:<osuser>` stream whose suffix differs from `paths.current_user()`,
  raising `ForeignStreamError` (`store.py:53-57`) — the check that produces a second log file when
  one checkout is used from a container with a different OS username
- `store.CONFIG_STREAM_ID` is the literal `"config"` (`store.py:23`), one stream shared by every
  writer, written at `config.py:38,71` and read at `config.py:112`, `ledger.py:536`
- `store._path_for_stream` maps a stream id to a path through its `<osuser>` suffix
  (`store.py:26-31`), so the filename is what distinguishes one writer's log from another's
- Every read enumerates log files from one filesystem directory and derives the stream id from
  each filename stem: `ledger._load` (`ledger.py:79-81`), `ledger.verify_all` (`ledger.py:539-542`),
  `ledger.diagnose` (`ledger.py:564-565`), `store.iter_all_logs` (`store.py:72-76`)
- `config.build_config` folds exactly one config stream read from one path (`config.py:112`), with
  latest-revision-wins resolved on `ts` alone and no tiebreak (`config.py:127-135`)
- `ledger.verify_all` reports `actor_mismatches` by comparing each record's `actor` against the log
  filename stem (`ledger.py:539-547`), rendered as "owning user" by `render.print_verify`
  (`render.py:534-537`)
- `ledger._check_seq` attributes `seq` gaps and duplicates to a filename stem (`ledger.py:38-59`,
  called at `ledger.py:82`)
- `store.LineFailure` identifies a failure by filesystem `Path` and line number (`store.py:78-82`),
  formatted into anomaly text at `ledger.py:83` and `config.py:115`
- `supersedes` resolves across the union of every writer's log rather than within one
  (`ledger.py:96-97`), and `decide_correct` validates its target against that same cross-log view
  (`decide.py:570-575`) — a correction to another writer's record needs no write to their stream
- `_fold_into` drops a repeated record id and reports `duplicate_record` (`ledger.py:370-379`), so
  the same record reaching the fold twice double-counts nothing
- `.gitattributes` declares `data/**/*.jsonl merge=union` in its single line
- `tests/conftest.py`'s `osuser` fixture monkeypatches `getpass.getuser` to `"alice"`
  (`conftest.py:36-39`); `data_dir` is a plain directory under `tmp_path` (`conftest.py:41-44`)
- `tests/fixtures/golden_log/` holds `config.jsonl.enc` and `log/alice.jsonl` sealed under
  `"config"` and `"log:alice"`, folded by `test_model_properties.py:34-39`
- `evals/conftest.py` pins `getpass.getuser` to `eval_fixtures.EVAL_OSUSER`
  (`evals/conftest.py:165`) and guards `store.append` against writing outside the eval root
  (`evals/conftest.py:178-183`)
- 414 tests pass at `f232559` (`uv run pytest -q`, 12s)

## Missing

### Identity

- No function turns a free-form string into a writer `<id>` — `paths.validate_osuser`
  (`paths.py:25-29`) rejects an unsafe OS username rather than normalising, and nothing derives the
  branch name, `log:<id>` and `config:<id>` from one canonical slug (§3)
- No code reads the checked-out branch, and no error type covers a data dir outside a git repo or a
  detached HEAD (`errors.py` defines nine `SumacError` subclasses, none about git)

### Paths and stream ids

- No path form names the single per-branch `data/log.jsonl.enc` of §2 — `paths.log_path` requires
  an OS username (`paths.py:44-45`) and `paths.all_log_paths` (`paths.py:48-51`) exists only to
  enumerate the per-user directory §2 removes
- No per-writer config stream id exists: `store.CONFIG_STREAM_ID` is one shared literal
  (`store.py:23`), and `config.add_location`/`add_product` take no writer id to build `config:<id>`
  from (`config.py:25,38,56,71`)

### Writes

- No git write path exists: every write ends at `store.append` (`cli.py:321,341,365`,
  `llm.py:1728`) with the file left uncommitted, so §2's one-commit-per-command and its
  count-only commit message have no implementation
- No command creates the repo shape §2 describes — `sumac init` (`cli.py:131-142`) makes neither
  the root commit holding only `data/vault.json` nor the `writer/<id>` branch it leaves checked out
- No `sumac init-writer` exists for joining an existing household repo from a second machine (§3)

### Reads across writers

- No read path resolves another writer's data from a git ref; all four enumeration sites read one
  filesystem directory (`ledger.py:79-81,539-542,564-565`, `store.py:72-76`)
- `store.verify_stream` takes a `Path` and reports failures as `Path:lineno` (`store.py:78-82,97`),
  with all three formatting sites assuming a filesystem path (`ledger.py:83`, `config.py:115`,
  `render.py:533`) — a stream read out of `writer/bob`'s ref has no path to name
- No code combines several writers' config records into one registry, and `config.build_config`'s
  latest-wins comparison has no tiebreak for two writers' records sharing a `ts`
  (`config.py:127-135`)
- No `sumac sync` command exists to fetch `writer/*` refs (`cli.py` defines 20 commands, none
  touching a remote)
- `ledger.verify_all`'s actor check and `ledger._check_seq`'s attribution are both keyed on a
  filename stem (`ledger.py:545-547`, `ledger.py:38-59`), with no writer-id equivalent

### Verify (§4)

- Nothing walks a branch's commits: `verify_all` checks one state of the files
  (`ledger.py:532-550`), so §4's consecutive-states append-only invariant is unimplemented

### Tests and fixtures

- No fixture builds a git repo, a root commit, or a writer branch (`conftest.py` defines four
  fixtures, all filesystem-only)
- `tests/test_store.py` asserts the OS-user guard and the per-user filesystem layout directly
  (`test_store.py:52-54,57-63,73-80,110-121`), and `tests/test_leak.py`'s
  `ALLOWED_PATH_COMPONENTS` lists `log` and `alice.jsonl` (`test_leak.py:52-58`)
- `tests/fixtures/golden_log/`'s lines are sealed under `"config"` and `"log:alice"`, so none of
  them authenticate under §2's stream ids — the corpus needs regenerating through
  `generate_golden_log.py` and `test_model_properties.py`'s expected holdings re-verifying
- `evals/fixtures.py` and `evals/conftest.py` build an inventory in a plain data dir under a pinned
  OS username (`evals/conftest.py:165,195`), with no writer id anywhere

### Docs

- `docs/FORMAT.md` §"Layout", §"AAD binding" and §"Versioning" describe `log/<osuser>.jsonl`, the
  `"config"` stream id, and `merge=union`; `docs/LAYOUT.md`'s table describes
  `data/config.jsonl.enc` as "Mutable by any user" and `data/log/<osuser>.jsonl` as "Mutable only
  by `<osuser>`"; `docs/PLAN.md` §"On-disk layout" documents the same tree — none describes a branch
- `README.md` §"Usage" shows `sumac init` as the whole of setup and documents no branch, remote,
  or sync step

### Migration

- No script implements §6 (`scripts/` holds ten shell scripts, all for model builds, benchmarks and
  eval runs)

## Divergence

- None yet — no code has changed to diverge from `docs/FORMAT.md`/`docs/LAYOUT.md`. This entry is
  the plan those two docs will need to be rewritten against once implemented.
