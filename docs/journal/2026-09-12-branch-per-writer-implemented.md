# 2026-09-12: Branch-Per-Writer — Implemented

**Status:** implemented; `uv run pytest -q` is green
**Scope:** `src/sumac/**`, `tests/**`, `evals/**`, `docs/FORMAT.md`, `docs/LAYOUT.md`, `README.md`,
`.gitattributes`, `scripts/migrate-branch-per-writer.py`
**Design:** `docs/journal/2026-09-11-branch-per-user-design.md` (cited as §N below);
implementation plan: `docs/plans/branch-per-writer.md`

---

## Current State

### Identity

- `writer.current_id` resolves a writer id from `SUMAC_WRITER_ID` if set, else from a checked-out
  `writer/<id>` branch, else raises `NotAWriterBranchError` (`writer.py:51-65`) — the two modes
  the rest of the code calls "filesystem mode" and "git mode"
- `writer.slug` lowercases, collapses non-alphanumeric runs to one `-`, strips leading and trailing
  `-`, and raises on an empty result (`writer.py:21-27`); `writer.branch`, `log_stream_id` and
  `config_stream_id` build the branch name and both stream ids from that one value
  (`writer.py:33-48`), so the git ref, the stream ids, and `sealedlog`'s AAD derive from a single
  slug (§3)
- `writer.default_id` is `slug(f"{getpass.getuser()}-{socket.gethostname()}")` (`writer.py:29-30`),
  used when `sumac init`/`init-writer` get no `--writer`
- No module compares any identity against `getpass.getuser()`; `paths.current_user`,
  `paths.validate_osuser` and `errors.ForeignStreamError` no longer exist

### Git access

- `gitrepo.py` holds every git invocation in the codebase — 15 functions over `subprocess.run`,
  raising `GitError` with the command and stderr on a non-zero exit (`gitrepo.py:15-21`); commits
  pass `-c commit.gpgsign=false` (`gitrepo.py:13`)
- `gitrepo.list_writer_refs` returns `{writer_id: ref}` from `refs/heads/writer/*` and
  `refs/remotes/*/writer/*`, local winning a collision (`gitrepo.py:50-69`)
- `gitrepo.read_blobs_batch` reads every `<commit>:<path>` spec through one `git cat-file --batch`
  process, parsing the batch protocol by byte offset and mapping missing specs to `""`
  (`gitrepo.py:113-138`) — one subprocess per writer per stream for a history walk, not one per
  commit
- No git library is a dependency; `pyproject.toml` is unchanged

### Storage

- `paths.py` is three functions over fixed literals — `vault_path`, `config_path`,
  `log_path(data_dir)` — with `LOG_FILENAME = "log.jsonl.enc"` (`paths.py:11-25`)
- `store._path_for_stream` maps a `config:` stream to `data/config.jsonl.enc` and a `log:` stream to
  `data/log.jsonl.enc` (`store.py:22-27`), so the stream's writer id no longer picks the path
- `store.append` assigns `seq` for `log:` streams and writes the sealed line; nothing else
  (`store.py:48-57`)
- `store.commit_records(data_dir, n)` commits the data dir with the message `sumac: N record(s)`,
  and returns without doing anything in filesystem mode or outside a repo (`store.py:60-71`)
- `store.verify_lines(lines, key, stream_id, source)` is the decode-and-collect-failures
  implementation and `verify_stream(path, …)` a wrapper over it (`store.py:104-133`);
  `store.LineFailure.source` is a string, so a failure can name `writer/bob-linux` as readily as a
  path (`store.py:88-91`)
- `sources.writer_sources` returns the current writer read from the worktree files plus one
  `git cat-file`-backed source per other `writer/*` ref, sorted by writer id; in filesystem mode it
  returns exactly one worktree source (`sources.py:53-63`)

### Reads

- `ledger._load`, `ledger.verify_all`, `ledger.diagnose` and `store.iter_all_logs` each iterate
  `sources.writer_sources` and derive the stream id from the source's writer id
  (`ledger.py:76-99`, `ledger.py:627-656`, `ledger.py:664-676`, `store.py:78-85`)
- `config._load_config_records` folds every source's `config:<id>` stream into one registry, with
  latest-revision-wins broken by `(ts, writer_id)` (`config.py:104-160`)
- `ledger.verify_all` reports an actor mismatch as `(source_label, actor, writer_id)` strings by
  comparing each record's `actor` against its source's writer id (`ledger.py:627-656`)
- `ledger._check_seq` attributes `seq` gaps and duplicates to a writer id (`ledger.py:38-59`)

### `sumac verify`'s history walk

- `ledger.verify_history` walks every writer ref's commits touching the log and config paths,
  oldest first, and reports a `HistoryViolation` wherever one state's decoded record sequence is
  not an order-preserving prefix of the next (`ledger.py:528-617`) — the §4 invariant over decoded
  records, so a re-seal that changes ciphertext and nothing else passes
- The checked-out writer's last committed state is additionally checked against its worktree state,
  reported with `to_commit="worktree"` (`ledger.py:597-617`)
- `verify_all` folds the violations into `VerifyResult.ok` (`ledger.py:620-656`) and
  `render.print_verify` prints each one; `sumac verify` exits 1 on any of them (`cli.py:515-523`)
- `verify_history` returns `[]` in filesystem mode, where there is no history to walk

### CLI

- `sumac init` slugs `--writer` (default `writer.default_id()`), writes `data/vault.json`,
  `git init -b writer/<id>`s the data dir's parent when it isn't already a repo, appends
  `data/ask_queue.json` to a `.gitignore` without clobbering an existing one, commits that as the
  root commit, and prints the branch it left you on (`cli.py:134-175`)
- `sumac init-writer` refuses an id that already has a ref, branches from
  `gitrepo.root_commit` of an existing writer ref, checks it out, and names the branch it moved off
  (`cli.py:177-205`) — it works from another writer's branch, which is where `git clone` leaves a
  new machine
- `sumac sync` fetches `refs/heads/writer/*` from `origin` and prints the writer ids found; it does
  not merge, push, or create any other ref (`cli.py:207-217`)
- Every writing command calls `store.commit_records` once with the number of records it wrote —
  `add`, `snapshot`, `correct`, and each `config` subcommand, with `add-array`/`add-grid`'s N
  locations landing in one commit (`cli.py:245-442`)
- `llm.py`'s accept path commits once for the whole accepted plan (`llm.py:1732`)
- `actor` comes from `writer.current_id` at all 12 former `paths.current_user()` call sites

### Tests

- 453 tests pass (`uv run pytest -q`, 20s), against 414 before the change
- `tests/conftest.py` provides `writer_id` (sets `SUMAC_WRITER_ID=alice-mac`), `git_env` (the four
  `GIT_*` identity variables a repo under `tmp_path` lacks), `git_data_dir` (a real repo on
  `writer/alice-mac` with a root commit), and `seed_writer` (builds another writer's branch and
  restores the original)
- New: `tests/test_writer.py` (10), `tests/test_sources.py` (5), `tests/test_verify_history.py` (6),
  `tests/test_migration.py` (7)
- `tests/test_cli.py` drives the CLI in git mode through `init --writer alice-mac` (103 tests) and
  covers the branch HEAD, the root commit's contents, `--writer` slugging, worktree cleanliness and
  commit subjects, `.gitignore` preservation, `init-writer` from a clone, and `sync`'s output
- `tests/test_leak.py` asserts no ref name, commit message, or tree path in the repo contains a
  location or product name, alongside the existing data-dir walk
- `tests/fixtures/golden_log/` is a flat `config.jsonl.enc` + `log.jsonl.enc` sealed under
  `config:alice-mac` / `log:alice-mac`; its fold is unchanged — no anomalies, `fridge: milk 6 l`,
  `pantry: {}`, 14 live records, 15 total
- `evals/conftest.py` sets `SUMAC_WRITER_ID` instead of pinning `getpass.getuser`, and
  `evals/fixtures.py` exports `EVAL_WRITER_ID`
- `tests/test_cli.py`'s `test_another_user_cannot_write_into_alices_log` is deleted: its subject was
  the OS-user guard §5 removes

### Migration

- `scripts/migrate-branch-per-writer.py` (352 lines) takes `--repo`, `--data-dir`, one of
  `--mapping`/`--mapping-file`, and `--dry-run`, and reads the legacy streams by passing the legacy
  stream ids (`log:<osuser>`, `config`) to `store.verify_lines`
- A legacy OS username absent from the mapping raises, naming it
- The mapping is many-to-one: several legacy streams merge into one writer, interleaved by
  `(ts, id)`, `seq` renumbered contiguously from 0, `actor` rewritten to the writer id, re-sealed
  under `log:<id>`; the shared config is partitioned by each record's `actor` and re-sealed under
  `config:<id>`
- It prints three check results — record equality modulo `seq`/`actor`, holdings equality, anomaly
  equality excluding `seq_*` — and writes no ref unless all three pass
- Branches are built with `hash-object`/`mktree`/`commit-tree`/`update-ref`, so no checkout happens
  and the legacy branch and working tree are never touched
- The new root tree carries `data/vault.json` byte-identical to the legacy copy plus the same
  `.gitignore` `sumac init` writes; `merge=union` needs no removal step because every new tree is
  built from scratch

### Docs

- `docs/FORMAT.md` documents the per-branch layout, `config:<id>`/`log:<id>` AAD binding, and an
  accepted-leakage list that now includes branch names and `sumac: N records` commit messages, with
  OS usernames leaking through git commit authorship rather than any path
- `docs/LAYOUT.md`'s table covers `data/vault.json` (written once, shared via the root commit) and
  the two per-branch files, and states what happens on a stray write to another writer's branch
- `README.md` §"Usage" shows `init`, `init-writer` and `sync`, and §"The branch-per-writer model"
  states that reads combine every writer, writes commit, and pushing is manual
- `.gitattributes` is deleted — its only line was the retired `merge=union` rule

## Missing

- No command pushes: `sumac sync` fetches only, and `README.md` documents `git push` as manual
  (§7's open question)
- No sumac command configures the remote's ref protection, so the non-fast-forward rejection §3 and
  §4 rely on is a remote-side setup step no code performs
- `sumac sync` fetches `origin` and takes no remote argument (`cli.py:207-217`)
- `sumac verify` re-walks a branch's full ancestry every run, with no bound on history length
  (§7's second open question)
- Nothing records the OS username as descriptive record metadata now that `actor` is the writer id
  (§7's fifth open question)
- No command retires a writer branch or blocks reuse of a retired `<id>` (§3's "retirement, not
  deletion" is a convention with no code)

## Divergence

- None. `docs/FORMAT.md`, `docs/LAYOUT.md` and `README.md` describe the implemented behaviour;
  `docs/PLAN.md` is left as the historical pre-implementation snapshot it has always been and
  documents the old `log/<osuser>.jsonl` layout as such.
