# Implementation plan: branch-per-writer

Design source of truth: `docs/journal/2026-09-11-branch-per-user-design.md` (cited below as §N).
Read §2, §3, §5 and §6 before writing code. This file says *how* to build it; the journal says
*why*, and the why is not repeated in code comments (keep docstrings terse and cite `§N`).

## 1. What changes, in one paragraph

Today a writer is an OS username: it names the file (`data/log/<osuser>.jsonl`), the stream
(`log:<osuser>`), and the `actor` on every record. After this change a writer is a git branch,
`writer/<id>`. Each branch carries exactly `data/log.jsonl.enc` (`log:<id>`) and
`data/config.jsonl.enc` (`config:<id>`); `data/vault.json` comes from the shared root commit.
Reading combines every writer's branch in memory; writing only ever touches the branch you are on,
and commits. `getpass.getuser()` stops being an identity anywhere in the codebase.

## 2. Decisions already taken (do not re-litigate)

- **Writer id resolution**, in order — this is the single rule, implemented once in
  `writer.current_id(data_dir)`:
  1. `SUMAC_WRITER_ID` set → that id, filesystem mode, single writer, no git calls.
  2. else `data_dir` inside a git repo whose HEAD is a branch named `writer/<id>` → git mode, id
     is `<id>`.
  3. else → raise `NotAWriterBranchError`. Never guess. (Rule 3 is what stops sumac reading the
     *app* repo's branch when a data dir happens to live inside some other git repo.)
- **No OS-user check anywhere** (§5). `store.append`'s `ForeignStreamError` guard is deleted, as
  is the exception class and its test.
- **`actor` is the writer id** (§5), so `actor`, the stream id, and the branch always agree.
- **Writes commit** (§2): one commit per command, message is a fixed literal plus a record count —
  never a product, location, quantity or free-text reason. `sumac: N records` is the format.
- **No pushing** from writing commands; `sumac sync` fetches only (§2). Pushing stays manual for
  now (§7).
- **Branch prefix stays `writer/`** (§7's open question resolved this way for implementation:
  discovery is one refspec glob and writer branches can't collide with `master`/`main`).
- **Aggregation reads refs, never checkouts** (§2). No sumac command switches, creates (except
  `init`/`init-writer`), or checks out a branch.

## 3. Target module map

New:

- `src/sumac/writer.py` — writer identity. Slugging (§3), stream-id construction, and
  `current_id`. The only place identity is decided.
- `src/sumac/gitrepo.py` — every git subprocess call in the codebase lives here, nowhere else.
- `src/sumac/sources.py` — enumerates the writers to read, and where their bytes come from.

Changed: `paths.py`, `store.py`, `config.py`, `ledger.py`, `decide.py`, `render.py`, `cli.py`,
`llm.py`, `errors.py`.

### `writer.py`

```python
WRITER_BRANCH_PREFIX = "writer/"
WRITER_ID_ENV = "SUMAC_WRITER_ID"

def slug(raw: str) -> str          # lowercase; non-alphanumeric runs -> one "-"; strip leading/trailing "-"
                                   # raise ValueError on empty result (§3)
def default_id() -> str            # slug(f"{getpass.getuser()}-{socket.gethostname()}")
def branch(writer_id: str) -> str  # f"writer/{writer_id}"
def id_from_branch(branch: str) -> str | None
def log_stream_id(writer_id: str) -> str     # f"log:{writer_id}"
def config_stream_id(writer_id: str) -> str  # f"config:{writer_id}"
def current_id(data_dir: Path) -> str        # the §2 resolution rule
```

`slug` is the one normaliser: the branch name, both stream ids, and therefore sealedlog's AAD all
derive from its output. Nothing validates an id a second time.

### `gitrepo.py`

Thin wrappers over `subprocess.run(["git", "-C", str(dir), ...], check=..., capture_output=True)`.
No other module may call git.

```python
def is_repo(path: Path) -> bool
def toplevel(path: Path) -> Path | None
def current_branch(path: Path) -> str | None          # None when detached
def rel_to_toplevel(path: Path) -> PurePosixPath      # data dir's path inside the repo
def list_writer_refs(path: Path) -> dict[str, str]    # {writer_id: ref}, local refs/heads/writer/*
                                                      # plus refs/remotes/*/writer/* (local wins)
def read_blob(path: Path, ref: str, rel: str) -> str  # "" when the path is absent at that ref
def init_repo(path: Path) -> None
def root_commit_with_vault(path: Path, rel_vault: str) -> str   # orphan commit holding only the vault
def create_branch_from(path: Path, branch: str, commit: str, *, checkout: bool) -> None
def commit_paths(path: Path, rels: list[str], message: str) -> None   # `git add` those paths, commit
def fetch_writers(path: Path, remote: str = "origin") -> None
def commits_touching(path: Path, ref: str, rels: list[str]) -> list[str]   # oldest-first, for §4
def root_commit(path: Path, ref: str = "HEAD") -> str  # git rev-list --max-parents=0
def blob_at(path: Path, commit: str, rel: str) -> str  # read_blob for a commit, for §4
```

Errors: a non-zero git exit raises `GitError` carrying the command and stderr. `sumac` shows it;
it never swallows it.

### `sources.py`

```python
@dataclass(frozen=True, slots=True)
class WriterSource:
    writer_id: str
    label: str                      # "writer/bob-linux" or the data dir path, for error text
    log_text: Callable[[], str]
    config_text: Callable[[], str]

def for_writing(data_dir: Path) -> str            # writer.current_id, re-exported for call sites
def writer_sources(data_dir: Path) -> list[WriterSource]
```

`writer_sources` returns, in git mode: the current writer read from the **worktree files** (so an
uncommitted append is visible to its own writer), plus one ref-backed source per other
`writer/<id>` ref. In filesystem mode: exactly one worktree source. Sort by `writer_id` so every
read is deterministic.

### `paths.py`

- Delete `current_user`, `validate_osuser`, `_SAFE_OSUSER_RE`, `log_dir`, `all_log_paths`.
- `LOG_FILENAME = "log.jsonl.enc"`; `log_path(data_dir) -> data_dir / LOG_FILENAME` (no id).
- Keep `vault_path`, `config_path`, `VAULT_FILENAME`, `CONFIG_FILENAME`.

### `store.py`

- `LineFailure.path: Path` → `LineFailure.source: str` (a ref-qualified label or a path string).
  Update the three formatting sites (`ledger.py`, `config.py`, `render.py`).
- `verify_lines(lines: Iterable[str], key, stream_id, source: str) -> tuple[list[dict], list[LineFailure]]`
  is the real implementation; `verify_stream(path, key, stream_id)` becomes a thin wrapper over it.
- `_path_for_stream`: `log:*` → `paths.log_path(data_dir)`, `config:*` → `paths.config_path(data_dir)`.
- `append`: delete the `ForeignStreamError` block; keep `seq` assignment for `log:` streams.
- `iter_all_logs` → iterate `sources.writer_sources`, yielding `(writer_id, obj)`.

### `config.py`, `ledger.py`, `decide.py`, `render.py`

- `config.add_location`/`add_product`/`retire_*` write to `writer.config_stream_id(actor)`.
- `config.build_config` folds **every** source's config stream. Latest-wins tie-break becomes
  `(ts, writer_id)` — without the second element two writers' same-`ts` records resolve by ref
  enumeration order.
- `ledger._load`, `verify_all`, `diagnose` iterate sources instead of `paths.all_log_paths`.
- `ledger.verify_all` compares each record's `actor` against `source.writer_id`; `VerifyResult`'s
  mismatch tuples carry the source label and writer id.
- `ledger._check_seq(writer_id, objs)`.
- `decide.py`'s three `f"log:{actor}"` (`:491,:546,:583`) → `writer.log_stream_id(actor)`;
  `Write("config", …)` (`:212`) → `writer.config_stream_id(actor)`.
- `render.print_verify`'s "owning user" wording → "writer".

### `cli.py`, `llm.py`

- Every `actor = paths.current_user()` → `actor = writer.current_id(data_dir)` (10 sites in
  `cli.py`, 2 in `llm.py`).
- `init` (§2): resolve id (`--writer`, else `writer.default_id()`); `git init -b writer/<id>` the
  repo when `data_dir`'s parent isn't already one — the `-b` is what keeps a `master`/`main` ref
  from ever existing (verified with git 2.39.5: the first commit lands on `writer/<id>` and
  `git rev-list --max-parents=0` finds it as the root); write `data/vault.json`; orphan root commit containing
  only the vault; create and check out `writer/<id>`; write a `.gitignore` holding
  `ask_queue.json`. Do not create `data/log/`. Print the branch it left you on.
- `init-writer` (new, §3): joins an existing repo — resolve id, refuse if `writer/<id>` already
  exists locally or on a fetched remote, branch from the repo's root commit
  (`git rev-list --max-parents=0`), check it out. It must work when HEAD is currently *another*
  writer's branch, because that is where `git clone` leaves a new machine (`origin/HEAD` follows
  the remote's), and it should say plainly which branch it moved you off and onto.
- `sync` (new, §2): `gitrepo.fetch_writers`; print the writer ids discovered. Fetch only.
- After a successful append, commit: one commit per command covering every record that command
  wrote, message `sumac: N records`. Put it in one `_commit_writes(data_dir, n)` helper used by
  `add`, `snapshot`, `correct`, every `config` subcommand, and `llm.py`'s accept path.

### `errors.py`

Delete `ForeignStreamError`. Add `GitError`, `NotAWriterBranchError`, `WriterBranchExistsError`.
Map them in the CLI's existing error handling the way the other `SumacError`s are.

## 4. Waves

Each wave is one subagent. Run in order; the suite is red until wave C.

**Wave A — core and reads.** `writer.py`, `gitrepo.py`, `sources.py`, `paths.py`, `store.py`,
`errors.py`, `config.py`, `ledger.py`, `decide.py`, `render.py`. New unit tests for `writer.slug`
and `sources.writer_sources` (both modes). Acceptance: `ruff check`, `ruff format --check`,
`ty check` clean; `uv run pytest tests/test_store.py -q` may fail only where it asserts the old
layout.

**Wave B — CLI and agent.** `cli.py`, `llm.py`: identity, `init`, `init-writer`, `sync`, commits.
Acceptance: `sumac init` in a temp dir produces a repo whose HEAD is `writer/<id>`, whose root
commit holds only `data/vault.json`, and `sumac add` after it leaves a clean worktree and one new
commit. Verify by hand in `/tmp`, not only by test.

**Wave C — tests, fixtures, evals.** `tests/conftest.py` (`osuser` fixture → `writer_id` fixture
setting `SUMAC_WRITER_ID`, plus a `git_data_dir` fixture that builds a real repo with a root commit
and a writer branch), every test that references the old layout, `tests/test_leak.py`'s
`ALLOWED_PATH_COMPONENTS`, `tests/fixtures/generate_golden_log.py` + regenerating
`tests/fixtures/golden_log/`, `evals/conftest.py` and `evals/fixtures.py`. Multi-writer behaviour
(aggregation, `verify`, config from two writers) gets tested through a real two-branch repo, not a
faked directory. Acceptance: `uv run pytest -q` green, 414 tests as the floor.

**Wave D — `sumac verify`'s history walk (§4).** `ledger.verify_history(data_dir, key)`: for each
writer ref, walk `gitrepo.commits_touching` oldest-first and check that each state's decoded record
sequence is an exact order-preserving prefix of the next (the §4 invariant, stated over decoded
records, not ciphertext bytes). Wire it into `sumac verify` and report a violating commit pair by
short SHA and writer id. Skip in filesystem mode. Tests build a repo with a legitimate append
chain and one with a rewritten line, and assert clean/violating respectively.

**Wave E — docs and migration.** `docs/FORMAT.md`, `docs/LAYOUT.md`, `README.md` rewritten against
the new layout; `scripts/migrate-branch-per-writer.py` implementing §6 with `--mapping` (JSON
`{legacy_osuser: writer_id}`), `--dry-run`, the modulo-`seq`-and-`actor` record equality check and
the fold-equality acceptance check; `tests/test_migration.py` builds a synthetic legacy repo (two
OS-user logs, one shared config), migrates it, and asserts identical holdings and anomalies.
Leave `docs/PLAN.md` alone — it is a historical snapshot.

## 5. Conventions

- Line length 100, `ruff` lint set `E,F,I,UP,B`, target py312.
- Docstrings terse; cite `docs/journal/2026-09-11-branch-per-user-design.md §N` rather than
  restating the reasoning.
- Typer bool options get their flag named explicitly (`typer.Option("--dry-run", …)`).
- No new dependency: git is called through `subprocess`, not a library.
- Never put inventory content in a commit message, a branch name, or a path (`docs/FORMAT.md`
  threat model).

## 6. Verification

```sh
~/.local/bin/uv run --frozen ruff format src tests evals scripts
~/.local/bin/uv run --frozen ruff check .
~/.local/bin/uv run --frozen ty check     # pre-existing diagnostics in tests/ are the baseline
~/.local/bin/uv run --frozen pytest -q    # 414 passing before this work starts
```

`--frozen` keeps `uv run` from rewriting `uv.lock`. Do not run `ruff format .` across the repo
root: it reformats a python code block inside
`docs/journal/2026-09-06-ask-latency-round-two.md`, which is unrelated churn — revert that file
with `git checkout --` if it shows up modified.

## 7. Environment notes for this machine

- `uv` is at `~/.local/bin/uv` and is **not** on the tool shell's `PATH` — invoke it as
  `~/.local/bin/uv run …`.
- A git identity is configured only locally in `/workspace/.git/config`, so a repo created under
  `/tmp` has none and `git commit` there fails with "Author identity unknown". Test fixtures and
  manual checks must set `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME`,
  `GIT_COMMITTER_EMAIL`.
- `gitrepo` passes `-c commit.gpgsign=false` on commits, so a household member with commit signing
  enabled isn't prompted once per append.
- Use the scratchpad dir for manual checks, not `/tmp` directly:
  `/tmp/claude-1000/-workspace/a086183f-064e-49af-a470-83f893feed17/scratchpad`.
