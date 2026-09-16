# 2026-09-14: Wiring `remotectrl` into sumac

**Status:** implemented
**Depends on:** `docs/journal/2026-09-13-remotectrl-design.md` (as amended 2026-09-14),
`docs/journal/2026-09-13-sumac-sources-design.md`
**Package version pinned against:** `remotectrl` `/mnt/remotectrl` @ `36d7cba` (requires-python
loosened to `>=3.12` same session — see that repo's own commit, uv init's unexamined default,
nothing in the code needs newer)

---

## 1. Pre-existing state check before writing anything

Re-read both docs end to end against what `remotectrl` actually shipped as. Both were already
current — I had amended §5/§6/§7 of the `remotectrl` design doc earlier the same night (generic
refspecs, `-> str` return, try-all-remotes postflight) while resolving remotectrl's own open
questions, before this integration pass started. Nothing further was stale.

Two things surfaced that needed a decision, not a docs staleness fix:

- **`src/sumac/sources.py` already exists**, for an unrelated concept (aggregating other
  writers' log/config text for reads — 2026-09-11 branch-per-user design). The `sources` design
  doc's *command* name doesn't collide with it (Typer command name vs. Python module are
  different namespaces) but importing a new module also named `sources` inside `cli.py`, which
  already does `from sumac import (..., sources, ...)`, would shadow that import. New module
  named `remote_sync.py` instead; the CLI command is still literally `sumac sources` via
  `@app.command(name="sources")` on a function named `sources_cmd` (same pattern `log_cmd`
  already uses to avoid shadowing the builtin `log`).
- **`remotectrl.api.run()` (the one-call preflight→op→postflight wrapper) discards preflight's
  warnings** — it only returns the commit hash. Rather than change remotectrl's public API/
  README under this integration's time pressure, sumac calls the granular functions directly
  (`remotectrl.config.resolve`, `.preflight.run_preflight`, `.onecommit.run_op`,
  `.postflight.run_postflight`) from `remote_sync.py`, surfacing warnings through `render`
  itself. This is exactly design doc §1's stated division of labor ("sumac supplies the domain
  mapping ... and calls the wrapper"), one level finer than `api.run`. remotectrl's own code and
  tests are untouched by this choice.

## 2. Integration point: `store.commit_records`, not each CLI command

`store.commit_records` is the single choke point every write command (`add`, `snapshot`,
`correct`, every `config` subcommand) already calls to produce its one git commit — confirmed by
grepping every call site in `cli.py`. Wrapping preflight/postflight there, once, satisfies "every
write command runs preflight/postflight around it" without touching any of those command
functions individually — the Kolmogorov-minimal shape design doc §2 asked for on the
`remotectrl` side applies just as much to where sumac calls into it.

`commit_records`'s existing skip conditions (`SUMAC_WRITER_ID` env override, filesystem mode with
no git repo) are preserved unchanged — both mean there is no git commit at all, so there is
nothing for `remotectrl` to wrap.

## 3. `remote_sync.py`

- `resolve_remotes(repo_root) -> dict[str, RemoteType]`: thin pass-through to
  `remotectrl.config.resolve`, no sumac-specific logic — config resolution is entirely
  remotectrl's job per §4.
- `synced_commit(repo_root, op) -> str`: preflight (printing warnings via `render.print_warning`,
  raising `SyncDivergenceError` on `DivergenceError`) → `onecommit.run_op` → postflight (raising
  `SyncPushError` on `PushError`). Both new exceptions are `SumacError` subclasses in
  `errors.py`, alongside the existing `GitError` etc., so `cli.main`'s existing
  `except (SumacError, SealError)` handler catches them with no changes to `main`.

`store.commit_records` now builds its commit as a closure and passes it to
`remote_sync.synced_commit(repo_root, ...)` instead of calling `gitrepo.commit_paths` directly.
A repo with no `.rc/remotes.toml` and no remotes configured resolves to `{}` (empty dict) — both
preflight and postflight are no-ops over an empty dict, so a household that hasn't set up sync
yet sees no behavior change at all.

## 4. `sumac sources`

Per the `sources` design doc §3. Two sumac-side git helpers added to `gitrepo.py`
(`remote_names`, `remote_url`) — sumac keeps its own git subprocess layer per that module's own
docstring ("every git subprocess call in sumac lives here — nowhere else"); it does not reach
into `remotectrl.gitwrap` for git calls that are sumac's own responsibility (listing/reading
`git remote` config), only for calls remotectrl's public API already owns (fetch/ahead-behind/
push, via `preflight`/`postflight`).

- **Listing** (no `--setup`, or config already exists and `--setup` not given): one row per
  `git remote -v` name. `TYPE` from `resolve_remotes`; `URL` from the new `gitrepo.remote_url`;
  `STATUS` from calling `remotectrl.preflight.run_preflight` with just that one remote and
  reporting its warnings, or `"diverged: ..."` if it raises `DivergenceError` — caught per-remote
  so one bad remote's listing doesn't abort the rest of the table (§3's listing is meant to show
  status for everything, not stop at the first problem).
- **Setup** (`--setup`, or no `.rc/remotes.toml` exists yet): §3's discovery default — exactly
  one remote and no config → auto-assign `mirror`, print what was chosen, no prompt (matches
  remotectrl design doc §3's own discovery default, restated in the `sources` doc). Two or more
  remotes → per-remote `unsynced`/`mirror`/`backup` picker (`prompt_ui.select`), defaulting each
  to its current assignment (or `unsynced` if none). Written to `.rc/remotes.toml` and committed
  via `gitrepo.commit_paths` (tracked config, per design doc §2) with message
  `"sumac: configure remotes"` — separate from `store.commit_records`'s per-write commits, since
  this isn't a data write.
- `sources` doc §4's "edit an existing assignment" open question: resolved pragmatically by
  making `--setup` re-runnable at any time, defaulting every prompt to the existing assignment —
  not a separate designed flow, just the same setup loop re-entered.

## 4a. Independent review of `remote_sync.py`/`store.py` (unit close-out)

A review agent given only the diff plus §5/§7 excerpts flagged two untested behaviors:
transport-failure-is-soft (mirror/backup fetch failing must warn, not block) was asserted in
prose but never exercised, and §7's "attempt every remote, aggregate failures" amendment was
only ever tested with a single configured remote. Both fixed: added
`test_transport_failure_warns_but_does_not_block` and
`test_multiple_remotes_one_fails_postflight_other_still_receives_push` to
`tests/test_remote_sync.py`. The reviewer's other note (test fixtures hardcode
`writer/alice-mac` as the branch name) is correct but out of scope — that's `git_data_dir`, an
existing sumac test fixture from the 2026-09-11 branch-per-user design, not something this
integration introduced or could change without touching unrelated fixtures.

Also flagged by the errors/gitrepo unit's review: no tests existed yet for `gitrepo.remote_names`
/`gitrepo.remote_url`. Covered — see `test_remote_names_lists_configured_remotes`,
`test_remote_url_reads_configured_url`, `test_remote_url_returns_none_for_unknown_remote` in the
same file.

## 4b. Independent review of `sources_cmd` (unit close-out)

A review agent given only the diff plus the `sources` design doc §3 excerpt raised two points:

- **Whether plain `sumac sources` (no `--setup`) should prompt when 2+ remotes exist and no
  config does yet**, or just list everything as `unsynced`. Re-reading §3 closely resolves this:
  the "Interactive setup" mode's trigger condition is stated as "no `.rc/remotes.toml` exists
  yet, **or** user explicitly re-runs setup" — two independent triggers, not one. The single-
  remote auto-default bullet under that same heading confirms first-encounter (no config) is
  squarely inside interactive-setup's scope, not listing's — listing's own heading covers
  "already configured, just checking status." Current behavior (`setup or first_time` triggers
  the assignment path) matches this reading; no change made.
- **Whether STATUS surfaces a pending-push marker**, per §3's explicit requirement. It does —
  `status_text` delegates to `remotectrl.preflight.run_preflight`, which reads and surfaces any
  pending marker as a warning before checking anything else (this was itself one of remotectrl's
  own fixed bugs earlier the same night). This was true but *untested* from sumac's side —
  added `test_status_surfaces_pending_push_marker` to `tests/test_sources_cmd.py`, confirming a
  marker written directly (simulating a prior failed push) shows up in `sumac sources`'s output.

## 5. What was deliberately not touched

No changes to `decide.py`, `ledger.py`, `models.py`, `queue.py`, `writer.py`, `vault.py`, or any
CLI command function's body beyond `sources_cmd` itself — the entire integration is
`remote_sync.py` (new), two additions to `gitrepo.py`, two new exceptions in `errors.py`, one
changed function body in `store.py`, and one new command in `cli.py`.
