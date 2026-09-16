# 2026-09-15: Read-path freshness — a gap neither prior design covered

**Status:** implemented
**Depends on:** `docs/journal/2026-09-13-remotectrl-design.md`,
`docs/journal/2026-09-14-remotectrl-integration.md`

---

## 1. The gap

Both prior docs scope preflight/postflight to *write* commands only. Neither ever asked whether a
*read* needs freshness too — and it does. remotectrl-design.md §7 says outright:

> Sumac's read layer already aggregates directly over `refs/remotes/*/writer/*` in memory
> ... without ever touching local `writer/<other>` branches.

That's true, but it only holds if those remote-tracking refs are fresh. Nothing fetches them
except a write's preflight (`store.commit_records` → `remote_sync.synced_commit`) or the
pre-existing, manual `sumac sync` command (2026-09-11 design). `find`, `status`, and `ask` (which
wraps `find`/`status`-shaped tool calls, per its own docstring — it is not a command in its own
right) call `ledger.build_inventory` directly and never fetch anything. A household that only ever
reads — asks "where's the pork" without also writing — can go arbitrarily stale against a `mirror`
remote another writer is actively updating, and be told a confident, wrong answer. Found by the
user running the freshly-integrated `sumac sources`/`ask` against real data and noticing `sources`
reported `umbrel` as "diverged: 1 commit behind" while `ask` answered anyway from stale local data.

## 2. Decision: fetch always, never block, warn loud and inline

A blocked *write* is justified (§5 of the companion doc): committing on top of stale state
corrupts the append-only log, a structural problem. A blocked *read* has no equivalent
corruption risk — the cost of a stale read is "this might have moved since," real but bounded and
recoverable. Blocking every read on any divergence trades a recoverable staleness problem for an
unrecoverable-in-the-moment one: the tool that's supposed to tell you where something is now
tells you nothing, exactly when you're mid-task and least able to go fix a remote's connectivity
or wait for another writer's machine to come online.

So: fetch every configured `mirror`/`backup` remote before a read, exactly like preflight does,
but never raise on divergence — fold what would have been `DivergenceError` into the same
warning channel as an ordinary transport failure. The warning must be impossible to miss: printed
directly above the read's own output, not a separate log line a user has no reason to scroll back
for.

## 3. What was added

- `remote_sync.fetch_before_read(repo_root) -> list[str]`: calls `remotectrl.preflight.run_preflight`
  exactly as `synced_commit` does, but catches `DivergenceError` and returns its message as an
  ordinary warning string instead of letting it propagate. Reuses remotectrl's fetch/ahead-behind
  logic entirely — no duplicated git plumbing, consistent with `remote_sync.py`'s existing role as
  "sumac calls remotectrl's granular functions, never reimplements them."
- `cli._warn_staleness(data_dir)`: resolves `repo_root`, no-ops if it's not a git repo (matches
  every other remotectrl-aware command's existing no-op-outside-git behavior), else calls the
  above and prints each warning via `render.print_warning` before the command's own output.
- Wired into `find`, `status`, and `ask` (before `_ask_loop`/`_ask_one`, so it fires once per
  invocation rather than once per LLM tool call) — the three commands that read `build_inventory`
  as their primary purpose, i.e. the ones the user was actually asking about. Not wired into the
  `build_inventory` calls inside `add`/`retire-location`/the `ask`-loop's edit path — those are
  already write commands whose own `store.commit_records` preflight will catch real divergence
  before anything commits; adding a second, non-blocking fetch immediately beforehand would just
  be a second network round-trip for a check the write path already makes authoritative.

## 4. Known limitation: only the first diverged remote is ever reported

`remotectrl.preflight.run_preflight` raises on the *first* divergence it finds and stops — it
was never amended the way postflight was (§7 of the companion doc, "attempt every remote").
`fetch_before_read` catches that single `DivergenceError`, so with two remotes both diverged at
once, only the first is ever surfaced to a read; the second's staleness goes unreported until the
first is resolved and a later read re-checks. Not fixed here — it's remotectrl's own iteration
order, out of scope for a sumac-side change, and a real but narrow edge case (two remotes
diverging simultaneously). Worth revisiting in remotectrl itself if it turns out to matter in
practice.

## 4a. Independent review (unit close-out)

A review agent given only the diff plus this doc raised three points, all addressed:

- **Duplication**: `fetch_before_read` and `synced_commit` each re-implemented the
  `run_preflight` → `f"{remote}: {message}"` formatting independently. Factored into a shared
  `_preflight_messages(repo_root, remotes)` helper that both now call; the two functions differ
  only in whether they catch `DivergenceError` or let it propagate.
- **Untested: multiple remotes in one read fetch.** Added
  `test_fetch_before_read_multiple_remotes_only_reports_first_divergence`, which pins down §4's
  known limitation as an explicit assertion (`len(warnings) == 1`) rather than leaving it as an
  unverified claim in prose.
- **Untested: `status`/`ask` CLI-level warnings, and `_warn_staleness`'s no-git-repo no-op.**
  Added `test_status_warns_on_stale_remote_but_still_answers`,
  `test_warn_staleness_is_silent_outside_a_git_repo`, and
  `test_ask_calls_warn_staleness_before_loading_the_model` (the last one spies on
  `cli._warn_staleness` and stubs `_import_llm` to confirm the wiring without needing the actual
  model).

## 5. Non-goals

- No change to remotectrl itself — `run_preflight`'s raise-on-divergence behavior is exactly right
  for writes; sumac downgrades it to a warning only at its own read call sites, not inside
  remotectrl.
- No caching/debouncing of the fetch across repeated reads in a session — out of scope for this
  fix; `ask --loop`'s LLM inventory cache (`_build_inventory`'s `_inventory_cache`) is untouched,
  and the fetch happens once per `ask` invocation, not once per cached inventory build.
