# 2026-09-16: `sumac sources` silently dropped `ahead` from a real divergence

**Status:** fixed

## The bug

The user hit this against real data: accidentally checked out on the wrong writer branch
(`writer/cm` instead of their own), committed there, which made that branch both 1 commit ahead
of `umbrel`'s copy *and* 1 commit behind it (someone/something else had also written there).
`sumac sources` reported the STATUS column as `"...is 1 commit(s) behind"` — no mention of the
local commit sitting there too. That reads as routine staleness ("will resolve on its own")
when the real state is a genuine divergence needing manual resolution — a materially dangerous
difference to gloss over, not a cosmetic one.

## Root cause

Two independent instances of the same pattern: computing both `ahead` and `behind`, but only
ever formatting `behind` into the message, regardless of whether `ahead` was also nonzero.

1. **`remotectrl.preflight.run_preflight`**'s own-branch check (the one `sumac sources`'
   `status_text` surfaces via `DivergenceError`) — fixed at the source; see that package's
   `docs/journal/2026-09-16-ahead-behind-message-bug.md` for the exact diff. This is the one
   that produced the exact string the user saw.
2. **`sumac.remote_sync.fetch_before_read`**'s backup-type own-branch check (added
   2026-09-15 as part of the read-path-freshness fix) had the identical pattern independently:

   ```python
   _, behind = ahead_behind(repo_root, branch, own_ref)
   if behind > 0:
       messages.append(f"{remote}: local branch {branch!r} is {behind} commit(s) behind ...")
   ```

   `ahead` was discarded outright (`_, behind = ...`), never even inspected.

## Fix

- Added `_own_branch_divergence_text(branch, ahead, behind)` to `remote_sync.py` — the single
  formatting rule every own-branch-divergence message in this module now goes through:
  `"diverged (N ahead, M behind)"` when both are nonzero, `"N commit(s) behind"` /
  `"N commit(s) ahead"` otherwise. `_resync_own_branch`'s existing (already-correct)
  divergence message and the backup-path check above both now call it, so a future fix here
  covers every display site at once rather than needing to be re-applied per call site.
- `remotectrl.preflight`'s own-branch message fixed at the source (see its journal entry) —
  this also fixes `sumac`'s write-path `SyncDivergenceError` (raised from `synced_commit`,
  which just wraps whatever `DivergenceError` says) and `status_text` (which, after this fix,
  no longer prefixes a redundant second `"diverged: "` in front of a message that already says
  `"has diverged (...)"`).

## Sibling-instance check

Grepped every `ahead_behind` call site in both `sumac/remote_sync.py` and every file in
`/mnt/remotectrl/src/remotectrl/`. The two fixed above were the only instances of "computes both,
formats one." Two other call sites correctly discard one side on purpose, not by bug:

- Both packages' **other-branch** checks (`mine=False`, violation is `ahead > 0`): `behind` isn't
  part of that violation — a branch being behind one this repo doesn't own is the expected
  steady state — so reporting only `ahead` there is correct, not the same bug.
- `remotectrl.postflight._unpushed_count` discards `behind` on purpose — it's counting unpushed
  local commits (`ahead`) for the pending-push marker, not formatting a divergence message.

## Test

`tests/test_sources_cmd.py::test_sources_reports_explicit_divergence_not_just_behind` reproduces
the user's exact scenario end-to-end through the real CLI command: clones a remote, commits once
on the remote's copy of the writer's own branch and once locally, configures it as `mirror`, runs
`sumac sources`, and asserts the STATUS column says `"diverged (1 ahead, 1 behind)"` and never
the old, misleading `"1 commit(s) behind"` alone. Confirmed failing against the pre-fix
`remotectrl` (reproduces the exact wrong wording — `"...is 1 commit(s) behind"`, no "ahead"
anywhere in the output) and passing after both fixes.

`tests/test_preflight.py::test_mirror_own_branch_ahead_and_behind_reports_both` (in the
`remotectrl` repo) covers the same case one layer down, directly against `run_preflight`.

## Unrelated but serious: `remotectrl` was never actually wired as an editable dependency

Found while debugging why this fix didn't seem to take effect on the first test run: sumac's
`pyproject.toml` was missing the `[tool.uv.sources]` entry pointing `remotectrl` at
`/mnt/remotectrl` — despite an earlier session's summary claiming this had already been done and
verified. `remotectrl>=0.1.0` was resolving from a registry as a frozen `0.1.0` wheel (built at
some earlier point in the local package index, coincidentally after most of that night's
remotectrl-side fixes but before this session's). Every test in this session up to this point
happened to still pass only because that frozen wheel already contained the earlier fixes —
tonight's `preflight.py` message fix was the first change made *after* that wheel was built, and
silently didn't take effect until this was caught.

Fixed: added `remotectrl = { path = "/mnt/remotectrl", editable = true }` to
`[tool.uv.sources]`, ran `uv lock && uv sync`, and confirmed
`python -c "import remotectrl.preflight as p; print(p.__file__)"` resolves to
`/mnt/remotectrl/src/remotectrl/preflight.py` directly (an editable `.pth`-based install, not a
copied wheel) before trusting any further test run tonight.
