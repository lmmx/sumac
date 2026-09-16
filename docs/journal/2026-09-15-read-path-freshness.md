# 2026-09-15: Read-path freshness — a gap neither prior design covered

**Status:** implemented (revised same day — see §2a)
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

## 2a. First pass (warn-only) was the wrong call — traced and revised

The first version of this fix fetched before a read and turned a would-be-blocking divergence into
a warning, same as a transport failure, and stopped there — stale data, with a loud note attached.
The user pushed back: warning-and-leaving-stale skips over an obvious third option. The fetch that
`fetch_before_read` already runs puts fresh data into a remote-tracking ref
(`refs/remotes/<remote>/<branch>`) *before* anything is read. What's actually stale isn't
sitting in `.git` unreachable — it's that nothing had ever brought the checked-out branch (and its
worktree, which is what `ledger.build_inventory` reads for the current writer — see
`sources._worktree_source`, `paths.log_path`/`config_path`) up to date *from* that ref. Tracing it
concretely:

- **Other writers' data** (`sources._ref_source`, used for every `writer/<id>` this repo isn't
  currently checked out on) already reads via `gitrepo.read_blob(data_dir, ref, ...)` straight off
  `refs/remotes/*/writer/*` — no local branch involved at all. This path was already correct the
  moment fetch ran; nothing to fix here.
- **The current writer's own data** (`sources._worktree_source`) reads plain files off disk in the
  worktree — governed by whichever commit the checked-out branch (`writer/<id>`) currently points
  to, not by any remote-tracking ref. A fetch alone never touches this. This is the actual stale
  path, and the one the reported "1 commit behind" divergence was about: `umbrel`'s copy of
  `writer/cm` (the current writer's own branch) had a commit local's checkout didn't.

That divergence is exactly the case §5 of remotectrl-design.md calls "something else wrote to a
branch only this writer instance should ever write to" — but for a `mirror` remote specifically,
"something else" here means *the same writer identity, from another device* (§1's household
scenario: the same person's phone and laptop can both be `writer/cm`, both pushing to `umbrel`).
Fast-forwarding the local checkout to match is safe precisely because it's a fast-forward — no
local commits are ever discarded, `git merge --ff-only` either succeeds cleanly or fails outright.
This is a different case from remotectrl-design.md §7's "Mirror preflight stays fetch-only... no
local fast-forward/merge of **other writers'** branches" — that non-goal is scoped to branches
this repo doesn't own; it says nothing about fast-forwarding a writer's own branch, and doing so
here doesn't reopen the cross-branch-merge surface area that non-goal rules out (§9).

A `backup` remote's own-branch divergence is different again: `backup` is meant to be
strictly single-owner from local only (remotectrl-design.md §3), so the remote ever having a
commit local lacks is anomalous, not routine multi-device use — auto-merging that would quietly
absorb a state that indicates something is actually wrong. That case, and a mirror's *other*
branch being ahead of a branch it doesn't own (also anomalous, per §5), still only warn — never
auto-resolved, never block.

## 2. Decision: resync what's safely resyncable, warn on everything else, never block

- **Mirror's own-branch divergence**: fast-forward the checked-out branch (and worktree) to the
  freshly-fetched remote-tracking ref. On success, nothing to report — the read proceeds on
  genuinely fresh data, not "fresh-with-an-asterisk." Fast-forward can only fail if local also has
  commits the remote lacks (`ahead > 0` alongside `behind > 0`) — a real, harder-to-resolve
  conflict, not the routine multi-device case — in which case it falls back to a warning rather
  than guessing.
- **Backup's own-branch divergence, and a mirror's other-branch-ahead case**: warn only, exactly
  as the first pass did — these indicate something anomalous (an unexpected write to a supposedly
  single-owner remote, or this repo having committed on a branch it doesn't own), not something
  safe to silently absorb.
- **Transport failure** (any remote unreachable): warn only, as before — advisory, matches
  preflight's own "warn and proceed" philosophy (§5).
- **Nothing ever hard-blocks a read.** The original reasoning for that still holds: a blocked read
  is a worse failure mode than a stale (or now, usually resynced) one — the tool that's supposed to
  answer "where's the pork" would otherwise answer nothing, exactly when connectivity or another
  device being offline is least convenient to go fix.

## 3. What was added

- `remote_sync.fetch_before_read(repo_root) -> list[str]`: its own per-remote loop (fetch, then
  check), not a call into `run_preflight` — needed because behavior now differs mid-check (resync
  own-branch mirror divergence instead of raising) in a way a black-box call to `run_preflight`
  can't produce. Built directly from `remotectrl`'s public granular pieces
  (`remotectrl.gitwrap.{fetch_all,current_branch,ahead_behind,ref_exists,local_branches}`,
  `remotectrl.behavior.BEHAVIOR`, `remotectrl.markers.read_marker`) — the same posture the
  integration doc already established ("sumac calls remotectrl's granular functions directly"),
  extended one layer further since this is the first read-side caller that needs to *act*
  differently per remote type mid-loop rather than just format `run_preflight`'s output.
- `_resync_own_branch(repo_root, remote, branch)`: the fast-forward-or-warn helper for the mirror
  own-branch case, described in §2.
- `gitrepo.fast_forward_to(repo_root, ref)`: new, in sumac's own git subprocess layer (per that
  module's docstring — every git call sumac itself is responsible for lives there, not
  `remotectrl.gitwrap`, since this is sumac's own decision about when to merge, not something
  `remotectrl`'s public API offers). `git merge --ff-only <ref>`; raises `GitError` on anything
  that isn't a clean fast-forward.
- `cli._warn_staleness(data_dir)`: resolves `repo_root`, no-ops if it's not a git repo (matches
  every other remotectrl-aware command's existing no-op-outside-git behavior), else calls
  `fetch_before_read` and prints each returned message via `render.print_warning` before the
  command's own output.
- Wired into `find`, `status`, and `ask` (before `_ask_loop`/`_ask_one`, so it fires once per
  invocation rather than once per LLM tool call) — the three commands that read `build_inventory`
  as their primary purpose. Not wired into the `build_inventory` calls inside
  `add`/`retire-location`/the `ask`-loop's edit path — those are already write commands whose own
  `store.commit_records` preflight is authoritative before anything commits.

## 4. Multiple diverged remotes are no longer a blind spot

The first pass inherited a `run_preflight` limitation: it raises on the first divergence and
checks no further remote, so with two remotes both diverged, only the first was ever reported.
Since `fetch_before_read` now runs its own loop instead of calling `run_preflight`, every
configured remote is fetched and checked regardless of what an earlier one found — this
limitation is gone for the read path as a side effect of §2's rewrite, not a separate fix.
(`run_preflight` itself, used by writes via `synced_commit`, is untouched and keeps this
limitation — out of scope here, same as before.)

## 4b. Independent review of the resync revision (unit close-out)

A review agent given only the revised diff plus this doc confirmed `fetch_before_read`'s loop
reproduces `run_preflight`'s per-remote-type dispatch faithfully (nothing silently skipped or
double-counted versus what `run_preflight` itself would flag), and that `fast_forward_to`
degrades safely on anything that isn't a clean fast-forward: mid-merge/mid-rebase or a dirty
worktree whose uncommitted changes collide with the incoming commit both make `--ff-only` fail
outright (`GitError` → warning, never a crash or silent data loss); a dirty worktree with no
colliding paths lets the fast-forward through with the uncommitted changes left untouched. This
last point was implicit in ordinary git semantics but never stated outright here — worth writing
down: a read run mid-`add`/`correct` (an in-progress write with uncommitted staged changes)
resyncs cleanly as long as the incoming commit doesn't touch the same files.

The one gap found: `test_fetch_before_read_fast_forwards_own_branch_on_a_mirror` only asserted
`local_head == remote_head` (a ref-level check) using an `--allow-empty` commit — it never
verified a worktree file actually changed on disk, which is the entire point of §2a's fix
(`_worktree_source` reads files, not just the branch tip). A `fast_forward_to` that only moved
the ref (e.g. `update-ref` instead of `merge --ff-only`) would have passed undetected. Fixed:
the test now writes a real file in the "remote" clone before committing, asserts it's absent
locally beforehand, and asserts its exact content is present in the worktree after the resync.

## 4a. Independent review (unit close-out, first pass)

Before the revision above, a review agent given only the first-pass diff raised three points, all
addressed at the time: a small duplication between `fetch_before_read` and `synced_commit`
(factored into a `_preflight_messages` helper, still used by `synced_commit`), an untested
multi-remote scenario (added, then superseded by §4's rewrite — the test was updated to assert
every remote is now reported, not just the first), and missing CLI-level coverage for
`status`/`ask`/the no-git-repo no-op path (added: `test_status_warns_on_stale_remote_but_still_answers`,
`test_warn_staleness_is_silent_outside_a_git_repo`, `test_ask_calls_warn_staleness_before_loading_the_model`).

## 5. Non-goals

- No change to remotectrl itself — `run_preflight`'s raise-on-divergence behavior is exactly right
  for writes; sumac's read path builds its own loop from remotectrl's public pieces rather than
  changing that package's behavior.
- No caching/debouncing of the fetch across repeated reads in a session — out of scope for this
  fix; `ask --loop`'s LLM inventory cache (`_build_inventory`'s `_inventory_cache`) is untouched,
  and the fetch happens once per `ask` invocation, not once per cached inventory build.
- No fast-forward for anything but a mirror's own branch — every other divergence case stays
  warn-only, deliberately, per §2's anomalous-vs-routine distinction.
