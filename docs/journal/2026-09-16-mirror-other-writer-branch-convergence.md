# 2026-09-16: A `mirror` never actually resynced other writers' local branches

**Status:** confirmed by the user, implemented (see §5 for the STATUS-column extension, added
in the same round as this fix, at the user's explicit request)

## What was found

Not a bug that crept in by accident — a real question that was never explicitly decided,
because a different, adjacent question (anomaly detection) got answered instead and was
mistaken, all night, for having also answered this one.

remotectrl-design.md §2's rule — *"only its designated writer should ever be ahead of a remote's
copy of [a branch]; every other party's local view of that branch should only ever be behind or
even"* — is an **anomaly-detection** rule: it says what's a *violation* (ahead, for a branch you
don't own) versus what's *normal* (behind). It says nothing about whether "behind" gets resolved
or just tolerated indefinitely. Those are two different questions:

1. Is being behind on a branch I don't own an error? — **No.** Answered clearly, all night,
   correctly.
2. Given that it's not an error, does a `mirror` actually catch that branch up locally, or does
   it just leave it stale forever while feeling correct because "correct" was only ever measured
   against question 1? — **Never actually decided.** `fetch_before_read`'s behavior (fetch into
   the tracking ref, never touch the local branch) silently answered "no" by omission, and that
   answer went unnoticed because nothing about it looked like a violation of question 1's rule.

The name **`mirror`** was chosen, earlier the same night, specifically to mean actual
bidirectional convergence — rejected alternatives included "synced" (too ambiguous) precisely so
the chosen name would carry a stronger promise than "a remote I happen to also fetch from." A
mirror that only ever updates a tracking ref and never brings a non-owned branch's *local* state
forward isn't mirroring that branch at all from this repo's point of view — it's just caching a
remote's data under a name that promises more.

## What this actually affects, traced concretely

`gitrepo.list_writer_refs` — the function `sources.writer_sources` (the read layer every
`find`/`status`/`ask` call goes through) uses to decide where to read each writer's data from —
resolves each writer id to **exactly one** ref:

```python
return {**remotes, **locals_}  # local wins on a collision
```

- **If this machine has never created a local branch for another writer** (the common case —
  nothing in sumac ever does this for you, and there's no reason to unless someone manually runs
  `git checkout -b writer/cm`): there is no collision. Reads go straight to
  `refs/remotes/<remote>/writer/cm`, which `fetch_before_read` already keeps fresh (traced and
  confirmed twice, in the previous entry). **Not a read-correctness bug in this case** — only a
  visibility gap (already fixed) and, now, a convergence gap (this entry).
- **If a local branch for that writer *does* exist on this machine** (a shared/testing machine
  that once ran `init-writer` under someone else's id, or a stale leftover from an earlier
  session) — **local wins**, and that local branch is never fast-forwarded by anything today. In
  this case the read layer silently serves stale data from a branch nothing keeps current. This
  is the case where the missing convergence is an active read-correctness bug, not just a
  visibility or "mirror doesn't fully mirror" naming gap.

Either way, the fix is the same and closes both: fast-forward, don't just fetch.

## Decision: fast-forward every other writer's *local* branch too, when one exists

For a `mirror` remote, after fetching: for every writer branch that has **both** a local ref and
a matching remote-tracking ref, fast-forward the local branch to the tracking ref — not just the
current writer's own branch (already done, `_resync_own_branch`), every writer's.

This is safe in a way merging isn't, and does not reopen anything §7/§9 ruled out. Those sections
banned **merging** another writer's branch locally ("no local fast-forward/merge of other
writers' branches" — read in its original context, that was written to rule out materializing a
second, redundant representation of other writers' data for no consumer, back when the read layer
was assumed to always read tracking refs directly). Fast-forwarding is a different, strictly
safer operation: this repo never writes to a branch it doesn't own, so there is no divergent
local history to protect — a fast-forward there can only ever succeed cleanly (if the local ref
is a pure ancestor of the tracking ref, which is the only way it could exist unless something
already went wrong) or fail cleanly (raising, same as `_resync_own_branch`'s existing fallback),
never require a conflict decision.

When no local branch exists for that writer (the common case), there's nothing to fast-forward —
the tracking ref alone is already what reads use, already fresh. This fix's practical effect is
therefore mostly about consistency and honesty (a `git log` on `writer/cm` locally isn't
misleadingly stale if you ever do check it out) in the common case, and closes a real, if narrow,
read-correctness bug in the shared/testing-machine case where a local branch for another writer
already exists.

## What's being asked for confirmation before implementing

- Generalize `_resync_own_branch` into something that runs for every writer branch with a local
  ref on a mirror remote, not just the currently-checked-out one — same fast-forward-or-warn
  shape, just not restricted to `mine`.
- This lives in `sumac.remote_sync`/`sumac.gitrepo` (already where `fast_forward_to` lives), not
  in `remotectrl` — this is domain-specific (mapping "writer branches" onto git branches) exactly
  the way the original design divided labor.
- Wire it into `fetch_before_read` (so a read resyncs every writer's local branch, when present),
  and into `sources`'s per-branch view too, since `sources` is itself a read.

## 4. How to fast-forward a branch that isn't checked out

`gitrepo.fast_forward_to` (`git merge --ff-only <ref>`) only works on `HEAD` — it can't move a
branch other than the one currently checked out. Confirmed by hand (two throwaway repos, not
guessed): `git fetch . <tracking-ref>:<local-branch>` fast-forwards a *non-checked-out* local
branch using git's own fetch fast-forward enforcement — it updates the ref when the tracking ref
is a strict descendant, and fails cleanly (nonzero exit, `! [rejected] ... (non-fast-forward)`,
no partial state) when it isn't, with no `+`/force prefix on the refspec. This is the mechanism
for resyncing every writer branch that isn't the one currently checked out; the existing
`git merge --ff-only` stays exactly as is for the currently-checked-out (own) branch.

## 5. `sources`'s STATUS column was answering a narrower question than it looked like

Same category of problem as the ahead/behind message bug from earlier tonight — a summary label
technically correct about a narrow thing, read as a broader claim than it can support. The
top-level `STATUS` column (`status_text`) only ever reflected the *current writer's own*
ahead/behind state on that remote. Once the per-branch table was added, this became visibly
inconsistent: the top row said `"up to date"` while the table right underneath it showed another
writer's branch sitting hours stale. "Up to date" read as a claim about the whole remote; it was
only ever a claim about one branch on it.

**Decision:** for a `mirror`, `STATUS` now means "my own branch is current, **and** every other
writer's local branch (where one exists) was successfully fast-forwarded this call." Any local
writer branch that couldn't be resynced cleanly (diverged, per §4) makes STATUS say
`"partial (writer/<id> out of date)"` naming the specific branch, rather than the generic
"diverged" already used for the current writer's own-branch case — the two situations need
different responses (yours needs your attention directly; someone else's needs a message to
them, or investigation of why their local checkout on this machine has diverged), so the wording
should point at which. The goal, same standard as the ahead/behind fix: never require checking
the per-branch table underneath to learn that something's actually wrong.

## 6. Independent review (unit close-out)

Caught one real bug and one real test gap, both fixed:

- **`status_text` silently dropped one problem when both existed at once.** The first version
  returned `own_warning` whenever it was set, never checking `other_warnings` too — an own-branch
  divergence would hide a simultaneous other-writer resync failure, or vice versa. Directly
  contradicts this fix's own stated goal ("never require checking the per-branch table underneath
  to learn something's wrong"). Fixed: both messages are now included, joined, whenever both are
  present. Added `test_status_text_reports_own_and_other_branch_problems_together`, confirmed to
  fail against the pre-fix (drop-one-side) code and pass after.
- **No test asserted the positive path** — that STATUS still says `"up to date"` when another
  writer's branch exists and resyncs *cleanly*, not just when it fails to. Without this, a stub
  that always reported `"partial (...)"` for any repo with any other local writer branch present
  (divergent or not) would have passed every other test in this fix. Added the assertion to
  `test_sources_shows_per_branch_view_for_other_writers`, which already has exactly that setup.
- **Noted but not fixed** — a narrow, safely-degrading edge case: `_sync_remote`'s
  `other == branch` guard only knows about *this* worktree's checked-out branch. A repo with a
  second `git worktree` checked out on a different writer branch would still attempt
  `fast_forward_branch` against it; git's own worktree-checkout guard rejects that, and the
  existing `except GitError` turns it into a warning — no corruption, just a warning where none
  is really warranted. Multi-worktree sumac usage isn't a scenario in scope tonight; flagged for
  later if it ever comes up.
