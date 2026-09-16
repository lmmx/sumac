# 2026-09-16: `sumac sources` has no display surface for other writers' branches

**Status:** diagnosed, confirmed by the user, implementing

## What was reported

The user, on `writer/lm`, ran `sumac ask`, then separately confirmed via raw git that
`umbrel`'s copy of `writer/cm` had moved (another household member had pushed new work) while
their local tracking ref for it looked stale by inspection. `sumac sources` reported `umbrel` as
"up to date" throughout. Read literally, that's a contradiction worth taking seriously: the tool
that's supposed to tell you the sync state said everything was fine while a real writer branch's
state was, by the user's own reading, not being reflected.

## What the trace actually found

Two concrete, reproducible experiments (not inspection, actual runs):

1. Calling `remotectrl.gitwrap.fetch_all(local_repo, "origin")` directly while `local_repo` is
   checked out on `writer/lm`, after a second clone commits and pushes to `writer/cm` on the
   shared remote: `refs/remotes/origin/writer/cm` moves to the new commit. The generic fetch
   refspec (`refs/heads/*:refs/remotes/<remote>/*`) is not scoped to the checked-out branch —
   confirmed working as designed.
2. Calling `sumac.remote_sync.fetch_before_read` (the actual function `ask`/`find`/`status` call)
   in the same setup, with `.rc/remotes.toml` configuring the remote as `mirror`: the tracking ref
   for `writer/cm` updates correctly to match the remote, and `fetch_before_read` returns no
   warnings.

**The fetch is not broken.** §1 of the original ask ("confirm whether the generic fetch is
actually being invoked, or something upstream scopes it to only the current branch") is answered:
it is invoked, unscoped, and correct.

## The actual gap

`remote_sync.fetch_before_read`'s only check against another writer's branch is
`check_other_branches` (mirrored from `remotectrl.preflight`'s own logic): for every branch in
`local_branches(repo_root)` other than the current one, if a matching remote-tracking ref exists,
flag it if **this repo** is ahead of it. Per remotectrl-design.md §5, that check exists to catch
one specific mistake: *"this local repo committed on a branch it doesn't own."* It was never
designed to answer, and cannot answer, "is `writer/cm`'s branch itself in a healthy state" —
that's `cm`'s own machine's problem, checked by `cm`'s own preflight in its own `mine=True` mode
against `umbrel`. `lm`'s machine has no way to know whether `cm` has uncommitted local work, or
whether `cm`'s last push succeeded from `cm`'s point of view — it only ever sees what's already on
the remote.

Given that scoping, `writer/cm` simply advancing (a normal, healthy push from another writer) was
*never going to produce a warning* — correctly, since it isn't an error. But that leaves a second,
separate problem, which is the real gap the user's expectation exposed: **`sumac sources`'s table
has one row per remote, never one row per writer branch.** Even after a completely successful,
correctly-unscoped fetch, there is no code path anywhere that surfaces *any* information about
`writer/cm`'s state — not its last commit, not when it was last fetched, nothing. "Up to date"
in the STATUS column is only ever answering "is *my own* branch (`writer/lm`) synced with this
remote?" A user reading that column as "is everything on this remote fine" — the entirely
reasonable reading, since a per-*remote* table has no visible qualifier saying otherwise — gets
misled not by wrong data, but by a question that was never being asked. Silence defaults to
reading as "fine," which is indistinguishable from actually fine unless you already know the
column's real, narrower scope.

## Fix: a per-branch view, not just per-remote

`sumac sources` gains a second level under each mirror remote: every writer branch known on that
remote (via the union of local `refs/heads/writer/*` and that remote's `refs/remotes/<remote>/
writer/*`, i.e. the same union `gitrepo.list_writer_refs` already builds for the read layer,
scoped to one remote), each with:

- whether it's this repo's own branch (`mine`) or another writer's (`theirs`);
- the remote-tracking ref's current commit (short hash + relative time), confirming the fetch
  actually landed something recognizable, not just "no error was raised";
- for `mine`: the existing ahead/behind divergence text (`_own_branch_divergence_text`);
- for `theirs`: whether this repo has ever committed there by mistake (the existing
  `check_other_branches` anomaly, now attached to the specific branch it's about instead of
  folded into the same collapsed remote-level message) — otherwise just the tracking ref's last
  commit info, with no invented "healthy"/"unhealthy" verdict, since this machine genuinely has
  no basis for one beyond that.

This doesn't invent a new health signal for other writers' branches (there isn't one available)
— it makes visible exactly what's true today (the fetch worked, here's what it pulled) instead of
folding "theirs" into a single per-remote row that only ever spoke about "mine." A `backup`
remote gets no per-branch breakdown — it's single-owner by definition (remotectrl-design.md §3),
so there's only ever one branch's state to show, already covered by the existing row.

## Non-goals

- No new divergence *detection* for other writers' branches — architecturally, this machine
  cannot assess another writer's branch health beyond what's already pushed, and inventing a
  synthetic verdict would be worse than showing the raw fetched state and letting the user judge.
- No change to `remotectrl` itself — this is entirely a sumac-side display change, built from
  data `remote_sync`/`gitrepo` can already read (last-commit info via existing git plumbing, no
  new remotectrl API needed).
