# 2026-09-13: `remotectrl` — Multi-Remote Sync Abstraction — Draft Design

**Status:** design resolved (see §6-§7 for the resolved one-commit assertion, marker storage, and
mirror-preflight scope), no code written
**Author:** drafted with Claude, 2026-09-13
**Scope:** new standalone package `remotectrl` (not sumac-specific); consumed by sumac's
branch-per-writer design (`docs/journal/2026-09-11-branch-per-user-design.md`)

---

## 1. Problem

sumac's branch-per-writer design (§2026-09-11) gives each writer instance its own `writer/<id>`
branch so pushes never collide. That solves *coordination on one remote*. It says nothing about
*which remote(s)* a writer instance talks to, or what "in sync" means when there's more than one.

The concrete case driving this: a household git repo with two remotes —

- `umbrel` — a shared home server. The user and their partner each push to their own
  `writer/<id>` branch there. Bidirectional in the sense that the repo as a whole moves in both
  directions, but per-branch each branch still has exactly one writer.
- `origin` (GitHub) — a private backup, single-owner. Only the user ever pushes to it. No one
  else's writes are expected there, ever.

The behavior needed — "before an operation, check I'm not missing anyone else's changes on
`umbrel`; after it, push my branch to `umbrel` and mirror it to `origin`; never surprise me by
silently losing a commit" — is generic git-remote-sync behavior with no sumac domain content
(no stream IDs, no encryption, no config aggregation). It belongs in its own package for the same
reason encryption-at-rest was delegated to `sealedlog` rather than folded into `sumac/store.py`:
it's a self-contained, non-trivial correctness problem, and sumac's job is to supply the domain
mapping (which branch is "mine") and call the wrapper, not to reimplement git-remote plumbing.

`remotectrl` operates on **plain system `git`** via subprocess (matching sumac's own
`gitrepo.py` style) — no GitPython, no dulwich. Callers never touch git plumbing directly.

## 2. Core principle

For any given branch, exactly one party is its designated writer. Ahead/behind is evaluated
**per branch**, not per remote:

> For a given branch, only its designated writer should ever be ahead of a remote's copy of it.
> Every other party's local view of that branch should only ever be behind or even.

This single rule produces both remote types' behavior as special cases — no separate rules
needed, and no contradiction between them (an earlier draft of this design conflated "mirror can
be behind" with "mirror falling behind is an error," which this framing resolves: it depends on
*whose* branch it is).

## 3. Remote types

Three states, configured per remote, not inferred from git history:

| Type | Meaning | Branches it applies to |
|---|---|---|
| `unsynced` | Default for any remote not explicitly configured. No preflight checks, no automatic push. | n/a — skipped entirely |
| `mirror` | Bidirectional collaboration remote. Each branch still has exactly one writer. | Own branch: push-only (ahead is normal, behind is an error). Others' branches: fetch-only (behind is normal, ahead is an error — would mean this local repo committed on someone else's branch by mistake). |
| `backup` | Single-owner backup target. Only the current user ever writes here, on any branch this repo pushes. | Push-only from local. Remote ever being ahead of local (having a commit local lacks) is an error state — nothing else should ever write here. |

Naming note: rejected "synced" for the bidirectional type (ambiguous — sounds like a
steady-state property, not a remote role) and "push-only mirror" for backup (self-contradictory —
"mirror" implies bidirectionality). Settled on `mirror` / `backup` as disjoint, role-describing
terms.

**Discovery default:** if a repo has exactly one configured-or-configurable remote and no
`remotectrl` config exists yet, treat it as `mirror` (the collaboration case is assumed to be
the base case; backups are additive). With two or more remotes and no config, all are `unsynced`
until a human designates roles explicitly — this is the interactive setup surfaced by sumac's own
`sources` command (see the companion journal entry).

**Resolved gap: config names a remote that doesn't exist on this clone.** `.rc/remotes.toml`
is a tracked file (§2 of the companion `sources` entry) — it travels to every clone via normal
git sync, but `git remote -v` is per-clone, untracked, local config. This means the two can
disagree: the household case that surfaces it is a `backup` remote (e.g. `origin`, a private
GitHub mirror only one person owns) that gets committed to `.rc/remotes.toml` and thus appears
in every other clone's copy of that file too — including a partner's clone that has never run
`git remote add origin` and never should.

This matters because backup's single-owner property (§3's table: "only the current user ever
writes here") was never actually an active check anywhere in this design — unlike mirror, which
checks ownership per-call via the currently-checked-out branch (§4), backup has no equivalent
ownership check. The property only held *incidentally*, because in the motivating scenario only
one person's machine happens to have `origin` configured at all. Once `.rc/remotes.toml` is
shared, that incidental protection needs to become an explicit rule: what does `remotectrl` do
on a clone where the config names a remote with no matching entry in that clone's `git remote -v`?

**Decision: treat it as `unsynced` — skip silently, do not error.** A config entry naming a
remote absent from the local `git remote -v` is resolved to "no such remote here" and dropped
from preflight/postflight for that call, exactly as if it weren't listed in `.rc/remotes.toml`
at all. This matches the existing soft-fail philosophy for anything outside the local clone's
control (§5's transport-failure handling) rather than the hard-fail philosophy reserved for
genuine ahead/behind divergence on a remote that *is* present. It also means the single-owner
property for `backup` is — correctly, and by design, not by accident — enforced by nothing more
than "who ran `git remote add`," the same way mirror's per-branch ownership is enforced by
nothing more than "who is currently checked out on that branch." `remotectrl` verifies
consequences of ownership (ahead/behind), never ownership itself, for either remote type.

## 4. Per-branch identity

`remotectrl` has no concept of "user" — only "the branch currently checked out." The caller
(sumac) is responsible for having already switched to the correct `writer/<id>` branch before
invoking any `remotectrl` operation; `remotectrl` reads `git symbolic-ref HEAD` (or equivalent)
at call time and treats that branch as "mine" for the duration of the call. This is why config
can live inside the repo rather than in a user-global location: which remote plays which role is
a property of the repository (its git remotes), not of the OS user running sumac, and the
identity of "which branch is mine" is already externally determined by branch-per-writer.

## 5. Preflight (before the wrapped operation)

**One predicate underlies every check in this section**, making the type-specific bullets
below configurations of it rather than separate rules (this is what §2's "single rule produces
both remote types' behavior as special cases" cashes out to concretely):

```
ok(ahead, behind, mine) = (behind == 0) if mine else (ahead == 0)
```

`mine=True` for the currently-checked-out branch; `mine=False` for any other branch being
checked. Every bullet below is this predicate applied to a specific set of branches, plus a
decision about whether a `False` result blocks or is expected.

**Amended 2026-09-14** (see companion package's `docs/journal/2026-09-14-*` for the scoping
that surfaced this): the original text below fetched with a hardcoded `writer/*` refspec,
which contradicts §4's claim that `remotectrl` has no concept of branch-naming convention.
Fetching is now generic, and "which branches count as ahead-behind-checkable at all" is
derived structurally rather than by name:

- Fetch **every** branch generically: `git fetch <remote> 'refs/heads/*:refs/remotes/<remote>/*'`
  — no naming assumption, just mirrors whatever the remote has into tracking refs.
- After fetching, only **local branches that already have a local ref** get checked against
  their same-named remote-tracking ref. A branch this repo has never committed on has no local
  ref to compare, so it's structurally excluded — no `writer/*` pattern-matching needed to
  decide "is this someone else's branch," because "has no local ref" already means exactly that.

Run once per configured remote, in remote-config order:

- **`unsynced`**: skip. No fetch, no checks.
- **`mirror`**: fetch generically (above).
  - Fetch itself fails (network down, host unreachable): **warn and proceed**. Preflight is
    best-effort visibility, not a hard dependency on remote availability.
  - Current (own) branch: `ok(ahead, behind, mine=True)` is `False` (behind > 0) → **block**
    before the operation runs. This is a real divergence — something else wrote to a branch only
    this writer instance should ever write to — and it must be resolved before appending another
    commit, because sumac's format is event-sourcing / append-only: committing on top of stale
    state would produce a log entry whose causal position is wrong, not just a git conflict.
  - Every other local branch with a matching remote-tracking ref: `ok(ahead, behind, mine=False)`
    is `False` (ahead > 0) → **block** (this local repo committed on a branch it doesn't own).
    `False` from `behind > 0` on someone else's branch is expected, no action — that's the normal
    steady state.
- **`backup`**: fetch the current branch (or whatever refs it holds) from `<remote>`.
  - Fetch fails: **warn and proceed** (advisory only, same as mirror's transport-failure case).
  - Current branch: `ok(ahead, behind, mine=True)` is `False` (behind > 0, i.e. remote *ahead*
    of local) → **error state**. A backup remote should never have a commit local lacks — that
    can only mean something else pushed there, which breaks the single-owner invariant the whole
    point of `backup` depends on. This is the identical call mirror makes for its own branch —
    `backup` never evaluates `mine=False` on anything, because it never fetches or reasons about
    any branch but its own.

The distinction that matters throughout: **transport failure vs. a real ahead/behind
divergence get independent handling.** Transport failure is soft (warn, proceed) for both remote
types. A genuine divergence (`ok(...)` is `False` in a checked case) is hard (block) for mirror's
own-branch case, mirror's other-branches case, and backup's own-branch case alike, because all
three indicate a correctness problem in the append-only model, not just unavailability.

## 6. The wrapped operation and its one-commit contract

`remotectrl`'s call shape is a before/after wrapper around an operation the *caller* is
responsible for making atomic:

```
remotectrl.run(repo_path, op: Callable[[], None]) -> None
```

Contract: `op` must produce **exactly one commit** on the currently checked-out branch. This is
the caller's responsibility to guarantee — `remotectrl` does not stage or commit anything itself.
`remotectrl` snapshots `HEAD` before calling `op` and diffs against it after `op` returns; if the
diff isn't exactly one new commit (zero, or more than one), that's a caller bug and `remotectrl`
raises loudly rather than attempt any push. No stronger check (e.g. requiring `op` to return the
expected commit hash) — the snapshot/diff is sufficient and keeps the call shape simple.

## 7. Post-op (after the wrapped operation, commit confirmed)

For each configured remote, in order:

- **`unsynced`**: skip.
- **`mirror`**: push the current branch to the remote branch of the same name:
  `git push <remote> HEAD:refs/heads/<current_branch>` — `<current_branch>` from §4's
  `symbolic-ref HEAD`, the same source of truth preflight already used, not a reconstructed
  `writer/<id>` path. `remotectrl` never assembles a branch name; it only ever pushes to the
  name the caller already had checked out.
- **`backup`**: push the current branch the same way.

**Failure handling — deliberately no silent retry, no queue.** If a push fails (network down,
rejected, whatever), `remotectrl` hard-errors. The local commit from `op` is left intact — it is
never rolled back, since the append-only log's local copy is itself authoritative data, not a
cache. The failure must leave a **persistent, visible marker** that surfaces on:

- the next status check (sumac's `sources` command, or a `remotectrl` status call), and
- the *next* op attempt's preflight — so an unpushed backlog isn't silently forgotten between
  sessions.

No automatic background retry: this is a household tool run interactively, and a silent retry
that eventually succeeds could mean the user never learns a remote was flaky, while a silent
retry that never succeeds is worse than a hard stop. A human should always see and clear this
state deliberately.

**Marker storage.** `.git/remotectrl/pending-push/<remote>.json` — untracked, local-only, one
file per remote (not a combined file — avoids read-modify-write races between two remotes'
failures landing at once). This is cache/derived state describing *this specific clone's*
relationship to a remote, not configuration, which is why it lives under `.git/` rather than
`.rc/`: config (`.rc/remotes.toml`, §2 of the companion entry) is a decision true regardless of
which clone you're on ("umbrel is a mirror"); this marker is an observation true only for this
working copy ("this clone has 2 unpushed commits"). It correctly disappears on re-clone — a fresh
clone has no local commits beyond what it just cloned, so there is nothing to report. Same
pattern as git's own `MERGE_HEAD` / `ORIG_HEAD` / `.git/rebase-apply/`: transient, local-only,
silently absent when not applicable. Deleted outright on successful push — kept simple, no
retained "last synced" timestamp for the healthy case.

Example shape:

```
.git/remotectrl/
  pending-push/
    umbrel.json     # { "branch": "writer/louis", "commits": 2, "since": "2026-09-13T17:45:00Z", "last_attempt_error": "..." }
```

**Mirror preflight stays fetch-only.** No local fast-forward/merge of other writers' branches.
The fetch step (§5, amended 2026-09-14: `refs/heads/*:refs/remotes/<remote>/*`, generic — not
the `writer/*`-specific refspec this bullet originally cited) updates remote-tracking refs only
and nothing else. Sumac's read layer already aggregates directly over `refs/remotes/*/writer/*`
in memory (2026-09-11 design, §2) without ever touching local `writer/<other>` branches;
materializing local branches for other writers would build a second, redundant representation
of the same data with its own drift risk, for no consumer that reads it. It would also reopen
exactly the cross-branch-merge surface area §9 declares a non-goal (a non-fast-forward update to
a local branch nobody is meant to be checked out on has no clean resolution), and would violate
§4's scoping of `remotectrl` to only ever caring about the currently-checked-out branch. This
argument is naming-independent — it holds regardless of what any branch is called, which is why
switching §5's fetch to a generic refspec changes nothing here except the stale citation.
Confirmed as designed, not changed.

Config schema and file location are covered in the companion sumac-side journal entry (§2):
`remotectrl` stays agnostic to config *storage* and just accepts a resolved
`{remote_name: RemoteType}` mapping as input.

## 9. Non-goals

- No opinion on merge strategy, conflict resolution, or history rewriting — the branch-per-writer
  design already ensures no cross-branch merges are needed.
- No support for more than one `mirror` remote in this draft (a household with a third shared
  location isn't a case we have yet) — could generalize later if needed, not designed against now.
- Not a general-purpose git sync tool for arbitrary branch topologies — the per-branch-one-writer
  invariant is load-bearing throughout and assumed, not enforced defensively against arbitrary
  repos.
