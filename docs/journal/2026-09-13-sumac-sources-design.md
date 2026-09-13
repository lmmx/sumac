# 2026-09-13: `sumac sources` — Remote Sync Configuration UI — Draft Design

**Status:** design discussion, no code written
**Author:** drafted with Claude, 2026-09-13
**Scope:** new `sumac sources` CLI command; consumes the standalone `remotectrl` package
(`docs/journal/2026-09-13-remotectrl-design.md`)

---

## 1. Relationship to `remotectrl`

This entry covers only sumac's own surface: the `sources` command and where its config lives.
All sync semantics (remote types, preflight/postflight behavior, ahead/behind rules) belong to
`remotectrl` and are intentionally not restated here — see the companion journal entry. Nothing
in this design is sumac-specific in principle; `remotectrl` itself must stay usable by any
git-backed application with a designated-writer-per-branch shape, not just sumac. `sumac sources`
is simply the first (and for now, only) consumer's config UI for it.

## 2. Config file

`remotectrl` config lives at **`.rc/remotes.toml`**, inside the repo, at the app-config path
`remotectrl` itself owns (not `.sumac/`, not `.git/`) — `.rc` is `remotectrl`'s own directory,
independent of whatever application embeds it. It is a tracked file, committed like any other
repo content, so the same mapping travels with every clone and every backup remote.

Rationale for referencing remotes **by git-configured name, not URL**: the same repo can be
cloned to different machines where a "the shared server" remote might be added under a different
name locally (unlikely in practice, but the git config is the single source of truth for
name→address resolution already — duplicating the address into `remotectrl`'s own config would
create a second place that could drift). `remotectrl` resolves `origin` / `umbrel` /
whatever-it's-named purely by looking it up in `git remote -v` at call time; it never stores or
reasons about URLs itself.

Sketch:

```toml
# .rc/remotes.toml
[remotes]
umbrel = "mirror"
origin = "backup"
```

Any remote present in `git remote -v` but absent from this file is implicitly `unsynced` — no
need to enumerate every remote explicitly, only the ones with a role assigned.

## 3. `sumac sources` command

Two modes:

**Listing (no configured remotes yet, or just checking status):**

```
$ sumac sources
NAME     TYPE       URL                                  STATUS
umbrel   mirror     ssh://umbrel.local/home/repo.git      up to date
origin   backup     git@github.com:louis/private.git      up to date
```

- `NAME` / `TYPE` come from `.rc/remotes.toml` (or `unsynced` if absent from it).
- `URL` is read live from `git remote -v` — `remotectrl` resolves names to addresses, and sumac's
  `sources` command surfaces that resolution to the user rather than making them go check
  `git remote -v` separately.
- `STATUS` reflects the same ahead/behind and pending-push-marker state `remotectrl` tracks
  (§5/§7 of the companion entry) — this doubles as the surfacing point for "commits not yet
  pushed to `<remote>`" markers left behind by a prior failed push, so a stale backlog is never
  silently forgotten.

**Interactive setup (no `.rc/remotes.toml` exists, or user explicitly re-runs setup):**

- Exactly one remote configured in git, no `remotectrl` config yet: default it to `mirror`
  automatically (per §3 of the companion entry's discovery default), print what was chosen, done
  — no prompt needed for the trivial case.
- Two or more remotes, no config yet: **do not guess**. List them all as `unsynced` and prompt the
  user to assign roles — pick one remote as `mirror`, optionally one (or more, per the companion
  entry's non-goals note on multiple mirrors) as `backup`. The picker allows explicitly leaving a
  remote `unsynced` — assigning roles is opt-in, not mandatory; a user can run `sumac sources` and
  decide not to configure sync at all.

## 4. Open questions

- Exact prompt UX for the interactive picker (single-select then multi-select, or one combined
  per-remote-select-a-type loop) — not designed yet, low-stakes, can be decided at implementation
  time.
- Whether `sumac sources` should also expose a way to *edit* an existing assignment (change
  `umbrel` from `mirror` to `unsynced` later) versus only running full setup once — probably yes,
  re-running the command against an existing config should let you change one entry rather than
  starting over, but not designed in detail here.
- Where `remotectrl`'s own CLI (if it has one) ends and sumac's `sources` command begins — this
  draft assumes `sumac sources` is sumac's own command that calls into `remotectrl`'s Python API,
  not a wrapper shelling out to a separate `remotectrl` CLI binary. Worth confirming once
  `remotectrl`'s public API shape (§6/§8 of the companion entry) is settled.
