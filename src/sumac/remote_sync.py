"""Wires `remotectrl`'s preflight/postflight around sumac's one-commit-per-command
contract. See docs/journal 2026-09-13-remotectrl-design.md §5-§7,
2026-09-14-remotectrl-integration.md §1, §3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from remotectrl import DivergenceError, GitError, PushError
from remotectrl.behavior import BEHAVIOR
from remotectrl.config import RemoteType, resolve
from remotectrl.gitwrap import (
    ahead_behind,
    current_branch,
    fetch_all,
    local_branches,
    ref_exists,
)
from remotectrl.markers import read_marker
from remotectrl.onecommit import run_op
from remotectrl.postflight import run_postflight
from remotectrl.preflight import run_preflight

from sumac import gitrepo, render
from sumac.errors import SyncDivergenceError, SyncPushError


def resolve_remotes(repo_root: Path) -> dict[str, RemoteType]:
    return resolve(repo_root)


@dataclass(frozen=True, slots=True)
class _RemoteSync:
    """Result of fetching and resyncing one remote once. `own_warning` is set
    only if the current writer's own branch has a real, unresolvable
    divergence (mirror) or the backup-should-never-be-ahead case; a mirror's
    own-branch divergence that *can* be resolved is fast-forwarded silently as
    a side effect of computing this, not reported. `other_warnings` maps
    writer_id -> warning for every *other* local writer branch this repo
    couldn't cleanly fast-forward (mirror only) — see docs/journal
    2026-09-16-mirror-other-writer-branch-convergence.md."""

    preamble: list[str]
    own_warning: str | None
    other_warnings: dict[str, str]


def _sync_remote(repo_root: Path, remote: str, remote_type: RemoteType) -> _RemoteSync:
    """Fetch `remote` and resync every local branch it can (own branch via
    `git merge --ff-only` on `HEAD`; every other local writer branch via
    `gitrepo.fast_forward_branch`, since that's the only mechanism that can
    move a branch that isn't checked out). Shared by `fetch_before_read`,
    `status_text`, and `branch_statuses` so all three see the exact same
    fetch+resync, not three slightly different reimplementations."""
    branch = current_branch(repo_root)
    preamble: list[str] = []

    pending = read_marker(repo_root, remote)
    if pending is not None:
        preamble.append(
            f"{remote}: {pending.commits} commit(s) still unpushed since "
            f"{pending.since} (last attempt: {pending.last_attempt_error})"
        )

    behavior = BEHAVIOR[remote_type]
    if not behavior.fetch:
        return _RemoteSync(preamble, None, {})

    try:
        fetch_all(repo_root, remote)
    except GitError as e:
        preamble.append(f"{remote}: {e}")
        return _RemoteSync(preamble, None, {})

    own_warning: str | None = None
    if behavior.check_own_branch:
        if remote_type == RemoteType.MIRROR:
            own_warning = _resync_branch(repo_root, remote, branch, mine=True)
        else:
            own_ref = f"refs/remotes/{remote}/{branch}"
            if ref_exists(repo_root, own_ref):
                ahead, behind = ahead_behind(repo_root, branch, own_ref)
                if behind > 0:
                    own_warning = (
                        f"{remote}: local branch {branch!r} has "
                        f"{_own_branch_divergence_text(branch, ahead, behind)} — "
                        "a backup remote should never be ahead of local"
                    )

    other_warnings: dict[str, str] = {}
    if behavior.check_other_branches:
        for other in local_branches(repo_root):
            if other == branch:
                continue
            warning = _resync_branch(repo_root, remote, other, mine=False)
            if warning:
                other_warnings[other.removeprefix("writer/")] = warning

    return _RemoteSync(preamble, own_warning, other_warnings)


def status_text(repo_root: Path, name: str, remote_type: RemoteType) -> str:
    """Human-readable status for one remote, for `sumac sources` (docs/journal
    2026-09-13-sumac-sources-design.md §3's STATUS column; honesty fix in
    2026-09-16-mirror-other-writer-branch-convergence.md §5). "Up to date"
    means my own branch is current AND every other writer's local branch
    (where one exists) resynced cleanly — not just my own, which is what this
    column meant before that fix."""
    if remote_type == RemoteType.UNSYNCED:
        return "unsynced"
    result = _sync_remote(repo_root, name, remote_type)
    if result.preamble:
        return "; ".join(result.preamble)
    # Every problem found gets surfaced — an own-branch divergence must never
    # silently hide a simultaneous other-writer resync failure, or vice versa
    # (a review caught this: an earlier version of this function returned
    # only `own_warning` whenever it was set, dropping `other_warnings`
    # entirely even when both were nonempty at once).
    parts = []
    if result.own_warning:
        parts.append(result.own_warning)
    if result.other_warnings:
        detail = "; ".join(f"writer/{wid} out of date" for wid in sorted(result.other_warnings))
        parts.append(f"partial ({detail})")
    if parts:
        return "; ".join(parts)
    return "up to date"


@dataclass(frozen=True, slots=True)
class BranchStatus:
    """One writer branch's state on one remote, for `sumac sources`'s per-branch
    view (docs/journal 2026-09-16-sources-per-branch-visibility-gap.md). `note`
    is `None` when there's nothing to flag — for `mine=False` this only ever
    means "this repo hasn't committed here by mistake, and could cleanly
    fast-forward if it had a local branch to resync," never a claim that the
    other writer's branch is itself healthy; this machine has no way to know
    that beyond what's already been pushed."""

    writer_id: str
    mine: bool
    commit_summary: str | None
    note: str | None


def branch_statuses(repo_root: Path, remote: str) -> list[BranchStatus]:
    """Every writer branch known on `remote` (local `writer/*` heads, unioned
    with that remote's own `refs/remotes/<remote>/writer/*` tracking refs) —
    the per-branch breakdown a single per-remote STATUS row can't represent.
    Resyncs the same way `status_text` does (mirror only; `backup` is
    single-owner by definition, so there's only ever one branch to show —
    keep using `status_text` for those)."""
    branch = current_branch(repo_root)
    local_ids = gitrepo.local_writer_ids(repo_root)
    writer_ids = sorted(local_ids | gitrepo.remote_writer_ids(repo_root, remote))
    sync = _sync_remote(repo_root, remote, RemoteType.MIRROR)

    rows = []
    for writer_id in writer_ids:
        their_branch = f"writer/{writer_id}"
        mine = their_branch == branch
        ref = f"refs/remotes/{remote}/{their_branch}"
        commit_summary = gitrepo.ref_summary(repo_root, ref)
        note = sync.own_warning if mine else sync.other_warnings.get(writer_id)
        rows.append(BranchStatus(writer_id, mine, commit_summary, note))
    return rows


def _preflight_messages(repo_root: Path, remotes: dict[str, RemoteType]) -> list[str]:
    """`run_preflight`'s warnings, formatted as `"<remote>: <message>"` — the one
    formatting rule shared by every non-raising caller of preflight in this
    module. Does not catch `DivergenceError`; callers decide whether that's fatal
    (`synced_commit`) or just another warning (`fetch_before_read`)."""
    warnings = run_preflight(repo_root, remotes)
    return [f"{w.remote}: {w.message}" for w in warnings]


def _own_branch_divergence_text(branch: str, ahead: int, behind: int) -> str:
    """Unambiguous ahead/behind wording for a branch's own-remote divergence.

    A real bug found 2026-09-16: an earlier version of this module (and, at the
    time, `remotectrl.preflight` itself — fixed there too, see that package's
    2026-09-16 journal entry) only ever mentioned `behind` in this message,
    silently dropping `ahead` even when both were nonzero — a real divergence
    (committed on the wrong branch locally, which is also missing the remote's
    commit) read as "just behind, will resolve on its own" when it actually
    needed manual resolution. Every caller in this module must go through this
    function rather than formatting `ahead`/`behind` inline, so a fix here
    covers every display site at once."""
    if ahead and behind:
        return f"diverged ({ahead} ahead, {behind} behind)"
    if behind:
        return f"{behind} commit(s) behind"
    return f"{ahead} commit(s) ahead"


def _resync_branch(repo_root: Path, remote: str, branch: str, *, mine: bool) -> str | None:
    """Fast-forward `branch` to `remote`'s copy of it, or return a warning if
    that isn't possible. Handles both roles a branch can have on a mirror:

    - `mine=True` (the currently-checked-out branch): being ahead is normal
      (that's just local work not yet pushed); being behind means this same
      writer identity pushed from another device (§2's household scenario) —
      safe to fast-forward via `git merge --ff-only` on `HEAD`.
    - `mine=False` (any other writer's branch): being ahead is the anomaly —
      this repo committed on a branch it doesn't own, per remotectrl-design.md
      §5 — and cannot be resolved by fast-forwarding (there's nothing to
      forward past a local-only commit). Being behind is the routine case a
      `mirror` is supposed to actually converge, not just tolerate — see
      docs/journal 2026-09-16-mirror-other-writer-branch-convergence.md; safe
      to fast-forward via `gitrepo.fast_forward_branch`, since this repo never
      writes to a branch it doesn't own, so there's no local history to
      protect against.

    A real ahead+behind divergence (either role) can never be resolved
    automatically — returns a warning, same wording either way
    (`_own_branch_divergence_text`)."""
    ref = f"refs/remotes/{remote}/{branch}"
    if not ref_exists(repo_root, ref):
        return None
    ahead, behind = ahead_behind(repo_root, branch, ref)
    if behind == 0:
        if not mine and ahead > 0:
            return (
                f"{remote}: local branch {branch!r} is {ahead} commit(s) "
                "ahead of a branch it doesn't own"
            )
        return None
    if ahead > 0:
        return (
            f"{remote}: local branch {branch!r} has "
            f"{_own_branch_divergence_text(branch, ahead, behind)} — "
            "cannot fast-forward, resolve manually"
        )
    try:
        if mine:
            gitrepo.fast_forward_to(repo_root, ref)
        else:
            gitrepo.fast_forward_branch(repo_root, branch, ref)
    except GitError as e:
        return f"{remote}: fast-forward of {branch!r} failed: {e}"
    return None


def fetch_before_read(repo_root: Path) -> list[str]:
    """Fetch and resync every configured remote before a read, per docs/journal
    2026-09-15-read-path-freshness.md §2-3 and
    2026-09-16-mirror-other-writer-branch-convergence.md. A mirror's own
    branch, and every other writer's *local* branch (where one exists), are
    fast-forwarded as a side effect of computing this — the fetch alone only
    ever updates a remote-tracking ref, never the local branches that older
    code (or a stale/testing clone) might still be reading from. Everything
    that can't be resolved automatically (a real divergence, a backup ahead of
    local, this repo having committed somewhere it doesn't own) still only
    warns here, never blocks: reading stale data is recoverable in a way
    committing on top of it is not."""
    remotes = resolve_remotes(repo_root)
    messages: list[str] = []
    for remote, remote_type in remotes.items():
        result = _sync_remote(repo_root, remote, remote_type)
        messages.extend(result.preamble)
        if result.own_warning:
            messages.append(result.own_warning)
        messages.extend(result.other_warnings.values())
    return messages


def write_remotes_config(repo_root: Path, assignments: dict[str, RemoteType]) -> None:
    """Overwrites `.rc/remotes.toml` with exactly `assignments` — the caller
    (`sumac sources --setup`) is expected to pass a role for every currently
    known git remote, not a partial update; there is no read-modify-write here
    because remotectrl's own config parser is private to that package (see
    docs/journal 2026-09-14-remotectrl-integration.md §4)."""
    rc_dir = repo_root / ".rc"
    rc_dir.mkdir(exist_ok=True)
    lines = ["[remotes]"]
    for name in sorted(assignments):
        lines.append(f'{name} = "{assignments[name].value}"')
    (rc_dir / "remotes.toml").write_text("\n".join(lines) + "\n")


def synced_commit(repo_root: Path, op: Callable[[], None]) -> str:
    """Preflight every configured remote, run `op` under the one-commit contract,
    then postflight-push to every configured remote. An empty `resolve_remotes`
    result (no `.rc/remotes.toml`, no git remotes) makes preflight/postflight
    no-ops, so a repo with no sync configured behaves exactly as before this
    wiring existed."""
    remotes = resolve_remotes(repo_root)

    try:
        warnings = _preflight_messages(repo_root, remotes)
    except DivergenceError as e:
        raise SyncDivergenceError(str(e)) from e
    for warning in warnings:
        render.print_warning(warning)

    commit = run_op(repo_root, op)

    try:
        run_postflight(repo_root, remotes)
    except PushError as e:
        raise SyncPushError(str(e)) from e

    return commit
