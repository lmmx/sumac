"""Wires `remotectrl`'s preflight/postflight around sumac's one-commit-per-command
contract. See docs/journal 2026-09-13-remotectrl-design.md §5-§7,
2026-09-14-remotectrl-integration.md §1, §3.
"""

from __future__ import annotations

from collections.abc import Callable
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


def status_text(repo_root: Path, name: str, remote_type: RemoteType) -> str:
    """Human-readable status for one remote, for `sumac sources` (docs/journal
    2026-09-13-sumac-sources-design.md §3's STATUS column). Runs preflight for
    just this one remote so one remote's divergence doesn't stop the rest of a
    multi-remote listing from being checked."""
    if remote_type == RemoteType.UNSYNCED:
        return "unsynced"
    try:
        warnings = run_preflight(repo_root, {name: remote_type})
    except DivergenceError as e:
        # `e`'s own message already says "diverged (...)" when ahead and behind
        # are both nonzero (fixed 2026-09-16 — see this module's
        # `_own_branch_divergence_text`, and remotectrl's own preflight.py fix),
        # so this no longer prefixes a second, redundant "diverged:".
        return str(e)
    if warnings:
        return "; ".join(w.message for w in warnings)
    return "up to date"


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


def _resync_own_branch(repo_root: Path, remote: str, branch: str) -> str | None:
    """A mirror's own-branch divergence means this same writer identity pushed
    from another device (§2 of the household scenario: one writer, possibly
    several machines) — safe to fast-forward, unlike another writer's branch
    (never merged, per remotectrl-design.md §7's "no local fast-forward/merge of
    other writers' branches" — a rule scoped to *other* branches, silent on a
    writer's own). Returns `None` on a successful resync (or nothing to do), or
    a warning message if fast-forward wasn't possible."""
    ref = f"refs/remotes/{remote}/{branch}"
    if not ref_exists(repo_root, ref):
        return None
    ahead, behind = ahead_behind(repo_root, branch, ref)
    if behind == 0:
        return None
    if ahead > 0:
        return (
            f"{remote}: local branch {branch!r} has "
            f"{_own_branch_divergence_text(branch, ahead, behind)} — "
            "cannot fast-forward, resolve manually"
        )
    try:
        gitrepo.fast_forward_to(repo_root, ref)
    except GitError as e:
        return f"{remote}: fast-forward of {branch!r} failed: {e}"
    return None


def fetch_before_read(repo_root: Path) -> list[str]:
    """Fetch every configured remote before a read, per docs/journal
    2026-09-15-read-path-freshness.md §2-3. A mirror's own-branch divergence is
    actually resynced (fast-forwarded) rather than just reported, since the
    fetch that already ran leaves the fresh data sitting in a remote-tracking
    ref — the read path's own commit and worktree are what's stale, and that's
    fixable, not just something to warn about. Everything else that would block
    a write (a backup remote ahead of local, or a mirror's *other*-branch ahead
    of a branch it doesn't own — never auto-merged, see `_resync_own_branch`)
    still only warns here, never blocks: reading stale data is recoverable in a
    way committing on top of it is not."""
    remotes = resolve_remotes(repo_root)
    messages: list[str] = []
    branch = current_branch(repo_root)

    for remote, remote_type in remotes.items():
        pending = read_marker(repo_root, remote)
        if pending is not None:
            messages.append(
                f"{remote}: {pending.commits} commit(s) still unpushed since "
                f"{pending.since} (last attempt: {pending.last_attempt_error})"
            )

        behavior = BEHAVIOR[remote_type]
        if not behavior.fetch:
            continue

        try:
            fetch_all(repo_root, remote)
        except GitError as e:
            messages.append(f"{remote}: {e}")
            continue

        if behavior.check_own_branch:
            if remote_type == RemoteType.MIRROR:
                warning = _resync_own_branch(repo_root, remote, branch)
                if warning:
                    messages.append(warning)
            else:
                own_ref = f"refs/remotes/{remote}/{branch}"
                if ref_exists(repo_root, own_ref):
                    ahead, behind = ahead_behind(repo_root, branch, own_ref)
                    if behind > 0:
                        messages.append(
                            f"{remote}: local branch {branch!r} has "
                            f"{_own_branch_divergence_text(branch, ahead, behind)} — "
                            "a backup remote should never be ahead of local"
                        )

        if behavior.check_other_branches:
            for other in local_branches(repo_root):
                if other == branch:
                    continue
                other_ref = f"refs/remotes/{remote}/{other}"
                if not ref_exists(repo_root, other_ref):
                    continue
                ahead, _ = ahead_behind(repo_root, other, other_ref)
                if ahead > 0:
                    messages.append(
                        f"{remote}: local branch {other!r} is {ahead} commit(s) "
                        "ahead of a branch it doesn't own"
                    )

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
