"""Wires `remotectrl`'s preflight/postflight around sumac's one-commit-per-command
contract. See docs/journal 2026-09-13-remotectrl-design.md §5-§7,
2026-09-14-remotectrl-integration.md §1, §3.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from remotectrl import DivergenceError, PushError
from remotectrl.config import RemoteType, resolve
from remotectrl.onecommit import run_op
from remotectrl.postflight import run_postflight
from remotectrl.preflight import run_preflight

from sumac import render
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
        return f"diverged: {e}"
    if warnings:
        return "; ".join(w.message for w in warnings)
    return "up to date"


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
        warnings = run_preflight(repo_root, remotes)
    except DivergenceError as e:
        raise SyncDivergenceError(str(e)) from e
    for warning in warnings:
        render.print_warning(f"{warning.remote}: {warning.message}")

    commit = run_op(repo_root, op)

    try:
        run_postflight(repo_root, remotes)
    except PushError as e:
        raise SyncPushError(str(e)) from e

    return commit
