"""remotectrl integration. See docs/journal 2026-09-14-remotectrl-integration.md.

`git_data_dir`'s repo is cloned (not independently `git init`'d) to build the
"remote" fixture in these tests, for the same reason remotectrl's own test suite
does this: two independently-created "identical content" commits do not reliably
share a hash (author/committer timestamps are part of it), so only a real clone
guarantees the shared ancestry `ahead_behind` needs to mean anything.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sumac import gitrepo, remote_sync, store
from sumac.errors import SyncDivergenceError, SyncPushError


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result


def _write_remotes_config(repo_root: Path, toml_body: str) -> None:
    rc_dir = repo_root / ".rc"
    rc_dir.mkdir(exist_ok=True)
    (rc_dir / "remotes.toml").write_text(toml_body)


def _clone_remote(repo_root: Path, tmp_path: Path, name: str = "remote") -> Path:
    remote = tmp_path / name
    _git(tmp_path, "clone", "-q", "-b", "writer/alice-mac", str(repo_root), str(remote))
    _git(remote, "config", "user.email", "test@example.invalid")
    _git(remote, "config", "user.name", "Test")
    _git(remote, "config", "receive.denyCurrentBranch", "updateInstead")
    _git(remote, "remote", "remove", "origin")
    return remote


def test_no_remotes_configured_is_a_noop(git_data_dir: Path, key: bytes) -> None:
    """No `.rc/remotes.toml`, no `git remote` — resolve_remotes is {}; already
    covered indirectly by test_store.py's commit_records tests, asserted here
    directly against remote_sync itself."""
    repo_root = git_data_dir.parent
    assert remote_sync.resolve_remotes(repo_root) == {}


def test_mirror_push_succeeds(git_data_dir: Path, key: bytes, tmp_path: Path) -> None:
    repo_root = git_data_dir.parent
    remote = _clone_remote(repo_root, tmp_path)
    _git(repo_root, "remote", "add", "origin", str(remote))
    _write_remotes_config(repo_root, '[remotes]\norigin = "mirror"\n')

    store.append(git_data_dir, key, "log:alice-mac", {"n": 1})
    store.commit_records(git_data_dir, 1)

    local_head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    remote_head = _git(remote, "rev-parse", "writer/alice-mac").stdout.strip()
    assert local_head == remote_head


def test_backup_ahead_of_local_blocks_with_sync_divergence_error(
    git_data_dir: Path, key: bytes, tmp_path: Path
) -> None:
    """§5: a backup remote ever being ahead of local is a real divergence —
    something else must have pushed there, which breaks single-owner."""
    repo_root = git_data_dir.parent
    remote = _clone_remote(repo_root, tmp_path)
    _git(remote, "commit", "--allow-empty", "-q", "-m", "someone else wrote here")
    _git(repo_root, "remote", "add", "origin", str(remote))
    _write_remotes_config(repo_root, '[remotes]\norigin = "backup"\n')

    store.append(git_data_dir, key, "log:alice-mac", {"n": 1})
    with pytest.raises(SyncDivergenceError):
        store.commit_records(git_data_dir, 1)


def test_push_failure_raises_sync_push_error_and_keeps_local_commit(
    git_data_dir: Path, key: bytes
) -> None:
    repo_root = git_data_dir.parent
    _git(repo_root, "remote", "add", "origin", "/nonexistent/does-not-exist")
    _write_remotes_config(repo_root, '[remotes]\norigin = "mirror"\n')

    store.append(git_data_dir, key, "log:alice-mac", {"n": 1})
    before = gitrepo.commits_touching(repo_root, "HEAD", ["data"])
    with pytest.raises(SyncPushError):
        store.commit_records(git_data_dir, 1)
    after = gitrepo.commits_touching(repo_root, "HEAD", ["data"])

    # The commit itself must never be rolled back on a push failure (§7).
    assert len(after) == len(before) + 1


def test_transport_failure_warns_but_does_not_block(
    git_data_dir: Path, key: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    """§5: transport failure (unreachable remote) is soft — warn and proceed,
    never a hard block, unlike a real ahead/behind divergence."""
    repo_root = git_data_dir.parent
    _git(repo_root, "remote", "add", "origin", "/nonexistent/does-not-exist")
    _write_remotes_config(repo_root, '[remotes]\norigin = "mirror"\n')

    store.append(git_data_dir, key, "log:alice-mac", {"n": 1})
    with pytest.raises(SyncPushError):
        # postflight still fails to push (nothing there to push to) — this test
        # only checks preflight did NOT hard-block on the unreachable fetch.
        store.commit_records(git_data_dir, 1)
    assert "origin" in capsys.readouterr().out


def test_multiple_remotes_one_fails_postflight_other_still_receives_push(
    git_data_dir: Path, key: bytes, tmp_path: Path
) -> None:
    """§7 (amended): every configured remote is attempted, not just the first —
    a failing remote must not prevent a later one from being pushed to."""
    repo_root = git_data_dir.parent
    good_remote = _clone_remote(repo_root, tmp_path, name="good-remote")
    _git(repo_root, "remote", "add", "bad", "/nonexistent/does-not-exist")
    _git(repo_root, "remote", "add", "good", str(good_remote))
    _write_remotes_config(repo_root, '[remotes]\nbad = "mirror"\ngood = "mirror"\n')

    store.append(git_data_dir, key, "log:alice-mac", {"n": 1})
    with pytest.raises(SyncPushError, match="bad"):
        store.commit_records(git_data_dir, 1)

    local_head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    good_head = _git(good_remote, "rev-parse", "writer/alice-mac").stdout.strip()
    assert local_head == good_head


def test_fetch_before_read_is_a_noop_with_no_remotes(git_data_dir: Path) -> None:
    repo_root = git_data_dir.parent
    assert remote_sync.fetch_before_read(repo_root) == []


def test_fetch_before_read_warns_but_never_raises_on_divergence(
    git_data_dir: Path, tmp_path: Path
) -> None:
    """docs/journal 2026-09-15-read-path-freshness.md §2: a real ahead/behind
    divergence that would block a write must only ever warn for a read."""
    repo_root = git_data_dir.parent
    remote = _clone_remote(repo_root, tmp_path)
    _git(remote, "commit", "--allow-empty", "-q", "-m", "someone else wrote here")
    _git(repo_root, "remote", "add", "origin", str(remote))
    _write_remotes_config(repo_root, '[remotes]\norigin = "backup"\n')

    warnings = remote_sync.fetch_before_read(repo_root)
    assert len(warnings) == 1
    assert "origin" in warnings[0]
    assert "behind" in warnings[0]


def test_fetch_before_read_multiple_remotes_only_reports_first_divergence(
    git_data_dir: Path, tmp_path: Path
) -> None:
    """Known limitation, docs/journal 2026-09-15-read-path-freshness.md §4:
    `run_preflight` raises on the first divergence and checks no further remote,
    so a second diverged remote goes unreported by this call alone."""
    repo_root = git_data_dir.parent
    bad_a = _clone_remote(repo_root, tmp_path, name="bad-a")
    _git(bad_a, "commit", "--allow-empty", "-q", "-m", "diverged a")
    bad_b = _clone_remote(repo_root, tmp_path, name="bad-b")
    _git(bad_b, "commit", "--allow-empty", "-q", "-m", "diverged b")
    _git(repo_root, "remote", "add", "bad-a", str(bad_a))
    _git(repo_root, "remote", "add", "bad-b", str(bad_b))
    _write_remotes_config(repo_root, '[remotes]\nbad-a = "backup"\nbad-b = "backup"\n')

    warnings = remote_sync.fetch_before_read(repo_root)
    assert len(warnings) == 1


def test_fetch_before_read_warns_on_transport_failure(git_data_dir: Path) -> None:
    repo_root = git_data_dir.parent
    _git(repo_root, "remote", "add", "origin", "/nonexistent/does-not-exist")
    _write_remotes_config(repo_root, '[remotes]\norigin = "mirror"\n')

    warnings = remote_sync.fetch_before_read(repo_root)
    assert len(warnings) == 1
    assert "origin" in warnings[0]


def test_remote_names_lists_configured_remotes(git_data_dir: Path) -> None:
    repo_root = git_data_dir.parent
    assert gitrepo.remote_names(repo_root) == []
    _git(repo_root, "remote", "add", "origin", "/nonexistent/does-not-exist")
    assert gitrepo.remote_names(repo_root) == ["origin"]


def test_remote_url_reads_configured_url(git_data_dir: Path) -> None:
    repo_root = git_data_dir.parent
    _git(repo_root, "remote", "add", "origin", "/some/path")
    assert gitrepo.remote_url(repo_root, "origin") == "/some/path"


def test_remote_url_returns_none_for_unknown_remote(git_data_dir: Path) -> None:
    repo_root = git_data_dir.parent
    assert gitrepo.remote_url(repo_root, "nonexistent") is None
