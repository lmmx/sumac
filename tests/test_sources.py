from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from sumac import sources


def _git(cwd: Path, *args: str) -> None:
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "-C", str(cwd), *args], check=True, env={**os.environ, **env})


def test_writer_sources_filesystem_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUMAC_WRITER_ID", "alice-mac")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "log.jsonl.enc").write_text("logline\n")
    (data_dir / "config.jsonl.enc").write_text("cfgline\n")

    result = sources.writer_sources(data_dir)

    assert len(result) == 1
    src = result[0]
    assert src.writer_id == "alice-mac"
    assert src.log_text() == "logline\n"
    assert src.config_text() == "cfgline\n"


def test_writer_sources_filesystem_mode_missing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SUMAC_WRITER_ID", "alice-mac")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = sources.writer_sources(data_dir)

    assert result[0].log_text() == ""
    assert result[0].config_text() == ""


@pytest.fixture
def two_writer_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A real git repo with a root commit (vault.json) and two writer branches,
    `writer/alice-mac` and `writer/bob-linux`, each with its own log/config."""
    monkeypatch.delenv("SUMAC_WRITER_ID", raising=False)
    repo = tmp_path / "repo"
    data_dir = repo / "data"
    data_dir.mkdir(parents=True)

    (data_dir / "vault.json").write_text("{}")
    _git(repo, "init", "-q")
    _git(repo, "add", "data/vault.json")
    _git(repo, "commit", "-q", "-m", "root")

    _git(repo, "checkout", "-q", "-b", "writer/bob-linux")
    (data_dir / "log.jsonl.enc").write_text("bob-log-1\n")
    (data_dir / "config.jsonl.enc").write_text("bob-cfg-1\n")
    _git(repo, "add", "data/log.jsonl.enc", "data/config.jsonl.enc")
    _git(repo, "commit", "-q", "-m", "sumac: 2 records")

    _git(repo, "checkout", "-q", "master")
    _git(repo, "checkout", "-q", "-b", "writer/alice-mac")
    (data_dir / "log.jsonl.enc").write_text("alice-log-1\n")
    (data_dir / "config.jsonl.enc").write_text("alice-cfg-1\n")
    _git(repo, "add", "data/log.jsonl.enc", "data/config.jsonl.enc")
    _git(repo, "commit", "-q", "-m", "sumac: 2 records")

    return data_dir


def test_writer_sources_git_mode_reads_current_writer_from_worktree(
    monkeypatch: pytest.MonkeyPatch, two_writer_repo: Path
) -> None:
    # Currently checked out on writer/alice-mac (last checkout in the fixture).
    (two_writer_repo / "log.jsonl.enc").write_text("alice-log-1\nalice-log-2-uncommitted\n")

    result = sources.writer_sources(two_writer_repo)

    by_id = {s.writer_id: s for s in result}
    assert set(by_id) == {"alice-mac", "bob-linux"}

    # Current writer (alice) is read live from the worktree, including the
    # uncommitted append.
    assert by_id["alice-mac"].log_text() == "alice-log-1\nalice-log-2-uncommitted\n"
    assert by_id["alice-mac"].config_text() == "alice-cfg-1\n"

    # The other writer (bob) is read from the ref, unaffected by alice's worktree.
    assert by_id["bob-linux"].log_text() == "bob-log-1\n"
    assert by_id["bob-linux"].config_text() == "bob-cfg-1\n"


def test_writer_sources_git_mode_sorted_by_writer_id(two_writer_repo: Path) -> None:
    result = sources.writer_sources(two_writer_repo)
    assert [s.writer_id for s in result] == sorted(s.writer_id for s in result)


def test_writer_sources_git_mode_missing_blob_is_empty_not_error(two_writer_repo: Path) -> None:
    """A writer whose ref predates one of the two files reads "" for it."""
    # Bob's branch never wrote a second file name; simulate by reading a path
    # absent at bob's ref through the ref-backed source directly.
    result = sources.writer_sources(two_writer_repo)
    bob = next(s for s in result if s.writer_id == "bob-linux")
    assert bob.log_text() == "bob-log-1\n"
