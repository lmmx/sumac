from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sumac import writer
from sumac.errors import NotAWriterBranchError


def test_slug_lowercases_and_collapses_runs() -> None:
    assert writer.slug("Louis's Laptop") == "louis-s-laptop"


def test_slug_strips_leading_and_trailing_hyphens() -> None:
    assert writer.slug("--bob--") == "bob"


def test_slug_empty_raises() -> None:
    with pytest.raises(ValueError):
        writer.slug("---")


def test_branch_and_id_from_branch_roundtrip() -> None:
    assert writer.branch("bob-linux") == "writer/bob-linux"
    assert writer.id_from_branch("writer/bob-linux") == "bob-linux"


def test_id_from_branch_none_for_non_writer_branch() -> None:
    assert writer.id_from_branch("master") is None
    assert writer.id_from_branch("main") is None


def test_stream_ids() -> None:
    assert writer.log_stream_id("bob") == "log:bob"
    assert writer.config_stream_id("bob") == "config:bob"


def test_current_id_env_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(writer.WRITER_ID_ENV, "alice-mac")
    assert writer.current_id(tmp_path) == "alice-mac"


def _git(cwd: Path, *args: str) -> None:
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    import os

    subprocess.run(["git", "-C", str(cwd), *args], check=True, env={**os.environ, **env})


def test_current_id_git_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(writer.WRITER_ID_ENV, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "vault.json").write_text("{}")
    _git(repo, "add", "vault.json")
    _git(repo, "commit", "-q", "-m", "root")
    _git(repo, "checkout", "-q", "-b", "writer/bob-linux")
    assert writer.current_id(repo) == "bob-linux"


def test_current_id_raises_off_writer_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(writer.WRITER_ID_ENV, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "vault.json").write_text("{}")
    _git(repo, "add", "vault.json")
    _git(repo, "commit", "-q", "-m", "root")
    with pytest.raises(NotAWriterBranchError):
        writer.current_id(repo)


def test_current_id_raises_outside_git_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(writer.WRITER_ID_ENV, raising=False)
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(NotAWriterBranchError):
        writer.current_id(plain)
