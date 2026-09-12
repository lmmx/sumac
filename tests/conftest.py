from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import nacl.pwhash.argon2id as argon2id
import pytest
from sealedlog import Vault

from sumac import FORMAT_VERSION, passphrase, writer
from sumac import vault as sumac_vault


@pytest.fixture(autouse=True)
def _fast_kdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Argon2id at MIN cost so tests don't pay real KDF latency."""
    original = sumac_vault.create

    def fast_create(
        pw: str,
        *,
        opslimit: int = argon2id.OPSLIMIT_MIN,
        memlimit: int = argon2id.MEMLIMIT_MIN,
    ) -> Vault:
        return original(pw, opslimit=opslimit, memlimit=memlimit)

    monkeypatch.setattr(sumac_vault, "create", fast_create)


@pytest.fixture(autouse=True)
def _reset_key_cache() -> Iterator[None]:
    passphrase.reset_cache()
    yield
    passphrase.reset_cache()


@pytest.fixture
def writer_id(monkeypatch: pytest.MonkeyPatch) -> str:
    """Filesystem mode (§2 step 1): single writer, no git calls."""
    monkeypatch.setenv(writer.WRITER_ID_ENV, "alice-mac")
    return "alice-mac"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def key(_fast_kdf: None) -> bytes:
    vault = sumac_vault.create("pw")
    return sumac_vault.unlock(vault, "pw")


@pytest.fixture
def git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repo under `tmp_path` has no git identity of its own; without this,
    commits fail with "Author identity unknown"."""
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def git_data_dir(tmp_path: Path, git_env: None, key: bytes) -> Path:
    """A real single-writer repo: `writer/alice-mac` checked out, root commit
    holding only `data/vault.json`. Does not set `SUMAC_WRITER_ID` — tests using
    this exercise git mode via the branch name (§2)."""
    repo_root = tmp_path / "repo"
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True)

    vault = sumac_vault.create("pw")
    doc = {"format_version": FORMAT_VERSION, **vault.to_dict()}
    (data_dir / "vault.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    _git(repo_root, "init", "-q", "-b", "writer/alice-mac")
    _git(repo_root, "add", "data/vault.json")
    _git(repo_root, "commit", "-q", "-m", "sumac: root")
    return data_dir


def seed_writer(
    repo_root: Path, key: bytes, writer_id: str, appends: Callable[[Path], None]
) -> None:
    """Branches `writer/<writer_id>` from the repo's root commit, checks it out,
    runs `appends(data_dir)` with `SUMAC_WRITER_ID` set so records seal under
    that writer's stream ids, commits, then restores the original branch (§2)."""
    from sumac import gitrepo

    original_branch = gitrepo.current_branch(repo_root)
    root = gitrepo.root_commit(repo_root)
    branch_name = writer.branch(writer_id)
    gitrepo.create_branch_from(repo_root, branch_name, root, checkout=True)

    data_dir = repo_root / "data"
    prior_env = os.environ.get(writer.WRITER_ID_ENV)
    os.environ[writer.WRITER_ID_ENV] = writer_id
    try:
        appends(data_dir)
    finally:
        if prior_env is None:
            os.environ.pop(writer.WRITER_ID_ENV, None)
        else:
            os.environ[writer.WRITER_ID_ENV] = prior_env

    rel_data = gitrepo.rel_to_toplevel(data_dir)
    _git(repo_root, "add", rel_data.as_posix())
    _git(repo_root, "commit", "-q", "-m", "sumac: seed")

    if original_branch is not None:
        _git(repo_root, "checkout", "-q", original_branch)
