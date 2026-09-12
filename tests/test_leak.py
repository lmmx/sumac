"""Walks a populated data dir and asserts nothing about locations or products leaks
through path names, file contents, or git metadata (refs, commit messages)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sumac import paths
from sumac.cli import app

runner = CliRunner()
ENV = {"SUMAC_PASSPHRASE": "test-pass"}

SECRETS = ["pantry", "fridge", "wine-cellar", "milk", "cabernet"]


@pytest.fixture(autouse=True)
def _writer_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMAC_WRITER_ID", "alice-mac")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")


def _run(data_dir: Path, *args: str):
    result = runner.invoke(app, [*args, "--data-dir", str(data_dir)], env=ENV)
    assert result.exit_code == 0, result.output
    return result


def _populate(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-location", "Wine Cellar", "--id", "wine-cellar")
    _run(data_dir, "add", "purchase", "milk", "2", "l", "--to", "fridge")
    _run(
        data_dir,
        "add",
        "movement",
        "cabernet",
        "3",
        "bottle",
        "--from",
        "wine-cellar",
        "--to",
        "pantry",
    )
    _run(data_dir, "snapshot", "fridge", "milk=1/l")


ALLOWED_PATH_COMPONENTS = {
    "data",
    "vault.json",
    "config.jsonl.enc",
    "log.jsonl.enc",
}


def test_path_components_are_fixed_literals_or_usernames(data_dir: Path) -> None:
    _populate(data_dir)
    for path in data_dir.rglob("*"):
        rel = path.relative_to(data_dir)
        for part in rel.parts:
            assert part in ALLOWED_PATH_COMPONENTS, f"leaky path component: {part!r} in {path}"


def test_file_contents_do_not_contain_secrets(data_dir: Path) -> None:
    _populate(data_dir)
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        for secret in SECRETS:
            assert secret.encode() not in raw, f"{secret!r} leaked in {path}"


def test_vault_json_has_no_ciphertext_correlated_names(data_dir: Path) -> None:
    _populate(data_dir)
    vault_text = paths.vault_path(data_dir).read_text()
    for secret in SECRETS:
        assert secret not in vault_text


# `SUMAC_WRITER_ID` (set by `_writer_id` above) puts every command in filesystem
# mode (§2 step 1), so `_populate` alone creates no git repo — these tests need
# their own repo, without that env var, to exercise `init`'s git path.


COMMIT_SUBJECT_RE = re.compile(r"^sumac: (\d+ records?|root)$")


def _populate_git(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUMAC_WRITER_ID", raising=False)
    _populate(data_dir)


def _git_log(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True
    )
    return result.stdout


def test_git_ref_names_leak_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_data_dir = tmp_path / "repo" / "data"
    _populate_git(repo_data_dir, monkeypatch)
    repo_root = repo_data_dir.parent

    refs = _git_log(repo_root, "for-each-ref", "--format=%(refname)")
    for secret in SECRETS:
        assert secret not in refs, f"{secret!r} leaked in a ref name"


def test_commit_messages_leak_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_data_dir = tmp_path / "repo" / "data"
    _populate_git(repo_data_dir, monkeypatch)
    repo_root = repo_data_dir.parent

    log = _git_log(repo_root, "log", "--all", "--format=%s")
    subjects = [line for line in log.splitlines() if line]
    assert subjects, "expected at least one commit"
    for subject in subjects:
        assert COMMIT_SUBJECT_RE.match(subject), f"unexpected commit subject: {subject!r}"
        for secret in SECRETS:
            assert secret not in subject, f"{secret!r} leaked in commit subject {subject!r}"


def test_commit_tree_paths_leak_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_data_dir = tmp_path / "repo" / "data"
    _populate_git(repo_data_dir, monkeypatch)
    repo_root = repo_data_dir.parent

    paths_touched = _git_log(repo_root, "log", "--all", "--name-only", "--format=")
    for secret in SECRETS:
        assert secret not in paths_touched, f"{secret!r} leaked in a commit-tree path"
