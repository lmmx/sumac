"""`sumac sources`. See docs/journal 2026-09-13-sumac-sources-design.md §3,
2026-09-14-remotectrl-integration.md §4.

Kept separate from test_cli.py (which imports `sumac.llm` at module scope, an
optional dependency not installed in every environment) — `cli.py` itself only
imports `llm` lazily, so this file avoids the CliRunner test being coupled to an
unrelated optional dependency.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sumac.cli import app

runner = CliRunner()
PASSPHRASE_ENV = {"SUMAC_PASSPHRASE": "test-pass"}


@pytest.fixture(autouse=True)
def _git_identity(git_env: None) -> None:
    pass


def _run(data_dir: Path, *args: str, input: str | None = None):
    return runner.invoke(app, [*args, "--data-dir", str(data_dir)], env=PASSPHRASE_ENV, input=input)


def _init(data_dir: Path) -> None:
    result = _run(data_dir, "init", "--writer", "alice-mac")
    assert result.exit_code == 0, result.output


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result


def test_no_remotes_prints_nothing_to_configure(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _init(data_dir)
    result = _run(data_dir, "sources")
    assert result.exit_code == 0, result.output
    assert "no git remotes configured" in result.output


def test_single_remote_auto_defaults_to_mirror(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _init(data_dir)
    repo_root = data_dir.parent
    _git(repo_root, "remote", "add", "origin", "/nonexistent/does-not-exist")

    result = _run(data_dir, "sources")
    assert result.exit_code == 0, result.output
    assert "defaulted to mirror" in result.output
    assert (repo_root / ".rc" / "remotes.toml").read_text() == '[remotes]\norigin = "mirror"\n'
    # tracked, committed (§2 of the companion design doc)
    subject = _git(repo_root, "log", "-1", "--format=%s").stdout.strip()
    assert subject == "sumac: configure remotes"


def test_listing_after_config_exists_does_not_reprompt(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _init(data_dir)
    repo_root = data_dir.parent
    _git(repo_root, "remote", "add", "origin", "/nonexistent/does-not-exist")
    _run(data_dir, "sources")  # first call: auto-configures

    result = _run(data_dir, "sources")  # second call: plain listing, no prompt
    assert result.exit_code == 0, result.output
    assert "origin" in result.output
    assert "mirror" in result.output


def test_status_surfaces_pending_push_marker(tmp_path: Path) -> None:
    """§3: STATUS doubles as the surfacing point for a prior failed push's
    unpushed-commits marker, so a stale backlog is never silently forgotten."""
    from remotectrl.markers import PendingPush, write_marker

    data_dir = tmp_path / "data"
    _init(data_dir)
    repo_root = data_dir.parent
    _git(repo_root, "remote", "add", "origin", "/nonexistent/does-not-exist")
    _run(data_dir, "sources")  # first call: auto-configures as mirror

    write_marker(
        repo_root,
        "origin",
        PendingPush(
            branch="writer/alice-mac",
            commits=3,
            since="2026-09-14T00:00:00+00:00",
            last_attempt_error="boom",
        ),
    )

    result = _run(data_dir, "sources")
    assert result.exit_code == 0, result.output
    assert "3 commit" in result.output


def test_setup_with_two_remotes_prompts_per_remote(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _init(data_dir)
    repo_root = data_dir.parent
    _git(repo_root, "remote", "add", "umbrel", "/nonexistent/umbrel")
    _git(repo_root, "remote", "add", "origin", "/nonexistent/origin")

    # prompt_ui.select reads one line per remote off stdin when not a tty —
    # answers in remote_names() order (alphabetical: origin, then umbrel).
    result = _run(data_dir, "sources", "--setup", input="backup\nmirror\n")
    assert result.exit_code == 0, result.output
    body = (repo_root / ".rc" / "remotes.toml").read_text()
    assert 'origin = "backup"' in body
    assert 'umbrel = "mirror"' in body
