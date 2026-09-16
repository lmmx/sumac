"""`sumac sources`. See docs/journal 2026-09-13-sumac-sources-design.md §3,
2026-09-14-remotectrl-integration.md §4.

Kept separate from test_cli.py (which imports `sumac.llm` at module scope, an
optional dependency not installed in every environment) — `cli.py` itself only
imports `llm` lazily, so this file avoids the CliRunner test being coupled to an
unrelated optional dependency.
"""

from __future__ import annotations

import re
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


def test_find_warns_on_stale_remote_but_still_answers(tmp_path: Path) -> None:
    """docs/journal 2026-09-15-read-path-freshness.md §3: a read command fetches
    and warns on divergence but never blocks — unlike a write."""
    data_dir = tmp_path / "data"
    _init(data_dir)
    repo_root = data_dir.parent

    remote = tmp_path / "remote"
    _git(tmp_path, "clone", "-q", "-b", "writer/alice-mac", str(repo_root), str(remote))
    _git(remote, "commit", "--allow-empty", "-q", "-m", "someone else wrote here")
    _git(repo_root, "remote", "add", "origin", str(remote))
    (repo_root / ".rc").mkdir(exist_ok=True)
    (repo_root / ".rc" / "remotes.toml").write_text('[remotes]\norigin = "backup"\n')

    result = _run(data_dir, "find", "nonexistent-product")
    assert result.exit_code == 0, result.output
    assert "origin" in result.output
    assert "behind" in result.output


def test_status_warns_on_stale_remote_but_still_answers(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _init(data_dir)
    repo_root = data_dir.parent

    remote = tmp_path / "remote"
    _git(tmp_path, "clone", "-q", "-b", "writer/alice-mac", str(repo_root), str(remote))
    _git(remote, "commit", "--allow-empty", "-q", "-m", "someone else wrote here")
    _git(repo_root, "remote", "add", "origin", str(remote))
    (repo_root / ".rc").mkdir(exist_ok=True)
    (repo_root / ".rc" / "remotes.toml").write_text('[remotes]\norigin = "backup"\n')

    result = _run(data_dir, "status")
    assert result.exit_code == 0, result.output
    assert "origin" in result.output
    assert "behind" in result.output


def test_warn_staleness_is_silent_outside_a_git_repo(tmp_path: Path) -> None:
    """`cli._warn_staleness` must no-op, not error, when `data_dir`'s parent
    isn't a git repo at all — same convention every other remotectrl-aware
    command already follows (docs/journal 2026-09-15-read-path-freshness.md §3)."""
    from sumac.cli import _warn_staleness

    data_dir = tmp_path / "no-git-here" / "data"
    data_dir.mkdir(parents=True)
    _warn_staleness(data_dir)  # must not raise


def test_ask_calls_warn_staleness_before_loading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirms `ask` is wired to the same staleness check as `find`/`status`,
    without actually loading the LLM."""
    import sumac.cli as cli_module

    data_dir = tmp_path / "data"
    _init(data_dir)
    repo_root = data_dir.parent
    _git(repo_root, "remote", "add", "origin", "/nonexistent/does-not-exist")
    (repo_root / ".rc").mkdir(exist_ok=True)
    (repo_root / ".rc" / "remotes.toml").write_text('[remotes]\norigin = "mirror"\n')

    calls: list[Path] = []
    original = cli_module._warn_staleness

    def _spy(data_dir: Path) -> None:
        calls.append(data_dir)
        original(data_dir)

    monkeypatch.setattr(cli_module, "_warn_staleness", _spy)

    def _boom(*, verbose: bool = False):
        raise RuntimeError("llm should not load in this test")

    monkeypatch.setattr(cli_module, "_import_llm", _boom)

    result = _run(data_dir, "ask", "where is the pork?")
    assert calls == [data_dir]
    assert "origin" in result.output


def test_sources_reports_explicit_divergence_not_just_behind(tmp_path: Path) -> None:
    """A real bug found 2026-09-16: `sumac sources` displayed "1 commit(s) behind"
    for a branch that was actually 1 ahead AND 1 behind — a real divergence that
    needs manual resolution, misread as routine staleness that would "resolve on
    its own". See docs/journal 2026-09-16-ahead-behind-divergence-message-bug.md.
    """
    data_dir = tmp_path / "data"
    _init(data_dir)
    repo_root = data_dir.parent

    remote = tmp_path / "remote"
    _git(tmp_path, "clone", "-q", "-b", "writer/alice-mac", str(repo_root), str(remote))
    _git(remote, "config", "user.email", "test@example.invalid")
    _git(remote, "config", "user.name", "Test")
    _git(remote, "remote", "remove", "origin")
    _git(remote, "commit", "--allow-empty", "-q", "-m", "someone else's commit on this branch")

    _git(repo_root, "remote", "add", "umbrel", str(remote))
    _git(repo_root, "commit", "--allow-empty", "-q", "-m", "committed on the wrong branch")
    (repo_root / ".rc").mkdir(exist_ok=True)
    (repo_root / ".rc" / "remotes.toml").write_text('[remotes]\numbrel = "mirror"\n')

    result = _run(data_dir, "sources")
    assert result.exit_code == 0, result.output
    # Rich wraps the STATUS cell across lines/box-drawing chars at narrow
    # widths, so strip everything but word characters before matching.
    condensed = re.sub(r"[^\w]", "", result.output)
    assert "1ahead1behind" in condensed
    assert "diverged" in condensed
    # the bug: this used to say "1 commit(s) behind" alone, never mentioning
    # ahead at all — assert that exact wrong wording is gone, not just that the
    # right wording is present.
    assert "1commitsbehind" not in condensed


def test_sources_shows_per_branch_view_for_other_writers(tmp_path: Path) -> None:
    """docs/journal 2026-09-16-sources-per-branch-visibility-gap.md: the fetch
    always worked (traced and confirmed), but nothing ever displayed another
    writer's branch state — `sources` only ever reported on the current
    writer's own branch, collapsed into one row per remote. This is the exact
    scenario reported: on `writer/lm`, another writer (`cm`) pushes to their
    own branch on a mirror remote — `sources` must now show that branch's
    fresh state explicitly, not silently."""
    data_dir = tmp_path / "data"
    _init(data_dir)  # writer/alice-mac, per _init's fixed --writer
    repo_root = data_dir.parent

    remote = tmp_path / "remote"
    _git(tmp_path, "clone", "-q", "-b", "writer/alice-mac", str(repo_root), str(remote))
    _git(remote, "config", "user.email", "test@example.invalid")
    _git(remote, "config", "user.name", "Test")
    _git(remote, "remote", "remove", "origin")
    _git(remote, "config", "receive.denyCurrentBranch", "updateInstead")
    _git(remote, "branch", "writer/bob-laptop")

    # bob pushes new work to his own branch, from a separate clone
    bob = tmp_path / "bob"
    _git(tmp_path, "clone", "-q", "-b", "writer/bob-laptop", str(remote), str(bob))
    _git(bob, "config", "user.email", "test@example.invalid")
    _git(bob, "config", "user.name", "Test")
    (bob / "bob-work.txt").write_text("bob's inventory update\n")
    _git(bob, "add", "bob-work.txt")
    _git(bob, "commit", "-q", "-m", "bob's commit")
    _git(bob, "push", "-q", "origin", "writer/bob-laptop")
    bob_head = _git(bob, "rev-parse", "writer/bob-laptop").stdout.strip()

    _git(repo_root, "remote", "add", "umbrel", str(remote))
    (repo_root / ".rc").mkdir(exist_ok=True)
    (repo_root / ".rc" / "remotes.toml").write_text('[remotes]\numbrel = "mirror"\n')

    # before: alice's clone has never fetched bob's branch at all
    before = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--verify",
            "-q",
            "refs/remotes/umbrel/writer/bob-laptop",
        ],
        capture_output=True,
        text=True,
    )
    assert before.returncode != 0

    result = _run(data_dir, "sources")
    assert result.exit_code == 0, result.output
    assert "bob-laptop" in result.output
    assert "theirs" in result.output
    assert "alice-mac" in result.output
    assert "mine" in result.output

    # the fetch that sources triggered must have actually landed bob's commit
    fetched = _git(repo_root, "rev-parse", "refs/remotes/umbrel/writer/bob-laptop").stdout.strip()
    assert fetched == bob_head

    # the positive case matters as much as the failure case: another writer's
    # branch simply advancing, with a clean resync, must still read "up to
    # date" at the top level — a stub that always says "partial" whenever any
    # other writer branch exists (divergent or not) must not pass this.
    condensed = re.sub(r"[^\w]", "", result.output)
    assert "uptodate" in condensed
    assert "partial" not in condensed


def test_sources_status_shows_partial_when_other_writer_cannot_fast_forward(
    tmp_path: Path,
) -> None:
    """Case 4 (docs/journal 2026-09-16-mirror-other-writer-branch-convergence.md
    §5): the top-level STATUS column must not say "up to date" when another
    writer's local branch on this machine has diverged and couldn't be
    resynced — that's exactly the "technically true about a narrow thing, read
    as a broader claim" bug the ahead/behind fix already closed once tonight."""
    data_dir = tmp_path / "data"
    _init(data_dir)
    repo_root = data_dir.parent

    remote = tmp_path / "remote"
    _git(tmp_path, "clone", "-q", "-b", "writer/alice-mac", str(repo_root), str(remote))
    _git(remote, "config", "user.email", "test@example.invalid")
    _git(remote, "config", "user.name", "Test")
    _git(remote, "remote", "remove", "origin")
    _git(remote, "config", "receive.denyCurrentBranch", "updateInstead")
    _git(remote, "branch", "writer/bob-laptop")

    bob = tmp_path / "bob"
    _git(tmp_path, "clone", "-q", "-b", "writer/bob-laptop", str(remote), str(bob))
    _git(bob, "config", "user.email", "test@example.invalid")
    _git(bob, "config", "user.name", "Test")
    _git(bob, "commit", "--allow-empty", "-q", "-m", "bob's new work")
    _git(bob, "push", "-q", "origin", "writer/bob-laptop")

    # alice's machine already has a local, now-divergent copy of bob's branch
    _git(repo_root, "branch", "writer/bob-laptop", "writer/alice-mac")
    _git(repo_root, "checkout", "-q", "writer/bob-laptop")
    _git(repo_root, "commit", "--allow-empty", "-q", "-m", "divergent local commit")
    _git(repo_root, "checkout", "-q", "writer/alice-mac")

    _git(repo_root, "remote", "add", "umbrel", str(remote))
    (repo_root / ".rc").mkdir(exist_ok=True)
    (repo_root / ".rc" / "remotes.toml").write_text('[remotes]\numbrel = "mirror"\n')

    result = _run(data_dir, "sources")
    assert result.exit_code == 0, result.output
    condensed = re.sub(r"[^\w]", "", result.output)
    assert "uptodate" not in condensed
    assert "partial" in condensed
    assert "bob" in condensed and "laptop" in condensed


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
