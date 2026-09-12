from __future__ import annotations

import json
from pathlib import Path

from sealedlog import aead
from sealedlog._aad import build_aad

from sumac import NAMESPACE, gitrepo, ledger, paths, store, writer
from tests.conftest import seed_writer


def _seal_line(key: bytes, stream_id: str, obj: dict) -> str:
    aad = build_aad(NAMESPACE, stream_id)
    return aead.seal(key, aad, json.dumps(obj).encode("utf-8"))


def test_clean_append_chain_verifies(git_data_dir: Path, key: bytes) -> None:
    """A legitimate append-and-commit history over several commits (§4)."""
    stream = writer.log_stream_id("alice-mac")
    for i in range(3):
        store.append(git_data_dir, key, stream, {"n": i})
        store.commit_records(git_data_dir, 1)

    assert ledger.verify_history(git_data_dir, key) == []


def test_rewritten_line_at_tip_is_a_violation(git_data_dir: Path, key: bytes) -> None:
    """Re-sealing a *different* record at an existing position violates §4 even
    though the git history is a fast-forward chain."""
    repo_root = git_data_dir.parent
    stream = writer.log_stream_id("alice-mac")
    store.append(git_data_dir, key, stream, {"n": 0})
    store.commit_records(git_data_dir, 1)
    store.append(git_data_dir, key, stream, {"n": 1})
    store.commit_records(git_data_dir, 1)

    # Rewrite the first line in place with different decoded content.
    log_path = paths.log_path(git_data_dir)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    lines[0] = _seal_line(key, stream, {"n": 999, "seq": 0})
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    gitrepo.commit_paths(repo_root, ["data"], "sumac: rewrite")

    violations = ledger.verify_history(git_data_dir, key)
    assert len(violations) == 1
    assert violations[0].writer_id == "alice-mac"
    assert violations[0].stream_id == stream


def test_truncated_log_is_a_violation(git_data_dir: Path, key: bytes) -> None:
    """Removing records shortens the decoded sequence — not a valid prefix
    relationship in either direction."""
    repo_root = git_data_dir.parent
    stream = writer.log_stream_id("alice-mac")
    store.append(git_data_dir, key, stream, {"n": 0})
    store.commit_records(git_data_dir, 1)
    store.append(git_data_dir, key, stream, {"n": 1})
    store.commit_records(git_data_dir, 1)

    log_path = paths.log_path(git_data_dir)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    log_path.write_text(lines[0] + "\n", encoding="utf-8")  # drop the second record
    gitrepo.commit_paths(repo_root, ["data"], "sumac: truncate")

    violations = ledger.verify_history(git_data_dir, key)
    assert len(violations) == 1
    assert violations[0].writer_id == "alice-mac"


def test_reseal_preserving_decoded_records_is_not_a_violation(
    git_data_dir: Path, key: bytes
) -> None:
    """§4: re-sealing changes ciphertext but not the decoded record sequence,
    so it must not be flagged."""
    repo_root = git_data_dir.parent
    stream = writer.log_stream_id("alice-mac")
    store.append(git_data_dir, key, stream, {"n": 0})
    store.commit_records(git_data_dir, 1)
    store.append(git_data_dir, key, stream, {"n": 1})
    store.commit_records(git_data_dir, 1)

    log_path = paths.log_path(git_data_dir)
    objs, _failures = store.verify_stream(log_path, key, stream)
    resealed = "\n".join(_seal_line(key, stream, obj) for obj in objs) + "\n"
    before = log_path.read_bytes()
    log_path.write_text(resealed, encoding="utf-8")
    assert log_path.read_bytes() != before  # fresh nonces really do change ciphertext
    gitrepo.commit_paths(repo_root, ["data"], "sumac: reseal")

    assert ledger.verify_history(git_data_dir, key) == []


def test_violation_on_second_writer_is_attributed_to_them(git_data_dir: Path, key: bytes) -> None:
    repo_root = git_data_dir.parent

    def bob_appends(data_dir: Path) -> None:
        store.append(data_dir, key, "log:bob-linux", {"n": 0})

    seed_writer(repo_root, key, "bob-linux", bob_appends)

    bob_branch = writer.branch("bob-linux")
    gitrepo.create_branch_from(repo_root, "bob-worktree", bob_branch, checkout=True)
    try:
        bob_stream = writer.log_stream_id("bob-linux")
        log_path = paths.log_path(git_data_dir)
        log_path.write_text(
            _seal_line(key, bob_stream, {"n": 999, "seq": 0}) + "\n", encoding="utf-8"
        )
        gitrepo.commit_paths(repo_root, ["data"], "sumac: bob rewrite")
        gitrepo._run(repo_root, ["update-ref", f"refs/heads/{bob_branch}", "HEAD"])
    finally:
        gitrepo._run(repo_root, ["checkout", "-q", "writer/alice-mac"])
        gitrepo._run(repo_root, ["branch", "-D", "bob-worktree"])

    violations = ledger.verify_history(git_data_dir, key)
    assert len(violations) == 1
    assert violations[0].writer_id == "bob-linux"


def test_filesystem_mode_has_no_history(data_dir: Path, writer_id: str, key: bytes) -> None:
    store.append(data_dir, key, writer.log_stream_id(writer_id), {"n": 1})
    assert ledger.verify_history(data_dir, key) == []
