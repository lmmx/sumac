from __future__ import annotations

from pathlib import Path

from sealedlog import aead
from sealedlog._aad import build_aad

from sumac import NAMESPACE, gitrepo, paths, store, writer
from tests.conftest import seed_writer


def test_append_and_iter_round_trip(data_dir: Path, writer_id: str, key: bytes) -> None:
    stream = writer.log_stream_id(writer_id)
    store.append(data_dir, key, stream, {"n": 1})
    store.append(data_dir, key, stream, {"n": 2})
    assert list(store.iter_stream(data_dir, key, stream)) == [
        {"n": 1, "seq": 0},
        {"n": 2, "seq": 1},
    ]


def test_append_assigns_monotone_seq_per_stream(data_dir: Path, writer_id: str, key: bytes) -> None:
    """docs/journal §3.7: seq is append-time envelope data, assigned by
    `store.append` itself (not the caller), monotone starting at 0, and
    independent per log stream."""
    stream = writer.log_stream_id(writer_id)
    for i in range(3):
        store.append(data_dir, key, stream, {"n": i})
    seqs = [obj["seq"] for obj in store.iter_stream(data_dir, key, stream)]
    assert seqs == [0, 1, 2]


def test_append_ignores_a_caller_supplied_seq(data_dir: Path, writer_id: str, key: bytes) -> None:
    """seq is append's job, not the caller's — a stray `seq` key on the
    object passed in must never survive; `store.append` is the sole source
    of truth for what gets written."""
    stream = writer.log_stream_id(writer_id)
    store.append(data_dir, key, stream, {"n": 1, "seq": 999})
    [obj] = list(store.iter_stream(data_dir, key, stream))
    assert obj["seq"] == 0


def test_config_stream_never_gets_a_seq(data_dir: Path, writer_id: str, key: bytes) -> None:
    """Config is latest-revision-wins, not an append-sequential segment —
    seq assignment is scoped to `log:`-prefixed streams only."""
    stream = writer.config_stream_id(writer_id)
    store.append(data_dir, key, stream, {"loc": "fridge"})
    [obj] = list(store.iter_stream(data_dir, key, stream))
    assert "seq" not in obj


def test_append_is_byte_append(data_dir: Path, writer_id: str, key: bytes) -> None:
    stream = writer.log_stream_id(writer_id)
    store.append(data_dir, key, stream, {"n": 1})
    path = paths.log_path(data_dir)
    before = path.read_bytes()
    store.append(data_dir, key, stream, {"n": 2})
    after = path.read_bytes()
    assert after.startswith(before)


def test_config_stream_unrestricted(data_dir: Path, writer_id: str, key: bytes) -> None:
    stream = writer.config_stream_id(writer_id)
    store.append(data_dir, key, stream, {"loc": "fridge"})
    assert list(store.iter_stream(data_dir, key, stream)) == [{"loc": "fridge"}]


def test_verify_stream_reports_line_and_position(
    data_dir: Path, writer_id: str, key: bytes
) -> None:
    path = paths.log_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-valid-base64!!!\n")

    ok, failures = store.verify_stream(path, key, writer.log_stream_id(writer_id))
    assert ok == []
    assert len(failures) == 1
    assert failures[0].lineno == 1


def test_verify_stream_reports_authenticated_non_json_line(
    data_dir: Path, writer_id: str, key: bytes
) -> None:
    """A line that authenticates but whose plaintext isn't JSON must be a LineFailure,
    not an uncaught JSONDecodeError — this is what makes verify_stream (and doctor,
    which relies on it) genuinely tolerant rather than tolerant-except-for-this."""
    path = paths.log_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = writer.log_stream_id(writer_id)
    aad = build_aad(NAMESPACE, stream)
    sealed = aead.seal(key, aad, b"not json{")
    path.write_text(sealed + "\n")

    ok, failures = store.verify_stream(path, key, stream)
    assert ok == []
    assert len(failures) == 1
    assert "JSON" in failures[0].error


def test_iter_all_logs_spans_users(git_data_dir: Path, key: bytes) -> None:
    """Multi-writer aggregation through real refs (§2), not a faked directory."""
    repo_root = git_data_dir.parent

    def alice_appends(data_dir: Path) -> None:
        store.append(data_dir, key, "log:alice-mac", {"n": 1})

    def bob_appends(data_dir: Path) -> None:
        store.append(data_dir, key, "log:bob-linux", {"n": 2})

    seed_writer(repo_root, key, "alice-mac", alice_appends)
    seed_writer(repo_root, key, "bob-linux", bob_appends)

    seen = sorted(store.iter_all_logs(git_data_dir, key))
    assert seen == [("alice-mac", {"n": 1, "seq": 0}), ("bob-linux", {"n": 2, "seq": 0})]


def test_assigned_seqs_backfills_position_for_records_without_a_stored_seq() -> None:
    """docs/journal §3.7: a record with no stored `seq` (everything written
    before Phase 7) is treated as having whatever position it holds among
    the decoded objects — the same backfill the v1 upcaster already relies
    on for old data."""
    objs = [{"n": 1}, {"n": 2, "seq": 5}, {"n": 3}]
    assert store.assigned_seqs(objs) == [0, 5, 2]


def test_commit_records_is_noop_in_filesystem_mode(
    data_dir: Path, writer_id: str, key: bytes
) -> None:
    store.append(data_dir, key, writer.log_stream_id(writer_id), {"n": 1})
    store.commit_records(data_dir, 1)  # must not raise absent a git repo


def test_commit_records_creates_one_commit_in_git_mode(git_data_dir: Path, key: bytes) -> None:
    repo_root = git_data_dir.parent
    store.append(git_data_dir, key, "log:alice-mac", {"n": 1})
    store.append(git_data_dir, key, "log:alice-mac", {"n": 2})

    before = gitrepo.commits_touching(repo_root, "HEAD", ["data"])
    store.commit_records(git_data_dir, 2)
    after = gitrepo.commits_touching(repo_root, "HEAD", ["data"])

    assert len(after) == len(before) + 1
    subject = _last_commit_subject(repo_root)
    assert subject == "sumac: 2 records"


def test_commit_records_singular_for_one_record(git_data_dir: Path, key: bytes) -> None:
    repo_root = git_data_dir.parent
    store.append(git_data_dir, key, "log:alice-mac", {"n": 1})
    store.commit_records(git_data_dir, 1)
    assert _last_commit_subject(repo_root) == "sumac: 1 record"


def _last_commit_subject(repo_root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo_root), "log", "-1", "--format=%s"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
