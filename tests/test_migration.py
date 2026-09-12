"""scripts/migrate-branch-per-writer.py against a synthetic legacy repo. See
docs/journal/2026-09-11-branch-per-user-design.md §6.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sealedlog import Vault, aead
from sealedlog._aad import build_aad

from sumac import FORMAT_VERSION, NAMESPACE
from sumac import vault as sumac_vault

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "migrate-branch-per-writer.py"
_spec = importlib.util.spec_from_file_location("migrate_branch_per_writer", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
migrate = importlib.util.module_from_spec(_spec)
sys.modules["migrate_branch_per_writer"] = migrate
_spec.loader.exec_module(migrate)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _seal(key: bytes, stream_id: str, obj: dict) -> str:
    aad = build_aad(NAMESPACE, stream_id)
    return aead.seal(key, aad, json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def _record(actor: str, seq: int, ts: str, product: str = "milk") -> dict:
    return {
        "schema_version": 2,
        "type": "acquired",
        "id": f"{actor}-{seq}",
        "ts": ts,
        "actor": actor,
        "supersedes": None,
        "seq": seq,
        "cmd_id": None,
        "payload": {"product_id": product, "to": "fridge", "amount": "1", "unit": "l"},
    }


def _location(actor: str, ts: str) -> dict:
    return {
        "schema_version": 2,
        "ts": ts,
        "actor": actor,
        "location": {
            "id": "fridge",
            "name": "Fridge",
            "parent_id": None,
            "metadata": {},
            "retired": False,
        },
    }


@pytest.fixture(autouse=True)
def _passphrase_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMAC_PASSPHRASE", "pw")


@pytest.fixture
def legacy_repo(tmp_path: Path, git_env: None) -> Path:
    """A legacy pre-migration repo: one shared branch, `data/vault.json`, a
    shared `data/config.jsonl.enc` sealed under `"config"` attributed to two
    OS usernames, and two legacy logs sealed under `log:alice` / `log:node`."""
    repo_root = tmp_path / "legacy-repo"
    data_dir = repo_root / "data"
    (data_dir / "log").mkdir(parents=True)

    vault = sumac_vault.create("pw")
    key = sumac_vault.unlock(vault, "pw")
    doc = {"format_version": FORMAT_VERSION, **vault.to_dict()}
    (data_dir / "vault.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    def ts(n: int) -> str:
        return t0.replace(hour=n % 24, day=1 + n // 24).isoformat()

    # config: fridge defined by alice, then by node (both fold into alice-mac)
    config_lines = [
        _seal(key, "config", _location("alice", ts(0))),
        _seal(key, "config", _location("node", ts(1))),
    ]
    (data_dir / "config.jsonl.enc").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    # alice's log: seq 0, 1 (own segment)
    alice_lines = [
        _seal(key, "log:alice", _record("alice", 0, ts(2), "milk")),
        _seal(key, "log:alice", _record("alice", 1, ts(4), "eggs")),
    ]
    (data_dir / "log" / "alice.jsonl").write_text("\n".join(alice_lines) + "\n", encoding="utf-8")

    # node's log: seq 0, 1 (its own segment, starting at 0 again -- the
    # duplicate-seq case the migration's renumbering removes)
    node_lines = [
        _seal(key, "log:node", _record("node", 0, ts(3), "bread")),
        _seal(key, "log:node", _record("node", 1, ts(5), "butter")),
    ]
    (data_dir / "log" / "node.jsonl").write_text("\n".join(node_lines) + "\n", encoding="utf-8")

    # a second, unrelated writer: bob
    bob_lines = [_seal(key, "log:bob", _record("bob", 0, ts(6), "rice"))]
    (data_dir / "log" / "bob.jsonl").write_text("\n".join(bob_lines) + "\n", encoding="utf-8")

    _git(repo_root, "init", "-q", "-b", "legacy")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-q", "-m", "legacy data")
    return repo_root


@pytest.fixture
def legacy_key(legacy_repo: Path) -> bytes:
    doc = json.loads((legacy_repo / "data" / "vault.json").read_text())
    return sumac_vault.unlock(Vault.from_dict(doc), "pw")


MAPPING = {"alice": "alice-mac", "node": "alice-mac", "bob": "bob-linux"}


def test_migration_creates_one_branch_per_writer(legacy_repo: Path, legacy_key: bytes) -> None:
    argv = [
        "--repo",
        str(legacy_repo),
        "--data-dir",
        "data",
        "--mapping",
        json.dumps(MAPPING),
    ]
    exit_code = migrate.main(argv)
    assert exit_code == 0

    branches = subprocess.run(
        ["git", "-C", str(legacy_repo), "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert set(branches) == {"legacy", "writer/alice-mac", "writer/bob-linux"}


def test_legacy_branch_untouched(legacy_repo: Path, legacy_key: bytes) -> None:
    before = subprocess.run(
        ["git", "-C", str(legacy_repo), "rev-parse", "legacy"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    migrate.main(["--repo", str(legacy_repo), "--mapping", json.dumps(MAPPING)])

    after = subprocess.run(
        ["git", "-C", str(legacy_repo), "rev-parse", "legacy"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert before == after

    current = subprocess.run(
        ["git", "-C", str(legacy_repo), "symbolic-ref", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert current == "legacy"


def test_merged_branch_has_contiguous_seq_and_rewritten_actor(
    legacy_repo: Path, legacy_key: bytes
) -> None:
    migrate.main(["--repo", str(legacy_repo), "--mapping", json.dumps(MAPPING)])

    blob = subprocess.run(
        ["git", "-C", str(legacy_repo), "cat-file", "blob", "writer/alice-mac:data/log.jsonl.enc"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [ln for ln in blob.splitlines() if ln.strip()]

    aad = build_aad(NAMESPACE, "log:alice-mac")
    objs = [json.loads(aead.open_(legacy_key, aad, ln)) for ln in lines]

    assert [o["seq"] for o in objs] == [0, 1, 2, 3]
    assert all(o["actor"] == "alice-mac" for o in objs)
    # merged by (ts, id): alice(0), node(0), alice(1), node(1)
    assert [o["id"] for o in objs] == ["alice-0", "node-0", "alice-1", "node-1"]


def test_record_equality_modulo_seq_and_actor(legacy_repo: Path, legacy_key: bytes) -> None:
    """The migrated alice-mac branch's records match the legacy alice+node
    records field-for-field except `seq`/`actor` (§6 step 4) -- checked via
    the script's own check function directly."""
    mapping = MAPPING
    legacy_logs = migrate._read_legacy_logs(legacy_repo / "data", legacy_key)
    per_writer_logs = migrate._merge_logs(legacy_logs, mapping, sorted(set(mapping.values())))
    assert migrate.check_record_equality(legacy_logs, mapping, per_writer_logs)


def test_holdings_identical_after_migration(legacy_repo: Path, legacy_key: bytes) -> None:
    mapping = MAPPING
    data_dir = legacy_repo / "data"
    legacy_logs = migrate._read_legacy_logs(data_dir, legacy_key)
    legacy_config = migrate._read_legacy_config(data_dir, legacy_key)
    writer_ids = sorted(set(mapping.values()))
    per_writer_logs = migrate._merge_logs(legacy_logs, mapping, writer_ids)
    per_writer_config = migrate._partition_config(legacy_config, mapping)

    holdings_ok, anomalies_ok = migrate.check_holdings_and_anomalies(
        legacy_logs, legacy_config, per_writer_logs, per_writer_config
    )
    assert holdings_ok
    assert anomalies_ok


def test_missing_mapping_entry_is_a_hard_error(legacy_repo: Path, legacy_key: bytes) -> None:
    incomplete = {"alice": "alice-mac", "node": "alice-mac"}  # bob missing
    with pytest.raises(RuntimeError, match="bob"):
        migrate.main(["--repo", str(legacy_repo), "--mapping", json.dumps(incomplete)])


def test_dry_run_writes_nothing(legacy_repo: Path, legacy_key: bytes) -> None:
    before = subprocess.run(
        ["git", "-C", str(legacy_repo), "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    exit_code = migrate.main(
        ["--repo", str(legacy_repo), "--mapping", json.dumps(MAPPING), "--dry-run"]
    )
    assert exit_code == 0

    after = subprocess.run(
        ["git", "-C", str(legacy_repo), "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert before == after
