#!/usr/bin/env python3
"""One-off migration: legacy shared-branch layout -> branch-per-writer.

Implements docs/journal/2026-09-11-branch-per-user-design.md §6, step by step.
Not a `sumac` subcommand: run once, by hand, by whoever holds the household
passphrase, against the household's existing data repo. Never deletes or
rewrites the legacy branch — the currently checked-out branch is never
touched, since every new object is created with git plumbing (hash-object,
mktree, commit-tree, update-ref) rather than a checkout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

from sealedlog import Vault as SealedVault
from sealedlog import aead
from sealedlog._aad import build_aad

from sumac import NAMESPACE, store, writer
from sumac import passphrase as sumac_passphrase
from sumac import vault as sumac_vault
from sumac.ledger import _event_records, _fold  # reuse the real fold (§6 step 5)
from sumac.models import Anomaly, Location, Quantity
from sumac.schemas import ConfigRecordSchema

# --- git plumbing: never checks out, never touches the working tree or HEAD ---


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], input=input_bytes, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.decode().strip()}")
    return result.stdout.decode()


def _ref_exists(repo: Path, ref: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", ref], capture_output=True
    )
    return result.returncode == 0


def _hash_blob(repo: Path, data: bytes) -> str:
    return _git(repo, "hash-object", "-w", "--stdin", input_bytes=data).strip()


def _mktree(repo: Path, entries: list[tuple[str, str, str, str]]) -> str:
    lines = "".join(f"{mode} {typ} {sha}\t{name}\n" for mode, typ, sha, name in entries)
    return _git(repo, "mktree", input_bytes=lines.encode()).strip()


def _build_tree(repo: Path, files: dict[str, bytes]) -> str:
    """`files`: relative posix path -> content. Builds nested tree objects
    bottom-up and returns the root tree sha, all via plumbing — no index, no
    checkout."""
    root: dict = {}
    for relpath, content in files.items():
        parts = relpath.split("/")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = content

    def build(node: dict) -> str:
        entries = []
        for name, val in sorted(node.items()):
            if isinstance(val, dict):
                entries.append(("040000", "tree", build(val), name))
            else:
                entries.append(("100644", "blob", _hash_blob(repo, val), name))
        return _mktree(repo, entries)

    return build(root)


def _commit_tree(repo: Path, tree: str, parents: list[str], message: str) -> str:
    args = ["commit-tree", tree]
    for p in parents:
        args += ["-p", p]
    return _git(repo, *args, input_bytes=message.encode()).strip()


def _seal_lines(key: bytes, stream_id: str, objs: list[dict]) -> bytes:
    """Matches `sealedlog.SealedLog.append`'s own wire format exactly, so the
    result reads back through ordinary `store`/`sealedlog` code."""
    aad = build_aad(NAMESPACE, stream_id)
    lines = [
        aead.seal(key, aad, json.dumps(obj, separators=(",", ":")).encode("utf-8")) for obj in objs
    ]
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


# --- args ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path, help="the household's legacy data repo")
    p.add_argument("--data-dir", default="data", help="data dir name under --repo")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--mapping", help='JSON {"legacy_osuser": "writer_id"}')
    group.add_argument("--mapping-file", type=Path)
    p.add_argument("--dry-run", action="store_true", help="check and report only, write nothing")
    return p.parse_args(argv)


def _load_mapping(args: argparse.Namespace) -> dict[str, str]:
    raw = args.mapping if args.mapping is not None else args.mapping_file.read_text()
    mapping = json.loads(raw)
    if not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
        raise RuntimeError("--mapping/--mapping-file must be a JSON object of osuser -> writer_id")
    return mapping


# --- reading the legacy repo (§6 step 1) ---


def _read_legacy_logs(data_dir: Path, key: bytes) -> dict[str, list[dict]]:
    logs: dict[str, list[dict]] = {}
    log_dir = data_dir / "log"
    for path in sorted(log_dir.glob("*.jsonl")) if log_dir.is_dir() else []:
        osuser = path.stem
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        objs, failures = store.verify_lines(lines, key, f"log:{osuser}", str(path))
        if failures:
            raise RuntimeError(f"{path}: {len(failures)} unreadable line(s), aborting")
        logs[osuser] = objs
    return logs


def _read_legacy_config(data_dir: Path, key: bytes) -> list[dict]:
    path = data_dir / "config.jsonl.enc"
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    objs, failures = store.verify_lines(lines, key, "config", str(path))
    if failures:
        raise RuntimeError(f"{path}: {len(failures)} unreadable line(s), aborting")
    return objs


# --- building the per-writer streams (§6 step 3) ---


def _merge_logs(
    legacy_logs: dict[str, list[dict]], mapping: dict[str, str], writer_ids: list[str]
) -> dict[str, list[dict]]:
    """Interleave every legacy log a writer's mapped osusers own by `(ts, id)`,
    renumber `seq` contiguously from 0, rewrite `actor` to the writer id."""
    result: dict[str, list[dict]] = {}
    for wid in writer_ids:
        merged = [
            obj for osuser, objs in legacy_logs.items() if mapping[osuser] == wid for obj in objs
        ]
        merged.sort(key=lambda o: (datetime.fromisoformat(o["ts"]), o["id"]))
        result[wid] = [{**obj, "seq": i, "actor": wid} for i, obj in enumerate(merged)]
    return result


def _partition_config(config_objs: list[dict], mapping: dict[str, str]) -> dict[str, list[dict]]:
    """Partition by each record's `actor` through `mapping`, preserving file
    order, rewriting `actor` to the writer id (§6 step 3)."""
    result: dict[str, list[dict]] = defaultdict(list)
    for obj in config_objs:
        wid = mapping[obj["actor"]]
        result[wid].append({**obj, "actor": wid})
    return dict(result)


# --- checks (§6 steps 4-5) ---


def _strip(obj: dict, exclude: set[str]) -> dict:
    return {k: v for k, v in obj.items() if k not in exclude}


def check_record_equality(
    legacy_logs: dict[str, list[dict]],
    mapping: dict[str, str],
    per_writer_logs: dict[str, list[dict]],
) -> bool:
    """§6 step 4: the legacy sequence, partitioned by the mapping, equals each
    new branch's sequence field-for-field except `seq` and `actor`, with `seq`
    contiguous from 0 per branch."""
    ok = True
    for wid, new_objs in per_writer_logs.items():
        expected = [
            obj for osuser, objs in legacy_logs.items() if mapping[osuser] == wid for obj in objs
        ]
        expected.sort(key=lambda o: (datetime.fromisoformat(o["ts"]), o["id"]))
        if [_strip(o, {"seq", "actor"}) for o in expected] != [
            _strip(o, {"seq", "actor"}) for o in new_objs
        ]:
            ok = False
        if [o["seq"] for o in new_objs] != list(range(len(new_objs))):
            ok = False
        if any(o["actor"] != wid for o in new_objs):
            ok = False
    return ok


def _known_locations(config_objs: list[dict]) -> dict[str, Location]:
    """Latest-`(ts, actor)`-wins per location id — mirrors
    `config._load_config_records`'s tie-break, reimplemented here so the
    equality check doesn't need a `data_dir`/git-backed `sources.writer_sources`
    to run against a plain list of decoded objects."""
    latest: dict[str, tuple[datetime, str, Location]] = {}
    for obj in config_objs:
        record = ConfigRecordSchema.model_validate(obj)
        if record.location is None:
            continue
        key_ = (record.ts, record.actor)
        prior = latest.get(record.location.id)
        if prior is None or key_ >= (prior[0], prior[1]):
            latest[record.location.id] = (record.ts, record.actor, record.location.to_domain())
    return {loc_id: loc for loc_id, (_, _, loc) in latest.items()}


def _fold_holdings(
    log_objs: list[dict], config_objs: list[dict]
) -> tuple[dict[str, dict[str, Quantity]], list[Anomaly]]:
    locations = _known_locations(config_objs)
    records = _event_records(log_objs)
    state, anomalies = _fold(records, locations)
    return state, anomalies


def check_holdings_and_anomalies(
    legacy_logs: dict[str, list[dict]],
    legacy_config: list[dict],
    per_writer_logs: dict[str, list[dict]],
    per_writer_config: dict[str, list[dict]],
) -> tuple[bool, bool]:
    """§6 step 5: `build_inventory` over the legacy repo and over the aggregate
    of the new branches must produce identical holdings, and identical
    anomalies except `seq_*` ones (deliberately different: renumbering is the
    point). Anomalies are compared by `(record_id, reason)`, not full detail
    text, since `detail` legitimately embeds the rewritten `actor`."""
    legacy_all_logs = [obj for objs in legacy_logs.values() for obj in objs]
    new_all_logs = [obj for objs in per_writer_logs.values() for obj in objs]
    new_all_config = [obj for objs in per_writer_config.values() for obj in objs]

    legacy_state, legacy_anoms = _fold_holdings(legacy_all_logs, legacy_config)
    new_state, new_anoms = _fold_holdings(new_all_logs, new_all_config)

    holdings_ok = legacy_state == new_state

    def _key(a: Anomaly) -> tuple[str | None, str]:
        return (a.record_id, a.reason)

    legacy_keys = sorted(_key(a) for a in legacy_anoms if not a.reason.startswith("seq_"))
    new_keys = sorted(_key(a) for a in new_anoms if not a.reason.startswith("seq_"))
    anomalies_ok = legacy_keys == new_keys

    return holdings_ok, anomalies_ok


# --- writing the new branches (§6 steps 2-3) ---


def _write_branches(
    repo: Path,
    data_dir_name: str,
    key: bytes,
    vault_bytes: bytes,
    per_writer_logs: dict[str, list[dict]],
    per_writer_config: dict[str, list[dict]],
) -> None:
    for wid in per_writer_logs:
        branch_name = writer.branch(wid)
        ref = f"refs/heads/{branch_name}"
        if _ref_exists(repo, ref):
            raise RuntimeError(f"{branch_name} already exists, aborting")

    rel = PurePosixPath(data_dir_name)
    # `.gitignore` for ask's per-machine cache, the same one `sumac init` writes
    # (§6's closing paragraph); `.gitattributes`' retired `merge=union` line needs
    # no removal step, since every new tree is built from scratch.
    gitignore = f"{(rel / 'ask_queue.json').as_posix()}\n".encode()
    root_tree = _build_tree(repo, {str(rel / "vault.json"): vault_bytes, ".gitignore": gitignore})
    root_commit = _commit_tree(repo, root_tree, [], "sumac: migration root")

    for wid, log_objs in per_writer_logs.items():
        config_objs = per_writer_config.get(wid, [])
        files = {
            str(rel / "vault.json"): vault_bytes,
            ".gitignore": gitignore,
            str(rel / "log.jsonl.enc"): _seal_lines(key, writer.log_stream_id(wid), log_objs),
            str(rel / "config.jsonl.enc"): _seal_lines(
                key, writer.config_stream_id(wid), config_objs
            ),
        }
        tree = _build_tree(repo, files)
        n = len(log_objs) + len(config_objs)
        plural = "record" if n == 1 else "records"
        commit = _commit_tree(repo, tree, [root_commit], f"sumac: {n} {plural}")
        _git(repo, "update-ref", f"refs/heads/{writer.branch(wid)}", commit)
        print(f"  created {writer.branch(wid)} ({n} records)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mapping = _load_mapping(args)

    repo = args.repo.resolve()
    data_dir = repo / args.data_dir

    key = sumac_vault.unlock(
        SealedVault.from_dict(json.loads((data_dir / "vault.json").read_text())),
        sumac_passphrase.resolve_passphrase(),
    )

    legacy_logs = _read_legacy_logs(data_dir, key)
    legacy_config = _read_legacy_config(data_dir, key)

    seen_osusers = set(legacy_logs) | {obj["actor"] for obj in legacy_config}
    missing = sorted(seen_osusers - set(mapping))
    if missing:
        raise RuntimeError(f"no mapping entry for legacy user(s): {missing}")

    writer_ids = sorted(set(mapping.values()))
    per_writer_logs = _merge_logs(legacy_logs, mapping, writer_ids)
    per_writer_config = _partition_config(legacy_config, mapping)

    record_ok = check_record_equality(legacy_logs, mapping, per_writer_logs)
    holdings_ok, anomalies_ok = check_holdings_and_anomalies(
        legacy_logs, legacy_config, per_writer_logs, per_writer_config
    )

    print(f"writers: {', '.join(writer_ids)}")
    print(f"record equality (modulo seq/actor): {'OK' if record_ok else 'FAILED'}")
    print(f"holdings equality: {'OK' if holdings_ok else 'FAILED'}")
    print(f"anomaly equality (excl. seq_*): {'OK' if anomalies_ok else 'FAILED'}")

    if not (record_ok and holdings_ok and anomalies_ok):
        print("one or more checks failed; nothing written", file=sys.stderr)
        return 1

    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    vault_bytes = (data_dir / "vault.json").read_bytes()
    _write_branches(repo, args.data_dir, key, vault_bytes, per_writer_logs, per_writer_config)
    print("legacy branch left untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
