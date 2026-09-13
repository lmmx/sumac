"""Encrypted JSONL append/iterate over a `stream_id`, backed by `sealedlog`.

A `stream_id` is `"config:<writer_id>"` or `"log:<writer_id>"`. See docs/journal
2026-09-11-branch-per-user-design.md §2, §5.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sealedlog import SealedLog, aead
from sealedlog._aad import build_aad
from sealedlog.errors import AuthenticationError

from sumac import NAMESPACE, gitrepo, paths, sources, writer


def _path_for_stream(data_dir: Path, stream_id: str) -> Path:
    if stream_id.startswith("config:"):
        return paths.config_path(data_dir)
    if stream_id.startswith("log:"):
        return paths.log_path(data_dir)
    raise ValueError(f"unknown stream_id: {stream_id!r}")


def _log_for_stream(data_dir: Path, key: bytes, stream_id: str) -> SealedLog:
    path = _path_for_stream(data_dir, stream_id)
    return SealedLog(path, key, stream_id, namespace=NAMESPACE)


def assigned_seqs(objs: list[dict]) -> list[int]:
    """The `seq` each record in `objs` (file order) is treated as having: its
    own stored `seq` if present, else its position among decoded records —
    the same backfill the v1 upcaster already relies on (docs/journal
    2026-08-30 §3.7): every record written before Phase 7 has no stored
    `seq` at all, and those files are append-only and have never been
    reordered, so position-in-file is a faithful stand-in. Corrupted lines
    (a `store.verify_stream` failure) aren't represented here at all — they
    already surface as their own `line_failure` anomaly; this only orders
    the lines that decoded successfully."""
    return [obj["seq"] if isinstance(obj.get("seq"), int) else i for i, obj in enumerate(objs)]


def append(data_dir: Path, key: bytes, stream_id: str, obj: dict) -> None:
    if stream_id.startswith("log:"):
        # seq is append-time, not decide-time (docs/journal §3.7): it depends
        # on what's already on disk, which `decide` (pure, no I/O) can't see.
        # Config records never get one — config is latest-revision-wins, not
        # an append-sequential segment gap detection applies to.
        existing, _failures = verify_stream(_path_for_stream(data_dir, stream_id), key, stream_id)
        next_seq = max(assigned_seqs(existing), default=-1) + 1
        obj = {**obj, "seq": next_seq}
    _log_for_stream(data_dir, key, stream_id).append(obj)


def commit_records(data_dir: Path, n: int) -> None:
    """One commit per command, covering every record it wrote; the message is a
    fixed literal plus a count and nothing else (§2 "Writes commit"; threat
    model in docs/FORMAT.md). Filesystem mode (§2's step 1) has no repo to
    commit to and skips silently."""
    if os.environ.get(writer.WRITER_ID_ENV) or not gitrepo.is_repo(data_dir):
        return
    repo_root = gitrepo.toplevel(data_dir)
    assert repo_root is not None  # `is_repo` above guarantees a toplevel
    rel_data = gitrepo.rel_to_toplevel(data_dir)
    plural = "record" if n == 1 else "records"
    gitrepo.commit_paths(repo_root, [rel_data.as_posix()], f"sumac: {n} {plural}")


def iter_stream(data_dir: Path, key: bytes, stream_id: str) -> Iterator[dict]:
    yield from _log_for_stream(data_dir, key, stream_id)


def iter_all_logs(data_dir: Path, key: bytes) -> Iterator[tuple[str, dict]]:
    for source in sources.writer_sources(data_dir):
        stream_id = writer.log_stream_id(source.writer_id)
        lines = source.log_text().splitlines()
        objs, _failures = verify_lines(lines, key, stream_id, source.label)
        for obj in objs:
            yield source.writer_id, obj


@dataclass(frozen=True, slots=True)
class LineFailure:
    source: str
    lineno: int
    error: str


def _read_lines(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def verify_lines(
    lines: Iterable[str], key: bytes, stream_id: str, source: str
) -> tuple[list[dict], list[LineFailure]]:
    """Open every line under `stream_id`, collecting failures instead of raising.

    `sealedlog.SealedLog` offers `__iter__` (stops at the first bad line) and
    `verify()` (never raises, but doesn't return decoded records) — neither
    gives both decoded objects and per-line failures in one pass, which
    `ledger.verify_all` needs. So this reimplements that one loop directly
    against `sealedlog.aead`, using the same AAD `SealedLog` builds internally.
    """
    aad = build_aad(NAMESPACE, stream_id)
    ok: list[dict] = []
    failures: list[LineFailure] = []
    for lineno, line in enumerate((line for line in lines if line.strip()), start=1):
        try:
            plaintext = aead.open_(key, aad, line.strip())
        except AuthenticationError as e:
            failures.append(LineFailure(source=source, lineno=lineno, error=str(e)))
            continue
        try:
            ok.append(json.loads(plaintext))
        except json.JSONDecodeError as e:
            failures.append(LineFailure(source=source, lineno=lineno, error=f"not valid JSON: {e}"))
    return ok, failures


def verify_stream(path: Path, key: bytes, stream_id: str) -> tuple[list[dict], list[LineFailure]]:
    return verify_lines(_read_lines(path), key, stream_id, str(path))
