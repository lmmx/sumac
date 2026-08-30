"""Encrypted JSONL append/iterate over a `stream_id`, backed by `sealedlog`.

A `stream_id` is `"config"` or `"log:<osuser>"`. `append` is the ownership
boundary: it refuses to write to any `log:<osuser>` stream other than the
caller's own — sealedlog's AAD binding then backs that up cryptographically,
since `append` is the only function that can produce a valid sealed line.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from sealedlog import SealedLog, aead
from sealedlog._aad import build_aad
from sealedlog.errors import AuthenticationError

from sumac import NAMESPACE, paths
from sumac.errors import ForeignStreamError

CONFIG_STREAM_ID = "config"


def _path_for_stream(data_dir: Path, stream_id: str) -> Path:
    if stream_id == CONFIG_STREAM_ID:
        return paths.config_path(data_dir)
    if stream_id.startswith("log:"):
        return paths.log_path(data_dir, stream_id.removeprefix("log:"))
    raise ValueError(f"unknown stream_id: {stream_id!r}")


def _log_for_stream(data_dir: Path, key: bytes, stream_id: str) -> SealedLog:
    path = _path_for_stream(data_dir, stream_id)
    return SealedLog(path, key, stream_id, namespace=NAMESPACE)


def append(data_dir: Path, key: bytes, stream_id: str, obj: dict) -> None:
    if stream_id.startswith("log:"):
        osuser = stream_id.removeprefix("log:")
        current = paths.current_user()
        if osuser != current:
            raise ForeignStreamError(f"cannot append to {stream_id!r} as {current!r}: not your log")
    _log_for_stream(data_dir, key, stream_id).append(obj)


def iter_stream(data_dir: Path, key: bytes, stream_id: str) -> Iterator[dict]:
    yield from _log_for_stream(data_dir, key, stream_id)


def iter_all_logs(data_dir: Path, key: bytes) -> Iterator[tuple[str, dict]]:
    for path in paths.all_log_paths(data_dir):
        osuser = path.stem
        stream_id = f"log:{osuser}"
        for obj in SealedLog(path, key, stream_id, namespace=NAMESPACE):
            yield osuser, obj


@dataclass(frozen=True, slots=True)
class LineFailure:
    path: Path
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


def verify_stream(path: Path, key: bytes, stream_id: str) -> tuple[list[dict], list[LineFailure]]:
    """Open every line of `path` under `stream_id`, collecting failures instead of raising.

    `sealedlog.SealedLog` offers `__iter__` (stops at the first bad line) and
    `verify()` (never raises, but doesn't return decoded records) — neither
    gives both decoded objects and per-line failures in one pass, which
    `ledger.verify_all` needs. So this reimplements that one loop directly
    against `sealedlog.aead`, using the same AAD `SealedLog` builds internally.
    """
    aad = build_aad(NAMESPACE, stream_id)
    ok: list[dict] = []
    failures: list[LineFailure] = []
    for lineno, line in enumerate(_read_lines(path), start=1):
        try:
            plaintext = aead.open_(key, aad, line)
        except AuthenticationError as e:
            failures.append(LineFailure(path=path, lineno=lineno, error=str(e)))
            continue
        try:
            ok.append(json.loads(plaintext))
        except json.JSONDecodeError as e:
            failures.append(LineFailure(path=path, lineno=lineno, error=f"not valid JSON: {e}"))
    return ok, failures
