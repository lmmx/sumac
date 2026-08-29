"""Encrypted JSONL append/iterate over a `stream_id`.

A `stream_id` is `"config"` or `"log:<osuser>"`. `append` is the ownership
boundary: it refuses to write to any `log:<osuser>` stream other than the
caller's own, so the AAD binding in `crypto.py` isn't just defense in depth —
it's backed by the one function that can produce a valid sealed line.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from sumac import crypto, paths
from sumac.errors import DecryptionError, ForeignStreamError

CONFIG_STREAM_ID = "config"


def _path_for_stream(data_dir: Path, stream_id: str) -> Path:
    if stream_id == CONFIG_STREAM_ID:
        return paths.config_path(data_dir)
    if stream_id.startswith("log:"):
        return paths.log_path(data_dir, stream_id.removeprefix("log:"))
    raise ValueError(f"unknown stream_id: {stream_id!r}")


def append(data_dir: Path, key: bytes, stream_id: str, obj: dict) -> None:
    if stream_id.startswith("log:"):
        osuser = stream_id.removeprefix("log:")
        current = paths.current_user()
        if osuser != current:
            raise ForeignStreamError(f"cannot append to {stream_id!r} as {current!r}: not your log")
    path = _path_for_stream(data_dir, stream_id)
    plaintext = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sealed = crypto.seal_line(key, stream_id, plaintext)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(sealed + "\n")


def _read_lines(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def iter_stream(data_dir: Path, key: bytes, stream_id: str) -> Iterator[dict]:
    path = _path_for_stream(data_dir, stream_id)
    for line in _read_lines(path):
        yield json.loads(crypto.open_line(key, stream_id, line))


def iter_all_logs(data_dir: Path, key: bytes) -> Iterator[tuple[str, dict]]:
    for path in paths.all_log_paths(data_dir):
        osuser = path.stem
        stream_id = f"log:{osuser}"
        for line in _read_lines(path):
            yield osuser, json.loads(crypto.open_line(key, stream_id, line))


@dataclass(frozen=True, slots=True)
class LineFailure:
    path: Path
    lineno: int
    error: str


def verify_stream(path: Path, key: bytes, stream_id: str) -> tuple[list[dict], list[LineFailure]]:
    """Open every line of `path` under `stream_id`, collecting failures instead of raising."""
    ok: list[dict] = []
    failures: list[LineFailure] = []
    for lineno, line in enumerate(_read_lines(path), start=1):
        try:
            ok.append(json.loads(crypto.open_line(key, stream_id, line)))
        except DecryptionError as e:
            failures.append(LineFailure(path=path, lineno=lineno, error=str(e)))
    return ok, failures
