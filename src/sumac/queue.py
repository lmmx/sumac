"""Local cache of `sumac ask --loop` requests that failed or were deferred
mid-review, so a request that crashes the Python process (mistral.rs's
max-seq-len KV-cache exhaustion — see
docs/journal/2026-09-01-ask-agent-design.md — is the specific crash this was
built for) or that the person doesn't want to decide on right now isn't
lost, and doesn't block moving on to the next request.

Plain, unencrypted JSON — unlike the rest of `data_dir`, which is encrypted
with the vault key throughout (`paths.py`'s own docstring: nothing here is a
domain fact about inventory, and losing or reading this file has no
correctness consequence, only convenience. Deliberately a cache, not a
vault-shaped log: entries are removed on retry, not folded from an
append-only history the way `ledger`/`config` records are.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

QUEUE_FILENAME = "ask_queue.json"


def queue_path(data_dir: Path) -> Path:
    return data_dir / QUEUE_FILENAME


@dataclass(frozen=True, slots=True)
class QueuedRequest:
    prompt: str
    reason: str
    added_at: str
    attempts: int = 0


def load(data_dir: Path) -> list[QueuedRequest]:
    path = queue_path(data_dir)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [QueuedRequest(**item) for item in raw]


def _save(data_dir: Path, items: list[QueuedRequest]) -> None:
    path = queue_path(data_dir)
    # Write to a temp file and rename over the target — an interrupted
    # write (the same kind of crash this cache exists to survive) must
    # never leave a half-written, unparseable queue file behind.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(item) for item in items], indent=2))
    tmp.replace(path)


def enqueue(data_dir: Path, prompt: str, reason: str, *, attempts: int = 0) -> None:
    items = load(data_dir)
    items.append(
        QueuedRequest(
            prompt=prompt,
            reason=reason,
            added_at=datetime.now(UTC).isoformat(),
            attempts=attempts,
        )
    )
    _save(data_dir, items)


def dequeue(data_dir: Path, index: int) -> QueuedRequest:
    """`index` is 0-based into whatever `load` currently returns. Raises
    `IndexError` for an out-of-range index — the caller (an interactive
    prompt) is expected to validate against a freshly displayed list, not
    guess blindly."""
    items = load(data_dir)
    item = items[index]
    del items[index]
    _save(data_dir, items)
    return item
