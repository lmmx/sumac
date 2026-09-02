"""`queue.py`: the local cache `sumac ask --loop` uses to survive a crash
(mistral.rs's max-seq-len KV-cache exhaustion) or a deferred request
without losing either. Plain JSON, not part of the encrypted vault — these
tests write directly to a `tmp_path`, no `data_dir`/`key` fixtures needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from sumac import queue


def test_load_on_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert queue.load(tmp_path) == []


def test_enqueue_then_load_round_trips(tmp_path: Path) -> None:
    queue.enqueue(tmp_path, "add 1 can of jam", "deferred")

    items = queue.load(tmp_path)

    assert len(items) == 1
    assert items[0].prompt == "add 1 can of jam"
    assert items[0].reason == "deferred"
    assert items[0].attempts == 0
    assert items[0].added_at  # a timestamp was stamped, not asserting its exact value


def test_enqueue_appends_preserving_order(tmp_path: Path) -> None:
    queue.enqueue(tmp_path, "first", "deferred")
    queue.enqueue(tmp_path, "second", "deferred")

    items = queue.load(tmp_path)

    assert [item.prompt for item in items] == ["first", "second"]


def test_enqueue_with_attempts_carries_the_retry_count(tmp_path: Path) -> None:
    queue.enqueue(tmp_path, "add 1 can of jam", "error: boom", attempts=2)

    items = queue.load(tmp_path)

    assert items[0].attempts == 2


def test_dequeue_removes_and_returns_the_item(tmp_path: Path) -> None:
    queue.enqueue(tmp_path, "first", "deferred")
    queue.enqueue(tmp_path, "second", "deferred")

    item = queue.dequeue(tmp_path, 0)

    assert item.prompt == "first"
    assert [i.prompt for i in queue.load(tmp_path)] == ["second"]


def test_dequeue_out_of_range_raises_index_error(tmp_path: Path) -> None:
    with pytest.raises(IndexError):
        queue.dequeue(tmp_path, 0)


def test_queue_file_survives_being_written_twice(tmp_path: Path) -> None:
    """The save path writes to a temp file and renames over the target —
    confirms a second write doesn't leave stale content or fail outright
    because the target already exists."""
    queue.enqueue(tmp_path, "first", "deferred")
    queue.enqueue(tmp_path, "second", "deferred")
    queue.dequeue(tmp_path, 0)
    queue.enqueue(tmp_path, "third", "deferred")

    assert [i.prompt for i in queue.load(tmp_path)] == ["second", "third"]
    assert not queue.queue_path(tmp_path).with_suffix(".tmp").exists()
