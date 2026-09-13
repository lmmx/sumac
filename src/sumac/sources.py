"""Enumerates the writers to read, and where their bytes come from. See docs/journal
2026-09-11-branch-per-user-design.md §2, §5.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sumac import gitrepo, paths, writer


@dataclass(frozen=True, slots=True)
class WriterSource:
    writer_id: str
    label: str
    log_text: Callable[[], str]
    config_text: Callable[[], str]


def for_writing(data_dir: Path) -> str:
    return writer.current_id(data_dir)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _worktree_source(data_dir: Path, writer_id: str, label: str) -> WriterSource:
    return WriterSource(
        writer_id=writer_id,
        label=label,
        log_text=lambda: _read_text(paths.log_path(data_dir)),
        config_text=lambda: _read_text(paths.config_path(data_dir)),
    )


def _ref_source(data_dir: Path, writer_id: str, ref: str) -> WriterSource:
    rel = gitrepo.rel_to_toplevel(data_dir)
    log_rel = (rel / paths.LOG_FILENAME).as_posix()
    config_rel = (rel / paths.CONFIG_FILENAME).as_posix()
    return WriterSource(
        writer_id=writer_id,
        label=f"writer/{writer_id}",
        log_text=lambda: gitrepo.read_blob(data_dir, ref, log_rel),
        config_text=lambda: gitrepo.read_blob(data_dir, ref, config_rel),
    )


def writer_sources(data_dir: Path) -> list[WriterSource]:
    current_id = writer.current_id(data_dir)
    if not gitrepo.is_repo(data_dir):
        return [_worktree_source(data_dir, current_id, str(data_dir))]

    sources = [_worktree_source(data_dir, current_id, f"writer/{current_id}")]
    for writer_id, ref in gitrepo.list_writer_refs(data_dir).items():
        if writer_id == current_id:
            continue
        sources.append(_ref_source(data_dir, writer_id, ref))
    return sorted(sources, key=lambda s: s.writer_id)
