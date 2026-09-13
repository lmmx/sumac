"""Writer identity: the single place a writer id is resolved. See docs/journal
2026-09-11-branch-per-user-design.md §2, §3, §5."""

from __future__ import annotations

import getpass
import os
import re
import socket
from pathlib import Path

from sumac import gitrepo
from sumac.errors import NotAWriterBranchError

WRITER_BRANCH_PREFIX = "writer/"
WRITER_ID_ENV = "SUMAC_WRITER_ID"

_SLUG_RUN_RE = re.compile(r"[^a-z0-9]+")


def slug(raw: str) -> str:
    """§3: lowercase, non-alphanumeric runs -> one "-", strip leading/trailing "-"."""
    normalized = _SLUG_RUN_RE.sub("-", raw.lower()).strip("-")
    if not normalized:
        raise ValueError(f"slug of {raw!r} is empty")
    return normalized


def default_id() -> str:
    return slug(f"{getpass.getuser()}-{socket.gethostname()}")


def branch(writer_id: str) -> str:
    return f"{WRITER_BRANCH_PREFIX}{writer_id}"


def id_from_branch(branch_name: str) -> str | None:
    if not branch_name.startswith(WRITER_BRANCH_PREFIX):
        return None
    return branch_name.removeprefix(WRITER_BRANCH_PREFIX)


def log_stream_id(writer_id: str) -> str:
    return f"log:{writer_id}"


def config_stream_id(writer_id: str) -> str:
    return f"config:{writer_id}"


def current_id(data_dir: Path) -> str:
    """§2's resolution rule: env wins, else the checked-out `writer/<id>` branch,
    else raise rather than guess."""
    env_id = os.environ.get(WRITER_ID_ENV)
    if env_id:
        return env_id
    if gitrepo.is_repo(data_dir):
        current = gitrepo.current_branch(data_dir)
        if current is not None:
            writer_id = id_from_branch(current)
            if writer_id is not None:
                return writer_id
    raise NotAWriterBranchError(
        f"{data_dir} is not on a writer/<id> branch and {WRITER_ID_ENV} is not set"
    )
