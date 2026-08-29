"""Data-dir layout: the single place path names are constructed.

Every path component here is a fixed literal or an OS username — never a
location or product name — so the on-disk layout leaks nothing about
inventory contents.
"""

from __future__ import annotations

import getpass
import re
from pathlib import Path

VAULT_FILENAME = "vault.json"
CONFIG_FILENAME = "config.jsonl.enc"
LOG_DIRNAME = "log"

_SAFE_OSUSER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def current_user() -> str:
    return getpass.getuser()


def validate_osuser(osuser: str) -> str:
    if not osuser or osuser in {".", ".."} or not _SAFE_OSUSER_RE.fullmatch(osuser):
        raise ValueError(f"unsafe username: {osuser!r}")
    return osuser


def vault_path(data_dir: Path) -> Path:
    return data_dir / VAULT_FILENAME


def config_path(data_dir: Path) -> Path:
    return data_dir / CONFIG_FILENAME


def log_dir(data_dir: Path) -> Path:
    return data_dir / LOG_DIRNAME


def log_path(data_dir: Path, osuser: str) -> Path:
    return log_dir(data_dir) / f"{validate_osuser(osuser)}.jsonl"


def all_log_paths(data_dir: Path) -> list[Path]:
    d = log_dir(data_dir)
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix == ".jsonl")
