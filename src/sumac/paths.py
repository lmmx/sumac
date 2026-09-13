"""Data-dir layout: the single place path names are constructed.

Every path component here is a fixed literal — never a location or product
name — so the on-disk layout leaks nothing about inventory contents.
"""

from __future__ import annotations

from pathlib import Path

VAULT_FILENAME = "vault.json"
CONFIG_FILENAME = "config.jsonl.enc"
LOG_FILENAME = "log.jsonl.enc"


def vault_path(data_dir: Path) -> Path:
    return data_dir / VAULT_FILENAME


def config_path(data_dir: Path) -> Path:
    return data_dir / CONFIG_FILENAME


def log_path(data_dir: Path) -> Path:
    return data_dir / LOG_FILENAME
