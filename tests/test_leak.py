"""Walks a populated data dir and asserts nothing about locations or products leaks
through path names or file contents."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sumac import paths
from sumac.cli import app

runner = CliRunner()
ENV = {"SUMAC_PASSPHRASE": "test-pass"}

SECRETS = ["pantry", "fridge", "wine-cellar", "milk", "cabernet"]


@pytest.fixture(autouse=True)
def _osuser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "alice")


def _run(data_dir: Path, *args: str):
    result = runner.invoke(app, [*args, "--data-dir", str(data_dir)], env=ENV)
    assert result.exit_code == 0, result.output
    return result


def _populate(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-location", "Wine Cellar", "--id", "wine-cellar")
    _run(data_dir, "add", "purchase", "milk", "2", "l", "--to", "fridge")
    _run(
        data_dir,
        "add",
        "movement",
        "cabernet",
        "3",
        "bottle",
        "--from",
        "wine-cellar",
        "--to",
        "pantry",
    )
    _run(data_dir, "snapshot", "fridge", "milk=1/l")


ALLOWED_PATH_COMPONENTS = {
    "data",
    "vault.json",
    "config.jsonl.enc",
    "log",
    "alice.jsonl",
}


def test_path_components_are_fixed_literals_or_usernames(data_dir: Path) -> None:
    _populate(data_dir)
    for path in data_dir.rglob("*"):
        rel = path.relative_to(data_dir)
        for part in rel.parts:
            assert part in ALLOWED_PATH_COMPONENTS, f"leaky path component: {part!r} in {path}"


def test_file_contents_do_not_contain_secrets(data_dir: Path) -> None:
    _populate(data_dir)
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        for secret in SECRETS:
            assert secret.encode() not in raw, f"{secret!r} leaked in {path}"


def test_vault_json_has_no_ciphertext_correlated_names(data_dir: Path) -> None:
    _populate(data_dir)
    vault_text = paths.vault_path(data_dir).read_text()
    for secret in SECRETS:
        assert secret not in vault_text
