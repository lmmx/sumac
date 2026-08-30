from __future__ import annotations

from pathlib import Path

import pytest
from sealedlog.errors import WrongPassphraseError
from typer.testing import CliRunner

from sumac import paths
from sumac.cli import app
from sumac.errors import VaultExistsError

runner = CliRunner()
PASSPHRASE_ENV = {"SUMAC_PASSPHRASE": "test-pass"}


@pytest.fixture(autouse=True)
def _osuser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "alice")


def _run(data_dir: Path, *args: str, env: dict[str, str] = PASSPHRASE_ENV):
    return runner.invoke(app, [*args, "--data-dir", str(data_dir)], env=env)


def test_init_creates_vault(data_dir: Path) -> None:
    result = _run(data_dir, "init")
    assert result.exit_code == 0, result.output
    assert paths.vault_path(data_dir).exists()
    assert paths.log_dir(data_dir).exists()


def test_init_twice_fails(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "init")
    assert result.exit_code != 0
    assert isinstance(result.exception, VaultExistsError)


def test_wrong_passphrase_fails(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "config", "show", env={"SUMAC_PASSPHRASE": "nope"})
    assert result.exit_code != 0
    assert isinstance(result.exception, WrongPassphraseError)


def test_add_location_and_show(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "config", "show")
    assert result.exit_code == 0
    assert "Fridge" in result.output


def test_add_change_and_status(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    result = _run(data_dir, "add", "purchase", "milk", "2", "l", "--to", "pantry")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "status")
    assert result.exit_code == 0
    assert "milk" in result.output
    assert "2" in result.output


def test_snapshot_and_find(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "snapshot", "fridge", "milk=3/l")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "find", "milk")
    assert result.exit_code == 0
    assert "fridge" in result.output


def test_verify_clean(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    result = _run(data_dir, "verify")
    assert result.exit_code == 0
    assert "verified" in result.output


def test_verify_detects_tampering(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    log_path = paths.log_path(data_dir, "alice")
    log_path.write_text("not-valid-base64!!!\n")
    result = _run(data_dir, "verify")
    assert result.exit_code != 0


def test_doctor_clean_log(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    result = _run(data_dir, "doctor")
    assert result.exit_code == 0, result.output
    assert "no anomalies" in result.output


def test_doctor_flags_unknown_location(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "hob-right-below-bottom")
    result = _run(data_dir, "doctor")
    assert result.exit_code == 1
    assert "unknown_location" in result.output


def test_log_shows_recorded_events(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    result = _run(data_dir, "log")
    assert result.exit_code == 0
    assert "purchase" in result.output


def test_add_array_creates_numbered_sublocations(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    result = _run(data_dir, "config", "add-array", "Shelf", "--parent", "fridge", "--count", "3")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "config", "show")
    assert result.exit_code == 0
    for i in (1, 2, 3):
        assert f"Shelf {i}" in result.output


def test_add_grid_creates_grid_sublocations(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    result = _run(
        data_dir, "config", "add-grid", "Bin", "--parent", "pantry", "--rows", "2", "--cols", "2"
    )
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "config", "show")
    assert result.exit_code == 0
    for cell in ("Bin R1C1", "Bin R1C2", "Bin R2C1", "Bin R2C2"):
        assert cell in result.output


def test_status_rolls_up_sublocations(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-location", "Door", "--id", "fridge-door", "--parent", "fridge")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge")
    _run(data_dir, "add", "purchase", "eggs", "6", "ct", "--to", "fridge-door")

    result = _run(data_dir, "status", "fridge")
    assert result.exit_code == 0
    assert "milk" in result.output
    assert "eggs" in result.output


def test_status_on_leaf_excludes_siblings(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-location", "Door", "--id", "fridge-door", "--parent", "fridge")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge")
    _run(data_dir, "add", "purchase", "eggs", "6", "ct", "--to", "fridge-door")

    result = _run(data_dir, "status", "fridge-door")
    assert result.exit_code == 0
    assert "eggs" in result.output
    assert "milk" not in result.output


def test_another_user_cannot_write_into_alices_log(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")

    monkeypatch.setattr("getpass.getuser", lambda: "bob")
    result = _run(data_dir, "add", "purchase", "eggs", "6", "ct", "--to", "pantry")
    assert result.exit_code == 0, result.output

    assert paths.log_path(data_dir, "alice").exists()
    assert paths.log_path(data_dir, "bob").exists()
    alice_lines = paths.log_path(data_dir, "alice").read_text().splitlines()
    assert len(alice_lines) == 1
