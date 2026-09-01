"""Tests for the in-process agent."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from sumac import agent, config, ledger, models, paths, store, vault as sumac_vault
from sumac.cli import app
from sealedlog import Vault
from typer.testing import CliRunner
import json

runner = CliRunner()
PASSPHRASE_ENV = {"SUMAC_PASSPHRASE": "test-pass"}


def _real_key(data_dir: Path) -> bytes:
    """Get the actual key from a vault."""
    vault_dict = json.loads(paths.vault_path(data_dir).read_text(encoding="utf-8"))
    vault = Vault.from_dict(vault_dict)
    return sumac_vault.unlock(vault, PASSPHRASE_ENV["SUMAC_PASSPHRASE"])


def _run(data_dir: Path, *args: str, env: dict[str, str] = PASSPHRASE_ENV):
    """Run a CLI command."""
    return runner.invoke(app, [*args, "--data-dir", str(data_dir)], env=env)


@pytest.fixture(autouse=True)
def _osuser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "alice")


def test_search_inventory_exact_match(data_dir: Path) -> None:
    """Test searching for a product that exists."""
    # Setup
    _run(data_dir, "init")
    key = _real_key(data_dir)
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge")

    # Test
    inventory = ledger.build_inventory(data_dir, key)
    cfg = config.build_config(data_dir, key)
    result = agent.search_inventory(
        "milk",
        inventory=inventory,
        locations=cfg.known_locations,
        products=cfg.known_products,
    )

    assert isinstance(result, agent.SearchResult)
    assert result.product_id == "milk"
    assert result.product_name == "Milk"
    assert len(result.locations) == 1
    assert result.locations[0].id == "fridge"
    assert result.locations[0].amount == Decimal("1")


def test_search_inventory_not_found(data_dir: Path) -> None:
    """Test searching for a product that doesn't exist."""
    # Setup
    _run(data_dir, "init")
    key = _real_key(data_dir)
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")

    # Test
    inventory = ledger.build_inventory(data_dir, key)
    cfg = config.build_config(data_dir, key)
    result = agent.search_inventory(
        "cheese",
        inventory=inventory,
        locations=cfg.known_locations,
        products=cfg.known_products,
    )

    assert isinstance(result, agent.AgentError)
    assert "cheese" in str(result).lower()


def test_search_inventory_ambiguous(data_dir: Path) -> None:
    """Test searching with ambiguous results."""
    # Setup
    _run(data_dir, "init")
    key = _real_key(data_dir)
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    _run(data_dir, "config", "add-product", "Almond Milk", "l", "--id", "almond-milk")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge")
    _run(data_dir, "add", "purchase", "almond-milk", "1", "l", "--to", "fridge")

    # Test
    inventory = ledger.build_inventory(data_dir, key)
    cfg = config.build_config(data_dir, key)
    result = agent.search_inventory(
        "milk",
        inventory=inventory,
        locations=cfg.known_locations,
        products=cfg.known_products,
    )

    assert isinstance(result, agent.AgentError)
    assert "ambiguous" in str(result).lower()


def test_consume_success(data_dir: Path) -> None:
    """Test consuming a product."""
    # Setup
    _run(data_dir, "init")
    key = _real_key(data_dir)
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    _run(data_dir, "add", "purchase", "milk", "2", "l", "--to", "fridge")

    # Test
    inventory = ledger.build_inventory(data_dir, key)
    cfg = config.build_config(data_dir, key)
    result = agent.consume(
        "milk",
        "1",
        inventory=inventory,
        locations=cfg.known_locations,
        products=cfg.known_products,
        cfg=cfg,
    )

    assert isinstance(result, agent.ToolSuccess)
    assert "consumed" in result.message.lower()
    assert len(result.writes) > 0

    # Apply writes and verify
    for write in result.writes:
        store.append(data_dir, key, write.stream, write.obj)

    new_inventory = ledger.build_inventory(data_dir, key)
    milk_at_fridge = new_inventory.at("fridge").get("milk")
    assert milk_at_fridge is not None
    assert milk_at_fridge.amount == Decimal("1")


def test_consume_not_found(data_dir: Path) -> None:
    """Test consuming a product that doesn't exist."""
    # Setup
    _run(data_dir, "init")
    key = _real_key(data_dir)
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")

    # Test
    inventory = ledger.build_inventory(data_dir, key)
    cfg = config.build_config(data_dir, key)
    result = agent.consume(
        "milk",
        "1",
        inventory=inventory,
        locations=cfg.known_locations,
        products=cfg.known_products,
        cfg=cfg,
    )

    assert isinstance(result, agent.AgentError)
    assert "not found" in str(result).lower() or "no" in str(result).lower()


def test_move_success(data_dir: Path) -> None:
    """Test moving a product between locations."""
    # Setup
    _run(data_dir, "init")
    key = _real_key(data_dir)
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge")

    # Test
    inventory = ledger.build_inventory(data_dir, key)
    cfg = config.build_config(data_dir, key)
    result = agent.move(
        "milk",
        "pantry",
        None,
        inventory=inventory,
        locations=cfg.known_locations,
        products=cfg.known_products,
        cfg=cfg,
    )

    assert isinstance(result, agent.ToolSuccess)
    assert "moved" in result.message.lower()
    assert len(result.writes) > 0

    # Apply writes and verify
    for write in result.writes:
        store.append(data_dir, key, write.stream, write.obj)

    new_inventory = ledger.build_inventory(data_dir, key)
    milk_at_fridge = new_inventory.at("fridge").get("milk")
    assert milk_at_fridge is None or milk_at_fridge.amount == Decimal(0)
    milk_at_pantry = new_inventory.at("pantry").get("milk")
    assert milk_at_pantry is not None
    assert milk_at_pantry.amount == Decimal("1")


def test_move_ambiguous_location(data_dir: Path) -> None:
    """Test moving when location is ambiguous."""
    # Setup
    _run(data_dir, "init")
    key = _real_key(data_dir)
    _run(data_dir, "config", "add-location", "Fridge Shelf 1", "--id", "fridge-1")
    _run(data_dir, "config", "add-location", "Fridge Shelf 2", "--id", "fridge-2")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge-1")

    # Test
    inventory = ledger.build_inventory(data_dir, key)
    cfg = config.build_config(data_dir, key)
    result = agent.move(
        "milk",
        "fridge",
        None,
        inventory=inventory,
        locations=cfg.known_locations,
        products=cfg.known_products,
        cfg=cfg,
    )

    assert isinstance(result, agent.AgentError)
    assert "ambiguous" in str(result).lower()


def test_ask_command_searches_inventory(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the ask command (without actually running model)."""
    # Setup
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    _run(data_dir, "init")
    key = _real_key(data_dir)
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge")

    # Test that process_tool_call works (this is what the LLM would call)
    result = agent.process_tool_call(
        "search_inventory",
        {"query": "milk"},
        data_dir=data_dir,
        key=key,
    )

    assert "milk" in result.lower()
    assert "fridge" in result.lower()
