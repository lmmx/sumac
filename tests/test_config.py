from __future__ import annotations

from pathlib import Path

from sumac import config
from sumac.models import Location


def test_add_and_load_location(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    locations = config.load_locations(data_dir, key)
    assert locations["fridge"].name == "Fridge"


def test_latest_revision_wins(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Renamed Fridge"))
    locations = config.load_locations(data_dir, key)
    assert len(locations) == 1
    assert locations["fridge"].name == "Renamed Fridge"


def test_multiple_locations(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    config.add_location(data_dir, key, osuser, Location(id="pantry", name="Pantry"))
    locations = config.load_locations(data_dir, key)
    assert set(locations) == {"fridge", "pantry"}
