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


def _tree() -> dict[str, Location]:
    return {
        "fridge": Location(id="fridge", name="Fridge"),
        "fridge-door": Location(id="fridge-door", name="Door", parent_id="fridge"),
        "fridge-shelf-1": Location(id="fridge-shelf-1", name="Shelf 1", parent_id="fridge"),
        "fridge-shelf-1-bin": Location(
            id="fridge-shelf-1-bin", name="Bin", parent_id="fridge-shelf-1"
        ),
        "pantry": Location(id="pantry", name="Pantry"),
    }


def test_descendants_includes_root_and_nested_children() -> None:
    locations = _tree()
    assert config.descendants(locations, "fridge") == {
        "fridge",
        "fridge-door",
        "fridge-shelf-1",
        "fridge-shelf-1-bin",
    }


def test_descendants_of_leaf_is_itself() -> None:
    locations = _tree()
    assert config.descendants(locations, "fridge-door") == {"fridge-door"}


def test_descendants_of_unknown_id_is_itself() -> None:
    locations = _tree()
    assert config.descendants(locations, "nonexistent") == {"nonexistent"}


def test_descendants_does_not_cross_siblings() -> None:
    locations = _tree()
    assert "pantry" not in config.descendants(locations, "fridge")


def test_location_path_joins_ancestor_names() -> None:
    locations = _tree()
    assert config.location_path(locations, "fridge-shelf-1-bin") == "Fridge > Shelf 1 > Bin"


def test_location_path_of_root_is_its_own_name() -> None:
    locations = _tree()
    assert config.location_path(locations, "fridge") == "Fridge"


def test_location_path_of_unknown_id_falls_back_to_id() -> None:
    locations = _tree()
    assert config.location_path(locations, "nonexistent") == "nonexistent"
