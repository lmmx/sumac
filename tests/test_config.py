from __future__ import annotations

from pathlib import Path

import pytest

from sumac import config
from sumac.errors import UnknownLocationError
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


def test_retire_location_marks_retired_without_deleting(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    config.retire_location(data_dir, key, osuser, "fridge")
    locations = config.load_locations(data_dir, key)
    assert locations["fridge"].retired is True
    assert locations["fridge"].name == "Fridge"


def test_retire_location_preserves_parent_and_name(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    config.add_location(
        data_dir, key, osuser, Location(id="fridge-door", name="Door", parent_id="fridge")
    )
    config.retire_location(data_dir, key, osuser, "fridge-door")
    loc = config.load_locations(data_dir, key)["fridge-door"]
    assert loc.retired is True
    assert loc.parent_id == "fridge"
    assert loc.name == "Door"


def test_retire_unknown_location_raises(data_dir: Path, osuser: str, key: bytes) -> None:
    with pytest.raises(UnknownLocationError):
        config.retire_location(data_dir, key, osuser, "nonexistent")


def test_build_config_splits_active_from_known(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    config.add_location(data_dir, key, osuser, Location(id="pantry", name="Pantry"))
    config.retire_location(data_dir, key, osuser, "pantry")

    cfg = config.build_config(data_dir, key)
    assert set(cfg.known_locations) == {"fridge", "pantry"}
    assert set(cfg.active_locations) == {"fridge"}
    assert cfg.anomalies == ()


def test_build_config_detects_direct_cycle(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, Location(id="a", name="A", parent_id="b"))
    config.add_location(data_dir, key, osuser, Location(id="b", name="B", parent_id="a"))

    cfg = config.build_config(data_dir, key)
    assert any(a.reason == "circular_parent" for a in cfg.anomalies)
    # both known and active views still resolve — a cycle is a diagnostic, not a crash
    assert set(cfg.known_locations) == {"a", "b"}
    assert set(cfg.active_locations) == {"a", "b"}


def test_build_config_detects_self_parent(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_location(data_dir, key, osuser, Location(id="a", name="A", parent_id="a"))
    cfg = config.build_config(data_dir, key)
    assert any(a.reason == "circular_parent" for a in cfg.anomalies)


def test_build_config_no_cycle_for_normal_tree(data_dir: Path, osuser: str, key: bytes) -> None:
    for loc in _tree().values():
        config.add_location(data_dir, key, osuser, loc)
    cfg = config.build_config(data_dir, key)
    assert cfg.anomalies == ()
