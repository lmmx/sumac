from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from sumac import SCHEMA_VERSION, config, paths, store
from sumac.errors import UnknownLocationError, UnknownProductError
from sumac.models import Location, Product, Quantity


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


def test_add_and_load_product(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_product(data_dir, key, osuser, Product(id="milk", name="Milk", unit="l"))
    products = config.load_products(data_dir, key)
    assert products["milk"].name == "Milk"
    assert products["milk"].unit == "l"


def test_product_latest_revision_wins(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_product(data_dir, key, osuser, Product(id="milk", name="Milk", unit="l"))
    config.add_product(data_dir, key, osuser, Product(id="milk", name="Whole Milk", unit="l"))
    products = config.load_products(data_dir, key)
    assert len(products) == 1
    assert products["milk"].name == "Whole Milk"


def test_retire_product_marks_retired_without_deleting(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_product(data_dir, key, osuser, Product(id="milk", name="Milk", unit="l"))
    config.retire_product(data_dir, key, osuser, "milk")
    products = config.load_products(data_dir, key)
    assert products["milk"].retired is True
    assert products["milk"].name == "Milk"


def test_retire_unknown_product_raises(data_dir: Path, osuser: str, key: bytes) -> None:
    with pytest.raises(UnknownProductError):
        config.retire_product(data_dir, key, osuser, "nonexistent")


def test_build_config_splits_active_from_known_products(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_product(data_dir, key, osuser, Product(id="milk", name="Milk", unit="l"))
    config.add_product(data_dir, key, osuser, Product(id="eggs", name="Eggs", unit="ct"))
    config.retire_product(data_dir, key, osuser, "eggs")

    cfg = config.build_config(data_dir, key)
    assert set(cfg.known_products) == {"milk", "eggs"}
    assert set(cfg.active_products) == {"milk"}


def test_locations_and_products_coexist_in_shared_stream(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """Locations and products both live in `store.CONFIG_STREAM_ID`; each loader
    must only see its own kind."""
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    config.add_product(data_dir, key, osuser, Product(id="milk", name="Milk", unit="l"))
    config.add_location(data_dir, key, osuser, Location(id="pantry", name="Pantry"))
    config.add_product(data_dir, key, osuser, Product(id="eggs", name="Eggs", unit="ct"))

    assert set(config.load_locations(data_dir, key)) == {"fridge", "pantry"}
    assert set(config.load_products(data_dir, key)) == {"milk", "eggs"}


def test_old_shape_location_record_without_product_key_still_loads(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """A config line written before products (or `retired`) existed has only
    `location`, no `product` key, and no `retired` key inside `location` — must
    keep validating unchanged."""
    obj = {
        "schema_version": SCHEMA_VERSION,
        "ts": datetime.now(UTC).isoformat(),
        "actor": osuser,
        "location": {"id": "fridge", "name": "Fridge", "parent_id": None, "metadata": {}},
    }
    store.append(data_dir, key, store.CONFIG_STREAM_ID, obj)
    locations = config.load_locations(data_dir, key)
    assert locations["fridge"].name == "Fridge"
    assert locations["fridge"].retired is False
    assert config.load_products(data_dir, key) == {}


def test_malformed_config_record_becomes_anomaly_others_still_resolve(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    """A record setting neither (or both) of location/product must not take the
    rest of config down with it — same blast-radius principle as the main log."""
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    store.append(
        data_dir,
        key,
        store.CONFIG_STREAM_ID,
        {
            "schema_version": SCHEMA_VERSION,
            "ts": datetime.now(UTC).isoformat(),
            "actor": osuser,
        },  # neither location nor product set
    )
    config.add_location(data_dir, key, osuser, Location(id="pantry", name="Pantry"))

    cfg = config.build_config(data_dir, key)
    assert any(a.reason == "invalid_config_record" for a in cfg.anomalies)
    assert set(cfg.known_locations) == {"fridge", "pantry"}


def test_schema_too_new_config_record_becomes_anomaly_others_still_resolve(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    obj = {
        "schema_version": SCHEMA_VERSION + 1,
        "ts": datetime.now(UTC).isoformat(),
        "actor": osuser,
        "location": {"id": "pantry", "name": "Pantry", "parent_id": None, "metadata": {}},
    }
    store.append(data_dir, key, store.CONFIG_STREAM_ID, obj)

    cfg = config.build_config(data_dir, key)
    assert any(a.reason == "schema_too_new" for a in cfg.anomalies)
    assert set(cfg.known_locations) == {"fridge"}


def test_corrupted_config_line_becomes_anomaly_others_still_resolve(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_location(data_dir, key, osuser, Location(id="fridge", name="Fridge"))
    config_path = paths.config_path(data_dir)
    with config_path.open("a", encoding="utf-8") as f:
        f.write("not-valid-base64!!!\n")
    config.add_location(data_dir, key, osuser, Location(id="pantry", name="Pantry"))

    cfg = config.build_config(data_dir, key)
    assert any(a.reason == "line_failure" for a in cfg.anomalies)
    assert set(cfg.known_locations) == {"fridge", "pantry"}


def test_product_conversions_round_trip(data_dir: Path, osuser: str, key: bytes) -> None:
    product = Product(
        id="rice-pudding",
        name="Rice Pudding",
        unit="g",
        conversions={"jar": Decimal("340")},
    )
    config.add_product(data_dir, key, osuser, product)
    loaded = config.load_products(data_dir, key)["rice-pudding"]
    assert loaded.unit == "g"
    assert loaded.conversions == {"jar": Decimal("340")}


def test_convert_identity_when_unit_matches_canonical(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_product(data_dir, key, osuser, Product(id="milk", name="Milk", unit="l"))
    cfg = config.build_config(data_dir, key)
    assert cfg.convert("milk", Decimal("2"), "l") == Quantity(Decimal("2"), "l")


def test_convert_applies_conversion_ratio(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_product(
        data_dir,
        key,
        osuser,
        Product(
            id="rice-pudding", name="Rice Pudding", unit="g", conversions={"jar": Decimal("340")}
        ),
    )
    cfg = config.build_config(data_dir, key)
    result = cfg.convert("rice-pudding", Decimal("2"), "jar")
    assert result is not None
    assert result.amount == Decimal("680")
    assert result.unit == "g"


def test_convert_returns_none_for_unknown_product(data_dir: Path, osuser: str, key: bytes) -> None:
    cfg = config.build_config(data_dir, key)
    assert cfg.convert("nonexistent", Decimal("1"), "l") is None


def test_convert_returns_none_for_unconvertible_unit(
    data_dir: Path, osuser: str, key: bytes
) -> None:
    config.add_product(data_dir, key, osuser, Product(id="milk", name="Milk", unit="l"))
    cfg = config.build_config(data_dir, key)
    assert cfg.convert("milk", Decimal("1"), "kg") is None


def test_can_convert_mirrors_convert(data_dir: Path, osuser: str, key: bytes) -> None:
    config.add_product(
        data_dir,
        key,
        osuser,
        Product(
            id="rice-pudding", name="Rice Pudding", unit="g", conversions={"jar": Decimal("340")}
        ),
    )
    cfg = config.build_config(data_dir, key)
    assert cfg.can_convert("rice-pudding", "g") is True
    assert cfg.can_convert("rice-pudding", "jar") is True
    assert cfg.can_convert("rice-pudding", "oz") is False
    assert cfg.can_convert("nonexistent", "g") is False
