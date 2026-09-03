"""Builds one realistic inventory via real `sumac` CLI invocations. See
docs/journal/2026-09-02-eval-suite.md for why a real CLI run rather than
writing encrypted records directly: the dataset stays expressible, and
reviewable, as plain `sumac` commands.

Vocabulary matches the real runs recorded in
docs/journal/2026-09-01-ask-agent-design.md — including the exact request
("Add 1 bag of Basmati Rice (1kg) next to the existing jug of Basmati
Rice") that produced the repeated-`sumac_discover_inventory` context
overflow diagnosed in docs/journal/2026-09-02-eval-suite.md's second
implementation pass. Kept as a single fixture, not a family per case
category — see that entry for why ten generated families were cut back to
one hand-picked one.
"""

from __future__ import annotations

import json
from pathlib import Path

from sealedlog import Vault
from typer.testing import CliRunner

from sumac import config as sumac_config
from sumac import passphrase as sumac_passphrase
from sumac import paths
from sumac import vault as sumac_vault
from sumac.cli import app
from sumac.models import Location

EVAL_PASSPHRASE = "sumac-eval-fixed-passphrase-not-a-secret"
EVAL_OSUSER = "sumac-eval"

# Single source of truth for the location tree, checked against what
# `config add-array`/`config add-grid` actually produce
# (`src/sumac/cli.py:195-241`) by `test_location_path_matches_real_config`
# in `test_fixtures.py` — see `location_path` below.
LOCATIONS: tuple[Location, ...] = (
    Location(id="fridge", name="Fridge"),
    Location(id="fridge-door", name="Door", parent_id="fridge"),
    Location(id="fridge-main-shelves", name="Main Shelves", parent_id="fridge"),
    *(
        Location(id=f"fridge-main-shelf-{i}", name=f"Shelf {i}", parent_id="fridge-main-shelves")
        for i in range(1, 5)
    ),
    Location(id="fridge-bottle-rack", name="Bottle Rack", parent_id="fridge"),
    Location(id="pantry", name="Pantry"),
    *(
        Location(id=f"pantry-white-unit-r{r}c{c}", name=f"White Unit R{r}C{c}", parent_id="pantry")
        for r in range(1, 4)
        for c in range(1, 5)
    ),
    Location(id="freezer", name="Big Freezer"),
    *(
        Location(id=f"freezer-drawer-{i}", name=f"Drawer {i}", parent_id="freezer")
        for i in range(1, 4)
    ),
)
LOCATIONS_BY_ID: dict[str, Location] = {loc.id: loc for loc in LOCATIONS}


def location_path(location_id: str) -> str:
    return sumac_config.location_path(LOCATIONS_BY_ID, location_id)


# id, amount, unit, location_id — the products every test in test_agent.py
# refers to by name. Kept as one flat table, not a dataclass-per-role
# system: there is only one fixture now, so nothing needs to parameterise
# over which family a role belongs to.
PRODUCTS: tuple[tuple[str, str, str, str], ...] = (
    ("Chopped Tomatoes", "1", "jar", "fridge-main-shelf-2"),
    ("Ocado Italian Chopped Tomatoes", "3", "cans", "pantry-white-unit-r2c3"),
    ("Salted Butter", "1", "pack", "freezer-drawer-1"),
    ("Unsalted Butter", "2", "packs", "freezer-drawer-2"),
    ("Butter Beans", "2", "cans", "pantry-white-unit-r1c2"),
    ("Basmati Rice", "1", "jug", "pantry-white-unit-r1c1"),
    ("Strawberry Jam", "1", "jar", "pantry-white-unit-r3c1"),
    ("Ragu", "2", "tubs", "freezer-drawer-3"),
    ("Fusilli Pasta", "500", "g", "pantry-white-unit-r2c1"),
)
# Never seeded — the "does add-with-no-match discover a new product" case.
ABSENT_PRODUCT = "Irn-Bru Zero"


def _seed_locations(invoke) -> None:  # noqa: ANN001
    invoke("config", "add-location", "Fridge", "--id", "fridge")
    invoke("config", "add-location", "Door", "--id", "fridge-door", "--parent", "fridge")
    invoke(
        "config", "add-location", "Main Shelves",
        "--id", "fridge-main-shelves", "--parent", "fridge",
    )  # fmt: skip
    invoke(
        "config", "add-array", "Shelf",
        "--count", "4", "--parent", "fridge-main-shelves", "--id-prefix", "fridge-main-shelf",
    )  # fmt: skip
    invoke(
        "config", "add-location", "Bottle Rack",
        "--id", "fridge-bottle-rack", "--parent", "fridge",
    )  # fmt: skip
    invoke("config", "add-location", "Pantry", "--id", "pantry")
    invoke(
        "config", "add-grid", "White Unit",
        "--rows", "3", "--cols", "4", "--parent", "pantry", "--id-prefix", "pantry-white-unit",
    )  # fmt: skip
    invoke("config", "add-location", "Big Freezer", "--id", "freezer")
    invoke(
        "config", "add-array", "Drawer",
        "--count", "3", "--parent", "freezer", "--id-prefix", "freezer-drawer",
    )  # fmt: skip


def build(base_dir: Path, *, passphrase: str = EVAL_PASSPHRASE) -> tuple[Path, bytes]:
    """Seeds `PRODUCTS` into `base_dir / "inventory"` and returns
    `(data_dir, key)`. `key` is read back through `sealedlog.Vault.unlock`
    directly rather than through `sumac.passphrase.get_key`, which caches
    the derived key in a process-global and ignores which `Vault` it was
    given on a cache hit (`src/sumac/passphrase.py:14,24-29`) — harmless
    with only one vault per process, but reading it back independently
    costs nothing and avoids relying on that being true forever."""
    data_dir = base_dir / "inventory"
    runner = CliRunner()

    def invoke(*args: str) -> None:
        result = runner.invoke(app, [*args, "--data-dir", str(data_dir)])
        if result.exit_code != 0:
            raise RuntimeError(f"eval seeding failed: {' '.join(args)!r}\n{result.output}")

    sumac_passphrase.reset_cache()
    invoke("init")
    _seed_locations(invoke)
    for product_id, amount, unit, location_id in PRODUCTS:
        invoke("add", "purchase", product_id, amount, unit, "--to", location_id)

    vault = Vault.from_dict(json.loads(paths.vault_path(data_dir).read_text(encoding="utf-8")))
    key = sumac_vault.unlock(vault, passphrase)
    return data_dir, key
