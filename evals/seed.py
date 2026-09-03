"""Builds one fixture family's data directory as real `sumac` CLI
invocations — see docs/journal/2026-09-02-eval-suite.md, "Fixture families".

Location *structure* is shared across every family (see the module
docstring in `vocab.py`); `ROLE_LOCATIONS` and `MOVE_DESTINATION` map a
`FamilyVocab` role to the location id that structure gives it, so
`generate.py` can build a case's expected location without hand-repeating
the tree here.
"""

from __future__ import annotations

import json
from pathlib import Path

from sealedlog import Vault
from typer.testing import CliRunner

from evals.vocab import FamilyVocab
from sumac import config as sumac_config
from sumac import passphrase as sumac_passphrase
from sumac import paths
from sumac import vault as sumac_vault
from sumac.cli import app
from sumac.models import Location

# Shared with `conftest.py`'s fixtures (rail 1) and used by
# `build_families_standalone` below — one definition, so a script-driven
# seed and a pytest-fixture-driven seed of the same families are directly
# comparable rather than accidentally using different passphrases/osusers.
EVAL_PASSPHRASE = "sumac-eval-fixed-passphrase-not-a-secret"
EVAL_OSUSER = "sumac-eval"

# The single source of truth for the shared location tree — both
# `_seed_locations`' CLI invocations below and `location_path` are checked
# against this, so a display path used to build a prompt can never drift
# from what the seeded `sumac` instance actually resolves it to. Verified
# to match what `config add-array`/`config add-grid` produce
# (`src/sumac/cli.py:195-241`): an array's id is `{prefix}-{i}`, name
# `f"{name} {i}"`; a grid's id is `{prefix}-r{r}c{c}`, name
# `f"{name} R{r}C{c}"`.
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
    *(
        Location(id=f"pantry-black-unit-r{r}c{c}", name=f"Black Unit R{r}C{c}", parent_id="pantry")
        for r in range(1, 5)
        for c in range(1, 3)
    ),
    Location(id="cupboard-below-hob", name="Cupboard Below The Hob"),
    Location(id="freezer", name="Big Freezer"),
    *(
        Location(id=f"freezer-drawer-{i}", name=f"Drawer {i}", parent_id="freezer")
        for i in range(1, 5)
    ),
    Location(id="storage", name="Storage"),
)

LOCATIONS_BY_ID: dict[str, Location] = {loc.id: loc for loc in LOCATIONS}


def location_path(location_id: str) -> str:
    """Display path for a location id in the shared tree, e.g. `"Pantry >
    White Unit R2C3"` — the same value `decide._resolve_location` accepts
    back and `config.location_path` would compute against a real, seeded
    `Config`, computed here from `LOCATIONS` directly so a prompt built by
    `generate.py` can never name a path the real tree doesn't produce."""
    return sumac_config.location_path(LOCATIONS_BY_ID, location_id)


# Every seeded product's location, by the `FamilyVocab` field name it comes
# from. `category_stocked` and the others below are deliberately spread
# across distinct grid cells and drawers so no two roles collide.
ROLE_LOCATIONS: dict[str, str] = {
    "unit_collision": "fridge-main-shelf-2",
    "near_miss_brand": "pantry-white-unit-r2c3",
    "discriminator_a": "freezer-drawer-1",
    "discriminator_b": "freezer-drawer-2",
    "shared_word_decoy": "pantry-white-unit-r1c2",
    "rice": "pantry-black-unit-r1c1",
    "consumption_target": "pantry-white-unit-r3c1",
    "movement_source": "freezer-drawer-3",
    "category_stocked": "pantry-white-unit-r2c1",
}

# `move.explicit`'s destination — empty in every family, so a move there is
# unambiguous. Matches the real "move 1 tub of ragu... to the fridge door"
# transcript's own destination.
MOVE_DESTINATION = "fridge-door"

# Free grid cell for `add.positional` (blocked — see the eval spec).
POSITIONAL_TARGET = "pantry-white-unit-r3c3"
POSITIONAL_GRID_COLOUR = "white"
POSITIONAL_ROW = 3
POSITIONAL_COL = 3


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
    invoke(
        "config", "add-grid", "Black Unit",
        "--rows", "4", "--cols", "2", "--parent", "pantry", "--id-prefix", "pantry-black-unit",
    )  # fmt: skip

    invoke("config", "add-location", "Cupboard Below The Hob", "--id", "cupboard-below-hob")

    invoke("config", "add-location", "Big Freezer", "--id", "freezer")
    invoke(
        "config", "add-array", "Drawer",
        "--count", "4", "--parent", "freezer", "--id-prefix", "freezer-drawer",
    )  # fmt: skip

    invoke("config", "add-location", "Storage", "--id", "storage")


def _seed_products(invoke, family: FamilyVocab) -> None:  # noqa: ANN001
    for role, location_id in ROLE_LOCATIONS.items():
        product = getattr(family, role)
        invoke("add", "purchase", product.id, product.amount, product.unit, "--to", location_id)


def build_family(base_dir: Path, family: FamilyVocab, *, passphrase: str) -> tuple[Path, bytes]:
    """Seeds `family` into `base_dir / family.id` and returns `(data_dir,
    key)`. `key` is read back through `sumac_vault.unlock` directly rather
    than through `sumac.passphrase.get_key` — that function caches the
    derived key in a process-global and ignores which `Vault` it was given
    on a cache hit (`src/sumac/passphrase.py:14,24-29`), so seeding more
    than one family's vault within a process would hand every family after
    the first the wrong raw key. `reset_cache()` before this family's own
    CLI commands keeps *those* commands correct (they all share this one
    vault, so cache reuse across them is fine); reading the key back
    independently here keeps the value this function returns correct
    regardless of what a later family's seeding does to that cache
    afterward."""
    data_dir = base_dir / family.id
    runner = CliRunner()

    def invoke(*args: str) -> None:
        result = runner.invoke(app, [*args, "--data-dir", str(data_dir)])
        if result.exit_code != 0:
            raise RuntimeError(
                f"eval seeding failed for {family.id}: {' '.join(args)!r}\n{result.output}"
            )

    sumac_passphrase.reset_cache()
    invoke("init")
    _seed_locations(invoke)
    _seed_products(invoke, family)

    vault = Vault.from_dict(json.loads(paths.vault_path(data_dir).read_text(encoding="utf-8")))
    key = sumac_vault.unlock(vault, passphrase)
    return data_dir, key


def build_families_standalone(
    n: int = 10, *, base_dir: Path | None = None
) -> dict[str, tuple[Path, bytes]]:
    """Seeds `FAMILIES[:n]` outside of pytest — for `report.py`'s baseline
    rows, which need seeded inventory but not a full pytest session.
    Mutates process-global state for the life of the calling process
    (`SUMAC_PASSPHRASE`, `getpass.getuser`) rather than restoring it on
    exit, unlike `conftest.py`'s equivalent fixtures — acceptable for a
    short-lived script's `main()`, not for reuse inside a longer-running
    process."""
    import getpass
    import os
    import tempfile

    from evals.vocab import FAMILIES

    os.environ["SUMAC_PASSPHRASE"] = EVAL_PASSPHRASE
    getpass.getuser = lambda: EVAL_OSUSER  # ty: ignore[invalid-assignment]
    root = base_dir or Path(tempfile.mkdtemp(prefix="sumac-eval-standalone-"))
    return {
        family.id: build_family(root, family, passphrase=EVAL_PASSPHRASE) for family in FAMILIES[:n]
    }
