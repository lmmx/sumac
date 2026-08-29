# sumac

**🍋 sumac: home grocery inventory app**

Encrypted-at-rest grocery inventory for a household sharing one git repo and one passphrase.
Locations, products, and quantities are never visible to someone holding the repo without the
passphrase — not even in file or directory names. See `docs/FORMAT.md` for the on-disk format
and threat model, and `docs/LAYOUT.md` for what's read-only vs mutable.

## Install

```sh
uv sync
```

Depends on [`sealedlog`](https://pypi.org/project/sealedlog/) (the encrypted append-only log
primitive) from PyPI.

## Passphrase

Set `SUMAC_PASSPHRASE`, or sumac will prompt interactively. The passphrase is shared by every
user of the household's vault.

## Usage

```sh
sumac init                                              # once, creates data/
sumac config add-location "Fridge" --id fridge
sumac config add-location "Pantry" --id pantry
sumac config show

sumac add purchase milk 2 l --to fridge
sumac add consumption milk 1 l --from fridge
sumac add movement rice 1 kg --from pantry --to fridge
sumac snapshot fridge "milk=1/l" "eggs=6/ct"            # reconciliation: resets fridge's products

sumac status                                             # current inventory, all locations
sumac status fridge                                      # current inventory, one location
sumac find milk                                           # where is milk right now?
sumac log                                                 # full ordered event log
sumac verify                                              # re-authenticate every line; check actors
```

All commands take `--data-dir` (default `data`, or `$SUMAC_DATA_DIR`).

### Locations nest

A location can have a parent, so shelves, doors, bins, drawers — anything — nest under a
container to arbitrary depth. There's no separate "shelf" or "grid" type; a sub-location is just
another location with `--parent` set.

```sh
sumac config add-location "Door" --id fridge-door --parent fridge
sumac config add-array "Shelf" --parent fridge --count 4       # Shelf 1..4 under fridge
sumac config add-grid "Bin" --parent pantry --rows 3 --cols 4  # Bin R1C1..R3C4 under pantry
sumac config show                                              # renders the tree
```

`sumac status <location>` and `sumac find` both include everything nested under a location, not
just that exact node — `sumac status fridge` sums the fridge itself, its door, and its shelves in
one pass. Query a sub-location directly (e.g. `sumac status fridge-door`) to scope to just that
node and its own descendants.

## Development

```sh
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```
