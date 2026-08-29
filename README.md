# sumac
🍋 sumac: home grocery inventory app

Encrypted-at-rest grocery inventory for a household sharing one git repo and one passphrase.
Locations, products, and quantities are never visible to someone holding the repo without the
passphrase — not even in file or directory names. See `docs/FORMAT.md` for the on-disk format
and threat model, and `docs/LAYOUT.md` for what's read-only vs mutable.

## Install

```sh
uv sync
```

Depends on [`sealedlog`](/mnt/sealedlog) (the encrypted append-only log primitive) as an
editable path dependency — see `[tool.uv.sources]` in `pyproject.toml` if your checkout lives
somewhere else.

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

## Development

```sh
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```
