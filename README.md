# sumac

**🍋 sumac: home grocery inventory app**

Encrypted-at-rest grocery inventory for a household sharing one git repo and one passphrase.
Locations, products, and quantities are never visible to someone holding the repo without the
passphrase — not even in file or directory names. See `docs/FORMAT.md` for the on-disk format
and threat model, and `docs/LAYOUT.md` for what's read-only vs mutable.

## Install

Published on PyPI as `sumac-home` (`sumac` was taken); the command is still `sumac`.

```sh
uv tool install sumac-home     # or: pip install sumac-home
```

For development (this checkout):

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

## Evaluating agent behavior

`sumac ask`'s agent (below) has a behavioural eval suite — a seeded inventory and a set of
find/add/remove/reject scenarios, run against either a real local model or a deployed Modal
endpoint. It's how a prompt or model change actually gets checked, not just tried once by hand.

```sh
uv run pytest evals -v --eval-model qwen3.5-4b
```

See `evals/README.md` for the full guide (comparing models, comparing prompt variants, reading the
output) and `docs/MODAL.md` if you want faster iteration against a deployed Modal endpoint instead
of local inference.

## Development

See `docs/DEVELOPMENT.md`.

## Optional: natural-language input (`sumac ask`)

`sumac ask` parses freeform text ("consume 1 jar of jam") into structured commands using a local
LLM via [`mistralrs`](https://github.com/EricLBuehler/mistral.rs). It's an optional dependency
group, the rest of `sumac` works without it.

```sh
uv sync --group ask          # CPU/Metal — no GPU required
```

### Reviewing what it proposes

`ask` never writes anything without showing you the plan first. Each proposed change is one line
of what changes at that location — `Fridge > Door   3 jar → 2 jar` — where the "after" comes from
folding the records the write-time gate decided on, so a consumption larger than the recorded stock
shows the zero its reconciliation produces rather than a negative number.

Alongside it, a few deterministic checks (no second model call) flag anything worth a closer look
before you accept:

| badge | what it means |
| --- | --- |
| `[unverified]` | the product name is in no search result and in no config record, so nothing the agent looked up supplied it |
| `[new product]` | accepting registers a product that doesn't exist yet |
| `[near-duplicate]` | the name is one edit away from a product you already have |
| `[new unit]` | the product is tracked in a different unit, with no conversion configured |

On a terminal the decision is an arrow-key menu (`↑`/`↓`, Enter, or the option's own letter; Esc
rejects); piped or scripted, it prints the same options and reads a typed line. On a plan with
more than one change, `p` opens a checklist to apply only some of them, and `e` picks a change and
then the one field to change. Product, unit and location fields open a picker over what the vault
already holds, filtered as you type — locations are the layout (a location has to be one that
exists), products the registry, and units every unit ever recorded, the ones already used for that
product first. Products and units also accept a value that isn't in the list: type it and the picker offers it as
new, which is how `sumac add` treats it too. The amount field takes digits only, with the arrow keys
stepping it up and down. Anything you type that
isn't an option is feedback the agent revises the plan with.

```sh
sumac ask "move the ragu to the fridge" --dry-run   # show the plan, write nothing
sumac ask "consume 1 jar of jam" --trace            # full tool-call arguments and raw results
sumac ask "where is the rice?" --stats              # per-round token counts and tok/s
sumac ask "find the butter" --stats --trace        # both, i.e. what was printed unconditionally before
```

`--debug` (raw per-round request/response dumps) implies `--stats`, and also restores mistral.rs's
own load logs — the DType, tokenizer, device map and the whole GGUF chat template — which are
otherwise suppressed. They are Rust `tracing` output filtered by `RUST_LOG`; set that yourself for
anything finer, and sumac will not override it:

```sh
RUST_LOG=info sumac ask "find the butter"                              # all of it, without --debug
RUST_LOG=mistralrs_core::gguf::chat_template=off,info sumac ask "..."  # everything but the template
```

In `--loop` mode the model is loaded once, on the first request, and reused for the rest of the
session: a fresh conversation per request, with the same loaded model behind it. Switching model
with `g` loads the new one and drops the old.

To iterate on the review screens themselves without loading a model:

```sh
uv run scripts/preview-ask-ui.py            # every screen, against fixed example plans
uv run scripts/preview-ask-ui.py --svg out/ # one SVG per screen
```

### NVIDIA GPU acceleration

```sh
uv sync --no-group ask --group ask-cuda
```

**Known upstream issue:** mistral.rs's published CUDA wheels for `0.9.1`/`0.9.2` have a broken
`RPATH` that prevents the extension from loading at all (tracked upstream:
[EricLBuehler/mistral.rs#2411](https://github.com/EricLBuehler/mistral.rs/issues/2411)). Until
that's fixed upstream, `ask-cuda` uses a wheel built from source with a local patch — see
`scripts/build-mistralrs-cuda.sh`.

To (re)build it:

```sh
./scripts/build-mistralrs-cuda.sh v0.9.2
```

This clones mistral.rs into `.build/` (gitignored), builds with `maturin` against whatever CUDA
compute capability is present on the machine you run it on (confirm with `cuobjdump --list-elf
<extension>.so | grep sm_` if you need to check), and patches the resulting wheel to embed a real
copy of `libcuda.so.1` from the system driver instead of the incorrectly-vendored one the build
produces by default — `uv` doesn't preserve symlinks from wheel zips on install, so it's a full
copy, not a link. The patched wheel lands in `vendor/wheels/` (gitignored — too large for git)
and `pyproject.toml`'s `ask-cuda` group points at it via a local path source.

**This wheel is tied to the host machine, in two separate ways.** The compiled extension is built
for whichever GPU's compute capability was present at build time — it is not portable to a
different GPU architecture. And the embedded `libcuda.so.1` is a byte-for-byte copy of *this
machine's* driver — if you update the NVIDIA driver, reinstall the OS, swap GPUs, or move to a
different machine, rebuild rather than reuse the wheel. Requires the CUDA toolkit and a Rust
toolchain (`rustc >= 1.94`) to build; neither is needed just to *run* `sumac`.

To verify a build run `uv run python -c "import mistralrs"` which should import silently without
erroring.
