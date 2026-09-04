# sumac: Eval Suite for `sumac ask` — Specification

**Status:** specification only — no code written, nothing in `evals/` yet
**Scope:** a new top-level `evals/` directory, `pyproject.toml`, `README.md`, and two commits to
`src/sumac/llm.py` pinning sampling and widening `AgentRunner.__init__`.
**Revision:** third draft. Changes from the second draft are listed under "What changed and why"
at the end of the preface.

---

## Terminology

`src/sumac/llm.py` is the *agent harness*: the thing that classifies a request, holds the
tool-calling loop, and produces an `AgentPlan`. What this entry specifies is an *eval suite* — a
dataset of requests, an expected outcome per request, and a scorer — run against that harness.
The directory is `evals/`, the word "harness" is left meaning `llm.py`.

## Motivation

Every behavioural claim in `docs/journal/2026-09-01-ask-agent-design.md` and
`docs/journal/2026-09-02-query-classifier.md` is recorded as "untested against a real model", and
the four real-model failures those entries describe — a duplicate `sumac_find_inventory` call, a
wrong-product answer from unranked results, a plain question that never triggered a search, and
the Moma pistachio milk request that searched four times and then gave up — were each found by a
person typing one request at a terminal and reading the output.

## What twenty real queries say about where the failures are

A batch of twenty `sumac ask --dry-run` requests, supplied as representative of real usage, splits
by **how the request names its location**, not by what it asks the agent to do:

| style | example | count | resolvable today |
| --- | --- | --- | --- |
| indirect via existing stock | "to the pantry, same spot as existing stock"; "with the other pasta" | 8 | yes — `_ADD_PROMPT` names this case explicitly |
| positional / ordinal | "the white pantry, 3rd position along, 3rd row down"; "left column, 4th row"; "1st drawer" | 8 | **no** |
| relational / descriptive | "the cupboard below the hob"; "the big freezer"; "storage"; "the fridge bottle rack" | 4 | only if a location's own name matches the wording |
| indirect via *absent* stock | "in the now-empty spot where the old stock was" | 1 | **no** |
| location id or display path | (none in this batch) | 0 | yes |

**The dominant style in real usage has no supported route.** `decide._resolve_location` accepts a
location id or an exact `config.location_path` display string and nothing else
(`src/sumac/decide.py:100-123`). The only way a model can obtain either is from a
`sumac_find_inventory` result that happens to contain the target location, because:

- no prompt injects the location tree — `_FIND_PROMPT`, `_ADD_PROMPT`, and `_REMOVE_PROMPT` are
  static strings with no configuration interpolated (`src/sumac/llm.py:324-404`);
- no tool enumerates locations — the four callbacks search products, consume, move, and discover
  (`src/sumac/llm.py:311-315,592-597`);
- `_DISCOVER_INVENTORY_SCHEMA`'s `to_location` property carries no `description` at all, so
  nothing states what form a location argument even takes (`src/sumac/llm.py:284-309`).

So "3rd position along, 3rd row down" is only resolvable by guessing both that the grid is called
`White Unit` and that its ids are shaped `r{row}c{col}`. The recorded transcript where a model did
produce `pantry-white-unit-r2c3` got it from a `sumac_find_inventory` result that listed it — for
a target location holding nothing, that route does not exist.

This changes what the suite is for. Cases in the two unresolvable styles are tagged `blocked`,
run, and reported in their own table, **excluded from the headline pass rate**: scoring them as
model failures would attribute a harness capability gap to the model. The capability gap itself
is recorded under Missing.

Three further shapes the batch contributes, all resolvable and all currently unexercised:

- **Missing amount or unit.** "Add Barilla rigatoni to the pantry, with the other pasta" and "Add
  M&S spaghetti to the pantry, with the other spaghetti" give no quantity. `sumac_discover_inventory`
  requires `amount` and `unit`, so a correct run asks rather than inventing a number.
- **Unit collision inside one product.** "Add 1 bag of basmati rice (1kg) to the pantry, next to
  the existing jug of basmati rice" — the stored unit and the requested unit differ, which is the
  `unit_unconvertible` path.
- **One-word discriminators.** "2 salted butters" and "2 unsalted butters" into the same drawer are
  distinct products separated by a prefix, which is the sharpest form of the wrong-product failure.

### A contamination rule this batch exposes

`_ADD_PROMPT` contains a worked example using the brand "Heinz" ("after 'Heinz Baked Beans' finds
nothing, search 'Baked Beans', not 'Heinz'"), and the batch contains "Add 1 Heinz tomato ketchup
(zero added sugar)". An eval case whose brand appears in the prompt that is being evaluated does
not measure brand-dropping behaviour; it measures recall of the prompt's own example.
`docs/journal/2026-09-02-query-classifier.md` records a prior investigation where a worked example
in a prompt turned out to be the actual cause of a behaviour under study.

**Rule:** no product, brand, or location name in `evals/vocab.py` may appear in
`CLASSIFIER_PROMPT`, `_FIND_PROMPT`, `_ADD_PROMPT`, or `_REMOVE_PROMPT`. `test_scoring.py` asserts
this over the vocabulary at collection time, so a future prompt edit that introduces an overlap
fails a model-free test rather than silently biasing a number.

## Runner: pytest, not Inspect AI

pytest for v1. Two reasons, both about existing repository structure rather than about Inspect:
`uv run pytest` is already the development loop with the vault/key fixtures a seeded data
directory needs, and keeping `cases.py`, `scoring.py`, and `generate.py` free of pytest imports
makes an Inspect `Task` wrapping them additive later.

What that costs, stated rather than argued away: epoch bookkeeping, the `.eval` log format,
`inspect view`, model-as-a-config-axis for preset sweeps, and scorer composition. `evals/run.py`,
`evals/report.py`, and `evals/compare.py` replace those, and that is several hundred lines this
repository then owns.

## Two commits to `src/sumac/llm.py`, before any eval code

The second draft proposed optional keyword arguments defaulting to `None`, on the grounds that
this left `sumac ask` byte-identical. That defence is wrong: what it preserves is not a chosen
baseline but whatever mistral.rs happens to default to, which can move under a dependency bump
with nothing catching it. `AgentRunner._build_request` sets `messages`, `model`, `tool_schemas`,
`tool_choice`, and `enable_thinking` and no sampling parameter at all
(`src/sumac/llm.py:763-771`).

**Commit 1 — pin sampling.** Module-level `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`,
`DEFAULT_MAX_TOKENS` in `llm.py`, applied in `_build_request`; a `seed` parameter on
`_build_runner` passed through to `mistralrs.Runner`. This is a deliberate behaviour change to
`sumac ask` and it is the one testable change in that commit. Proposed starting values:
temperature `0.2` (tool-call reliability on a 1-4B model favours low temperature, and the
classifier in particular is a four-way decision that should not be sampled), `top_p` `0.95`,
`max_tokens` `1024` (the largest single round in the recorded transcripts emitted 432 completion
tokens, so this leaves headroom without permitting a runaway).

**Commit 2 — widen the constructor.** `AgentRunner.__init__` gains `temperature`, `top_p`,
`max_tokens`, and `seed`, each defaulting to the commit-1 constants rather than to `None`, plus a
public `classify()` alongside `_classify`.

`evals/agent.py` and the `EvalAgentRunner` subclass the second draft specified then do not exist,
`evals/` reaches into no private name, and the eval measures the shipped configuration. The
second draft's Divergence entry about evaluating a configuration that never ships is deleted
rather than carried.

## Epochs are separate pytest sessions

`mistralrs.ChatCompletionRequest` has no `seed` field; `mistralrs.Runner.__init__` does
(`mistralrs/__init__.pyi:129-200,728-758`). The second draft recorded that as a fact and kept an
in-process epoch loop over one session-scoped runner. That is a reproducibility failure, not a
documentation note: with one shared runner the RNG stream position at case 47 of epoch 3 depends
on how many tokens every prior case generated, so `-k` filters change results, re-running a single
failing case draws a different sample than the one that failed, and `compare.py` pairing two JSON
files silently compares different sampling positions if anything upstream moved.

`evals/run.py` orchestrates instead. One epoch is one pytest session at one `--eval-seed`:

```
uv run python -m evals.run --epochs 8 --out runs/2026-09-02-qwen35/
  → pytest evals --eval-seed 1 --eval-json runs/.../seed-01.json
  → pytest evals --eval-seed 2 --eval-json runs/.../seed-02.json
  ...
```

Each session builds its own `Runner` at that seed, so any epoch reproduces exactly from its seed
alone. `run.py` does not abort on a non-zero pytest exit — a failing epoch still writes its JSON,
and a stopped run loses the reliability signal it was started for. One model load per epoch costs
seconds against a ten-minute epoch.

pytest therefore has no `--epochs` option and asserts only the single epoch in front of it.
**pass^k is computed at aggregation time by `report.py` across the per-seed files**, not asserted
inside a session.

## Statistical size, as a sizing rationale only

At n=18 hand-written cases and a pass rate near 0.8 the standard error is 9.4pp and the 95%
interval spans about +/-18pp; that size detects nothing. The paired comparison is what matters,
with `SE(delta) = sqrt(q/n)` over the discordant fraction `q`, giving `n = q (2.8/delta)^2` at 80%
power:

| target effect | q=0.08 | q=0.15 |
| --- | --- | --- |
| 10pp | 63 cases | 118 cases |
| 5pp | 251 cases | 470 cases |

Clustering reduces that. With `c` clusters of `m` cases the design effect is `1 + (m-1) rho`:

| layout | rho=0.05 | rho=0.10 |
| --- | --- | --- |
| 1 family x 120 | n_eff 20 | n_eff 11 |
| 10 families x 12 | n_eff 77 | n_eff 57 |
| 20 families x 6 | n_eff 94 | n_eff 80 |

**These tables size the suite; they do not describe any result.** `q` and `rho` above are assumed
values, and both fall out of a real comparison for free. Quoting an assumed MDE next to a measured
difference is precisely the failure the analysis exists to prevent. So `compare.py` reports the
*observed* discordance from its own McNemar table and the *observed* intraclass correlation, and
prints the realised MDE for that comparison — see "Comparing two runs".

Default layout: 10 families x ~12 generated cases. Expected `n_eff` somewhere in the 57-77 band,
so the suite is built to catch large regressions and outright breakage, and the `hard` cases are a
named tripwire rather than a percentage.

Runtime, extrapolating from the recorded qwen3-4b runs (~200 tok/s, a five-round add completing in
~2.5s of generation): 3-5s per case, ~6-10 minutes per epoch over 120 cases, roughly an hour for
`--epochs 8`. `--families 2` gives a ~90-second development loop.

## Layout

```
evals/
├── README.md            # how to run, how to add a case, how to read the summary
├── __init__.py
├── conftest.py          # options, safety rails, seeded families, per-session model
├── vocab.py             # hand-written vocabulary + reject prompts, one block per family
├── seed.py              # builds one family's inventory as sumac CLI invocations
├── generate.py          # templated case generation; gold derived from template params
├── cases.py             # expectation types, hand-written adversarial table, assembly
├── scoring.py           # canonicalisation, write-set F1, trace and reply assertions
├── baselines.py         # do-nothing / reject-everything / always-propose stubs
├── run.py               # orchestrator: N pytest sessions, one seed each
├── report.py            # aggregate per-seed JSONs: pass^k, baselines, per-template, blocked
├── compare.py           # paired McNemar + two-way cluster bootstrap over two run directories
├── test_routing.py      # QueryKind classification                        [pytest.mark.model]
├── test_proposals.py    # full propose(), write-set and trace scoring     [pytest.mark.model]
└── test_scoring.py      # scorer self-tests, null baselines, vocab hygiene — no model
```

`testpaths` stays `["tests"]`, so `uv run pytest` does not collect `evals/`.
`test_routing.py` and `test_proposals.py` carry `pytestmark = pytest.mark.model`, so
`pytest evals -m "not model"` runs everything that needs no GPU. The model fixture **skips**, not
errors, when the GGUF is absent from the local cache.

## Fixture families

Ten structurally-equivalent inventories, `fam-01`..`fam-10`, each carrying the same structural
properties with different vocabulary: a unit collision, a shared-word trap, a one-word
discriminator pair, a near-miss brand, and an absent product. Each family also names locations in
the descriptive style the query batch uses — a "Cupboard Below The Hob", a "Big Freezer", a
"Fridge Bottle Rack" — so relational references resolve through the display-path route rather
than being blocked by construction.

Vocabulary lives in `evals/vocab.py`, hand-written, one block per family, reviewable in a diff.
Family assembly is deterministic from `(family_index, vocab_block)`; nothing is committed
encrypted.

Family 1 reuses the vocabulary of the recorded real runs, so the transcripts in
`docs/journal/2026-09-01-ask-agent-design.md` stay directly comparable — minus any name colliding
with the prompt constants under the contamination rule above:

| product | amount | unit | location | structural role |
| --- | --- | --- | --- | --- |
| Chopped Tomatoes | 1 | jar | `fridge-main-shelf-2` | unit collision (registered in jars) |
| Ocado Italian Chopped Tomatoes | 3 | cans | `pantry-white-unit-r2c3` | near-miss brand |
| Salted Butter | 1 | pack | `freezer-drawer-1` | one-word discriminator |
| Unsalted Butter | 2 | packs | `freezer-drawer-1` | one-word discriminator |
| Butter Beans | 2 | cans | `pantry-white-unit-r1c2` | shared-word decoy |
| Basmati Rice | 1 | jug | `pantry-black-unit-r1c1` | unit collision against "bag" |
| Strawberry Jam | 1 | jar | `pantry-white-unit-r3c1` | consumption target |
| Ragu | 2 | tubs | `freezer-drawer-1` | movement source |
| Irn-Bru Zero | — | — | — | absent product |

Locations keep the structure the transcripts show (`fridge`, `fridge-door`,
`fridge-main-shelf-1..4`, `pantry-white-unit-r1c1..r3c4`, `pantry-black-unit-r1c1..r4c2`,
`freezer-drawer-1..3`) plus the descriptively-named ones above.

Seeding runs through `sumac.cli.app` via `typer.testing.CliRunner`, one invocation per command,
with `--data-dir` bound to that family's temp directory. `passphrase.get_key` caches the derived
key in a module global (`src/sumac/passphrase.py:14,24-29`), so Argon2id runs once per process,
not once per invocation.

## Case generation

`evals/generate.py` instantiates templates against each family, deriving the gold from the
template's own parameters rather than from a hand annotation. Templates carry two or three
phrasing variants, all drawn from the real query batch's wording.

| template | prompt shape | gold | tag |
| --- | --- | --- | --- |
| `add.location_path` | Add {n} {unit} of {product} to {location_path} | discovery to {location_id} | |
| `add.indirect_stock` | Add {n} {unit} of {product} to the {room}, same spot as existing stock | discovery to {product}'s location | |
| `add.new_product` | Add {n} {unit} of {absent_product} to {location_path} | discovery of an unseeded product | |
| `add.brand_variant` | Add {n} {unit} of {brandless_name} to {location_path} | discovery onto the branded stock | |
| `add.discriminator` | Add {n} {unit} of {unsalted_variant} to {location_path} | discovery onto the right one of the pair | |
| `add.unit_collision` | Add 1 bag of {product} ({size}) next to the existing {stored_unit} | NoWrites — `unit_unconvertible` | |
| `add.missing_amount` | Add {product} to the {room}, with the other {category} | AskOrAct, branch reported | |
| `add.positional` | Add {n} {unit} of {product} to the {grid_colour} pantry, {ordinal} position along, {ordinal} row down | discovery to the implied `r{row}c{col}` | `blocked` |
| `add.absent_spot` | Add {n} {unit} of {product} to the {room}, in the now-empty spot where the old stock was | discovery to the emptied location | `blocked` |
| `remove.partial` | Remove {n} {unit} of {product} from {location_path} | consumption from {location_id} | |
| `remove.all` | we finished the {product} | consumption of the stocked amount | |
| `move.explicit` | move {n} {unit} of {product} from {src_path} to {dst_path} | movement {src_id} to {dst_id} | |
| `find.where` | where is the {product}? | find called; reply names {location_name} | |
| `find.quantity` | how much {product} do we have? | find called; reply names amount adjacent to unit | |
| `find.shared_word` | do we have any {shared_word}? | reply names {intended}, ordered before {decoy} | |
| `reject.out_of_domain` | from that family's own reject list | classification only, no domain tool call | |

**Reject prompts vary per family.** The second draft drew them from one family-independent list,
which would have produced thirty rows over about three distinct items — inflating `n` and
deflating every interval. Each family's `vocab.py` block carries its own reject prompts, so the
rows are genuinely independent items.

The adversarial cases stay hand-written, tagged `hard`, counted once (not per family), and
reported as a named list of what flipped rather than as a rate:

| id | prompt | kind | expect |
| --- | --- | --- | --- |
| `hard-unit-conflict` | Add 1 can of Chopped Tomatoes to Fridge > Main Shelves > Shelf 2 | add | NoWrites |
| `hard-ambiguous-product` | Add 1 can of chopped tomatoes, along with the existing 3 cans | add | AskOrAct, branch reported |
| `hard-vague-move` | move the ragu to the fridge | remove | AskOrAct, branch reported |
| `hard-joke-inventory-word` | Tell me a joke about tomatoes | reject | NoWrites |
| `hard-gibberish` | asdf | reject | NoWrites |
| `hard-duplicate-search-bait` | Add 6 {brand} {product} to the same pantry cupboard as existing stock | add | discovery, find called at most twice |
| `hard-odd-destination` | Add 1 {brand} Margherita pizza to the fridge bottle rack | add | discovery to the bottle rack, unaltered |

`hard-unit-conflict` expects `kind=add` with zero writes: the classifier is right, `decide`
rejects the can-into-jar, and a correct run ends with an explanation. `hard-odd-destination`
checks the agent records the location a person named rather than substituting a more sensible one.

## Scoring

### Write-set comparison

Each `ProposedWrite` reduces to `(kind, product_id_folded, amount, unit_canonical,
from_location_id, to_location_id)`; plans compare as sets.

- **Location.** `ProposedWrite.from_location`/`to_location` hold the raw string the model emitted,
  unresolved (`src/sumac/llm.py:651-751`). `canonical_location(cfg, value)` resolves a display path
  to its id, passes an id through, and returns the raw value unchanged otherwise, so an
  unresolvable location scores as a mismatch rather than raising.
- **Unit.** An explicit synonym table over the units the families use — `can/cans`, `jar/jars`,
  `carton/cartons`, `pack/packs`, `tub/tubs`, `box/boxes`, `bag/bags`, `bottle/bottles`,
  `tin/tins`, `jug/jugs`, plus `kg`, `g`, `l`, `ml`, `ct` unfolded. Units outside the table compare
  verbatim. Stripping a trailing `s`, as the first draft specified, maps `boxes` to `boxe`.
- **Product.** Case-insensitive exact comparison, no fuzzy matching.
- **Amount.** `Decimal` equality.

Exact set match gates the assertion; set-F1 is reported so a plan with two of three writes right
is distinguishable from one with none.

### Trace and reply assertions

`AgentPlan.trace` carries every `ToolCallRecord` with name, arguments, and raw result, making the
three recorded failure modes checkable without a judge:

```python
@dataclass(frozen=True, slots=True)
class TraceExpectation:
    called: tuple[str, ...] = ()  # never-searched
    not_called: tuple[str, ...] = ()
    max_calls: Mapping[str, int] = ()  # duplicate-search
    reply_mentions: tuple[str, ...] = ()  # wrong-product-from-unranked-results
    reply_before: tuple[tuple[str, str], ...] = ()
    reply_amount: tuple[Decimal, str] | None = None
```

Two corrections to the second draft's substring checks, both from cases where the check would
mark correct behaviour wrong:

- **`reply_amount` is a regex requiring the amount adjacent to its unit**, not a bare substring.
  Asserting that a reply "names the amount" for a small integer is near-vacuous — `2` occurs in
  `R2C3`, in dates, and in unrelated counts.
- **`reply_before` replaces `reply_excludes` for the shared-word trap.** Excluding the decoy
  outright fails a correct reply that names it in order to dismiss it ("you have Salted Butter;
  the Butter Beans aren't butter"). The assertion is that the intended product is mentioned, and
  appears before the decoy if the decoy appears at all.

These remain case-folded substring and ordering checks over `reply_text` and are brittle by
construction; the trade is determinism with no second model in the loop. `ledger.search_inventory`
returns both `Salted Butter` and `Butter Beans` as whole-word matches for "butter", so the tool
result contains both and only the reply distinguishes a correct answer from the recorded failure.

### Ask-versus-act

`AskOrAct` cases report which branch fired rather than collapsing to a maximum. `report.py` shows
the per-epoch split (`ask 6 / act 2`) for `ambiguous`-tagged cases; the split is the information,
and a case that always passes carries none.

The v1 discriminator reads as *asked* when all four hold: writes empty, `reply_text` contains a
question mark, no tool call outside `sumac_find_inventory`, and `reply_text` under a stated length
bound. A bare question mark alone conflates asking with rambling without acting. The real fix is a
structured needs-clarification outcome on `AgentPlan`, kept out of the two sampling commits and
recorded under Missing.

## Null baselines, always reported

`evals/baselines.py` provides `DoNothingRunner`, `RejectEverythingRunner`, and
`AlwaysDiscoverRunner` as `llm.SendsCompletions` stubs needing no model. `test_scoring.py` asserts
the scorer is not trivially satisfiable, and **`report.py` prints each baseline's score as a fixed
row in every summary beside the model's.** Asserting only that a baseline fails somewhere hid the
fact that in the first draft `RejectEverythingRunner` passed all four `reject` cases and all four
`find` cases — a 44% null floor. The trace assertions above remove that particular floor by
requiring a tool call the stubs never make; showing the floor every run is what makes the next
one visible.

## Comparing two runs

`evals/compare.py` takes two run directories and reports, over cases present in both:

- the paired difference with an exact McNemar test on discordant cases;
- **the observed discordance `q` from that McNemar table, the observed intraclass correlations,
  and the realised MDE for this comparison** — computed, not assumed;
- a **two-way cluster bootstrap resampling both families and templates**. Family is not the only
  clustering axis: ten instances of `add.location_path` across ten families are plausibly more
  correlated with one another than with other templates in the same family, since the model either
  handles a phrasing or does not. That is crossed, not nested, and a family-only bootstrap
  understates variance when template effects dominate;
- per-template pass rates as a first-class table, which is the diagnostic view anyway;
- the named list of `hard` cases that changed state, with no p-value attached;
- the `blocked` table, separately, never folded into the headline.

### Provenance, and refusing to compare unlike runs

Each per-seed JSON carries: git SHA, model preset name, GGUF filename and file hash, temperature,
top_p, max_tokens, seed, the family set, the template set, and a hash of the four prompt constants
(`CLASSIFIER_PROMPT`, `_FIND_PROMPT`, `_ADD_PROMPT`, `_REMOVE_PROMPT`). `compare.py` diffs those
headers and **hard-errors on a model or family-set mismatch**, warning on a prompt-hash mismatch —
a prompt change is usually the thing being measured, a different family set is a different
population. `--families 2` therefore produces numbers that cannot be compared against a full run
by accident.

## Safety rails

The real household inventory lives outside this repository and `sumac` resolves its data directory
from `SUMAC_DATA_DIR`, defaulting to `./data`. Four rails, in `evals/conftest.py`:

1. A session fixture overrides `SUMAC_DATA_DIR` and `SUMAC_PASSPHRASE` for the whole run via
   `pytest.MonkeyPatch`, so ambient values are never read.
2. An autouse session fixture asserts every family's data directory is inside
   `tmp_path_factory`'s base directory and raises `pytest.UsageError` otherwise.
3. `sumac.store.append` is wrapped for the session with a guard raising unless its `data_dir`
   argument is under the eval temp root. Seeding writes through it and passes; anything else fails
   loudly.
4. No eval calls `AgentRunner.commit`. Scoring reads `plan.writes` from `propose()`, the same
   "compute the plan, write nothing" path `sumac ask --dry-run` uses.

Rail 3 holds if rails 1 and 2 are edited wrongly, since it checks the argument at the point of the
write rather than the configuration that produced it.

## Packaging changes

```toml
[dependency-groups]
evals = ["pytest>=8", "pytest-timeout>=2"]

[tool.pytest.ini_options]
markers = ["model: eval that loads a real local model"]

[tool.hatch.build.targets.sdist]
exclude = ["/evals"]
```

`pytest-timeout` is an outer backstop only and cannot interrupt a blocking call inside the Rust
extension holding the GIL; the real bound is `DEFAULT_MAX_TOKENS` from commit 1 together with
`MAX_TOOL_ROUNDS = 20`. The wheel already excludes `evals/` —
`[tool.hatch.build.targets.wheel] packages = ["src/sumac"]` names `src/sumac` and nothing else.

```sh
uv sync --no-group ask --group ask-cuda --group evals
uv run pytest evals -m "not model"                    # scorer, baselines, vocab hygiene; no GPU
uv run pytest evals --eval-seed 1                     # one epoch, 10 families, ~10 min
uv run pytest evals --families 2 --eval-seed 1        # development loop, ~90s
uv run python -m evals.run --epochs 8 --out runs/a/
uv run python -m evals.report runs/a/
uv run python -m evals.compare runs/a/ runs/b/
```

## Build order

Everything model-free comes first, and is verifiable in a container with no GPU:

1. `vocab.py`, `seed.py` — then confirm ten families build and `ledger.build_inventory` reports
   the seeded quantities.
2. `generate.py`, `cases.py` — then print the generated case count per family and per template.
3. `scoring.py`, `baselines.py`, `test_scoring.py` — then **report the baseline floor numbers for
   the generated table before any model-loading code is written.** If the floor is high, the case
   mix is wrong and no amount of model work fixes it.
4. `conftest.py`, `test_routing.py`, `test_proposals.py`, `run.py`, `report.py`, `compare.py`.

The two `llm.py` commits land before step 4, since `conftest.py` depends on the widened
constructor.

## What changed and why, from the second draft

- Sampling constants are pinned in `llm.py` rather than passed from `evals/`; `EvalAgentRunner` and
  the Divergence entry about measuring an unshipped configuration are both gone.
- Epochs became separate pytest sessions with independent seeds, because a shared session-scoped
  `Runner` makes results order-dependent and unreproducible.
- `q` and `rho` moved from asserted MDE to observed statistics printed by `compare.py`.
- Clustering became two-way over families and templates.
- Reject prompts vary per family instead of being replicated across all ten.
- `reply_excludes` became `reply_before`; amount checks became adjacency regexes.
- The ask heuristic gained three conditions beyond a question mark.
- Run provenance and a compare-time mismatch gate were added.
- The `model` marker is now applied rather than only declared, and the model fixture skips rather
  than errors on a missing GGUF.
- The location-reference taxonomy, the `blocked` tag, and the prompt-contamination rule are new,
  from the twenty-query batch.

---

# 2026-09-02: Eval Suite Specification

## Current State

- `tests/` holds 14 test modules driven by `uv run pytest` with `testpaths = ["tests"]` in
  `pyproject.toml`; `docs/journal/2026-09-02-query-classifier.md` records 251 passing tests
  repo-wide at that entry's commit.
- `tests/test_llm.py` (25 tests) drives `AgentRunner` through `FakeRunner`, a hand-built
  `llm.SendsCompletions` returning one scripted response per round, against a real encrypted
  `data_dir`/`key` built by `tests/conftest.py`'s fixtures.
- `AgentRunner.__init__` accepts `runner: SendsCompletions | None`, substituting a caller-supplied
  object for the `mistralrs.Runner` that `llm._build_runner` would otherwise construct
  (`src/sumac/llm.py:573-608`).
- `AgentRunner.propose` returns an `AgentPlan` carrying `writes` and `trace`, with no
  `store.append` on that path — writes happen only in `AgentRunner.commit`
  (`src/sumac/llm.py:902-956`).
- `AgentRunner._build_request` sets `messages`, `model`, `tool_schemas`, `tool_choice`, and
  `enable_thinking`, and no sampling parameter (`src/sumac/llm.py:763-771`).
- `llm._build_runner` constructs `mistralrs.Runner(which=which)` and passes no `seed`
  (`src/sumac/llm.py:548-563`).
- `mistralrs.ChatCompletionRequest` exposes `temperature`, `top_p`, `top_k`, `min_p`, and
  `max_tokens` and carries no `seed` field; `mistralrs.Runner.__init__` accepts `seed`
  (`mistralrs/__init__.pyi:129-200,728-758`).
- `_FIND_PROMPT`, `_ADD_PROMPT`, and `_REMOVE_PROMPT` are static strings with no configuration
  interpolated, so no location tree reaches the model through any prompt
  (`src/sumac/llm.py:324-404`).
- `AgentRunner.tool_callbacks` registers four callbacks — find, consume, move, discover — and none
  enumerates locations (`src/sumac/llm.py:592-597`).
- `_DISCOVER_INVENTORY_SCHEMA`'s `to_location` property is `{"type": "string"}` with no
  `description`, so the schema states nothing about what form a location argument takes
  (`src/sumac/llm.py:284-309`).
- `_ADD_PROMPT` contains a worked example naming the brand "Heinz"
  (`src/sumac/llm.py:340-364`).
- `decide._resolve_location` accepts a location id or an exact `config.location_path` display
  string and raises `Rejected("unknown_location", ...)` for anything else
  (`src/sumac/decide.py:100-123`).
- `decide._resolve_product` auto-registers an unknown product under the unit it is first given and
  raises `Rejected("unit_unconvertible", ...)` for a later unconvertible unit against a registered
  one (`src/sumac/decide.py:139-206`).
- `passphrase.get_key` caches the derived key in a module-level global for the life of the
  process, so repeated in-process CLI invocations run Argon2id once
  (`src/sumac/passphrase.py:14,24-29`).
- `[tool.hatch.build.targets.wheel]` names `packages = ["src/sumac"]`, so no directory outside
  `src/sumac` reaches the wheel.

## Stubbed

- None found.

## Missing

- No `evals/` directory exists; every file, family, template, case, and scorer described above is
  unwritten.
- No `evals` entry exists in `pyproject.toml`'s `[dependency-groups]`, no `markers` entry in
  `[tool.pytest.ini_options]`, and no `[tool.hatch.build.targets.sdist]` table.
- No test or eval runs `AgentRunner` against a real model — `tests/test_llm.py` scripts every
  model response through `FakeRunner`.
- **Nothing lets a model name a location it has not already seen in a search result.** No prompt
  carries the location tree, no tool enumerates locations, and
  `_DISCOVER_INVENTORY_SCHEMA.to_location` documents no format, while
  `decide._resolve_location` accepts only an id or an exact display path. Eight of the twenty
  real queries name a location positionally ("3rd position along, 3rd row down", "left column,
  4th row", "1st drawer") and one names an emptied location; none of those is resolvable by any
  route the agent has. A location-listing tool, or interpolating the location tree into the
  per-kind prompts, would unblock them; neither exists.
- `sumac_discover_inventory` requires `amount` and `unit`, and nothing in `_ADD_PROMPT` states what
  to do when a request supplies neither ("Add Barilla rigatoni to the pantry, with the other
  pasta"). The `add.missing_amount` template's correct outcome is therefore unspecified by the
  harness, not merely unmeasured.
- `AgentPlan` carries no field distinguishing a clarifying question from inaction; both surface as
  `writes == ()`, and the `AskOrAct` branch attribution infers the difference from `reply_text`.
- v1 scores proposed writes, not resulting inventory state, so write-set equivalence classes score
  as failures. A `move.explicit` case expects one `ChangeKind.MOVEMENT`; a plan proposing a
  `CONSUMPTION` from the source plus a `DISCOVERY` at the destination reaches the same inventory
  and scores zero. Both primitives are exposed as separate tool callbacks
  (`src/sumac/llm.py:592-597`), so the equivalence class is reachable. The mechanism a state-diff
  scorer would use exists: copy a family's data directory, apply the plan's writes through
  `decide.decide_change` and `store.append` into the copy, and diff `ledger.build_inventory`
  against a gold state. Unresolved in that design: what a partial application means when write 2
  of 3 is rejected at apply time.
- No Inspect AI task, `@modelapi` provider, or model-preset sweep exists; running against more
  than one `ModelPreset` requires re-invoking `run.py` per preset. Epoch bookkeeping, the `.eval`
  log format, `inspect view`, and scorer composition are reimplemented by `evals/run.py`,
  `evals/report.py`, and `evals/compare.py`.
- No baseline, threshold, or risk-coverage curve exists for `reject` as a tunable abstention
  decision — the classifier's output is a hard enum, and `ChatCompletionRequest.logprobs` and
  `top_logprobs` are never set.
- No Hypothesis state machine covers the propose/mutate/commit interleaving that
  `AgentRunner.commit`'s re-validation exists to handle; `tests/test_llm.py` covers `commit`
  against unchanged state only.
- Reply assertions are case-folded substring, ordering, and adjacency checks over `reply_text`;
  nothing validates that a reply naming the right product also names the right location, and
  nothing validates a `find` reply that is correct but phrased outside those patterns.

## Divergence

- `README.md` documents `uv run pytest` under Development and makes no claim about evals, so
  nothing in it diverges yet — an `evals/` directory added without a corresponding README section
  would introduce one, as would the two `llm.py` commits landing without a note that `sumac ask`
  now runs at a pinned temperature.

---

# 2026-09-02: Eval Suite Implementation State

## Current State

- `src/sumac/llm.py` carries `DEFAULT_TEMPERATURE = 0.2`, `DEFAULT_TOP_P = 0.95`,
  `DEFAULT_MAX_TOKENS = 1024`; `AgentRunner._build_request` passes all three on every
  `mistralrs.ChatCompletionRequest` it builds, where previously none was set.
- `AgentRunner.__init__` accepts `temperature`, `top_p`, `max_tokens` (defaulting to the three
  constants above) and `seed` (defaulting to `None`); `llm._build_runner` accepts `seed` and
  passes it to `mistralrs.Runner`. `AgentRunner.classify(prompt)` is a public alias for
  `_classify`. `sumac ask`'s own call sites pass none of these explicitly, so its behaviour is
  the same defaults as before, now pinned rather than inherited from mistral.rs.
  `tests/test_llm.py` gained five tests covering this (`test_classify_public_alias_...`,
  `test_build_request_passes_default_sampling_config`,
  `test_build_request_passes_custom_sampling_config`,
  `test_build_runner_passes_seed_to_mistralrs_runner`,
  `test_build_runner_defaults_seed_to_none`); 286 tests pass repo-wide (`tests/`), up from 281
  before this entry's changes.
- `evals/` exists (`vocab.py`, `seed.py`, `generate.py`, `cases.py`, `scoring.py`, `baselines.py`,
  `conftest.py`, `test_scoring.py`, `test_routing.py`, `test_proposals.py`, `run.py`, `report.py`,
  `compare.py`, `README.md`, `__init__.py`; 2884 lines total).
- `evals/vocab.py` defines ten `FamilyVocab`s (`fam-01`..`fam-10`) with independent product/brand
  vocabulary and independent reject-prompt triples; `fam-01` matches the vocabulary of the real
  runs in `docs/journal/2026-09-01-ask-agent-design.md`.
- `evals/seed.py`'s `LOCATIONS` tuple is the single source for the shared location tree;
  `location_path()` computes display paths from it via `sumac.config.location_path` directly, so
  a prompt `generate.py` builds cannot name a path the seeded tree doesn't produce — confirmed
  against a real seeded `Config` by `test_location_path_matches_seeded_config`. `build_family`
  reads its returned key back through `sealedlog.Vault.unlock` directly rather than through
  `sumac.passphrase.get_key`, avoiding that function's process-global cache
  (`src/sumac/passphrase.py:14,24-29`) — verified directly: two families seeded in one process
  get different keys, and the wrong family's key fails to decrypt the other's data
  (`Decryption failed` `Anomaly`s), confirming the bug this works around is real, not
  hypothetical.
- `evals/generate.py` implements 14 single-instance templates plus `reject.out_of_domain` (3
  cases per family) against every `FamilyVocab`; `evals/cases.py` adds 7 hand-written `hard` cases
  against `fam-01` only. `evals/cases.all_cases()` over all ten families returns 177 cases: 20
  tagged `blocked` (`add.positional`, `add.absent_spot` — no route from a positional or
  emptied-location reference to a resolvable location, per Missing below), 7 tagged `hard`.
- `evals/scoring.py` implements `canonical_unit` (an explicit synonym table, not stemming — the
  first draft's stemming rule would have mapped `"boxes"` to `"boxe"`), `canonical_location`
  (display path or id, raw value otherwise), `score_writes` (set equality plus F1),
  `check_trace` (`called`/`not_called`/`max_calls`/`reply_mentions`/`reply_before`/`reply_amount`),
  `classify_ask_or_act`, and `score_case` (kind AND outcome AND trace, all three, or it doesn't
  count as passed).
- `evals/baselines.py` implements `DoNothingRunner`, `RejectEverythingRunner`,
  `AlwaysDiscoverRunner`, and `run_baseline`/`BaselineResult`, shared by `test_scoring.py` and
  `report.py`. Run against the real 177-case table over all ten real seeded families:
  do-nothing 32/177 (18.1%), reject-everything 32/177 (18.1%), always-discover 11/177 (6.2%) —
  all below the 50% floor `test_null_baselines_never_pass_the_full_suite` asserts.
- `evals/conftest.py` implements all four safety rails from the spec, plus a fifth not in the
  original spec: `_gguf_cached_locally` checks the local Hugging Face cache for the target GGUF
  file *before* `eval_runner` ever constructs a real `mistralrs.Runner`, skipping immediately if
  absent. This was added after a real incident during this implementation: an earlier version of
  `eval_runner` guarded `_build_runner` with try/except-then-skip only, and running the full
  suite without excluding `pytest.mark.model` triggered a real ~2.5GB download (a Qwen3.5-4B GGUF)
  that reached 1.3GB before being caught and killed — try/except catches an immediate error, not
  a slow download that never raises. The cache-check fix was verified afterward: the full suite,
  run without excluding `model`, now skips both model-gated modules in ~12s with no network
  activity (confirmed via `ss -tnp` showing no new connections and the Hugging Face cache
  directory staying empty).
- `evals/test_scoring.py` (15 tests) runs without a GPU or a downloaded GGUF; verified directly in
  this session (`uv run pytest evals -m "not model"`, 15 passed in ~12s against all ten families).
  Covers vocab hygiene (no product/brand name collides with `_ADD_PROMPT`'s "Heinz"/"Baked Beans"
  worked example, checked against every family), reject-prompt independence across families,
  `location_path` correctness, the null-baseline floor, and the specific fixes named in the eval
  spec (`reply_before` accepting a decoy named to be dismissed, `reply_amount` requiring
  adjacency, `canonical_unit`'s explicit table).
- `evals/report.py` and `evals/compare.py` are verified against synthetic per-seed JSON files
  constructed by hand in this session (not against a real model — none was run) — both aggregate
  pass^k, print the classification confusion matrix, print per-template rates, print the
  hand-written `hard`-case tripwire list, and (`report.py` only) re-seed real families and print
  fresh null-baseline rows. `compare.py`'s McNemar exact test, observed discordance, ANOVA-based
  ICC(1) per clustering axis, realised MDE, and two-way (family x template) pairs cluster
  bootstrap all ran against the synthetic data and produced numbers in the expected ranges (e.g.
  a template deliberately degraded in the synthetic "B" run showed up in its per-template rate
  and in the discordant case list) without raising.
- `evals/test_routing.py` and `evals/test_proposals.py` (`pytest.mark.model`) are unverified
  against a real model — this container has no GPU and no cached GGUF, and downloading one to
  test them was avoided deliberately (see the incident above). Confirmed only to skip cleanly.
- `pyproject.toml` gained `[dependency-groups] evals = ["pytest>=8", "pytest-timeout>=2"]`,
  `[tool.hatch.build.targets.sdist] exclude = ["/evals"]`, and
  `[tool.pytest.ini_options] markers = ["model: ..."]`.
- `README.md` (top-level) gained one line under Development pointing to `evals/README.md`.

## Stubbed

- None found.

## Missing

- Every item under Missing in this entry's specification section above is still missing —
  location-tool/positional-reference resolution, the missing-amount harness behaviour, a
  structured ask-vs-inaction field on `AgentPlan`, the state-diff scorer, Inspect AI integration,
  abstention thresholds, and the Hypothesis commit-staleness state machine. None of that scope was
  implemented in this pass; all are still open exactly as recorded there.
- `evals/report.py`'s `_provenance` (`evals/conftest.py`) records the GGUF's repo id and filename
  but not a file hash — locating the actual downloaded blob depends on mistral.rs's own Hugging
  Face cache layout, which this suite doesn't introspect beyond the presence check
  `_gguf_cached_locally` uses.
- `evals/compare.py`'s ICC and two-way cluster bootstrap are implemented per the eval spec's
  design and produced plausible numbers against synthetic data in this session, but have not been
  checked against a dataset with a known, independently-computed ICC or bootstrap CI — the
  synthetic-data run confirms the code runs and responds to an engineered signal, not that either
  statistic is numerically correct in general.
- No CI workflow runs `uv run pytest evals -m "not model"` on push — `.github/` is unmodified by
  this entry.

## Divergence

- None found against `README.md`, which now names `evals/` and points to `evals/README.md`,
  matching what exists.

---

# 2026-09-03: Eval Suite Reduction Pass

## Context

A real run against a real model (`unsloth/Qwen3.5-4B-GGUF` or `Qwen3-4B-Instruct-2507-GGUF`,
both `max_seq_len: 4096` per the logs in `docs/journal/2026-09-01-ask-agent-design.md`) against
the request this entry's fixtures already encoded verbatim — "Add 1 bag of Basmati Rice (1kg)
next to the existing jug of Basmati Rice" — produced a repeated `sumac_discover_inventory` loop
that overflowed the model's context window. Separately, the 177-generated-case suite the previous
implementation pass built was judged too large and too statistically elaborate to answer the
question actually being asked at this stage: given one request, did the agent do the right thing.
This entry records the diagnosis of that failure and the resulting cut of the suite from 177
generated cases across ten fixture families down to 23 hand-picked ones across one.

## Diagnosis: the Basmati Rice loop

`AgentRunner._run_loop` (`src/sumac/llm.py:802-894`) bounds round *count* per call
(`MAX_TOOL_ROUNDS = 20`), not accumulated *token* count. `self._messages` is never trimmed and is
shared across every `_run_loop()` call a single `propose()` makes: the initial call,
`_maybe_force_action`'s one extra call (triggered whenever an `add`/`remove`-classified request
produces no writes), and up to `SELF_REVIEW_ROUNDS` more from `_maybe_self_review` — each
appending onto the same growing list, with no check anywhere on total size.
`decide._resolve_product` rejects a bag-vs-jug unit mismatch with `unit_unconvertible` on every
attempt; a model that cannot resolve that just keeps retrying `sumac_discover_inventory`, and
`_build_request` resends the full `tool_schemas` on every round of the client-side loop
(`src/sumac/llm.py` module docstring). `evals/test_termination.py`'s
`test_repeated_rejection_terminates_within_round_cap` reproduces the mechanism deterministically,
with no real model: a fake runner that repeats the same rejected call forever drives exactly 41
rounds (1 classify + 20 main loop + 20 from `_maybe_force_action`'s retry — `_maybe_self_review`
never fires here, since it short-circuits on empty `plan.writes` before making a further round)
before `propose()` returns cleanly with no write. 41 rounds of repeated tool-schema-plus-rejection
payload is a plausible way to exceed a 4096-token context well before any round cap fires, which
is consistent with what the real run hit.

**Not fixed here.** The round cap is a real termination guarantee; it is not a context-size
guarantee, and nothing else is one. A fix (a token-based bound, refusing to retry an identical
rejected call twice, or trimming message history) is an `src/sumac/llm.py` behaviour change of
the same weight as the sampling-pinning commit earlier in this entry, and wants an explicit
decision rather than a silent edit alongside an eval-suite rewrite.

## What was deleted

`evals/generate.py` (the template engine), `evals/cases.py` (`EvalCase`/`WriteSpec`/expectation
type system), `evals/vocab.py` (ten `FamilyVocab`s), `evals/scoring.py` (canonicalisation-as-a-
framework, `TraceExpectation`, `CaseScore`), `evals/baselines.py` (null baselines),
`evals/run.py` (epoch orchestrator), `evals/report.py`, `evals/compare.py` (McNemar/ICC/MDE/
two-way cluster bootstrap), `evals/test_scoring.py`, `evals/test_routing.py`,
`evals/test_proposals.py`, `evals/seed.py` (superseded by `evals/fixtures.py`, below). All of it
existed to support 177 generated cases and cross-epoch aggregation; neither survived this pass.

## What remains

- `evals/fixtures.py` — one seeded inventory (the `docs/journal/2026-09-01-ask-agent-design.md`
  vocabulary: Chopped Tomatoes / Ocado Italian Chopped Tomatoes, Salted / Unsalted Butter, Butter
  Beans, Basmati Rice, Strawberry Jam, Ragu, Fusilli Pasta, plus the never-seeded Irn-Bru Zero),
  built via real `sumac` CLI invocations against a trimmed location tree (fridge with a door, main
  shelves, and a bottle rack; a pantry with one 3x4 grid; a freezer with three drawers — the
  second grid and two standalone locations from the ten-family version dropped as unused).
- `evals/conftest.py` — the same four safety rails as before, one `inventory` fixture, an
  `agent_runner_factory` fixture (kept: the local-GGUF-cache check before ever constructing a real
  `mistralrs.Runner`, added after the real near-download incident recorded in this entry's first
  implementation pass), and five plain assertion functions
  (`assert_write`/`assert_no_writes`/`assert_classified`/`assert_tool_called`/`is_ask_or_act`)
  replacing the generic scorer.
- `evals/test_agent.py` — 22 named tests against a real model, one `def test_*` per scenario,
  covering: find (existing item, missing item, quantity, shared-word decoy, tool-use scoping),
  add (explicit location, indirect location, new product, discriminator variant, the Basmati Rice
  unit conflict, an unusual-but-valid destination, bounded duplicate search, two ways of not
  inventing a missing amount), remove (partial, full), move (explicit, vague-but-answerable), and
  reject (out-of-domain, gibberish, an inventory word inside an out-of-domain request).
- `evals/test_termination.py` — the one deterministic, no-model test: the round-cap mechanism
  proof described above.

## Case matrix reduction

| category (from the product-question list) | old (generated) | new (hand-picked) |
| --- | --- | --- |
| successful / failed find | 60 across 10 families | `test_find_existing_item`, `test_find_missing_item` |
| find quantity / shared-word decoy | 60 across 10 families | `test_find_quantity`, `test_find_shared_word_picks_right_product` |
| successful add (explicit / indirect location) | 20 | `test_add_existing_item_full_path`, `test_add_existing_item_indirect_location` |
| missing item -> discover | 10 | `test_add_missing_item_discovers_new_product` |
| add without confusing near-miss/discriminator products | 20 | `test_add_discriminator_variant_not_confused`, `test_duplicate_search_bounded` |
| unit-conflict rejection (Basmati Rice) | 10 (`add.unit_collision`) | `test_add_unit_conflict_rejected_basmati_rice` |
| successful remove (partial / full) | 20 | `test_remove_partial`, `test_remove_all` |
| move (explicit / vague) | 20 | `test_move_explicit`, `test_move_vague_asks_or_acts` |
| reject / clarify | 32 + 2 hard | `test_reject_out_of_domain_weather`, `test_reject_gibberish`, `test_reject_joke_with_inventory_word`, `test_ambiguous_product_asks_or_acts` |
| don't invent a missing required field | 10 (`add.missing_amount`) | `test_missing_amount_does_not_invent_values`, `test_multi_item_add_without_amounts_does_not_invent_values` |
| unusual destination respected | 1 (hard) | `test_odd_destination_respected` |
| positional/emptied-location reference | 20 (`blocked`, excluded from headline) | dropped — recorded under Blind spots in `evals/README.md`, not tested until `llm.py` gains a way to resolve one |
| termination under pathological retry | none | `test_repeated_rejection_terminates_within_round_cap` |

"consumption vs waste vs purchase/discovery" is not a separate row: `AgentRunner.tool_callbacks`
only ever emits `ChangeKind.DISCOVERY`, `CONSUMPTION`, or `MOVEMENT`
(`src/sumac/llm.py:311-315`) — `PURCHASE` and `WASTE` have no route through the agent at all, only
through `sumac add` directly. Every add case is structurally a discovery and every "remove without
a named destination" case is structurally a consumption; nothing here tests a distinction the
agent is unable to make, and `evals/README.md` records this under Blind spots rather than a case
that would trivially pass without checking anything real.

## Verified this session

`uv run pytest evals -v`: 23 collected, `test_repeated_rejection_terminates_within_round_cap`
passes deterministically, all 22 `test_agent.py` cases skip cleanly (no local GGUF cache, no
network attempt — confirmed by a ~1.2s total run time) rather than erroring. Full repo suite still
286 passing; `ruff check .`, `ruff format --check .`, `ty check .` all clean.

## Missing

- `evals/test_agent.py` is unverified against a real model — this container has no GPU and no
  cached GGUF. In particular, `test_add_unit_conflict_rejected_basmati_rice` — the direct
  reproduction of the diagnosed failure — has not been run for real; that is the first thing to
  run once a model is available.
- The real fix for the diagnosed termination mechanism (token-bounded history, or refusing to
  retry an identical rejected call) is not implemented — see "Not fixed here" above.
- Location-reference resolution (positional and emptied-location references) remains unaddressed
  and untested, as it was in the previous pass.

---

# 2026-09-03: Assertion Fixes, Location-Language Audit, and a Review Pass

## Context

Three rounds since the reduction pass, each requested separately: (1) two tests whose assertions
required `NoWrites` for an omitted amount/unit were amended to accept the agent inferring a
plausible default instead, since that's the desired behaviour, not the missing-amount case being
tested; (2) a real-model run surfaced `test_add_unit_conflict_rejected_basmati_rice` failing
(the user's own view: adding a bag alongside an existing jug of the same product should be
accepted, not rejected — a `decide.py`/tool-calling design question, not an eval bug), and a
separate audit of every ADD/MOVE prompt in `test_agent.py` for realism (natural spoken language,
not the internal `X > Y` display-path syntax, and a concrete destination or legitimate
existing-stock inference rather than a bare "the pantry"); (3) a review pass re-reading this
entry's specification against the current suite.

## Current State

- `test_add_product_with_omitted_amount` (renamed from `test_missing_amount_does_not_invent_values`)
  and `test_add_multiple_products_with_omitted_amounts` (renamed from
  `test_multi_item_add_without_amounts_does_not_invent_values`) now assert a write happened —
  positive amount, non-empty unit, correct product identity — rather than `NoWrites`. Butter is
  checked only for containing "butter" and not "beans" (three plausible identities — a new
  "Butter" registration, "Salted Butter", or "Unsalted Butter" — are all accepted; only landing on
  "Butter Beans" fails it).
- `test_add_basmati_rice_in_different_unit` (renamed from
  `test_add_unit_conflict_rejected_basmati_rice`, by the user's own edit) now asserts the write is
  accepted under the same product identity (`product_id == "basmati rice"`, `unit == "bag"`) rather
  than rejected. This fails against a real model today — `decide._resolve_product` has no
  registered bag-to-jug conversion for Basmati Rice, so `decide` rejects it — and is left failing
  deliberately, as the marker of a real `decide.py`/`llm.py` gap (accept-with-confirmation, not
  flat rejection) that's explicitly out of scope for the eval suite itself.
- `test_add_existing_item_full_path` and `test_move_explicit` no longer put the internal
  `X > Y` display-path syntax into a prompt — natural phrasing instead ("the second shelf of the
  third white pantry cupboard", "the third drawer of the big freezer").
- `test_add_missing_item_discovers_new_product`'s destination changed from bare "the pantry" (a
  grouping node nothing is ever seeded on directly in this fixture) to the same natural-language
  grid-cell phrasing.
- `test_add_discriminator_variant_not_confused`'s prompt dropped its named destination entirely,
  relying only on "with the existing stock" — forces the agent to search and resolve Unsalted
  Butter's own location rather than being handed it, which a request naming the drawer outright
  wouldn't test.
- `test_add_product_with_omitted_amount`'s accepted resolved location was tightened from
  `("pantry", "pantry-white-unit-r2c1")` to `"pantry-white-unit-r2c1"` only — `_ADD_PROMPT`
  instructs using the found location over the person's literal wording in exactly this case, so
  the "pantry" fallback was never actually correct, just left lenient.

## Missing (found in the review pass, not yet fixed)

- **`test_add_existing_item_full_path` and `test_add_missing_item_discovers_new_product` are
  currently untestable.** Their natural-language phrasing ("the second shelf of the third white
  pantry cupboard") depends on a row-means-shelf, column-means-cupboard convention that exists
  nowhere in the system — not in `_ADD_PROMPT`, not in any tool schema, not in a location's own
  `name` field (a search result would return "White Unit R2C3", not "second shelf, third
  cupboard"). `grep -n 'shelf\|cupboard\|row\|column' src/sumac/llm.py` finds nothing. For the
  new-product case there is additionally no existing stock to search and ground against at all.
  This is the same "positional reference has no route to a location id" gap this entry's earlier
  sections named `blocked` and deferred — reintroduced into two now-strictly-asserted tests by the
  location-language audit round, not by design. Proposed fix (not applied): use locations whose
  `name` field is itself the natural phrase (`fridge-door`, `fridge-bottle-rack`,
  `freezer-drawer-N`) for tests needing an explicit, non-inferred destination; reserve grid cells
  for cases where the destination is inferred via a prior search-and-match, which is the only path
  by which a grid cell's canonical id ever reaches the model.
- **Two small, cheap, deterministic checks from the pre-reduction suite were not carried
  forward**, and would have caught the above before a real-model run: a check that no fixture
  product/brand name collides with `_ADD_PROMPT`'s "Heinz"/"Baked Beans" worked example, and a
  check that `fixtures.location_path()` (what every prompt is built from) matches what a real
  seeded `Config` resolves. `fixtures.py:36` still cites `test_location_path_matches_config` in a
  comment — that test does not exist in the current suite.
- `evals/README.md`'s Blind Spots section states positional location references are "not covered
  here", which the two tests above currently contradict.

## Under consideration, not decided

An external design outline (ChatGPT, pasted by the user) proposes restructuring the suite around
named scenarios, a small typed `EvalResult`, evaluator functions replacing inline assertions, and
a runner producing a version-comparison report — while explicitly keeping pytest as the execution
mechanism and explicitly not reintroducing YAML scenario files, an LLM judge, or the deleted
statistical machinery. Not acted on this session; the review above (untestable tests, two missing
regression checks) was treated as the more urgent, concrete work. Revisit once those are fixed.

---

# 2026-09-03: Scenario/Evaluator Refactor

## Context

An external design review (ChatGPT, pasted by the user) proposed restructuring the suite around
named scenarios, a small typed result, evaluator functions in place of inline assertions, and a
runner producing a category/dimension summary — while explicitly keeping pytest as the execution
mechanism and explicitly not introducing YAML, an LLM judge, or the deleted statistical machinery.
The previous entry recorded this as "under consideration, not decided." This entry is that
decision, made after the user confirmed the direction directly. The two bugs found in the prior
review pass (two ADD scenarios depending on a shelf/cupboard naming convention nowhere in the
system; the contamination and location-path regression checks not carried forward) were fixed in
the same pass, since every affected file was being rewritten anyway.

## Current State

- `evals/evaluators.py` is new: an `EvalResult` dataclass (`scenario`, `category`, `checks:
  dict[str, bool]`, `failures: list[str]`, `note: str | None`, a `passed` property, a `check(name,
  ok, message)` method) and seven `evaluate_*` functions (`evaluate_classification`,
  `evaluate_no_writes`, `evaluate_write`, `evaluate_tools`, `evaluate_only_tools`,
  `evaluate_reply_mentions`, `evaluate_reply_order`, `evaluate_ask_or_act`) that mutate a passed-in
  `EvalResult` in place. These are the old `conftest.py` assertion helpers (`assert_write`,
  `assert_classified`, `assert_tool_called`, `is_ask_or_act`) with the same logic, changed from
  raising to recording a named check — nothing about what's verified changed, only whether a
  partial pass is visible when the final `assert` in a test fails.
- `evals/conftest.py` gained a `result` fixture (function-scoped, derives `scenario` from the
  test's own name and `category` from the test module's `_CATEGORY` constant, yields a fresh
  `EvalResult`, captures it into a session list on teardown regardless of pass/fail) and a
  `pytest_sessionfinish` hook printing a category tally, a per-scenario failure list, and an
  ask-vs-act branch tally, plus an optional `--eval-json PATH` writing the same data as JSON. Lost
  the `assert_no_writes`/`assert_write`/`assert_classified`/`assert_tool_called`/`is_ask_or_act`
  functions (moved to `evaluators.py`) and the `UNIT_SYNONYMS`/`_canon_unit`/`_canon_location`
  helpers (moved with them).
- `evals/test_agent.py` (22 scenarios, one file) is gone, replaced by `evals/test_find.py` (5),
  `evals/test_add.py` (10), `evals/test_remove.py` (4 — consumption and movement, both classified
  `REMOVE`), `evals/test_reject.py` (3) — 22 scenarios, same count, split by capability, each file
  carrying a `_CATEGORY` constant the `result` fixture reads.
- `evals/test_fixtures.py` is new (2 tests, no model): `test_no_product_name_leaks_into_prompt_constants`
  and `test_location_path_matches_real_config`, restoring the two checks the reduction pass had
  dropped — see the previous entry's Missing section for why the second one specifically would
  have caught the bug below before a real-model run needed to.
- **The two untestable ADD scenarios are fixed.** `test_add.py::test_existing_item_explicit_location`
  (renamed from `test_existing_item_full_path`) now targets `fridge-main-shelf-2` via "the second
  shelf of the fridge" — the fridge's main-shelf array is genuinely numbered "Shelf 1".."Shelf 4",
  so "the second shelf" is the location's own name, not an invented convention.
  `test_add.py::test_missing_item_discovers_new_product` now targets `fridge-door` via "the fridge
  door" — again the location's actual name. Neither depends on the pantry grid's
  row-means-shelf/column-means-cupboard convention that exists nowhere in `_ADD_PROMPT`, any tool
  schema, or a location's own `name` field.
- `evals/fixtures.py`'s stale comment (citing a test that hadn't existed since the reduction pass)
  now cites `test_location_path_matches_real_config` in `test_fixtures.py`, which exists.
- `evals/README.md` rewritten: new layout, an example of writing a scenario, an example of the
  summary output, an updated Blind Spots section (the location-reference gap is now correctly
  described as "beyond an id, an exact display path, or a location whose own name is the natural
  phrase," not blanket-unresolvable), and a "Deliberately not here (yet)" section explaining the
  YAML/LLM-judge/comparison-tool decisions from the design review.

## Verified this session

`uv run pytest evals/test_termination.py evals/test_fixtures.py -v`: 3 passed, including the
restored `test_location_path_matches_real_config` against the real seeded config (confirms the two
relocated ADD scenarios' new targets — `fridge-main-shelf-2`, `fridge-door` — actually resolve).
`uv run pytest evals -v`: 25 collected, 3 passed (the two fixture checks plus termination), 22
skipped cleanly (no cached GGUF in this container, confirmed no network activity, ~1.2s total).
The summary printer and `--eval-json` payload shape were smoke-tested directly against synthetic
`EvalResult`s (not a real run) and produce the output shown in `evals/README.md`. Full repo suite:
286 passing. `ruff check .`, `ruff format --check .` (evals/ only — the journal's own embedded
code-block formatting is a pre-existing cosmetic nit, unrelated), `ty check .` all clean.

## Missing

- Every `test_add.py`/`test_find.py`/`test_remove.py`/`test_reject.py` scenario is unverified
  against a real model this session — same limitation as every entry before this one; this
  container has no GPU and no cached GGUF.
- No cross-run comparison tool exists yet, deliberately — see `evals/README.md`'s "Deliberately not
  here (yet)". `--eval-json` output is unconsumed until one is written.
- `test_add.py::test_basmati_rice_in_different_unit` is still expected to fail for real — the
  `decide.py` gap it marks is unfixed, out of scope for this suite.

---

# 2026-09-03: First Real-Model Comparison, Registry Trim

## Context

First real-model runs of the scenario/evaluator suite (previous entries could only smoke-test
against synthetic `EvalResult`s — no GPU/cached GGUF in-container). Run outside this container,
across seven presets:

```sh
for model in qwen3.5-4b qwen3-4b lfm2.5-2.6b lfm2.5-1.2b qwen3.5-2b qwen3-1.7b qwen3-0.6b; do
  uv run pytest evals --eval-json "runs/${model}.json" -v --eval-model "$model"
done
```

Aggregated with:

```sh
jq -c -s '
  map({
    model,
    scenarios: (.results | length),
    passed: ([.results[] | select(.passed)] | length),
    pass_rate: (([.results[] | select(.passed)] | length) / (.results | length) * 100),
    checks: (
      [.results[].checks | to_entries[]]
      | group_by(.key)
      | map({
          check: .[0].key,
          passed: ([.[] | select(.value == true)] | length),
          total: length,
          rate: (([.[] | select(.value == true)] | length) / length * 100)
        })
    )
  })
  | sort_by(-.pass_rate)
' *.json
```

`--eval-model qwen3.5-2b` initially produced no `runs/qwen3.5-2b.json` at all — the loop above
silently didn't write one (the fixture skips cleanly when the target GGUF isn't in the local HF
cache, by design; nothing had downloaded qwen3.5-2b's weights yet). The first aggregation pass
over `*.json` therefore only covered six of the seven presets, and dropped the missing one without
flagging it rather than erroring — caught by counting files in `runs/`, not from the aggregation
output itself. Separately, `--eval-model` alone doesn't fetch a preset's weights; getting
qwen3.5-2b evaluated needed a `sumac ask --model qwen3.5-2b` run first to populate the HF cache, at
which point the loop was rerun for that one preset and `runs/qwen3.5-2b.json` landed. All seven
presets have a real run as of this entry.

`DEFAULT_MODEL_PRESET` was also found pointing at `MODEL_PRESETS[4]` (`qwen3.5-2b`) on disk,
uncommitted — a temporary edit made to work around the same cache-population problem (`sumac ask`
had no other convenient way to target a specific preset's weights for download), left in place
afterward. Reverted to index 0 in this pass.

## Current State

Verified directly from `runs/*.json` (not just the pasted interpretation of the aggregation
above):

| model | pass/22 | rate | classification | tool_scope | writes |
|---|---|---|---|---|---|
| qwen3.5-4b | 21 | 95% | 100% | 100% | 100% |
| lfm2.5-2.6b | 18 | 82% | 91% | 100% | 95% |
| qwen3.5-2b | 18 | 82% | 95% | 100% | 100% |
| qwen3-1.7b | 11 | 50% | 77% | 100% | 85% |
| qwen3-4b | 7 | 32% | 64% | 100% | 65% |
| qwen3-0.6b | 1 | 5% | 50% | 100% | 50% |
| lfm2.5-1.2b | 0 | 0% | 18% | 100% | 40% |

`qwen3-4b`'s weak showing (32%, `outcome` check at 0%, `tool:sumac_find_inventory` at 60%) is
suspected — not confirmed — to be an artifact of the specific quant used
(`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`; "2507" names an interim dated release, not qwen3.5's later
one), rather than the 4B size class itself being weak — qwen3.5-4b, same size class, newer release,
scores 95%. Not investigated further; the preset is dropped either way (see below), so
distinguishing "bad quant" from "bad model" for qwen3-4b specifically wasn't pursued.

`MODEL_PRESETS` in `src/sumac/llm.py` trimmed from seven presets to three, keeping every preset
that scored at or above 82% and dropping every one at or below 50%:

```python
MODEL_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset("qwen3.5-4b", ...),  # default — 95%
    ModelPreset("qwen3.5-2b", ...),  # 82%
    ModelPreset("lfm2.5-2.6b", ...),  # 82%
)
```

Dropped: `qwen3-4b`, `qwen3-1.7b`, `qwen3-0.6b`, `lfm2.5-1.2b`. `DEFAULT_MODEL_PRESET` stays index 0
(`qwen3.5-4b`, the top scorer). `evals/README.md`'s two example commands referencing the
now-removed `qwen3-4b` preset updated to `qwen3.5-2b`.

## Verified this session

`ruff check src/sumac/llm.py evals/` and `ty check src/sumac/llm.py`: clean. `python -m py_compile`
on the edited file: clean. Full `pytest` run not possible in this container (no GPU, no cached
GGUFs, and this container's `.venv` points at a host-only interpreter path) — the registry trim
itself needs no model to be structurally correct (`model_preset()` does a plain dict lookup), but
hasn't been exercised against `evals/conftest.py`'s `--eval-model` option resolution for real here.

## Missing

- The registry trim and `DEFAULT_MODEL_PRESET` fix are unverified by an actual `pytest` run in any
  environment this session — confirm with `uv run pytest evals -v --eval-model qwen3.5-4b` (or the
  other two remaining presets) outside this container.
- The qwen3-4b bad-quant hypothesis is untested — would need a different quant of the same
  qwen3.5-generation 4B model run through the same suite to confirm, and isn't planned now that
  the preset is dropped.
- Local HF cache still holds weights for the four dropped presets — cleanup is manual, out of
  scope here (the user's own).

---

# 2026-09-03: Quant Variants, SmolLM3-3B, Model-Pull/Benchmark DX, Eval Timing

## Context

Follow-up to the registry trim above. The user wanted to keep experimenting inside the
same-size-class field (more quantizations of the three kept presets; a same-size-class contender,
SmolLM3-3B) without the DX friction that shaped the previous entry — manually editing
`DEFAULT_MODEL_PRESET` and running `sumac ask` once per model just to prime the HF cache before a
benchmark loop — and wanted eval latency tracked alongside correctness, not just pass/fail.

Before writing any code, confirmed by direct HF lookups (not guessed):
- `unsloth/Qwen3.5-4B-GGUF` and `unsloth/Qwen3.5-2B-GGUF` each offer `Q4_K_S`, `Q4_K_M` (already in
  the registry), and `UD-Q4_K_XL` quant files — filenames and sizes matched what the user had
  independently seen on the qwen3.5-4b model card.
- `LiquidAI/LFM2.5-2.6B-GGUF` (LiquidAI's own repo, not unsloth) has no `Q4_K_S` or `UD` variant —
  only `Q4_0`/`Q4_K_M`/`Q5_K_M`/`Q6_K`/`Q8_0`/`F16`/`BF16`/a QAD checkpoint. No quant expansion
  added there.
- `unsloth/SmolLM3-3B-GGUF` exists, offers the same three 4-bit quants, and its own chat template
  documents the tool-call wrapping `<tool_call>\n{"name": <fn>, "arguments": <args>}\n</tool_call>`
  — byte-identical to what `_render_tool_call`'s `ToolCallFormat.QWEN` branch already produces —
  with no `tool_calls`-specific rendering of its own (an assistant turn's `content` is echoed
  verbatim by its template). Strong evidence `ToolCallFormat.QWEN` is directly reusable, but this
  is a documented-template read, not a real run — flagged as unverified everywhere it's referenced
  below.
- Granite 4.0's tool-call format was not confirmed to the same byte-exact level and would need a
  genuinely new `ToolCallFormat` branch to hand-render correctly (the client-side tool-calling loop
  requires an exact match — see the module docstring). By the user's own choice, deferred to a
  follow-up once actually researched, rather than guessed now.

## Current State

- `src/sumac/llm.py`: `MODEL_PRESETS` grew from 3 to 10 entries. The original three keep their
  names/positions unchanged (nothing already referencing `qwen3.5-4b`/`qwen3.5-2b`/`lfm2.5-2.6b`
  breaks). Four new quant-suffixed presets for the two unsloth-published models
  (`qwen3.5-4b-Q4_K_S`, `qwen3.5-4b-UD-Q4_K_XL`, `qwen3.5-2b-Q4_K_S`, `qwen3.5-2b-UD-Q4_K_XL`), and
  three new SmolLM3-3B presets (`smollm3-3b-Q4_K_S`, `smollm3-3b-Q4_K_M`,
  `smollm3-3b-UD-Q4_K_XL`), all `ToolCallFormat.QWEN`, with an inline comment recording the
  chat-template evidence and the unverified status. `DEFAULT_MODEL_PRESET` unchanged (still index
  0, `qwen3.5-4b`).
- `src/sumac/llm.py` also gained `is_cached(model: ModelPreset) -> bool` — the same HF-cache-probe
  logic that used to live only as `evals/conftest.py`'s private `_gguf_cached_locally`, moved here
  so both the eval fixture and the new CLI commands below share one implementation.
  `evals/conftest.py`'s `agent_runner_factory` now calls `llm.is_cached` instead of keeping its own
  copy.
- `src/sumac/cli.py`: new `sumac models` sub-Typer (`models_app`, registered the same way as the
  existing `config_app`), with `models list [--names-only]` (every preset, cached-or-not; the
  `--names-only` form is plain one-name-per-line for scripting) and `models pull [NAME...]`
  (defaults to every registry preset; skips ones already cached; loads each uncached one via
  `llm._build_runner` just long enough to trigger `mistralrs`' own download-on-load, then drops
  it). The existing inline `try/except ImportError` around `from sumac import llm` in `ask()` was
  extracted into `_import_llm()`, now shared by `ask`, `models list`, and `models pull` — three
  call sites of the same guard was worth naming.
- `scripts/benchmark-models.sh` (new, mirrors `scripts/build-mistralrs-cuda.sh`'s existing
  dev-tooling convention): `sumac models pull` → `pytest evals --eval-model NAME --eval-json
  runs/NAME.json` looped over `sumac models list --names-only` → `jq -c -s -f evals/report.jq
  runs/*.json`. `evals/report.jq` is the aggregation query from this file's own 2026-09-03 entry
  above, checked in instead of retyped by hand each time — the README previously listed a
  cross-run comparison tool under "Deliberately not here (yet)"; this is that tool, now that
  there's real multi-run data to justify it, and that bullet is removed.
- `evals/evaluators.py`: `EvalResult` gained `duration_s: float = 0.0`. `evals/conftest.py`'s
  `result` fixture times the test body (`time.perf_counter()` around the `yield` — the once-per-
  session model load isn't included, only the per-request latency each scenario actually
  measures). `_print_summary` prints a total wall-clock line; the `--eval-json` payload carries
  `duration_s` per scenario and a top-level `total_duration_s`.
- `evals/README.md` updated throughout: the "Running" section's cache-priming instruction now
  points at `sumac models pull` instead of "run `sumac ask` once"; a new "Comparing models" section
  documents `models list`/`models pull`/`scripts/benchmark-models.sh`; the sample output block
  shows the new time line; `report.jq` added to the file tree.

## Verified this session

`ruff check` and `ruff format --check` on every edited `.py` file: clean (one caught issue fixed —
a missing blank line before `MAX_TOOL_ROUNDS`'s comment block in `llm.py`). `ty check`: clean.
`bash -n scripts/benchmark-models.sh`: clean. `evals/report.jq` smoke-tested directly against the
real `runs/*.json` from the previous entry's model comparison (`jq -c -s -f evals/report.jq
runs/*.json`) — reproduces the same pass-rate ranking, `total_duration_s` reads `null` for those
older files as expected (they predate this session's timing field). `MODEL_PRESETS` checked for
duplicate names (`grep -oP 'ModelPreset\("\K[^"]+' | sort | uniq -d`, empty) and correct count (10
entries, matching 3 original + 4 quant variants + 3 SmolLM3).

## Missing

- Nothing here has been run against a real model or a real HF download in this container (no GPU,
  no network attempt made deliberately) — `sumac models pull`, `sumac models list`,
  `scripts/benchmark-models.sh`, and every new preset's actual load are all unverified outside it.
- **SmolLM3-3B's `ToolCallFormat.QWEN` reuse is a documented-template read, not a confirmed
  match** — before trusting any SmolLM3 benchmark score, run it once with `--eval-debug` (or
  `sumac ask --debug`) and inspect the tool-call round-trip by eye for a malformed replay. If it's
  wrong, it needs its own `ToolCallFormat` branch, not a registry-only fix.
- Granite 4.0 is still not in the registry — deferred, per the Context section above, until its
  tool-call chat template is actually worked out rather than guessed.
- The new quant-variant presets' filenames were confirmed against HF's file listing but not
  against an actual download — a typo or a since-renamed file would surface as a 404 the first
  time `sumac models pull` tries it, not before.

---

# 2026-09-03: Confirmed Broken — UD-Q4_K_XL and SmolLM3-3B

## Context

First real run of the additions from the entry above (quant variants + SmolLM3-3B), via `sumac
models pull` and `scripts/benchmark-models.sh`, outside this container. Two of that entry's
additions are confirmed non-functional with the installed `mistralrs` — not a benchmark loss, a
load-time failure before inference ever starts:

- **`UD-Q4_K_XL` (any model)**: fails for all three presets tried (`qwen3.5-4b-UD-Q4_K_XL`,
  `qwen3.5-2b-UD-Q4_K_XL`, `smollm3-3b-UD-Q4_K_XL`), same error shape —
  `GGUF tensor ... uses dtype IQ4_XS (23) ...; direct GGUF loading currently supports F32, F16,
  BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, and Q2_K through Q8_K`. Unsloth's "Dynamic 2.0"
  quantization mixes `IQ4_XS` tensors into what's nominally a "Q4_K_XL" file; this installed
  `mistralrs` version's direct-GGUF loader doesn't support that dtype at all (MXFP4 has an explicit
  GPT-OSS-only path; IQ4_XS has no path). This is a mistralrs limitation, not a per-model issue —
  no `UD-*` quant of anything is expected to load until that changes.
- **SmolLM3-3B (every quant, not just UD)**: `smollm3-3b-Q4_K_S` and `smollm3-3b-Q4_K_M` both fail
  with `GGUF BPE pre-tokenizer 'smaug-bpe' is not supported for standalone conversion; use the
  original tokenizer.json or add its exact tokenizer.ggml.pre profile` — a tokenizer-construction
  failure before the model itself loads, unrelated to quantization or to the `ToolCallFormat.QWEN`
  hypothesis from the previous entry (never reached far enough to test it). SmolLM3-3B is not
  usable via this mistralrs version's direct GGUF loading path at all, regardless of quant.

`qwen3.5-4b-Q4_K_S` and `qwen3.5-2b-Q4_K_S` both loaded and pulled successfully — the only two of
the previous entry's five new-model/new-quant additions that actually work.

Separately, `scripts/benchmark-models.sh` stalled after one model: `qwen3.5-4b`'s run had 1 real
(expected — `add.basmati_rice_in_different_unit`, deliberately left failing) scenario failure, so
`pytest` exited 1, and the script's `set -euo pipefail` aborted the whole loop right there instead
of moving to the next preset. Bug in the script, not in the eval suite or the registry.

## Current State

- `src/sumac/llm.py`: `MODEL_PRESETS` reduced from 10 back to 5 — the original 3 plus
  `qwen3.5-4b-Q4_K_S` and `qwen3.5-2b-Q4_K_S` only. `qwen3.5-4b-UD-Q4_K_XL`,
  `qwen3.5-2b-UD-Q4_K_XL`, and all three `smollm3-3b-*` presets removed, with a short comment
  pointing back at this entry rather than restating the failure detail in code.
- `scripts/benchmark-models.sh`: the per-model `pytest` line now ends `|| true`, with a comment
  explaining why (a scenario failing is an expected, informative result some of the time — see the
  eval README's Blind Spots — and shouldn't abort the rest of the model comparison).

## Verified this session

Real-model evidence only (no container run needed/attempted for this entry) — `qwen3.5-4b`'s real
`scripts/benchmark-models.sh` run: 25 collected, 21/22 correctness (`time 40.6s`), the one failure
being the already-documented `add.basmati_rice_in_different_unit`. `ruff check`/`ruff format
--check`/`ty check` on the edited `llm.py`: clean. `bash -n scripts/benchmark-models.sh`: clean.
`MODEL_PRESETS` name count re-checked (5, no duplicates).

## Missing

- `scripts/benchmark-models.sh` with the `|| true` fix hasn't been re-run for real yet — should
  now get through all 5 registry presets and produce a `runs/*.json` per preset plus the
  `evals/report.jq` summary table in one pass.
- The local HF cache still holds the fetched-but-broken GGUF files for the removed presets
  (`Qwen3.5-4B-UD-Q4_K_XL.gguf`, `Qwen3.5-2B-UD-Q4_K_XL.gguf`, and the whole `SmolLM3-3B-GGUF`
  repo, all three of its fetched quants) — safe to delete (the user's own action, per standing
  preference; `SmolLM3-3B-GGUF` can go entirely, the two Qwen repos should keep their working
  `Q4_K_M`/`Q4_K_S` files and only drop the `UD-Q4_K_XL` blob).
- If `mistralrs` gains `IQ4_XS`/dynamic-quant support in a later version, or SmolLM3 GGUFs start
  shipping with a recognized pre-tokenizer profile, both are worth retrying — not planned now.

---

# 2026-09-03: Decision — qwen3.5-4b Q4_K_M, Registry Pruned to One

## Context

Final real run of the comparison, across the 5 working presets from the previous entry
(`qwen3.5-4b`, `qwen3.5-4b-Q4_K_S`, `lfm2.5-2.6b`, `qwen3.5-2b`, `qwen3.5-2b-Q4_K_S`). Results
(pass/22, `total_duration_s` — wall-clock sum of per-scenario latency, model load excluded):

| model | pass | rate | time |
|---|---|---|---|
| qwen3.5-4b (Q4_K_M) | 21 | 95.5% | 41.1s |
| qwen3.5-4b-Q4_K_S | 21 | 95.5% | 53.4s |
| lfm2.5-2.6b | 20 | 90.9% | 303.3s |
| qwen3.5-2b (Q4_K_M) | 17 | 77.3% | 60.0s |
| qwen3.5-2b-Q4_K_S | 14 | 63.6% | 38.2s |

Two things flagged as worth naming, neither chased further this session by the user's own choice
(deliberately stopping model exploration here — see Decision below):
- The 4B pair's timing (Q4_K_M faster than Q4_K_S, 41.1s vs 53.4s) runs opposite to what file-size
  difference alone would predict (2.74GB vs 2.59GB, ~6%) and each model was only run once —
  plausibly single-run noise rather than a real per-quant speed difference, unlike the 2B pair,
  where Q4_K_S being both faster *and* less accurate is the physically expected direction and a
  much larger, more trustworthy effect (77.3% → 63.6%).
- lfm2.5-2.6b's 303s (5-8x every other preset here, despite being smaller than qwen3.5-4b) wasn't
  root-caused — plausibly a GPU-fallback or kernel-support gap in this `mistralrs` version for
  LFM2.5's architecture rather than an inherent property of the model, but not confirmed.

## Decision

**qwen3.5-4b (`Qwen3.5-4B-Q4_K_M.gguf`) is the model.** Accuracy is effectively tied with every
alternative that could load, latency is second-best (within noise of the best), and its failure
breakdown is a single known, already-documented edge case
(`add.basmati_rice_in_different_unit` — a `decide.py` unit-conversion gap, not a tool-calling or
classification problem). Model exploration is paused here, deliberately — the user is moving on to
prompt/tool-schema work in `src/sumac/llm.py` next, not further model comparison.

## Current State

- `src/sumac/llm.py`: `MODEL_PRESETS` reduced to one entry — `qwen3.5-4b`
  (`unsloth/Qwen3.5-4B-GGUF` / `Qwen3.5-4B-Q4_K_M.gguf`). `qwen3.5-2b`, `lfm2.5-2.6b`,
  `qwen3.5-4b-Q4_K_S`, and `qwen3.5-2b-Q4_K_S` all removed, with a comment on the remaining entry
  pointing back at this entry for why. The preset's own `name` was never quant-suffixed (it was
  always `"qwen3.5-4b"`, distinct from its `quantized_filename`) — nothing to rename there.
  `DEFAULT_MODEL_PRESET = MODEL_PRESETS[0]` still holds, now trivially the only entry.
- **Two real unit tests (`tests/`, not `evals/`) depended on the registry having more than one
  entry and would have broken silently on the next run** — caught and fixed in this pass, not
  left as debris:
  - `tests/test_llm.py::test_run_loop_appends_lfm_formatted_assistant_message_when_configured`
    called `llm.model_preset("lfm2.5-2.6b")` to get an LFM-format preset for a fully-faked
    `AgentRunner` run (no real GGUF ever touched). Changed to construct a throwaway
    `ModelPreset("test-lfm", "unused/repo", "unused.gguf", ToolCallFormat.LFM)` directly — this
    test exercises `_render_tool_call`'s LFM branch reaching `_run_loop`, not anything about the
    real model registry, so it shouldn't have depended on the registry's contents in the first
    place.
  - `tests/test_cli.py::test_ask_regenerate_reuses_the_prompt_with_a_different_model` did
    `next(p for p in llm.MODEL_PRESETS if p != llm.DEFAULT_MODEL_PRESET)` to drive the "g"
    regenerate flow's "different model" case — `StopIteration` with only one preset registered.
    Changed to `monkeypatch.setitem(llm._MODEL_PRESETS_BY_NAME, ...)` a throwaway second preset
    for the duration of the test — `_prompt_regenerate` only needs `llm.model_preset(name)` to
    resolve, and this test's `AgentRunner` is fully faked too.
- `evals/README.md`'s two example commands (`--eval-model`, `--eval-json` path) updated from
  `qwen3.5-2b` to `qwen3.5-4b`; the "Comparing models" section rewritten to state the current
  one-preset state and decision directly, with `sumac models pull qwen3.5-4b lfm2.5-2.6b`'s
  now-dead second name dropped from its example.
- `scripts/benchmark-models.sh` and `evals/report.jq` untouched — both are name-agnostic (driven
  by `sumac models list --names-only`), so they work unchanged for a one-preset registry and stay
  ready if model comparison resumes later.
- HF cache cleanup is the user's own action (standing preference — see the previous entry); not
  done here. `LiquidAI/LFM2.5-2.6B-GGUF`, `unsloth/Qwen3.5-2B-GGUF`, and `unsloth/SmolLM3-3B-GGUF`
  are entirely unused now and safe to remove wholesale (`hf cache rm model/<repo> -y`, or the
  interactive `hf cache delete` picker). `unsloth/Qwen3.5-4B-GGUF` must stay (holds the winning
  `Q4_K_M` file) — the `Q4_K_S`/`UD-Q4_K_XL` blobs inside that same repo are now unused too, but
  `hf cache`'s delete tooling only operates at whole-revision granularity (no per-file delete —
  see `huggingface/huggingface_hub` issue #2219), so removing just those two files means either
  leaving them (harmless, ~5.6GB) or manually deleting the specific blobs by hand.

## Verified this session

`ruff check`/`ruff format --check`/`ty check` on every edited `.py` file (`llm.py`, `cli.py`,
`tests/test_llm.py`, `tests/test_cli.py`, `evals/`): clean. No other `MODEL_PRESETS`-size
assumption found repo-wide (`grep -rn "next(p for p in llm.MODEL_PRESETS\|len(llm.MODEL_PRESETS)"`
— empty after the fix above). A real `pytest` run of the two fixed tests was not possible in this
container (same limitation as every session so far — this container's Python doesn't match the
`.venv`'s ABI, on top of no GPU); both fixes were verified by reading, not by running.

## Missing

- The two fixed tests (`test_run_loop_appends_lfm_formatted_assistant_message_when_configured`,
  `test_ask_regenerate_reuses_the_prompt_with_a_different_model`) need a real `pytest` run outside
  this container to confirm — they were unit tests with fully-faked completions before this pruning
  ever touched them, so they *should* be unaffected by the model registry shrinking, but that's
  reasoning, not a confirmed run.
- The `lfm2.5-2.6b` 303s latency and the 4B Q4_K_M-vs-Q4_K_S timing anomaly are both left
  unexplained, by choice — noted for whoever revisits model exploration later, not investigated
  further now.

---

# 2026-09-03: Added qwen3.8-4b-distill (Untried Candidate)

## Context

User surfaced a community model via a tweet: `empero-ai/Qwen3.8-4B-Distill-GGUF`, a
full-parameter distillation of Alibaba's Qwen3.8-Max (2.4T-A95B MoE, a real recent flagship
release — confirmed via search, not assumed) onto the Qwen3.5-4B architecture. Claimed MMLU
+19.9pts / GSM8K -6.5pts vs. base Qwen3.5-4B per the tweet. Verified directly against the HF repo
before adding anything (not taken from the tweet alone): `Qwen3.8-4B-Q4_K_M.gguf`, 2.783GB,
labeled "Recommended" on the repo — matches the tweet exactly. No tool-calling documentation on
the model card.

## Current State

- `src/sumac/llm.py`: `MODEL_PRESETS` gained `qwen3.8-4b-distill`
  (`empero-ai/Qwen3.8-4B-Distill-GGUF` / `Qwen3.8-4B-Q4_K_M.gguf`, `ToolCallFormat.QWEN`).
  `DEFAULT_MODEL_PRESET` unchanged (still `qwen3.5-4b`, index 0) — this is a candidate to
  benchmark, not yet a replacement for the settled default from the previous entry.
- Same architecture family as `qwen3.5-4b` (Qwen3.5's hybrid Gated DeltaNet/attention layers,
  per the model card's own llama.cpp-build-version warning) — meaningfully better odds of loading
  cleanly than SmolLM3-3B or Granite were, since this `mistralrs` already loads that architecture
  successfully today. `ToolCallFormat.QWEN` is inherited from that architecture match, not
  confirmed against this specific fine-tune's own chat template — same caveat pattern as the
  SmolLM3 addition two entries back, and same required check before trusting a benchmark: run it
  for real and inspect the tool-call round-trip (`--eval-debug`) before trusting the score.

## Missing

- Entirely unverified against a real run — no pull, no load, no benchmark attempted this session.
  `sumac models pull qwen3.8-4b-distill` then `uv run pytest evals --eval-model
  qwen3.8-4b-distill --eval-json runs/qwen3.8-4b-distill.json` is the next step, outside this
  container.

---

# 2026-09-03: Added gemma-4-e2b and spark-x2.5-4b — Two Different Risk Levels

## Context

User named two more candidates from a tweet/browsing: `unsloth/gemma-4-E2B-it-qat-GGUF` and
`XHToken/Spark-X2.5-4B`. Both researched against their actual HF repos before adding anything —
neither was taken at face value.

- **The QAT repo the user named only ships `UD-Q2_K_XL`/`UD-Q4_K_XL`** — no plain Q4_K_M/Q4_K_S at
  all. `UD-Q4_K_XL` is the exact IQ4_XS-tensor quant type already confirmed unloadable under this
  `mistralrs` two entries back (every `UD-*` preset tried failed the same way, across three
  different model families). Rather than register a preset already known to hit that same wall,
  used the sibling **non**-QAT repo, `unsloth/gemma-4-E2B-it-GGUF`, which does have a plain
  `gemma-4-E2B-it-Q4_K_M.gguf` (3.11GB) — same model family, sidesteps a failure mode we already
  have hard evidence for.
- Gemma 4's tool-call syntax is genuinely different from Qwen/LFM —
  `<|tool_call>call:name{key:value,...}<tool_call|>`, unquoted argument values, asymmetric
  open/close tags. Reconstructed from published research (not a real chat template read
  byte-for-byte, unlike QWEN/LFM/SmolLM3's QWEN-reuse) — search results also surfaced a confirmed
  upstream bug (an LM Studio issue: Gemma 4's own official chat template calls a
  `format_type_argument` macro it never defines), meaning even first-party tooling has had to patch
  around this template. New `ToolCallFormat.GEMMA` added on that basis, explicitly flagged as the
  least-confident format in the module — more uncertain than the SmolLM3 addition two entries back,
  which was a verified-identical match to an existing format, not new code.
- **Spark-X2.5-4B has real, not just absent, evidence against it loading here**: its own model
  card instructs installing a *fork* of llama.cpp (`git clone
  https://github.com/XHToken/llama.cpp.git`) for its hybrid attention architecture (one
  full-attention layer per three sliding-window layers) — meaning mainline llama.cpp/GGUF doesn't
  support its custom ops at all, and `mistralrs` (which implements standard GGUF architectures, not
  community forks) has no more reason to support them than mainline llama.cpp does. Also: this
  repo ships exactly one GGUF, unquantized, 8.23GB — no quantized variant exists to pick instead.

## Current State

- `src/sumac/llm.py`: `ToolCallFormat` gained `GEMMA`. `_render_tool_call` gained a `GEMMA` branch
  (unquoted `key:value` pairs, no escaping — the real escaping rule was never confirmed, so none
  was invented). A new unit test,
  `test_llm.py::test_render_tool_call_gemma_uses_call_colon_syntax`, pins down what this module's
  own function produces — it guards against a regression in *this* code, explicitly not a claim
  that Gemma 4 actually expects that output.
- `MODEL_PRESETS` gained two entries: `gemma-4-e2b` (`unsloth/gemma-4-E2B-it-GGUF` /
  `gemma-4-E2B-it-Q4_K_M.gguf`, `ToolCallFormat.GEMMA`) and `spark-x2.5-4b`
  (`XHToken/Spark-X2.5-4B-GGUF` / `Spark-X2.5-4B.gguf`, `ToolCallFormat.QWEN` as a low-confidence
  placeholder — the repo mentions Hermes-style tool use but documents no template of its own, and
  it mostly doesn't matter if the model can't load to begin with). `DEFAULT_MODEL_PRESET` unchanged
  (still `qwen3.5-4b`). Registry is now 4 presets.
- `ruff check`/`ruff format --check`/`ty check` on `llm.py` and `tests/test_llm.py`: clean. The new
  `_render_tool_call` GEMMA branch's exact string output was also hand-verified against the new
  test's expected value directly in a standalone snippet (this container can't run the real
  `pytest` suite — see every prior entry).

## Missing

- Neither preset has been pulled or run for real. `spark-x2.5-4b` in particular is a real
  candidate for simply refusing to load, given the fork requirement — worth trying only if the
  8.23GB download is cheap enough to spend on confirming that.
- `gemma-4-e2b`'s `ToolCallFormat.GEMMA` rendering is the most speculative piece of code in this
  module right now. If it loads but behaves oddly on tool-heavy scenarios (garbled replies,
  repeated identical tool calls, never settling), check the escaping gap first via `--eval-debug`
  before concluding the model itself is weak.

---

# 2026-09-03: Confirmed Broken — gemma-4-e2b, spark-x2.5-4b — and Tokens/Sec

## Context

Real-run results for the two previous entry's additions, both hard failures at load time — not
tool-calling or accuracy problems, the model never starts:

- `gemma-4-e2b`: `GGUF architecture 'gemma4' is supported only as a multimodal model. Pass its
  companion projector with --mmproj <file>, or load from a GGUF repository that publishes one;
  text-only 'gemma4' checkpoints are not supported`. This `mistralrs` treats Gemma 4 as
  inherently multimodal at the architecture level — a text-only checkpoint with no `mmproj`
  sibling (which `unsloth/gemma-4-E2B-it-GGUF` doesn't publish) can't load regardless of quant.
  The `ToolCallFormat.GEMMA` best-effort rendering added alongside this preset was never
  exercised — the failure is earlier than that.
- `spark-x2.5-4b`: `Unknown normal-model GGUF architecture 'spark2_5'` — confirms the previous
  entry's inference from the model card's "install our llama.cpp fork" instruction: mainline
  GGUF/llama.cpp (and `mistralrs`, which implements that spec, not community forks) has no
  registration for this architecture at all.

Separately, the user asked to add tokens/sec to the eval output alongside the existing wall-clock
latency (`duration_s`) — mistral.rs already reports per-round `usage.completion_tokens` /
`usage.total_time_sec` (visible in the console's own `round N: ... tok/s ...` log line), just
never aggregated or exposed anywhere a test could read it.

## Current State

- `src/sumac/llm.py`: `MODEL_PRESETS` back down to 2 — `qwen3.5-4b` (default) and
  `qwen3.8-4b-distill` (still unverified against a real run — untouched by this entry).
  `gemma-4-e2b`/`spark-x2.5-4b` removed with a one-line pointer back to this entry.
  `ToolCallFormat.GEMMA` and its `_render_tool_call` branch/unit test are left in place — unused by
  any current preset, same as `ToolCallFormat.LFM` already was after `lfm2.5-2.6b` was pruned;
  infrastructure independent of what's currently registered.
- `AgentRunner` gained token-throughput tracking: `self._completion_tokens`/
  `self._generation_time_sec`, accumulated by a new `_record_usage` method (replaces the two direct
  `_print_usage(...)` call sites in `_classify`/`_run_loop` — printing behavior unchanged, it now
  also folds the same numbers into a running total), and a `tokens_per_sec` property computing
  `completion_tokens / generation_time_sec` (summed across every round across every
  `propose()`/`revise()` call this instance has made, not averaged per-round — averaging would
  over-weight short rounds). Never reset; a fresh `AgentRunner` per test scenario is what scopes it.
- `evals/conftest.py` gained a single shared `agent` fixture (`agent_runner_factory` + `result` as
  dependencies), replacing four identical one-line `agent` fixtures duplicated across
  `test_add.py`/`test_find.py`/`test_remove.py`/`test_reject.py` — on teardown it writes
  `result.tokens_per_sec = agent.tokens_per_sec`. Depending on `result` means this fixture's
  teardown runs *before* `result`'s own (pytest tears down in reverse dependency order), so the
  write always lands before `result` is captured into the session list.
- `evals/evaluators.py`: `EvalResult` gained `tokens_per_sec: float | None = None` — not a check
  (nothing to pass/fail), same category as `duration_s`.
- `_print_summary` prints a `tok/s` line (mean across scenarios that have one — `test_termination.py`
  and `test_fixtures.py` don't use the `agent` fixture at all, so they contribute `None` and are
  excluded from the mean, same as they always were `duration_s`-adjacent no-ops). `--eval-json`
  payload gained `tokens_per_sec` per scenario and a top-level `mean_tokens_per_sec`.
  `evals/report.jq` passes `mean_tokens_per_sec` through into the comparison table.

## Verified this session

`ruff check`/`ruff format --check`/`ty check` on every edited file: clean. `evals/report.jq`
re-validated against a synthetic run JSON carrying the new field (real `runs/*.json` files from
prior entries were already cleared by the user) — parses and passes the field through correctly.
No other `def agent(` definitions remain outside the new shared one
(`grep -rn "def agent(" evals/*.py` → one hit, `conftest.py`). `test_termination.py`/
`test_fixtures.py` confirmed to never reference `agent`/`agent_runner_factory`, so the fixture
consolidation doesn't touch them.

## Missing

- None of this — the registry prune or the tok/s instrumentation — has been run for real in any
  environment this session; same limitation as every entry so far (no GPU, no matching Python ABI
  in this container). The `tokens_per_sec` property's arithmetic was checked by reading, not by
  running a real `AgentRunner` against a real `Usage` object.
- `qwen3.8-4b-distill` (the one still-unverified addition left in the registry) hasn't been pulled
  or benchmarked yet, tok/s or otherwise.

---

# 2026-09-03: Higher-Bit Quants of qwen3.5-4b

## Context

User wanted to benchmark less-lossy quants of the settled default, prompted by a claim (from a
Claude conversation elsewhere) that tool-calling token positions take disproportionately more
distributional damage from 4-bit quantization than prose does, with Q6_K/fp8 suggested as a
near-lossless alternative worth the <1GB size cost on a 4B model.

Added the three plain K-quants `unsloth/Qwen3.5-4B-GGUF` offers above Q4_K_M — `Q5_K_M` (3.14GB),
`Q6_K` (3.53GB), `Q8_0` (4.48GB) — all in the same confirmed-loadable dtype family as the existing
Q4_K_M/Q4_K_S presets (`F32/F16/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, Q2_K through Q8_K`, per the
mistralrs error message from two entries back). Deliberately skipped this same repo's
`UD-Q5_K_XL`/`UD-Q6_K_XL`/`UD-Q8_K_XL` and plain `IQ4_NL`/`IQ4_XS` — all outside that dtype family
(`IQ*`-tensor-containing, same failure mode already confirmed for `UD-Q4_K_XL`), so not worth
registering without evidence they'd load any better at a different bit width.

Not chased: the specific "public nvfp4 quant with KL-divergence 0.1049/1.3375" claim from the
Claude conversation — NVFP4 isn't a GGUF quantization format at all (it's NVIDIA's own
TensorRT-LLM-oriented format), so a repo matching that description almost certainly isn't loadable
through this project's GGUF-only `mistralrs` pipeline regardless of the number's accuracy, which
wasn't independently verified either. The KV-cache-dtype suggestion and the other three
directions offered alongside it (best-of-n selection, silent-failure taxonomy, constrained
decoding) weren't requested this round — the ask was scoped to quant benchmarking only.

## Current State

`MODEL_PRESETS` now holds 4 entries: `qwen3.5-4b` (default, Q4_K_M), `qwen3.5-4b-Q5_K_M`,
`qwen3.5-4b-Q6_K`, `qwen3.5-4b-Q8_0`.

## Verified this session

`ruff check`/`ruff format --check`/`ty check` on `llm.py`: clean. Filenames/sizes confirmed against
the real HF file listing (not guessed).

## Missing

Unpulled, unbenchmarked — `sumac models pull` then `scripts/benchmark-models.sh` is the next step,
same as every addition this session.

---

# 2026-09-03: Quant Question Closed — Q4_K_M Confirmed Best, Registry Back to One

## Context

Real run of the three higher-bit quants added in the previous entry, against the same 22
scenarios. Result: every one of Q5_K_M/Q6_K/Q8_0 scored *worse* than Q4_K_M (20/22 vs 21/22) and
*slower* (79-88 tok/s vs 97 tok/s) — higher precision cost both accuracy and throughput here, the
opposite of the hypothesis that prompted trying them.

Both remaining failures were traced to specific causes, neither a quantization artifact:
- **Basmati Rice unit mismatch (`bag` expected, `jug` got)** — happens identically on all four
  quants, including Q4_K_M. Already-documented: `decide._resolve_product` has no bag-to-jug
  conversion registered, a `decide.py` gap, not a model or quant problem.
- **Butter amount mismatch (`2` expected, `4` got) — Q5/Q6/Q8 only, not Q4_K_M.** The model's own
  tool call passed `amount="4"`, and its reply text shows why: it resolved "add 2 more" against the
  *post-add* total (2 existing + 2 new = 4) instead of the requested delta. A genuine
  higher-quant-only regression on this specific scenario, not a mistralrs/runtime bug.

## Decision

Quant exploration is closed. `qwen3.5-4b` (`Q4_K_M`) remains the only preset in the registry —
`qwen3.5-4b-Q5_K_M`, `qwen3.5-4b-Q6_K`, `qwen3.5-4b-Q8_0` removed. Model/quant selection work for
`sumac ask` is done; the user is moving on to other work.

## Current State

`MODEL_PRESETS` holds one entry again: `qwen3.5-4b`. HF cache cleanup for the three removed quants
is the user's own action, as with every prior removal this session.

## Verified this session

`ruff check`/`ruff format --check`/`ty check` on `llm.py`: clean.

## Missing

The Basmati Rice bag/jug gap (`decide._resolve_product`) and the higher-quant-only amount
resolution bug remain unfixed — both out of scope for this suite, noted here in case either is
revisited later. Neither blocks anything; `qwen3.5-4b` at 21/22 is the shipped answer.

---

# 2026-09-03: Added qwen3.5-9b (Unbenchmarked, Kept for Later)

## Context

User wants a larger same-family model on hand for a future comparison, expecting it to be slower
but not chasing that trade-off right now. Confirmed `unsloth/Qwen3.5-9B-GGUF`'s `Q4_K_M` filename
and size against the real HF listing (5.68GB) before adding — same plain-K-quant family already
proven to load under this `mistralrs`, same architecture/tool-call format as the working default,
so no new compatibility risk expected.

## Current State

`MODEL_PRESETS`: `qwen3.5-4b` (default) and `qwen3.5-9b`
(`unsloth/Qwen3.5-9B-GGUF`/`Qwen3.5-9B-Q4_K_M.gguf`, `ToolCallFormat.QWEN`). Not pulled, not run.

## Verified this session

`ruff check`/`ruff format --check`/`ty check` on `llm.py`: clean.

## Missing

Unbenchmarked by design — the user is keeping this on hand rather than comparing now.

---

# 2026-09-03: Repeated-Epoch Comparison (epoch-benchmark.sh, epoch_report.py)

## Context

A real qwen3.5-4b vs qwen3.5-9b run showed the 9B one write short (19/20 vs 20/20) — a
single-epoch difference too small to tell apart from noise on its own. The user asked for a
routine to run repeated trials and report a rate rather than a one-off pass/fail, and shared two
AI-drafted designs (their own ChatGPT conversation, and a from-Claude one) for comparison, asking
explicitly to be told where either was poor design rather than have it implemented as-is.

This suite already tried something like this once, in much more elaborate form —
`docs/journal/2026-09-02-eval-suite.md`'s original (pre-reduction) design had `evals/run.py`
orchestrating epochs and `evals/compare.py` doing a paired McNemar test, observed intraclass
correlation, a two-way cluster bootstrap, and a realised MDE, all deleted in the reduction pass as
not earning their complexity at ~22-25 hand-picked scenarios. That same original design had
already worked out the one piece worth keeping, for a reason neither pasted AI proposal mentioned:
**one epoch must be one separate `pytest` process at its own explicit `--eval-seed`, not N
repetitions sharing one loaded model** — `mistralrs.ChatCompletionRequest` has no seed field, only
`Runner.__init__` does, so a shared Runner's RNG stream position at attempt `k` depends on how
many tokens every prior attempt generated; reordering or filtering scenarios would silently change
results under a shared-Runner design. Both pasted proposals defaulted to that unsafe shape (a
single harness looping N times), just via different external mechanisms.

Two other pieces of the ChatGPT proposal were cut, not merely trimmed, for reasons specific to
this harness rather than general disagreement with the source: "10 warm-up runs, discarded" is
MCMC-style burn-in reasoning applied where it doesn't fit — each epoch here is already an
independent, freshly-seeded draw from a stationary distribution (same model, same temperature)
from its first token, there is no non-stationary initial state to warm out of. "Paired/interleaved
run ordering, randomized between repetitions" defends against cache/order effects that don't exist
here — each epoch is a fully separate process with nothing carried over from the last, so there is
no shared state for interleaving to protect against.

## Current State

- `scripts/epoch-benchmark.sh N`: for every preset `sumac models list` returns, runs `N` separate
  `pytest --eval-model NAME --eval-seed K --eval-json runs/epochs/NAME/epoch-KK.json` invocations
  (`K` = 1..N), reusing the `--eval-seed`/`--eval-json` flags `conftest.py` already had wired up
  for exactly this (unused for it until now). `|| true` on the pytest line, same reasoning as
  `benchmark-models.sh`: one epoch's real scenario failures shouldn't abort the rest. Calls `sumac
  models pull` first.
- `evals/epoch_report.py` (new, zero new dependencies — stdlib only): reads every
  `runs/epochs/<model>/epoch-*.json`, groups by each file's own `"model"` field (not directory
  naming, so a misplaced file still groups correctly), and prints (1) an overall table — epochs,
  total attempts, pass rate, mean tok/s, mean per-scenario latency, per model; (2) a per-scenario
  pass-count table (`p/N`) with one column per model — the part that actually answers "did this
  regression show up consistently, or was that one run noise"; (3) a "disagreeing scenarios"
  subset, filtered to where the raw pass count differs by more than 1 across models; (4) an
  "always-failing" subset (every model, every epoch) — the marker for an application bug
  (`add.basmati_rice_in_different_unit`) rather than a model difference, surfaced separately so it
  doesn't get read as one. No p-values, confidence intervals, or effect sizes anywhere in the
  output — plain counts and rates only.
- `evals/README.md`: new "Is a difference real, or one noisy run?" subsection under "Comparing
  models" explaining the routine and what was deliberately left out and why; "Deliberately not here
  (yet)"'s epochs/seeds bullet corrected (they're back, in reduced form) and the file tree updated.

## Verified this session

`ruff check`/`ruff format --check`/`ty check` on `evals/epoch_report.py`: clean. `bash -n
scripts/epoch-benchmark.sh`: clean. `evals/epoch_report.py` functionally smoke-tested against
synthetic epoch JSON (10 epochs x 2 models x 3 scenarios, one scenario built to fail identically on
both models, one built to regress only on the second) — correctly separated the two into
"always-failing" vs "disagreeing" in the printed output, matching the intended diagnostic split.

## Missing

- Not run against real models this session — `scripts/epoch-benchmark.sh` needs a real GPU/cache,
  same limitation as everything else here. The next step is literally `scripts/epoch-benchmark.sh
  10` against the current registry (`qwen3.5-4b`, `qwen3.5-9b`) to find out whether the 9B write
  regression that prompted this is real.

---

# 2026-09-03: Traces Now Captured in --eval-json

## Context

The 10-epoch qwen3.5-4b vs qwen3.5-9b comparison surfaced two real, consistent per-scenario
regressions (`add.missing_item_discovers_new_product` 10/10 vs 3/10,
`add.multiple_products_with_omitted_amounts` 10/10 vs 7/10) — not noise, per `epoch_report.py`'s
disagreement filter. The obvious next step, diffing the actual failing traces between models, was
blocked: nothing captured `AgentPlan.trace` anywhere. It existed only in memory for the duration of
whichever test built it, then was discarded.

## Current State

- `src/sumac/llm.py`: `AgentRunner` gained `trace_history` — every `ToolCallRecord` dispatched
  across every `propose()`/`revise()` call the instance has made, never reset (same accumulate-
  don't-reset shape as `tokens_per_sec`'s counters). A new `_record_call` helper appends to both
  the existing per-call `self._trace` (unchanged — still resets at the top of `propose()`/
  `revise()`, still what `AgentPlan.trace` is built from) and the new `self._trace_history`;
  replaces the three direct `self._trace.append(...)` call sites in `_classify`/`_run_loop`.
- `evals/evaluators.py`: `EvalResult` gained `trace: list[dict]`.
- `evals/conftest.py`'s `agent` fixture now also writes `result.trace` on teardown (each
  `ToolCallRecord` flattened to `{"name", "arguments", "result"}`), and the `--eval-json` payload
  carries it per scenario as `"trace"`. Not printed in the console summary — too verbose for the
  table — but present in the JSON output, which is where a failing scenario actually gets debugged
  from now (e.g. pasted to an outside model for comparison, as the user was already doing for the
  epoch results themselves).
- `evals/README.md` updated to describe it and where to find it.

## Verified this session

`ruff check`/`ruff format --check`/`ty check` on every edited file: clean. Not exercised against a
real model or a real `--eval-json` write this session — same limitation as everything else here
(no GPU, no matching Python ABI in this container).

## Missing

- The actual 4B-vs-9B trace diff for `add.missing_item_discovers_new_product` and
  `add.multiple_products_with_omitted_amounts` — the reason this was built — hasn't been done yet.
  Re-running `scripts/epoch-benchmark.sh` now will capture it; the previous 10-epoch run's traces
  are gone, since this landed after that run finished.
