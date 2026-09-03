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
    called: tuple[str, ...] = ()          # never-searched
    not_called: tuple[str, ...] = ()
    max_calls: Mapping[str, int] = ()     # duplicate-search
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
