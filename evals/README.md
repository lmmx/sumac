# sumac eval suite

Behavioural tests for `sumac ask`'s agent (`src/sumac/llm.py`) against a real local model. Lives
in this repo, doesn't ship with it — not under `src/`, excluded from the sdist
(`[tool.hatch.build.targets.sdist] exclude = ["/evals"]`), and the wheel already excludes anything
outside `src/sumac`.

One seeded inventory, 25 scenarios split by capability, each a plain `def test_*`. No generated
case matrix, no epochs, no statistical apparatus. Full rationale in
`docs/journal/2026-09-02-eval-suite.md`, across all four of its dated sections — in particular the
reduction pass that cut a 177-case generated suite down, and the review pass after it that found
two of the reduced tests had become untestable and restored two cheap regression checks.

## Install

```sh
uv sync --no-group ask --group ask-cuda --group evals    # or --group ask for CPU/Metal
```

## Running

```sh
uv run pytest evals/test_termination.py evals/test_fixtures.py -v   # no model, ~1s
uv run pytest evals -v                                              # everything
uv run pytest evals/test_add.py -v --eval-model qwen3.5-4b          # one category, one preset
uv run pytest evals -v --eval-json runs/qwen3.5-4b.json             # also write results to JSON
```

`test_add.py`/`test_find.py`/`test_remove.py`/`test_reject.py` skip cleanly — no network attempt,
no GPU touched — if the target GGUF isn't already in the local Hugging Face cache. Run `sumac
models pull [NAME...]` first (defaults to every preset in the registry) so the weights are cached,
then these actually run instead of skipping.

## Comparing models

`DEFAULT_MODEL_PRESET` is `qwen3.5-4b` (`Qwen3.5-4B-Q4_K_M.gguf`), the winner of a real
multi-model, multi-quant comparison — the full field it beat (qwen3.5-2b, lfm2.5-2.6b, both
models' Q4_K_S quants) and why each was ruled out (accuracy, latency, or — for `UD-Q4_K_XL` and
SmolLM3-3B — not loading under this `mistralrs` at all) is in
`docs/journal/2026-09-02-eval-suite.md`'s 2026-09-03 entries. `MODEL_PRESETS` otherwise holds
whatever's currently being tried against that default — see the journal's later entries for each
addition's own compatibility notes before trusting its benchmark score. This tooling is what
you'd reach for to run that comparison:

```sh
sumac models list                    # every ModelPreset, and whether it's cached locally
sumac models pull                    # download every preset's GGUF (skips ones already cached)
sumac models pull qwen3.5-4b         # or just specific ones

scripts/benchmark-models.sh          # pull what's missing, run the suite once per preset,
                                      # print a pass-rate/latency table (evals/report.jq)
```

`sumac models pull` replaces manually editing `DEFAULT_MODEL_PRESET` and running `sumac ask` once
per model just to prime the cache — it loads each uncached preset just long enough to trigger
`mistralrs`' own download-on-load, the same mechanism `sumac ask` already relies on, then drops it.
`scripts/benchmark-models.sh` is `sumac models pull` plus a `pytest --eval-model NAME --eval-json
runs/NAME.json` loop over every registry preset, finished with `jq -c -s -f evals/report.jq
runs/*.json` — the aggregation query from that same comparison, checked in instead of retyped by
hand each time.

## What's here

```
evals/
├── conftest.py       # safety rails, the inventory fixture, agent_runner_factory, result collection
├── evaluators.py      # EvalResult + evaluate_* functions — the checks, not tied to any one test
├── fixtures.py         # one realistic seeded inventory, built via real `sumac` CLI commands
├── report.jq            # the multi-model summary table query — see scripts/benchmark-models.sh
├── test_find.py         # 5 scenarios
├── test_add.py           # 10 scenarios
├── test_remove.py         # 4 scenarios (consumption and movement — both classified REMOVE)
├── test_reject.py          # 3 scenarios
├── test_fixtures.py         # 2 deterministic checks on the fixture itself — no model
└── test_termination.py       # 1 deterministic check that the agent's round cap actually bounds
                               #   a model stuck repeating the same rejected call — no model
```

25 scenarios total. Split by capability (find/add/remove/reject) rather than one file, now that
each category has its own file-level `_CATEGORY` constant driving the scenario id and summary —
still small enough to read end to end.

### How a scenario is written

```python
def test_discriminator_variant_not_confused(agent, cfg, result) -> None:
    plan = agent.propose("Add 2 more packs of Unsalted Butter, with the existing stock")
    evaluate_classification(result, plan, QueryKind.ADD)
    evaluate_write(
        result, plan, cfg,
        kind=ChangeKind.DISCOVERY, product_id="Unsalted Butter",
        amount="2", unit="packs", to_location="freezer-drawer-2",
    )
    assert result.passed, result.failures
```

`evaluate_*` functions (`evaluators.py`) mutate a plain `EvalResult` in place — one named check per
dimension (`classification`, `product`, `amount`, `unit`, `location`, a `tool:<name>` per tool
called, `reply`, `outcome` for the ask-or-act scenarios). The final `assert` is what makes pytest
report the test PASSED/FAILED by name; the `result` fixture captures the same `EvalResult`
regardless, so a test that fails partway still shows exactly which checks it got right.

### Reading the output

```
SUMAC AGENT EVALUATION
  add        8/10
  find       5/5
  remove     3/4
  reject     3/3
  overall    19/22
  time       46.8s

FAILURES
  add.discriminator_variant_not_confused
    - expected to_location 'freezer-drawer-2', got 'freezer-drawer-1' (resolved: 'freezer-drawer-1')
  remove.move_vague_asks_or_acts
    - neither asked nor acted (branch=inaction)

ask-vs-act branches: {'branch=act': 2, 'branch=ask': 1}
```

Printed once at the end of the session (`pytest_sessionfinish` in `conftest.py`) — a category
tally, a total wall-clock time (sum of each scenario's own `EvalResult.duration_s`, timed around
just the test body — the once-per-session model load isn't counted), then every failing scenario
with its specific failed checks, then how the ask-or-act scenarios resolved. `--eval-json PATH`
additionally writes the same data as JSON, including each scenario's `duration_s` and a
`total_duration_s` — one run's worth; see "Comparing models" above for turning several of these
into one table.

## Safety rails

The real household inventory lives outside this repository. The seeded inventory lives under a
`pytest`-owned temp directory; `SUMAC_DATA_DIR`/`SUMAC_PASSPHRASE` are overridden for the session
so an ambient value is never read; `sumac.store.append` is wrapped to refuse any write outside
that temp root; no eval calls `AgentRunner.commit` — every assertion reads `plan.writes` from
`propose()`, the same "compute the plan, write nothing" path `sumac ask --dry-run` uses.

## Deliberately not here (yet)

- **Generated cases, epochs, seeds beyond one, null baselines, McNemar/ICC/MDE/cluster bootstrap.**
  All built once, all deleted, in the reduction pass this suite went through before reaching this
  shape — see the journal for what each did and why none of it earned its complexity at 25 cases.
- **YAML scenario files.** Considered (an external design review suggested it) and explicitly not
  done — the scenarios carry real semantic nuance in their prompts and docstrings (why this
  wording, why this location, why this outcome is acceptable) that YAML would flatten.
- **An LLM judge.** Every scenario here has a deterministic ground truth (a specific product,
  amount, unit, location) — grading a deterministic answer with a judge would be a downgrade, not
  an upgrade.

## Blind spots

- **`waste` and `purchase` have no route through the agent.** `AgentRunner.tool_callbacks` only
  ever emits `ChangeKind.DISCOVERY`, `CONSUMPTION`, or `MOVEMENT`
  (`src/sumac/llm.py:311-315`) — those two other `ChangeKind`s exist in the domain model but only
  `sumac add` reaches them directly. Nothing here can test a distinction the agent is structurally
  unable to make.
- **Location references beyond an id, an exact display path, or a location whose own name is the
  natural phrase aren't resolvable.** A positional reference ("3rd shelf along" against a grid
  cell with no natural name for its position) or an emptied-location reference has no route to a
  location id — `decide._resolve_location` accepts only an id or an exact display path, no prompt
  carries the location tree, and no tool enumerates locations. `test_fixtures.py`'s
  `test_location_path_matches_real_config` exists specifically to catch a scenario that
  accidentally depends on this gap (a real regression during this suite's own development — see
  the journal) before a real-model run has to.
- **`test_basmati_rice_in_different_unit`** is expected to fail against a real model today —
  `decide` has no bag-to-jug conversion for Basmati Rice registered, so it rejects the write. Left
  failing deliberately, as the marker of a real `decide.py`/`llm.py` gap that's out of scope for
  this suite to fix.
