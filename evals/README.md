# sumac eval suite

An eval suite for `sumac ask`'s in-process agent (`src/sumac/llm.py`), run against a real local
model. It lives in this repo but does not ship with the library — it is not under `src/`, and
`[tool.hatch.build.targets.sdist] exclude = ["/evals"]` keeps it out of the sdist; the wheel
already excludes anything outside `src/sumac`.

Full design rationale, the statistics behind the case count, and what each layer catches is in
`docs/journal/2026-09-02-eval-suite.md`. This file is the how-to.

## Install

```sh
uv sync --no-group ask --group ask-cuda --group evals    # or --group ask for CPU/Metal
```

The suite needs `mistralrs` importable the same way `sumac ask` does — that's what `--group ask`
or `--group ask-cuda` provides. It does **not** need a GPU or a downloaded model to run the
model-free layer (below).

## Two layers

**Model-free** (`test_scoring.py`): the scorer's own self-tests, vocabulary hygiene checks, and
the three null-baseline agents run against the full generated case table. No GGUF, no GPU. This
is what CI can run on every commit.

**Model-gated** (`test_routing.py`, `test_proposals.py`, `pytest.mark.model`): classification and
full `propose()` runs against a real local model. These skip cleanly — no network download, no
GPU touched — when the target GGUF isn't already in the local Hugging Face cache
(`evals/conftest.py`'s `_gguf_cached_locally` check runs before any real `mistralrs.Runner` is
constructed). Run `sumac ask` once against the model you want first, so its weights are cached,
then these tests will actually run instead of skipping.

## Running

```sh
uv run pytest evals -m "not model"        # scorer, baselines, vocab hygiene — no GPU, ~12s
uv run pytest evals -m model               # routing + proposals, one epoch, needs a cached model
uv run pytest evals --families 2 -m model  # smaller population for a faster dev loop, ~90s-ish
uv run pytest evals/test_scoring.py        # just the model-free module
```

For more than one epoch, use the orchestrator rather than pytest's own loop — see "Epochs", below.

```sh
uv run python -m evals.run --epochs 8 --out runs/2026-09-02-qwen35/
uv run python -m evals.report runs/2026-09-02-qwen35/
uv run python -m evals.compare runs/before/ runs/after/
```

## Epochs are separate processes, not a pytest loop

`mistralrs.ChatCompletionRequest` has no `seed` field; `mistralrs.Runner.__init__` does, and one
`AgentRunner` session shares one `Runner` across every case. Reusing that session across epochs
would make the RNG stream position at case N depend on how many tokens every prior case
generated — so `-k` filters would change what gets sampled, and re-running one failing case
wouldn't reproduce the sample that failed. `evals.run` sidesteps this: one pytest session per
`--eval-seed`, each its own process, each reproducing exactly from its own seed. `evals.report`
aggregates the resulting `seed-NN.json` files for pass^k; `evals.compare` does the paired
comparison between two such directories.

## Reading the output

- **pass^k, not pass@k.** A case counts as passed only if every epoch passed it. "Succeeded once
  in eight tries" isn't reported as a success rate — see the eval spec's "Epochs and pass^k".
- **Headline vs. blocked.** Cases tagged `blocked` (currently `add.positional`, `add.absent_spot`)
  test a location reference the harness has no route to resolve — see the eval spec's
  "location-reference taxonomy". They're run and reported, but excluded from the headline number:
  scoring them as failures would blame the model for a harness gap.
- **Null baselines are always shown beside the real score.** `report.py` re-seeds and re-runs
  `do-nothing`, `reject-everything`, and `always-discover` fresh every time — if the model's
  number is close to one of these, the case table (or the model) has a problem worth looking at
  before trusting anything above it.
- **`hard` cases are a tripwire, not a percentage.** Seven hand-written adversarial cases (run once,
  against `fam-01` only) are too few to support a rate — `report.py`/`compare.py` list them by
  name when one starts or stops passing, which is the signal that matters at this sample size.
- **`ask-vs-act` branches, not a pass rate.** The two `AskOrAct`-tagged cases report which branch
  fired (`ask`/`act`/`inaction`) per epoch rather than collapsing to pass/fail — see
  `scoring.classify_ask_or_act`.

## Adding a case

Prefer a template in `generate.py` over a hand-written case — it runs against all ten families for
free and its gold is derived from the template's own parameters, so it can't drift from its prompt.
Add a hand-written case to `cases.py`'s `_hard_cases()` only for something templating can't express
cleanly (the way `hard-ambiguous-product` needs a genuinely ambiguous product name, not a
parameterised one).

Either way, run `uv run pytest evals/test_scoring.py -k prompt_constants` after adding any new
product/brand/location name — it fails if the name collides with a worked example in one of
`sumac.llm`'s prompt constants (see `vocab.py`'s module docstring for why that matters).

## Safety

The real household inventory lives outside this repository. Every family this suite builds lives
under a `pytest`-owned temp directory; `SUMAC_DATA_DIR`/`SUMAC_PASSPHRASE` are overridden for the
whole session so an ambient value is never read, and `sumac.store.append` is wrapped to refuse any
write outside that temp root. No eval calls `AgentRunner.commit` — every score comes from
`propose()`, the same "compute the plan, write nothing" path `sumac ask --dry-run` uses. Full detail
in the eval spec's "Safety rails".
