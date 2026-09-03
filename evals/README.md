# sumac eval suite

Behavioural tests for `sumac ask`'s agent (`src/sumac/llm.py`) against a real local model. Lives
in this repo, doesn't ship with it — not under `src/`, excluded from the sdist
(`[tool.hatch.build.targets.sdist] exclude = ["/evals"]`), and the wheel already excludes anything
outside `src/sumac`.

Deliberately small: one seeded inventory, ~23 individually-named tests, no generated case matrix,
no epochs, no statistics. Full rationale in `docs/journal/2026-09-02-eval-suite.md` — in
particular its second implementation pass, which cut a much larger generated-case version back to
this.

## Install

```sh
uv sync --no-group ask --group ask-cuda --group evals    # or --group ask for CPU/Metal
```

## Running

```sh
uv run pytest evals/test_termination.py -v   # deterministic, no model, ~1s
uv run pytest evals/test_agent.py -v          # needs a cached GGUF — see below
uv run pytest evals -v                        # both
```

`test_agent.py`'s tests skip cleanly — no network attempt, no GPU touched — if the target GGUF
isn't already in the local Hugging Face cache. Run `sumac ask` once against the model you want
first, so the weights are cached, then these actually run instead of skipping.

```sh
uv run pytest evals/test_agent.py -v --eval-model qwen3-4b
```

## What's here

- **`conftest.py`** — four safety rails (below), one seeded `inventory` fixture, an
  `agent_runner_factory` fixture that builds a fresh `AgentRunner` per test over one shared loaded
  model, and a handful of plain assertion functions (`assert_write`, `assert_no_writes`,
  `assert_classified`, `assert_tool_called`, `is_ask_or_act`). No generic scorer class, no
  expectation dataclasses — every test calls these directly and reads top to bottom.
- **`fixtures.py`** — one realistic inventory (the vocabulary from the real runs in
  `docs/journal/2026-09-01-ask-agent-design.md`), seeded via real `sumac` CLI commands.
- **`test_agent.py`** — the behavioural cases, one `def test_*` per named scenario. `pytest -v`
  shows every one passing or failing by name.
- **`test_termination.py`** — one deterministic test (no model) proving `AgentRunner`'s round cap
  actually stops a model that keeps repeating the same rejected tool call, rather than looping or
  hanging. Does not fix, or attempt to reproduce, the real context-overflow failure it's named
  after — that diagnosis is in the journal entry, and the fix (if any) is a separate `llm.py`
  decision.

That's the whole suite. If you can read `conftest.py` and `test_agent.py` in one sitting, that was
the design goal.

## Safety rails (kept from the earlier, larger version)

The real household inventory lives outside this repository. The seeded inventory lives under a
`pytest`-owned temp directory; `SUMAC_DATA_DIR`/`SUMAC_PASSPHRASE` are overridden for the session
so an ambient value is never read; `sumac.store.append` is wrapped to refuse any write outside
that temp root; no eval calls `AgentRunner.commit` — every assertion reads `plan.writes` from
`propose()`, the same "compute the plan, write nothing" path `sumac ask --dry-run` uses.

## Deliberately not here (yet)

Cut in the second implementation pass, on the grounds that none of it was answering "did the agent
do the right thing" any faster or more clearly than a short list of named tests does:

- **Generated cases.** Ten fixture families and a template engine produced 177 cases; almost none
  of that variation was informative at the scale this suite runs at. One hand-picked inventory,
  ~23 hand-picked cases.
- **Epochs, seeds, `pass^k`.** No orchestrator, no per-seed JSON, no aggregation. One model, one
  seed (optional, for reproducing a single run), one pass.
- **Null baselines.** With generated cases dominated by `NoWrites` expectations, a do-nothing agent
  could score misleadingly well, which is why they existed. With ~23 cases split across find/add/
  remove/reject and a good share requiring a specific write, that risk is much smaller, and a
  human reading 23 named results doesn't need a computed floor to sanity-check them.
- **McNemar / ICC / MDE / cluster bootstrap.** All implemented once, all deleted — this suite isn't
  trying to detect a 5-percentage-point regression yet, and none of that machinery earns its
  complexity at 23 cases.

Revisit any of these if the suite outgrows what a person can read and reason about directly —
not before.

## Blind spots

- **`waste` and `purchase` have no route through the agent.** `AgentRunner.tool_callbacks` only
  ever emits `ChangeKind.DISCOVERY`, `CONSUMPTION`, or `MOVEMENT`
  (`src/sumac/llm.py:311-315`) — those two other `ChangeKind`s exist in the domain model but only
  `sumac add` reaches them directly. Nothing here can test a distinction the agent is structurally
  unable to make.
- **Location references beyond an id or exact display path aren't resolvable.** A positional
  reference ("3rd shelf along") or an emptied-location reference has no route to a location id —
  `decide._resolve_location` accepts only those two forms, no prompt carries the location tree, and
  no tool enumerates locations. Not covered here; would need an `llm.py` change first.
- **`test_add_unit_conflict_rejected_basmati_rice`** is the real-model reproduction of the
  diagnosed context-overflow failure. If it starts failing (or hanging), that's the signal —
  don't just raise `at_most`.
