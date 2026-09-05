# sumac: `sumac ask` Confirmation UX — Plan and Preview Build

**Status:** plan, written before the code it describes. The sections after "Build order" record what
the accompanying commits on `claude/sumac-ask-verification-ux-q6culs` actually landed; everything
before it is the design record.
**Scope:** `src/sumac/cli.py`'s `ask` command and its two decision loops, `src/sumac/render.py`'s
plan/trace rendering, one new `src/sumac/prompt_ui.py`, one new `scripts/preview-ask-ui.py`. No
change to `decide.py`, `ledger.py`'s fold semantics, `store.py`, `events.py`, or any tool schema
or prompt text in `llm.py`.

---

## 1. Why this, and why now

`docs/journal/2026-09-01-ask-agent-design.md` §5 states the reason `sumac ask` has a confirm step
at all: every other command in this CLI is determinate before it runs, and `ask` is the one entry
point whose input requires interpretation, so the interpretation is what gets reviewed. The
mechanism §12/§14 chose — preview the whole plan once, then accept/reject/feedback — is built and
works (`cli.py:591-660`, `cli.py:723-829`).

What is not built is a preview that is quick to check. The confirm step's value is bounded by how
much a person can verify in the seconds they spend reading it, and three things currently reduce
that:

- **The plan is not the most prominent thing on screen when the decision is made.**
  `render.print_trace` (`render.py:338-354`) prints a table of every tool call with its raw JSON
  result verbatim, unconditionally, immediately before `render.print_plan` (`cli.py:607` then
  `cli.py:615`; `cli.py:752` then `cli.py:758`). `_print_usage` (`llm.py:733-747`) prints a
  per-round token/timing line, also unconditionally — `docs/journal/2026-09-01-ask-agent-design.md`'s
  tail section records under Missing that "no flag or setting suppresses `render.print_trace`'s or
  `_print_usage`'s output". A three-round find-then-write request therefore puts three usage lines
  and a JSON-bearing table above the panel the person is being asked to approve.
- **The preview describes the write, not its effect.** `render.print_plan` (`render.py:356-384`)
  renders kind, amount, unit, product, endpoints, and `current_amount` as "already there". Its own
  docstring records why there is no "after" number: a naive current-minus-amount subtraction can
  disagree with `decide_change`'s shortfall reconciliation, and a wrong number is worse than none.
  That reasoning holds for a subtraction computed at render time; it does not hold for a projection
  computed by folding the writes `decide_change` itself returned, which `_propose_write` already
  computes and discards (`llm.py:1032` binds them to `_writes` and never reads them again).
- **Nothing in the preview distinguishes a write grounded in the vault from one the model
  invented.** `docs/journal/2026-09-04-basmati-rice-unit-mismatch.md` traces a real 0/10 eval
  failure to exactly that: `_maybe_force_action` and `_maybe_self_review` stack, and the plan a
  human would have been shown is `sumac_discover_inventory(product_id="Basmati Rice Bag",
  unit="bag")` — a product id absent from the vault and from every tool result in the same trace,
  carrying `decide._resolve_product`'s auto-registration warning as its only signal. That warning
  does reach the preview (`ProposedWrite.warnings` → `render.py:379-380`), as one yellow line among
  the others.

The interaction is also plainer than the rest of the CLI's rendering. Every decision is
`typer.prompt("Choice", default="a")` (`cli.py:617`, `cli.py:760`) against a printed key table
(`render.print_decision_options`, `render.py:386-397`) — type a letter, press Enter, and for a
compound plan the only granularity available is all-or-nothing plus `_prompt_edit`'s per-field
retype (`cli.py:520-589`).

## 2. What this plan does not change

- No new required dependency. The interactive selector is `termios`/`tty` from the standard library
  plus `rich`, which is already a hard dependency (`pyproject.toml:10`).
- No behavior change when stdin is not a TTY. `tests/test_cli.py` drives every `ask` test through
  `CliRunner(input=...)`, and `evals/` runs headless; both keep the existing typed-line prompt, on
  the same keys, reading the same way.
- No change to what is committed. `AgentRunner.commit` (`llm.py:1351`) re-decides against fresh
  state per `docs/journal/2026-09-01-ask-agent-design.md` §14; a projection shown in the preview is
  descriptive of the moment it was computed and is never replayed as a write.
- No LLM-based review pass. Every check below is deterministic and reads only `Config`, the current
  `Inventory`, and `AgentPlan.trace` — no extra round-trip, no extra latency, and reproducible
  regardless of the eval-reproducibility problem `docs/journal/2026-09-04-trace-and-verdict-redesign.md`
  records for anything that goes through the model.

## 3. The five changes

### 3.1 Arrow-key decision menu, with the typed prompt as the non-TTY path

New `src/sumac/prompt_ui.py` exposing `select(options, ...)` and `multiselect(items, ...)`.
`select` renders the same `(key, description)` pairs `_decision_options` (`cli.py:471-485`) already
builds, one per row with the current row highlighted, and reads single keypresses: up/down (and
`k`/`j`) move, Enter chooses, a row's own key character chooses it directly, `Ctrl-C`/`Esc` returns
the reject option. The free-text feedback row opens a line editor when chosen, since it is the one
option a keystroke cannot express.

`select` checks `sys.stdin.isatty()` and, when false, falls through to exactly the call site that
exists today: `render.print_decision_options(options)` then `typer.prompt("Choice", default="a")`.
The keys and their meanings are unchanged in both paths, so `"a"`/`"r"`/`"e"`/`"d"`/`"g"`/`"s"` and
free text keep working from a pipe, a test, and a script.

### 3.2 A plan preview that shows the effect, not just the write

`ledger.project(inventory, locations, objs)` — new, built from `_apply_sides`/`_commit`
(`ledger.py:288-332`) rather than reimplementing their arithmetic — folds a list of already-decided
record dicts onto a copy of an existing `Inventory` and returns the resulting `Inventory`.
`_propose_write` (`llm.py:997-1094`) passes the `decide_change` writes it currently discards
through it and records the resulting per-location before/after on the `ProposedWrite` as a new
`effects: tuple[LocationEffect, ...]` field. `current_amount` stays, unchanged, so existing tests
and callers are untouched.

The preview's numbers then come from `decide`, including the shortfall `Counted` correction
`docs/journal/2026-08-30_decide-pattern-data-integrity-upgrade.md` §3.5 emits — the case
`render.print_plan`'s docstring cites as the reason not to compute an "after" by subtraction.
`render.print_plan` renders each write as two lines — kind, amount and product, then the location
path and `before → after` per endpoint — with a single count line above the whole plan.

Two things this does not claim: the projection is per-write against the inventory each
`_propose_write` call loaded, so two writes in one plan touching the same product both project from
the same base (`docs/journal/2026-09-01-ask-agent-design.md` §15's first accepted limitation,
unchanged here); and commit re-decides, so a projection is a statement about the moment it was
computed, labeled as such.

### 3.3 Grounding badges: what in this plan is not in your vault

`sumac/review.py` — new, pure, no I/O — takes a plan and a `Config` and returns a per-write list of
findings:

- `new-product`: `product_id` is not in `cfg.known_products`.
- `new-unit`: the product is known and `cfg.can_convert(product_id, unit)` is false.
- `unknown-location`: an endpoint is not in `cfg.known_locations`.
- `ungrounded`: the `product_id` appears in no `ToolCallRecord.result` in `plan.trace` and is not a
  registered product, so neither the vault nor the model's own searches supplied it. This is the
  check that applies to `docs/journal/2026-09-04-basmati-rice-unit-mismatch.md`'s fabricated
  `"Basmati Rice Bag"`, using the same trace data that entry reconstructed the failure from after
  the fact.
- `near-match`: `decide.near_matches` (`decide.py:63-64`) finds a registered product within
  difflib's existing 0.6 cutoff of an unregistered `product_id` — the "did you mean" the
  auto-registration warning already carries, shown as a badge.

Findings render as a badge on the write's own row and a single count line above the plan
("2 changes · 1 creates a new product"), so the count is stated before the detail.

### 3.4 Per-write selection for compound plans

`prompt_ui.multiselect` over `plan.writes`, reachable from a new `p` ("Pick which changes to
apply") option present only when `len(plan.writes) > 1`. Space toggles, Enter applies the checked
subset by committing a `dataclasses.replace(plan, writes=<subset>)`. The non-TTY path does not
offer `p`, since there is no line-typed equivalent and the existing keys cover every case a script
can express.

### 3.5 Trace and usage demoted below the plan

`render.print_trace` gains a compact mode: one line per call, `tool(args) → <n> matches` for a
search and the decided effect for a write, with the raw JSON result kept for `--trace`. `ask` gains
`--trace` (full table, today's behavior) and `--stats` (the `_print_usage` lines), the latter
threaded into `AgentRunner` as `show_usage`, which defaults True, so `evals/` and the benchmark
scripts keep printing what they always did and only `sumac ask` changes. That closes the Missing
bullet
`docs/journal/2026-09-01-ask-agent-design.md`'s tail section carries.

## 4. A preview harness for the rendering

`scripts/preview-ask-ui.py` renders every one of the above against hand-built `AgentPlan`s —
a single consumption, a compound move, a plan carrying every badge in §3.3, a read-only reply — with
no `mistralrs` import, no vault, and no model call. `--svg <dir>` writes each scene through
`rich.console.Console.save_svg` for side-by-side comparison of a rendering change.

Rendering and interaction are the parts of `sumac ask` that can be iterated deterministically, and
`docs/journal/2026-09-04-trace-and-verdict-redesign.md` records why the model side cannot:
`mistralrs.Runner`'s RNG stream position depends on everything that ran earlier in the same session,
so two prompt-wording sessions are not cleanly comparable. A rendering change compared through this
harness has no such dependency, since the input is fixed data.

## 5. Build order

1. `prompt_ui.py` with its non-TTY fallback and its own tests (§3.1).
2. `ledger.project` + `ProposedWrite.effects` + `render.print_plan` (§3.2).
3. `review.py` + badges in the preview (§3.3).
4. `--trace`/`--stats` and compact trace rendering (§3.5).
5. `multiselect` and the `p` option (§3.4).
6. `scripts/preview-ask-ui.py` (§4).

Each step keeps `pytest` green (314 tests pass at the commit this plan is written against) and adds
tests for what it introduces.

## 6. Considered, not planned

- **An LLM review pass over the plan.** The user's framing offers this or human confirmation as
  alternatives. Every check in §3.3 is deterministic and costs no round-trip, and a model reviewing
  its own plan is the mechanism `docs/journal/2026-09-04-basmati-rice-unit-mismatch.md` steps 4-5
  trace the fabricated product to (`_maybe_self_review` firing on a jug-substituted plan produced
  the "Basmati Rice Bag" call). Adding a second such pass before the human is not planned here.
- **Committing a subset by re-deciding only the subset.** §3.4 commits through the existing
  `AgentRunner.commit`, which already re-decides each write independently; no new commit path.
- **A TUI framework.** `rich` is already a dependency, and `rich.live.Live` plus raw-mode key reads
  covers a single-select and a multi-select; `textual`/`prompt_toolkit` would add a dependency for
  two widgets.
- **Undo of a committed `ask`.** `sumac correct` (`cli.py:318`) supersedes one record by id; an
  "undo the last plan" wrapper over it is a separate change with its own design questions (what
  identifies a plan's records as one group — no `cmd_id` is shared across the writes of one plan
  today).

---

# 2026-09-04: `sumac ask` Confirmation UX — Plan

## Current State

- `docs/journal/2026-09-04-ask-confirmation-ux.md` (this file) plans five changes to `sumac ask`'s
  confirmation interface and one preview harness; no code accompanies the commit that added it.
- `src/sumac/cli.py`'s `ask` command takes `PROMPT`, `--loop`, `--dry-run`, `--debug`, and
  `--data-dir` (cli.py:424-438), and routes to `_ask_one` (cli.py:591) or `_ask_loop` (cli.py:662).
- `_ask_one` and `_ask_loop_request` each read a decision with `typer.prompt("Choice", default="a")`
  (cli.py:617, cli.py:760) after `render.print_decision_options` (render.py:386-397) prints the
  option table `_decision_options` (cli.py:471-485) builds.
- `render.print_trace` prints every `ToolCallRecord`'s name, arguments, and raw JSON result in one
  table (render.py:338-354), called before `render.print_plan` on every loop iteration in both
  decision loops (cli.py:607, cli.py:752).
- `_print_usage` prints one token/timing line per round with no flag gating it (llm.py:733-747).
- `render.print_plan` renders one `rich.panel.Panel` per `ProposedWrite` with kind, amount, unit,
  product, endpoints, `current_amount` as "already there", and each `warnings` entry, and states in
  its docstring that no computed "after" total is shown (render.py:356-384).
- `AgentRunner._propose_write` binds `decide.decide_change`'s returned writes to `_writes` and reads
  only the returned messages (llm.py:1032-1044), so the decided writes are discarded before
  `ProposedWrite` is constructed (llm.py:1056-1065).
- `pytest` reports 314 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- Nothing in this plan is implemented at the commit that added it — `src/sumac/prompt_ui.py`,
  `src/sumac/review.py`, `ledger.project`, `ProposedWrite.effects`, `ask --trace`, `ask --stats`,
  and `scripts/preview-ask-ui.py` do not exist there.
- No `sumac ask` interaction reads a single keypress; every prompt in the `ask` code path is a
  line-buffered `typer.prompt`.
- Nothing in the `ask` code path compares a `ProposedWrite`'s `product_id` against `AgentPlan.trace`
  or against `Config.known_products` for display — `decide._resolve_product`'s auto-registration
  warning (decide.py:191-195), carried in `ProposedWrite.warnings`, is the only signal in the
  preview that a product is unregistered.

## Divergence

- `README.md` documents `sumac ask` as parsing freeform text via a local LLM and does not describe
  the confirmation interface, so nothing in it diverges from what this plan changes.

---

# 2026-09-04: `sumac ask` Confirmation UX — Implementation State

## Current State

- `src/sumac/prompt_ui.py` (new) defines `Option`, `Choice`, `interactive()`, `read_key()`,
  `select()`, and `multiselect()`; `interactive()` returns False unless `termios`/`tty` imported
  and both `sys.stdin` and `sys.stdout` are ttys, and `select()` prints
  `render.print_decision_options` and reads `typer.prompt("Choice", default=...)` when it does.
- `prompt_ui.select` returns an option's `key`, the typed line for an option with
  `prompt_for_text=True`, and `"r"` for Escape or Ctrl-C — the same strings the typed path
  produces, so `cli.py`'s accept/reject/edit branches match on one set of values regardless of
  which path produced the answer.
- `prompt_ui.read_key` returns `"\x1b[A"`/`"\x1b[B"` whole for the arrow keys, distinguishing a
  bare Escape from an escape sequence by a zero-length `select.select` poll rather than a blocking
  read of the following bytes.
- `ledger._fold_into(state, records, locations)` (extracted from `_fold`) applies records onto a
  caller-supplied state dict; `ledger._fold` calls it with an empty dict and returns
  `(state, anomalies)` as before.
- `ledger.project(inventory, locations, objs)` parses already-serialized record dicts through
  `RecordSchema.model_validate(...).to_domain()` and `upcast.upcast`, then folds them onto a copy
  of `inventory.by_location` via `_fold_into`, returning a new `Inventory`.
- `llm.LocationEffect(location_id, product_id, unit, before, after)` and
  `llm.ProposedWrite.effects: tuple[LocationEffect, ...] = ()` record a write's per-location
  before/after; `llm._effects` computes them from `ledger.project` over the log-stream writes
  `decide.decide_change` returned, filtering out the config-stream writes an auto-registration adds.
- `ProposedWrite.current_amount` is unchanged and still populated — `render._effect_text` falls
  back to it when `effects` is empty, so a `ProposedWrite` built by hand renders as it did before.
- `src/sumac/review.py` (new) defines `Finding(code, label, detail, explain)`, `review_write`,
  `review_plan`, and `headline`; `review_plan` returns one findings tuple per write, positionally
  aligned with `plan.writes`.
- `review.review_write` emits `ungrounded` when a `product_id` is in neither `cfg.known_products`
  nor the concatenated results of the plan's `sumac_find_inventory` calls (`review.READ_TOOLS`),
  plus `new-product`, `near-match` (via `decide.near_matches`), `new-unit` (a known product whose
  unit `cfg.can_convert` rejects), and `unknown-location`.
- `review.READ_TOOLS` excludes the three write tools — `llm._propose_write`'s result JSON echoes
  the `product_id` it was called with, which a grounding check over every trace entry would count
  as a search result (`test_grounding_ignores_a_write_tool_echoing_its_own_argument`,
  tests/test_review.py).
- `Finding.explain` is True only for `ungrounded` and `unknown-location`; `render.print_plan`
  prints a detail line for those and shows every other finding as a badge only, alongside the
  `decide` warnings already carried on the write.
- `review.headline` returns `""` for a single write with no findings, and otherwise a count line
  ("2 changes · 1 creates a new product") counting writes rather than findings.
- `render.print_plan(plan, *, findings=(), locations=None, header="")` prints two lines per write —
  kind/amount/product with badges, then location path and `before → after` per endpoint — plus one
  indented line per explaining finding and per `decide` warning, via `render._indented`'s `Padding`
  so a wrapped line keeps its indent.
- `render.print_trace(trace, *, verbose=False)` prints one summary line per call by default —
  `render._trace_summary` reports product/location counts for a search result, a status or
  rejection reason for a write result, and the first 60 characters of anything it cannot parse —
  and the previous full table when `verbose`.
- `sumac ask` takes `--trace` and `--stats` alongside `--dry-run`/`--debug`, carried through
  `cli._AskView`; `--stats` reaches `AgentRunner(show_usage=...)` (llm.py), whose default stays
  True so `evals/` and the benchmark scripts print what they always did.
- `cli._build_agent` passes `show_usage=view.stats or view.debug`, so `--debug` prints the per-round
  usage lines without `--stats` alongside it (`test_ask_debug_implies_stats`, tests/test_cli.py).
- `_print_usage` (llm.py) is unchanged by this branch — `--stats` restores its lines verbatim,
  round label, token counts, `tok/s`, `total_time_sec` and round preview included; `--trace`
  restores `print_trace`'s previous full table verbatim. `sumac ask --stats --trace` reproduces
  exactly what printed unconditionally before.
- `cli._decision_options` returns `list[prompt_ui.Option]` and takes `pick`, which `cli._decide_prompt`
  passes only when a plan has more than one write and `prompt_ui.interactive()` is True.
- `cli._pick_writes` runs `prompt_ui.multiselect` over a plan's writes and returns a
  `dataclasses.replace`d plan; the narrowed plan re-enters the same preview and decision prompt
  rather than committing from the checklist.
- `cli._build_agent` constructs every `AgentRunner` in the `ask` code path — four call sites across
  the two decision loops previously repeated the constructor call.
- `scripts/preview-ask-ui.py` (new) renders seven scenes (single, compound, fabricated, read-only,
  trace, typed, checklist) against hand-built `AgentPlan`s with no vault and no `mistralrs` import;
  `--svg <dir>` writes each through `rich.console.Console.save_svg`.
- `tests/test_prompt_ui.py` (15 tests) drives `select`/`multiselect` with `interactive` and
  `read_key` monkeypatched to a scripted keypress list, and asserts the non-TTY path calls
  `typer.prompt`.
- `tests/test_review.py` (10 tests) covers each finding code, the write-tool echo exclusion, which
  findings explain themselves, and `headline`'s two shapes.
- `tests/test_ledger.py` gains three `project` tests, including the shortfall case where folding
  gives zero where subtraction gives -2; `tests/test_llm.py` gains four `effects` tests;
  `tests/test_cli.py` gains five (`before → after` in the preview, the `[unverified]` badge, the
  compact trace, `--trace`'s table, `--stats` reaching the agent).
- `tests/test_preview_script.py` imports `scripts/preview-ask-ui.py` by path and runs every scene
  against a recording console, restoring `render.console`/`prompt_ui.console` afterwards.
- `README.md`'s `sumac ask` section documents the preview, the four badges, the two input paths,
  `p`, `--trace`/`--stats`, and `scripts/preview-ask-ui.py`.
- `pytest` reports 354 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

---

# 2026-09-05: An Amount Field That Takes Numbers

## Current State

- `e`'s amount field was `typer.prompt`, which accepted "three", showed it on the menu as the
  amount, and reported it invalid only at Done — where `_apply_edit` returned the plan unchanged,
  discarding every other field edited in the same pass.
- `prompt_ui.number(current, *, title, step, minimum)` returns the accepted value or `None`. A key
  that is not a digit, a decimal point, or a control it knows does nothing at all, and Enter is
  inert until what has been typed parses, so nothing invalid leaves the widget
  (`test_letters_do_nothing_in_the_amount_field` and
  `test_enter_is_inert_until_what_is_typed_parses`, tests/test_prompt_ui_pty.py).
- Up and Down step by `step` (default 1), covering the common case of a quantity being wrong by
  one, clamped at `minimum` (default 0), since `decide` rejects a non-positive amount and stepping
  into negatives would produce a rejection two screens later. Stepping preserves a fraction: 0.5 up
  one is 1.5.
- `prompt_ui._plain` formats with `format(value, "f")`, not `str()`, which uses scientific notation
  for some values a repeated decrement produces.
- `cli._apply_edit` returns `AgentPlan | None`, `None` meaning `decide_change` rejected the edit;
  `_prompt_edit`'s interactive path loops, re-entering `_edit_fields_by_menu` with the values
  already typed (its new `resume` argument), so correcting a rejection takes one field rather than
  five.
- `_prompt_edit`'s non-TTY path stays a single pass: a piped answer cannot react to a rejection, so
  re-prompting would consume the next scripted line as a field value or block on empty input
  (`test_a_rejected_edit_off_a_terminal_still_leaves_the_plan_alone`, tests/test_cli.py, feeds the
  same input the original invalid-amount test always did).
- `scripts/preview-ask-ui.py` gains an `amount` scene, drawn valid, fractional, and mid-typing.
- `pytest` reports 426 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- `step` is fixed at 1 wherever `cli` calls `number`: no smaller step for a fractional amount, and
  no larger one for a count in the dozens.
- The amount field has no unit context on screen: it says `Amount?` alone, while the menu row
  behind it shows the unit.
- The non-TTY walkthrough still takes any string for the amount and reports it invalid afterwards —
  unchanged, since a piped caller has no widget to type into.

---

# 2026-09-05: An Edited Plan Carrying the Model's Reply and Trace

## Current State

- A real run edited a plan's product and location and received a preview carrying the model's
  original reply — "A packet of Ham has been added to the top shelf of the fridge." — under a write
  that by then read `Billy Bear Ham` at `Fridge > Door Shelves`.
- `cli._apply_edit` replaced `AgentPlan.writes` and left `reply_text` and `trace` unchanged: both
  describe what the model proposed, and after a hand edit neither describes the plan on screen. Both
  are now set to `""`/`()`; both remain in the transcript above, where they describe the state at the
  time they were printed. `_pick_writes` does the same, for the same reason: the reply describes
  every change proposed, not the subset kept.
- The same run showed the edited write with a location and no before/after: `_apply_edit` dropped
  `effects` rather than recomputing them, and `render._effect_text`'s `current_amount` fallback was
  empty for that write. `_apply_edit` now recomputes them from the records its own `decide_change`
  call returned, through `llm.effects` (renamed from `_effects` for this caller).
- `_apply_edit` also resolves the edited endpoints through `decide.resolve_location`, so a
  hand-entered display path records as an id — the same reason `_propose_write` does.
- `ProposedWrite.edited_fields: frozenset[str]` names the fields a person set by hand; `_apply_edit`
  computes it by comparing the edited write against the original and unions it with what earlier
  edits had already set.
- `review.review_write` skips `ungrounded` when `"product_id"` is in `edited_fields`: the check
  reports a name the model produced without a source, and a name typed into the edit menu has one.
  `new-product` still applies, as do the unit and location checks, and editing any other field
  leaves the grounding check in force
  (`test_editing_another_field_leaves_the_grounding_check_alone`, tests/test_review.py).
- `pytest` reports 416 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- Nothing replaces the cleared `reply_text`: an edited plan has no sentence describing it, only its
  rows. A generated summary of the edited write would not be the model's reply, and is not built.
- `edited_fields` is not displayed: the preview does not mark which values a person set, so an
  edited write and a proposed one look alike apart from the badges that no longer apply.
- `_pick_writes` clears `reply_text` for the whole plan even when the kept subset is the one the
  reply described.

---

# 2026-09-04: Picking a Unit or a Product, or Typing a New One

## Current State

- `prompt_ui.pick` takes `allow_new` and `new_hint`; `_visible_rows` appends a row carrying the
  filter text itself when `allow_new` is set, the filter is non-empty, and no existing row has that
  exact value.
- The added row goes last, and the cursor resets to the top of the matches on every keystroke, so
  typing a value that already exists selects the existing row with one Enter, and a partial match
  ("jarful" against a "jar" row) needs one arrow-down to reach the new row
  (`test_a_partial_match_still_reaches_the_new_row`, tests/test_prompt_ui_pty.py).
- With nothing matching, the added row is the only row and is already under the cursor, so typing a
  new unit and pressing Enter selects it directly; `pick` with an empty list and `allow_new` still
  accepts a value, which a vault with nothing recorded yet requires.
- `pick` without `allow_new` adds no row, which keeps the location picker to the closed set
  `decide.resolve_location` accepts
  (`test_without_allow_new_a_typed_value_is_not_offered`, tests/test_prompt_ui_pty.py).
- `cli._unit_rows(observed, cfg, product_id)` lists every unit the vault has used
  (`ledger.observed_product_units`) plus every registered canonical unit and conversion key, ordered
  with the units already used for `product_id` first and the rest by total frequency: units are
  reused far more often than new ones are introduced, so the few a household uses should not need
  typing. Each row notes `already used for <product>` or `<n> uses`.
- `cli._product_rows(cfg)` lists every unretired registered product with the unit it is tracked in,
  ordered by id; the display name is in `search`, so filtering finds a product by either.
- `cli._edit_fields_by_menu` routes `product_id` and `unit` through `_choose_from_rows`
  (`allow_new=True`), the two location fields through `_choose_location` (`allow_new` unset), and
  `amount` through `typer.prompt` — a number is not a value to choose from a list.
- The unit rows are built from `values["product_id"]` when the unit row is chosen, so editing the
  product first and the unit second offers the new product's units.
- `_edit_fields_by_walkthrough` is unchanged: off a terminal every field is still typed.
- `scripts/preview-ask-ui.py` gains a `value-pickers` scene showing the unit list, a typed new unit,
  and a product filter that offers both a match and a new value.
- `pytest` reports 410 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- `_edit_fields_by_menu` calls `config.build_config` (and, for units,
  `ledger.observed_product_units`, which folds every record in the log) each time a field is chosen,
  rather than once per edit. Unmeasured against a real vault's log size.
- A picked product does not adjust the unit: choosing "Strawberry Jam" leaves a `packet` unit in
  place, which `decide`'s unconvertible-unit warning reports at re-validation rather than the picker
  offering that product's own unit.
- `sumac add` still takes typed values for every field; the pickers exist only inside `sumac ask`'s
  edit menu.

---

# 2026-09-04: Editing a Location by Picking It

## Current State

- `e`'s location fields took free text, so correcting a wrongly-chosen location meant typing an id
  from memory, which is how the wrong location arose. It is also the one input in the edit menu that
  can be wrong in a way `decide` rejects outright.
- Locations are a closed set: `decide.resolve_location` rejects one that is not configured, and
  unlike a product there is no auto-registration. Products keep free text for that reason;
  locations do not require it.
- `prompt_ui.Row(value, label, search)` and `prompt_ui.pick(rows, *, title, current)` present a long
  list: arrows move, printable keys filter (`search` is matched, falling back to `label`), backspace
  widens the filter, Enter chooses, Escape returns `None`. Distinct from `select`, which handles a
  handful of fixed options each carrying its own accelerator key; in `pick` every printable key is
  filter text.
- `pick` shows `_PICK_HEIGHT` (12) rows at a time around the cursor, with `↑ n more` / `↓ n more`
  counts and a `[shown of total]` header, since a household's layout is longer than a terminal is
  tall.
- `pick` does not return a value when nothing matches the filter and Enter is pressed, rather than
  choosing a row that is not displayed (`test_enter_on_an_empty_filter_result_does_nothing`,
  tests/test_prompt_ui_pty.py, backspaces afterwards and chooses from the restored list).
- `cli._location_rows` builds the rows from `config.location_path`, ordered by path, retired
  locations excluded: the order `sumac config show --locations-only` uses, so a container and
  everything nested under it stay together. `search` carries both the path and the id.
- `cli._choose_location` is called from `_edit_fields_by_menu` for `from_location`/`to_location`,
  opening on the location the write already names (`current`), and leaves the field unchanged when
  cancelled. Every other field still opens `typer.prompt`.
- `_edit_fields_by_walkthrough` is unchanged, so the non-TTY path still types a location id.
- `scripts/preview-ask-ui.py` gains a `location-picker` scene, drawn unfiltered and filtered.
- `pytest` reports 400 passing tests repo-wide, nine of them driving `pick` over a real pty;
  `ruff format --check .`, `ruff check .`, and `ty check` each report no findings.

## Missing

- The product field is still free text. A vault of ~469 products is a longer list to scroll than
  the locations, and a product may legitimately be new — `decide` auto-registers one — so a picker
  there needs a "type a new name" row that the location picker does not.
- Nothing offers the picker outside `e`: `sumac add --to` still takes a typed id or display path.
- `pick`'s filter is a case-insensitive substring over path and id, with no fuzzy or out-of-order
  matching, so "fridge door" finds `Fridge > Door` and "door fridge" does not.
- The layout is what made "the top shelf of the fridge" ambiguous in the run that prompted this: a
  shelf above the fridge and the top shelf inside it are different places with similar names.
  Nothing here distinguishes them; the picker only makes the choice visible.

---

# 2026-09-04: No Way to Look Up a Location

## Current State

- Two real runs of `sumac ask "add a packet of ham to the top shelf of the fridge"` show the same
  sequence: `sumac_discover_inventory(to_location="top shelf of the fridge")` rejected
  `unknown_location`, then `sumac_find_inventory(query="fridge")` seventeen times, each returning
  zero products, until the round cap. One run then wrote to `fridge-main-shelf-3-bottle-rack`, a
  location id that appears in the same trace's earlier `sumac_find_inventory(query="milk")` result
  and in no other result; the other stopped and asked the person for a valid identifier.
- `ledger.search_inventory` matches products only, so `query="fridge"` returning nothing was
  correct: no product has that name. `sumac_find_inventory` was the only search tool, which left no
  route from a place named in words to a location id except guessing one and being rejected.
- `decide._resolve_location`'s `Rejected` carries `suggestions=near_matches(value,
  active_locations)`; difflib's 0.6 cutoff scores a phrase like "top shelf of the fridge" against no
  id at all, so the rejection reaching the model was empty of candidates
  (`test_an_unknown_location_rejection_names_real_candidates`, tests/test_llm.py, asserts
  `suggestions == "[]"` for exactly that input).
- `config.search_locations(locations, query)` returns every active location whose id, name, or
  display path contains `query`, ordered by path. Matching the path lets a query for a container
  return the locations nested inside it: "Shelf 1" names nothing about a fridge, but its path
  does.
- `AgentRunner._sumac_find_inventory` returns `{"products": [...], "locations": [...],
  "location_match_count": n}`; `locations` carries `location_id` and `location_path`, capped at
  `_MAX_LOCATION_MATCHES` (20) with the full count alongside. `_FIND_INVENTORY_SCHEMA`'s description
  states that searching a place is how a phrase becomes a `location_id`.
- `AgentRunner._propose_write` adds `known_locations` to an `unknown_location` rejection's detail:
  the locations whose id or path shares a word with the rejected value, or the whole layout capped
  at 20 when none does, so the rejection always names some valid locations.
- `AgentRunner._searched` records each search result within one `propose`/`revise` call; an
  identical repeat returns the same payload with `repeated_query: true` and a hint stating that,
  mirroring `_propose_write`'s existing `already_proposed`. Cleared alongside `_trace` at the top of
  `propose` and `revise`.
- `pytest` reports 388 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- No real-model run confirms any of this changes the outcome: the six new tests drive the tool
  callbacks directly, and the failure they were written from is a model's response to what the tools
  return, which only a real run can show.
- `evals/` has no scenario for a request naming a location in plain words rather than by id, so the
  suite would not have caught this and does not yet measure the fix.
- `sumac find` (`cli.py`) still searches products only; `config.search_locations` is called from
  `llm.py` alone. `sumac config show`'s tree is the closest existing equivalent.
- Nothing bounds how many times a different search may run: the repeat guard catches only an
  identical query, and `MAX_TOOL_ROUNDS` remains the only cap on a model varying its wording each
  time.

---

# 2026-09-04: A Valid Location Reported as a New One

## Current State

- A real run proposed `sumac_discover_inventory(to_location="Fridge Top Shelf")`, which the preview
  rendered with a `[new location]` badge, a `'Fridge Top Shelf' is not a configured location`
  warning, and an effect of `— → —`: no before, no after.
- `decide.resolve_location` accepts a location's display path as well as its id (an established
  behaviour: a path pasted into `--to`), so that write resolved, passed the gate, and would have
  committed to the correct location. The same run rejected `to_location="fridge door"` with
  `unknown_location`, which is what an invalid location produces.
- Cause: `AgentRunner._propose_write` recorded the model's raw endpoint strings on its
  `ProposedWrite` while `decide_change` resolved them to ids internally, so every downstream lookup
  keyed on a string that is not a location id — `review.review_write` against `cfg.known_locations`,
  `llm._effects` against `Inventory.at`, and `render._where_text` against `config.location_path`.
  Reproduced against a two-location vault before the fix.
- `decide._resolve_location` is now public as `decide.resolve_location`; `_propose_write` calls it
  on both endpoints after `decide_change` returns — where it cannot raise, both having resolved
  once already — and records the ids.
- The duplicate-call guard (`candidate in self._pending`) now sees through a second call naming the
  same location by its other name
  (`test_the_same_write_named_two_ways_is_only_proposed_once`, tests/test_llm.py).
- `llm._effects` returns `()` when `ledger.project` reports anomalies, rather than reading
  before/after from a fold that did not apply the records; `render.print_plan` then falls back to
  `current_amount`'s "already there".
- `evals/evaluators.py`'s `_canon_location` existed to resolve those raw strings for scoring; its
  first branch now answers every write, and it is kept to handle a regression rather than
  removed.
- `pytest` reports 382 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- `ProposedWrite.product_id` still holds the model's raw string. Unlike a location, `decide` does
  not resolve a product to a different id: an unknown one auto-registers under the name given, so
  there is nothing to resolve to, and `review`'s `new-product`/`near-match` findings report that
  case.
- `ProposedWrite.amount`/`unit` hold what was requested, not `_resolve_product`'s canonical
  `Quantity`, so a write in a convertible alt-unit previews in the unit asked for while the log
  records the canonical one. `LocationEffect.unit` takes its unit from the projection rather than
  the write, so the before/after line is already in the stored unit.
- Nothing checks a proposed product name against the search results the agent received beyond
  substring presence: "Ham Packet" alongside a `packet` unit is flagged `[unverified]`, not
  identified as a unit duplicated into a product name.

---

# 2026-09-04: One Model Load Per Session, Not Per Request

## Current State

- `sumac ask --loop` built a fresh `mistralrs.Runner` for every request: `_ask_loop_request`
  constructs an `AgentRunner` per request (deliberately — each request is its own conversation with
  no memory of the last), and `AgentRunner.__init__` called `_build_runner` whenever no `runner` was
  passed, so each request also reloaded the GGUF.
- `llm.shared_runner(model, seed=None)` returns the most recently built backend when its
  `(model.name, seed)` matches, and builds one otherwise; `AgentRunner.__init__` calls it instead of
  `_build_runner` when no `runner` is injected.
- `llm._SHARED_RUNNER` holds exactly one backend and is cleared before the next is built: two GGUFs
  resident at once can exhaust a GPU that fits either alone when switching models mid-session.
  Clearing drops this module's reference only; a caller still holding the previous `AgentRunner`
  keeps that backend alive until it releases it
  (`test_only_one_backend_is_held_at_a_time`, tests/test_llm.py).
- `llm.release_shared_runner()` drops the cached backend; nothing in `sumac ask` calls it, since a
  session ends with the process.
- An injected `runner` neither consults nor populates the cache, so `evals/conftest.py` — which has
  built exactly one `base_runner` per run and passed it to every scenario's `AgentRunner` since the
  eval suite existed — is unaffected
  (`test_an_injected_backend_never_builds_or_caches`, tests/test_llm.py).
- Reuse carries mistral.rs's RNG stream position and prefix cache across requests in one session, so
  a request's sampling depends on what ran before it: the order-dependence
  `docs/journal/2026-09-04-trace-and-verdict-redesign.md` records for the eval suite, which is why
  that suite pins a seed and this does not.
- `pytest` reports 379 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- No measurement of the saved latency is recorded: the five tests count `_build_runner` calls
  against a monkeypatched builder, and nothing in the suite loads a real model.
- Nothing preloads the model before the first `--loop` prompt, so the first request of a session
  still pays the load with the person waiting at the prompt.
- `sumac ask` without `--loop` handles one request per process, so the cache never serves a second
  caller there.

---

# 2026-09-04: mistral.rs's Own Load Logs Quieted

## Current State

- Every `sumac ask` invocation printed mistral.rs's load logs above its own output: the DType, the
  GGUF tokenizer summary, the device map, the version and git revision, the PTX preload, the
  modalities, the prefix-caching notice, and the discovered GGUF chat template in full, which is
  several hundred lines of Jinja printed on one line.
- mistral.rs logs through Rust's `tracing` with a filter built from `RUST_LOG` — confirmed against
  the built extension (`.venv/.../mistralrs.abi3.so` carries the `RUST_LOG` string and
  `tracing_subscriber::filter::env::builder::Builder` symbols from `mistralrs_core`), not assumed
  from its documentation.
- Every line in that load output is `INFO`, so `cli.QUIET_RUST_LOG` is `"warn"`: the level that
  suppresses all of it and still reports warnings and errors. `cli.VERBOSE_RUST_LOG` is `"info"`,
  reproducing the previous output exactly.
- `cli._set_rust_log(verbose)` sets `RUST_LOG` and is called from `_import_llm`, immediately before
  `from sumac import llm` — the filter is built once, when the Rust side installs its subscriber, so
  a value set after the extension is first imported has no effect.
- `_set_rust_log` never overrides a `RUST_LOG` already in the environment
  (`test_an_existing_rust_log_is_never_overridden`, tests/test_cli.py), so a per-target filter finer
  than either level stays available.
- `sumac ask --debug` passes `verbose=True`, so the flag that shows the raw per-round dumps also
  restores mistral.rs's own logs; `sumac models pull` passes `verbose=True` unconditionally, since
  mistral.rs's progress is the only sign a multi-gigabyte download is proceeding.
- `pytest` reports 374 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- No test observes mistral.rs printing less: the three tests assert what `_set_rust_log` puts in
  the environment, and nothing in the suite loads a real model.
- `evals/` and the benchmark scripts import `sumac.llm` directly rather than through
  `cli._import_llm`, so they are unaffected and still print mistral.rs's load logs.

---

# 2026-09-04: `e` Converted to the Same Menu as Every Other Decision

## Current State

- `e` (Edit) reached `_prompt_edit`'s original numbered `typer.prompt("Edit which one? (number)")`
  on a terminal, unchanged by the keypress-menu work: `p`'s checklist and the decision prompt were
  the only two interactions converted, and a real run reached the numbered one.
- `cli._prompt_edit` now reads which write to edit through `cli._choose_write_to_edit` — one write
  skips the question, more than one gets `prompt_ui.select` on a terminal and the numbered list plus
  a typed index otherwise — and which fields to change through `cli._edit_fields_by_menu`
  (interactive) or `cli._edit_fields_by_walkthrough` (everything else).
- `_edit_fields_by_menu` shows one row per editable field carrying its current value, changes only
  the field chosen, and redraws until "Done", so correcting one mistyped location no longer requires
  an Enter through each of the four fields that were already correct.
- `_edit_fields_by_walkthrough` prompts product_id, unit, amount, then whichever endpoints the write
  has, each defaulting to its current value — the same sequence and the same prompts `e` has always
  read, so `tests/test_cli.py`'s two piped edit tests pass unchanged.
- `cli._editable_fields` drops the `from_location`/`to_location` row for a write that has no such
  endpoint: `decide_change` rejects a purchase carrying a `from_location`, so a row offering to fill
  one in would produce a rejection.
- `_edit_fields_by_menu` seeds its value dict from all five fields, not just the editable ones, so
  an endpoint the write does not have reaches `_apply_edit` as `None` rather than missing.
- `cli._apply_edit` validates through `decide.decide_change` and returns the plan unchanged on
  `Rejected` or an unparseable amount, as `_prompt_edit` always did, and drops the edited write's
  `effects`: the projection described the write the model proposed rather than this one, and
  `render.print_plan` falls back to `current_amount` for a write without one.
- `prompt_ui.select` answering `"r"` for Escape means "cancel this edit" inside the field menu:
  `_edit_fields_by_menu` returns `None`, `_prompt_edit` returns the plan unchanged, and the decision
  prompt asks again (`test_ask_edit_menu_escape_cancels_the_edit_not_the_plan`, tests/test_cli.py).
- `render.write_summary(write, locations)` is the single label for a write in a menu row, used by
  the edit picker, `cli._pick_writes`' checklist, and the preview harness. It replaces
  `_prompt_edit`'s `(from None to fridge-door)` field dump and `_pick_writes`' own phrasing, which
  omitted the location.
- `scripts/preview-ask-ui.py` gains an `edit` scene drawing both menus.
- `pytest` reports 371 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- The interactive edit path is tested with `prompt_ui.select` monkeypatched
  (`tests/test_cli.py`'s `_patch_menu`), not over a pty. `tests/test_prompt_ui_pty.py` covers
  `select` itself, so what is untested is `_prompt_edit`'s wiring against real keypresses.
- No edit menu row shows a location's display path; `from`/`to` show the raw id, which is the value
  to be retyped.

---

# 2026-09-04: Arrow Keys Read as Escape — Fixed

## Current State

- The first real-terminal run of `prompt_ui.select` exited on the Down arrow: `read_key` returned a
  bare `"\x1b"`, which `select` answers as `"r"` (reject), ending the request with nothing
  written.
- Cause: `sys.stdin` is a buffered `TextIOWrapper`, and `sys.stdin.read(1)` on a terminal decodes
  every byte already available into the wrapper's own buffer before returning the first. A Down
  arrow's `\x1b[B` arrives in one burst, so after `"\x1b"` was returned the `"[B"` remained in
  Python's buffer, while `select.select`, which polls the file descriptor, reported nothing pending.
  Reproduced against a `pty.openpty()` pair before the fix was written.
- `prompt_ui.read_key` reads the file descriptor with `os.read(fd, 1)` instead, extending the read
  only for an escape sequence (polling `_ESCAPE_TIMEOUT` per byte while the accumulated bytes are
  still a prefix in `_PARTIAL_ESCAPES`) or a multi-byte UTF-8 character
  (`_utf8_continuation_bytes`).
- `read_key` reads one byte rather than a chunk: a chunked read returns two keypresses already in
  the tty buffer as one merged string that matches no option, dropping both
  (`test_two_keypresses_arriving_together_are_read_separately`, tests/test_prompt_ui_pty.py).
- `prompt_ui.UP` and `prompt_ui.DOWN` are tuples carrying both cursor-key encodings: `ESC [ A`/`B`,
  and the application-cursor-mode `ESC O A`/`B` a terminal in DECCKM (tmux among others) sends.
- `prompt_ui.raw_mode` is a context manager held across a whole `select`/`multiselect` loop rather
  than re-entered per keypress: restoring canonical mode between reads allowed a keystroke arriving
  during a redraw to be echoed and line-buffered by the tty.
- `raw_mode` calls `tty.setcbreak(fd, termios.TCSAFLUSH)`, not `tty.setraw`: `setraw` also clears
  `OPOST`, which maps `\n` to `\r\n` on output, so every line `rich.live.Live` redrew inside it
  would be indented one column further than the last. `TCSAFLUSH` discards input queued before the
  mode change, so a keystroke typed during the seconds of model inference before a plan appeared is
  not counted as a decision about it.
- `prompt_ui.interactive` probes `termios.tcgetattr(sys.stdin.fileno())` alongside the two
  `isatty()` checks, so a stdin that claims to be a terminal but has no readable attributes falls
  back to the typed prompt instead of raising inside `raw_mode` mid-decision.
- `tests/test_prompt_ui_pty.py` (12 tests) drives `read_key`, `select`, and `multiselect` over a
  `pty.openpty()` pair with `os.fdopen(slave, "r")` as `sys.stdin`, the same buffered wrapper the
  bug occurred in. Four of them fail against the previous `read_key`, verified by reinstating it.
- `tests/test_prompt_ui.py`'s scripted-keypress tests stub `raw_mode` to `contextlib.nullcontext`
  alongside `interactive`/`read_key`: pytest's captured stdin has no terminal attributes to set,
  and those tests are about what a keypress means, not how it is read.
- `pytest` reports 367 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- No test covers `select`'s `Live` redraw: the pty tests assert what `select` returns, not what it
  drew.
- `multiselect` still has no non-TTY path (unchanged from the entry above).

## Missing

- No real-model run exercises any of this — every test drives a scripted `SendsCompletions` or a
  `_FakeAgentRunner`, and no `sumac ask` invocation against a real GGUF is recorded in this entry.
- No test drives `prompt_ui` against a real terminal: `interactive()` is monkeypatched to True and
  `read_key` replaced, so `termios.tcgetattr`/`tty.setraw` and the `rich.live.Live` redraw are
  exercised by no test. The arrow-key failure that gap allowed, and the pty tests added for it, are
  recorded in the entry below.
- `prompt_ui.multiselect` has no non-TTY path — `cli._decide_prompt` omits the `p` option entirely
  when `interactive()` is False, so a pipe and a script cannot apply a subset of a compound plan.
- `llm._effects` projects each write against the inventory that write's own `_propose_write` call
  loaded, so two writes in one plan touching the same product both project from the same base —
  `docs/journal/2026-09-01-ask-agent-design.md` §15's first accepted limitation, unchanged.
- `review`'s `ungrounded` check matches a `product_id` as a case-insensitive substring of the
  concatenated search results, so a product id that is a substring of an unrelated one a search
  returned counts as grounded.
- Nothing undoes a committed `ask` as a unit — `sumac correct` (cli.py) supersedes one record by
  id, and the writes of one plan share no identifier that would group them.

## Divergence

- None found. `README.md`'s new `sumac ask` review section was written against the flags
  `cli.py`'s `ask` signature declares and the badge labels `review.py` emits.
