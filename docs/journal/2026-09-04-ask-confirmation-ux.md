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

What is not built is the reviewing itself being easy. The confirm step's value is bounded by how
much a person can actually check in the seconds they spend looking at it, and three things
currently work against that:

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
  unit="bag")` — a product id that exists nowhere in the vault and appears in no tool result in the
  same trace, carrying `decide._resolve_product`'s auto-registration warning as its only signal.
  That warning does reach the preview (`ProposedWrite.warnings` → `render.py:379-380`) as one
  yellow line among the others.

The interaction mechanics are also plainer than the rest of the CLI's rendering. Every decision is
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
the reject option. The free-text feedback row opens a line editor when chosen, so free-text
feedback stays reachable — it is the one option that cannot be a keystroke.

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

This makes the preview's numbers `decide`'s own, including the shortfall `Counted` correction
`docs/journal/2026-08-30_decide-pattern-data-integrity-upgrade.md` §3.5 emits — the exact case
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
  registered product — the model produced an identity that neither the vault nor its own searches
  supplied. This is the check that fires on `docs/journal/2026-09-04-basmati-rice-unit-mismatch.md`'s
  fabricated `"Basmati Rice Bag"`, on the same trace data that entry reconstructed the failure from
  after the fact.
- `near-match`: `decide.near_matches` (`decide.py:63-64`) finds a registered product within
  difflib's existing 0.6 cutoff of an unregistered `product_id` — the "did you mean" the
  auto-registration warning already carries, hoisted to the badge line.

Findings render as a badge on the write's own row and a single count line above the plan
("2 changes · 1 creates a new product"), so the decision's headline states the risk before the
detail does.

### 3.4 Per-write selection for compound plans

`prompt_ui.multiselect` over `plan.writes`, reachable from a new `p` ("Pick which changes to
apply") option present only when `len(plan.writes) > 1`. Space toggles, Enter applies the checked
subset by committing a `dataclasses.replace(plan, writes=<subset>)`. The non-TTY path does not
offer `p` at all, since there is no meaningful line-typed equivalent and the existing keys already
cover every case a script can express.

### 3.5 Trace and usage demoted below the plan

`render.print_trace` gains a compact mode — one line per call, `tool(args) → <n> matches` for a
search and the decided effect for a write, with the raw JSON result kept for `--trace`. `ask` gains
`--trace` (full table, today's behavior) and `--stats` (the `_print_usage` lines), the latter
threaded into `AgentRunner` as `show_usage` — which defaults True, so `evals/` and the benchmark
scripts keep printing what they always did and only `sumac ask` changes. That closes the Missing
bullet
`docs/journal/2026-09-01-ask-agent-design.md`'s tail section carries.

## 4. A preview harness, because the model is not the thing being iterated on

`scripts/preview-ask-ui.py` renders every one of the above against hand-built `AgentPlan`s —
a single consumption, a compound move, a plan carrying every badge in §3.3, a read-only reply — with
no `mistralrs` import, no vault, and no model call. `--svg <dir>` writes each scene through
`rich.console.Console.save_svg` for side-by-side comparison of a rendering change.

This exists because rendering and interaction are the parts of `sumac ask` that can be iterated
deterministically, and `docs/journal/2026-09-04-trace-and-verdict-redesign.md` records why the
model-side cannot: `mistralrs.Runner`'s RNG stream position depends on everything that ran earlier
in the same session, so two prompt-wording sessions are not cleanly comparable. A rendering change
compared through this harness has none of that exposure — the input is fixed data.

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
  alternatives. Every check in §3.3 is deterministic and costs no round-trip; a model asked to
  review its own plan is the mechanism `docs/journal/2026-09-04-basmati-rice-unit-mismatch.md`
  step 4-5 traces the fabricated product *to* (`_maybe_self_review` firing on a jug-substituted
  plan produced the "Basmati Rice Bag" call). Adding a second such pass in front of the human is
  not planned here.
- **Committing a subset by re-deciding only the subset.** §3.4 commits through the existing
  `AgentRunner.commit`, which already re-decides each write independently; no new commit path.
- **A TUI framework.** `rich` is already a dependency and `rich.live.Live` plus raw-mode key reads
  covers a single-select and a multi-select; `textual`/`prompt_toolkit` would be a new dependency
  for two widgets.
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
  lands on zero and subtraction would give -2; `tests/test_llm.py` gains four `effects` tests;
  `tests/test_cli.py` gains five (`before → after` in the preview, the `[unverified]` badge, the
  compact trace, `--trace`'s table, `--stats` reaching the agent).
- `tests/test_preview_script.py` imports `scripts/preview-ask-ui.py` by path and runs every scene
  against a recording console, restoring `render.console`/`prompt_ui.console` afterwards.
- `README.md`'s `sumac ask` section documents the preview, the four badges, the two input paths,
  `p`, `--trace`/`--stats`, and `scripts/preview-ask-ui.py`.
- `pytest` reports 354 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

---

# 2026-09-04: `e` Given the Same Menu as Every Other Decision

## Current State

- `e` (Edit) reached `_prompt_edit`'s original numbered `typer.prompt("Edit which one? (number)")`
  on a terminal, unchanged by the keypress-menu work — `p`'s checklist and the decision prompt were
  the only two interactions converted, and a real run landed on the numbered one.
- `cli._prompt_edit` now reads which write to edit through `cli._choose_write_to_edit` — one write
  skips the question, more than one gets `prompt_ui.select` on a terminal and the numbered list plus
  a typed index otherwise — and which fields to change through `cli._edit_fields_by_menu`
  (interactive) or `cli._edit_fields_by_walkthrough` (everything else).
- `_edit_fields_by_menu` shows one row per editable field carrying its current value, retypes only
  the field chosen, and redraws until "Done" — correcting one mistyped location no longer costs an
  Enter through each of the four fields that were already right.
- `_edit_fields_by_walkthrough` prompts product_id, unit, amount, then whichever endpoints the write
  has, each defaulting to its current value — the same sequence and the same prompts `e` has always
  read, so `tests/test_cli.py`'s two piped edit tests pass unchanged.
- `cli._editable_fields` drops the `from_location`/`to_location` row for a write that has no such
  endpoint — `decide_change` rejects a purchase carrying a `from_location`, so a row offering to
  fill one in could only ever produce a rejection.
- `_edit_fields_by_menu` seeds its value dict from all five fields, not just the editable ones, so
  an endpoint the write does not have reaches `_apply_edit` as `None` rather than missing.
- `cli._apply_edit` validates through `decide.decide_change` and returns the plan unchanged on
  `Rejected` or an unparseable amount, as `_prompt_edit` always did, and drops the edited write's
  `effects` — the projection described the write the model proposed, not this one, and
  `render.print_plan` falls back to `current_amount` for a write without one.
- `prompt_ui.select` answering `"r"` for Escape means "cancel this edit" inside the field menu:
  `_edit_fields_by_menu` returns `None`, `_prompt_edit` returns the plan unchanged, and the decision
  prompt asks again (`test_ask_edit_menu_escape_cancels_the_edit_not_the_plan`, tests/test_cli.py).
- `render.write_summary(write, locations)` is the one label for a write in a menu row, used by the
  edit picker, `cli._pick_writes`' checklist, and the preview harness — replacing `_prompt_edit`'s
  `(from None to fridge-door)` field dump and `_pick_writes`' own location-less phrasing.
- `scripts/preview-ask-ui.py` gains an `edit` scene drawing both menus.
- `pytest` reports 371 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- The interactive edit path is tested with `prompt_ui.select` monkeypatched
  (`tests/test_cli.py`'s `_patch_menu`), not over a pty — `tests/test_prompt_ui_pty.py` covers
  `select` itself, so what is untested is `_prompt_edit`'s wiring against real keypresses.
- No edit menu row shows a location's display path; `from`/`to` show the raw id, which is what has
  to be retyped.

---

# 2026-09-04: Arrow Keys Read as Escape — Fixed

## Current State

- The first real-terminal run of `prompt_ui.select` exited on the Down arrow: `read_key` returned a
  bare `"\x1b"`, which `select` answers as `"r"` (reject), ending the request with nothing written.
- Cause: `sys.stdin` is a buffered `TextIOWrapper`, and `sys.stdin.read(1)` on a terminal decodes
  every byte already available into the wrapper's own buffer before returning the first — a Down
  arrow's `\x1b[B` arrives in one burst, so after `"\x1b"` came back the `"[B"` sat in Python's
  buffer while `select.select`, which polls the file descriptor, saw nothing pending. Reproduced
  directly against a `pty.openpty()` pair before the fix was written.
- `prompt_ui.read_key` reads the file descriptor with `os.read(fd, 1)` instead, extending the read
  only for an escape sequence (polling `_ESCAPE_TIMEOUT` per byte while the accumulated bytes are
  still a prefix in `_PARTIAL_ESCAPES`) or a multi-byte UTF-8 character
  (`_utf8_continuation_bytes`).
- `read_key` reads one byte rather than a chunk — a chunked read returns two keypresses already in
  the tty buffer as one merged string that matches no option, dropping both
  (`test_two_keypresses_arriving_together_are_read_separately`, tests/test_prompt_ui_pty.py).
- `prompt_ui.UP` and `prompt_ui.DOWN` are tuples carrying both cursor-key encodings — `ESC [ A`/`B`
  and the application-cursor-mode `ESC O A`/`B` a terminal in DECCKM (tmux among others) sends.
- `prompt_ui.raw_mode` is a context manager held across a whole `select`/`multiselect` loop rather
  than re-entered per keypress: restoring canonical mode between reads let a keystroke arriving
  during a redraw be echoed and line-buffered by the tty.
- `raw_mode` calls `tty.setcbreak(fd, termios.TCSAFLUSH)`, not `tty.setraw` — `setraw` also clears
  `OPOST`, which is what maps `\n` to `\r\n` on output, so every line `rich.live.Live` redrew
  inside it would staircase. `TCSAFLUSH` discards input queued before the mode change, so a
  keystroke typed during the seconds of model inference before a plan appeared is not counted as a
  decision about it.
- `prompt_ui.interactive` probes `termios.tcgetattr(sys.stdin.fileno())` alongside the two
  `isatty()` checks, so a stdin that claims to be a terminal but has no readable attributes falls
  back to the typed prompt instead of raising inside `raw_mode` mid-decision.
- `tests/test_prompt_ui_pty.py` (12 tests) drives `read_key`, `select`, and `multiselect` over a
  `pty.openpty()` pair with `os.fdopen(slave, "r")` as `sys.stdin` — the same buffered wrapper the
  bug lived in. Four of them fail against the previous `read_key`, verified by reinstating it.
- `tests/test_prompt_ui.py`'s scripted-keypress tests stub `raw_mode` to `contextlib.nullcontext`
  alongside `interactive`/`read_key`: pytest's captured stdin has no terminal attributes to set,
  and those tests are about what a keypress means, not how it is read.
- `pytest` reports 367 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Missing

- No test covers `select`'s `Live` redraw itself — the pty tests assert what `select` returns, not
  what it drew.
- `multiselect` still has no non-TTY path (unchanged from the entry above).

## Missing

- No real-model run exercises any of this — every test drives a scripted `SendsCompletions` or a
  `_FakeAgentRunner`, and no `sumac ask` invocation against a real GGUF is recorded in this entry.
- No test drives `prompt_ui` against a real terminal: `interactive()` is monkeypatched to True and
  `read_key` replaced, so `termios.tcgetattr`/`tty.setraw` and the `rich.live.Live` redraw are
  exercised by no test — the arrow-key failure that gap allowed, and the pty tests added for it,
  are recorded in the entry below.
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
