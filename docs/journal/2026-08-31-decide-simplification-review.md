# sumac: `decide` Simplification Review and Follow-Up Plan

**Status:** review complete, plan not yet implemented
**Author:** drafted with Claude, 2026-08-31
**Scope:** `src/sumac/decide.py`, with follow-on changes in `config.py`, `ledger.py`

---

## 1. Context

`decide.py` was read by an outside model (ChatGPT) in response to the question: *"Do you think this could be simplified? It seems to be inventing a wheel, if not reinventing one."* The reply argued that the module has accumulated ten responsibilities, that `_check_endpoint_shape` could be eliminated by replacing `ChangeKind` with event-shaped command types, that `serialize_event`'s `match` is a hand-maintained serializer registry better expressed as `event.to_record()`, and that `Write`, the `(writes, messages)` return, and decide-layer UUID generation should all be reconsidered.

That assessment was then reviewed independently against the actual source, its own docstrings, the callers, and the 2026-08-30 journal. This entry records the outcome of that review and the change plan it produced. **No code was changed in this pass.** §5 is the payload: it is a plan to be picked up later, written to be implementable without re-deriving the reasoning.

The reviewing model had no access to the 2026-08-30 journal when it wrote its assessment. Several of its recommendations collide with decisions recorded there, which is the main reason the verdict lands where it does — and is worth remembering as evidence that this file's rationale does not survive being read in isolation.

---

## 2. Verdict

**Roughly 30% right.** The description of the problem is fair; the prescription mostly is not.

**It recommends a design that is already implemented.** The "deeper simplification" — distinct event types so an `Acquired` has no `frm` to forget — is §3.3 of the 2026-08-30 journal, shipped in Phase 4b. `events.py` is exactly that shape. The project already paid for the migration and carries a v1→v2 upcaster in perpetuity as the price. The assessment is arguing for the design it is looking at.

**It mistakes the CLI boundary for a domain-model gap.** `_check_endpoint_shape` does not guard the events; it guards the *command surface*. `sumac add` takes `(kind, --from, --to)` (cli.py:234-243), so something must total-map that triple onto six event types. Five command classes relocate that case analysis into a CLI dispatch factory; they do not delete it. Three specific reasons it gets worse there, not better:

- `CORRECTION` genuinely admits either shape — exactly one of `from`/`to`, both valid (decide.py:78-85) — and maps to *two* event types (decide.py:219-228). The branch survives verbatim.
- §4's catalogue requires `Rejected("missing_endpoint")` specifically. That entry exists *because* `InventoryChange.__post_init__` raised `ValueError` rather than `Rejected` — the error channel was the bug. Constructor validation reintroduces it.
- Validation in the event constructors would break fold totality, the one rule the whole prior journal exists to protect. `events.py`'s dataclasses are also the *read* type: `schemas.py`'s `to_domain()` builds them from whatever is on disk (schemas.py:175-183), and `_load` catches only `(ValidationError, ValueError)` (ledger.py:91). `Rejected` extends `SumacError`, so a validating constructor would make a legacy record **crash `build_inventory`** instead of becoming an `invalid_record` anomaly. That is precisely the §3.1 violation routes B/C/D were rejected to avoid.

**The serializer-registry complaint sees 1 of 7 registration points.** Adding an event type today requires edits in: the `events.py` dataclass; the `Event` union (events.py:110); a new pydantic ingest schema; `_V2_PAYLOAD_BY_TYPE` (schemas.py:332); `RecordSchema.type`'s `Literal` (schemas.py:347); `RecordSchema.payload`'s union (schemas.py:359); `ledger._fold`'s match arms (ledger.py:338-384); and `serialize_event`. Moving serialization onto the event fixes one, and leaves the hand-written *deserializer* — which is strictly more elaborate, including a before-validator that exists because pydantic's union resolution silently misreads empty v1 snapshots as v2 (schemas.py:371-392). The asymmetry is inherent to a versioned wire format with two live schema generations.

`event.to_record()` also inverts a load-bearing dependency. `events.py` knows nothing about `SCHEMA_VERSION`, `uuid4`, actors, or wire dicts, which is what lets `_fold` be driven by Hypothesis-generated events with no files and no crypto (ledger.py:276-289). And the proposed generic envelope serializer cannot work as described: `asdict()` does not give `str(amount)` for the `Decimal` fields, it flattens `Snapshot.entries`' nested dataclasses with `Decimal`s intact, and deriving `type_name` from the class name would couple **a permanent, append-only, encrypted on-disk format to Python class identifiers** — renaming `Acquired` would silently change the wire format and desync schemas.py's `Literal`, visible only when folding records written years earlier. The explicit `type_name = "acquired"` is the correct decoupling.

**Several "reconsider" items are decisions already made against production data.** `Write(stream, obj)` is specified in §3.5 and is the minimum pair `store.append` needs — the store is stream-addressed, ownership is enforced on the stream id (store.py:52), and one `decide_change` call can write to *two different streams* (`config` and `log:<actor>`), which a bare event list cannot express. The `(writes, messages)` return is §3.5's deliberate anti-`--force` channel. Auto-registration is §3.5a, forced by 472 unregistered product ids in the real log. Decide-layer `uuid4` is addressed in the module docstring: ids carry no meaning decide's logic depends on, which is why time, actor, and inventory *are* injected and ids are not.

The honest summary: the assessment is a competent generic architecture review of a file whose specifics defeat most of its recommendations.

---

## 3. What the assessment got right

**The responsibility list is fair as description.** 468 lines covering validation, resolution, event construction, serialization, and shortfall reconciliation is a lot to hold at once. That the reasons are good does not make the file easy to read cold.

**"Isolate the shortfall logic" is correct**, and for the right reason. The block at decide.py:369-415 is 47 lines, ~30 of them comment, sitting mid-function, carrying three distinct behaviours (skip on unit mismatch, emit `Counted`, emit note) and the single most interesting rule in the module. Unlike the rest of the assessment's suggestions it costs nothing structurally. Adopted as §5.2.

**Declining to simplify away the domain rules** deserves credit; the assessment states it explicitly rather than reflexively minimizing.

---

## 4. What it missed

Three real defects, none of which appear in the assessment.

### 4.1 `serialize_event` cannot serialize `Correction`, and says it can

`events.Correction` is a member of the `Event` union (events.py:110), but `serialize_event`'s `match` has no arm for it (decide.py:245-316). It falls to `case _: raise TypeError`, annotated `# pragma: no cover - exhaustive given events.Event`. **That annotation is false.**

It is unreachable today only because `decide_correct` hand-builds its envelope (decide.py:455-467), duplicating all eight fields `serialize_event` already constructs. This is the module's one genuine DRY violation, and it is the one the assessment missed while critiquing the `match` it lives in.

### 4.2 The ordering invariant is guarded, but probabilistically

An earlier draft of this review claimed the microsecond-offset ordering hack (decide.py:405) was untested. **That was wrong**, and the correction matters. `test_insufficient_stock_counted_actually_precedes_the_movement_after_reload` (test_ledger.py:749) drives the full write → store → reload → fold path and asserts the end state; its docstring names the bug and notes that decide.py's own unit tests "only ever checked list order." The gap was found and closed at the right layer.

The residual point is narrower. That guard is **non-deterministic**: remove the offset and the two records tie on `(ts, actor)`, the sort falls to a random `uuid4`, and the test detects the regression with p≈0.5 per run — it flakes in CI rather than failing cleanly, and can pass locally. Making the ordering deterministic converts a coin-flip into a real guard. This lowers the urgency of the fix relative to the first draft's framing, and is why §5.4 is sequenced last rather than first.

### 4.3 `nominal_basis` has no producer

§3.3 states that "the raw user input and the conversion basis are recorded alongside for audit," and §3.5's sketch writes `nominal_basis=cfg.basis(p, u)`. In the implementation, `_build_event` never sets it, `Config` has no `basis()` method, and **every event ever written has `nominal_basis=None`**. The field is plumbed through `events.py`, `schemas.py`, and `serialize_event` with nothing on the other end.

Because `cfg.convert` is applied at decide-time and frozen into the event, `sumac add purchase jam 2 jar` stores `680 g` and the fact that the user said "2 jars" is destroyed — irreversibly, in a log that by design is never rewritten. A data-fidelity gap in the module the assessment called over-engineered.

---

## 5. Change plan

Four items. Items 5.1, 5.2 and 5.3 are self-contained and may be done in any order or in parallel; 5.4 is sequenced last and is the only one that can silently change an answer the tool has already given.

| # | Change | Wire format | Read path | Risk |
|---|---|---|---|---|
| 5.1 | `decide_correct` → `serialize_event` | unchanged (must verify byte-identical) | untouched | low |
| 5.2 | Extract `_reconcile_shortfall` | untouched | untouched | none |
| 5.3 | `Config.basis()` producer | new values in an existing field | untouched | medium |
| 5.4 | `seq`-based ordering | **untouched** | changed for every record | high |

### 5.1 Route `decide_correct` through `serialize_event`

Two defaulted keyword params, so every existing call site is unchanged:

```python
def serialize_event(
    event: events.Event,
    *,
    actor: str,
    occurred_at: datetime,
    cmd_id: str,
    supersedes: str | None = None,
    record_id: str | None = None,
) -> dict:
```

`supersedes` replaces the hardcoded `"supersedes": None` (decide.py:324). `record_id` replaces the internal `str(uuid4())` (decide.py:321) as `record_id or str(uuid4())`. Add the missing arm ahead of `case _`:

```python
case events.Correction(reason=r):
    type_name = "correction"
    payload = {"reason": r}
```

Then rewrite the false pragma. With `Correction` handled the claim becomes true, but nothing in the language enforces it, so phrase it as a guard rather than an exhaustiveness proof: `# pragma: no cover - defensive; every events.Event member is handled above`.

`decide_correct`'s tail collapses to one `serialize_event` call passing `supersedes=target_id` and `record_id=record_id`.

**Why `record_id` exists.** `decide_correct` must know the generated id *before* serializing, to run the `supersede_self` check (decide.py:452) that §4's catalogue requires and `test_correct_self_supersede_is_rejected` (test_decide.py:467) verifies by patching `sumac.decide.uuid4`. Note that patch would also hit `serialize_event`'s own `uuid4`, so generating the id inside would appear to work for that test while leaving `decide_correct` unable to check. Keep generation in `decide_correct`.

**Risk.** Output must stay byte-identical to the current literal: same eight keys, same insertion order (`schema_version, type, id, ts, actor, supersedes, cmd_id, payload`), `type="correction"`, `payload={"reason": reason}`. It does — but assert it in a test rather than trusting the eyeball. A key-set or key-order drift writes a shape `RecordSchema` (`extra="forbid"`) would refuse to read back, and in an append-only log that record is unfixable.

### 5.2 Extract the shortfall block

Lift decide.py:369-415 to module level:

```python
def _reconcile_shortfall(
    event: events.Event,
    canon: Quantity,
    inventory: Inventory,
    *,
    actor: str,
    occurred_at: datetime,
    cmd_id: str,
) -> tuple[list[Write], list[str]]:
```

Returns `([], [])` when no correction is needed. `product_id` need not be a parameter — it is on every event that has a `frm`. The call site in `decide_change` becomes an `extend` of each returned list.

While moving it, replace `getattr(event, "frm", None)` (decide.py:375) with a `match` over `Consumed | Discarded | Moved`. The `getattr` is a duck-typed probe across a union no type checker can verify, and it is silently coupled to `Counted` using `at` rather than `frm` — rename that field and the shortfall logic would start firing on counts. The match states the "delta events with a source side" set explicitly, which the comment at decide.py:374 already says in prose.

Keep all three long comments attached to the code they explain; the ordering comment travels with the `counted_at` line.

**Risk.** None. Pure refactor. The existing shortfall tests (test_decide.py:313-400) drive `decide_change` and must pass untouched — if any needs editing, the extraction changed behaviour and is wrong.

Sequenced before 5.4 deliberately: it localizes the microsecond offset into one small function, turning 5.4 into a one-line change instead of surgery inside a 90-line body.

### 5.3 Give `nominal_basis` a producer

A new method on `Config`, beside `convert` (config.py:172):

```python
def basis(self, product_id: str, amount: Decimal, unit: str) -> dict[str, str] | None:
    product = self.known_products.get(product_id)
    if product is None or unit == product.unit:
        return None
    ratio = product.conversions.get(unit)
    if ratio is None:
        return None
    return {"raw_amount": str(amount), "raw_unit": unit, "ratio": str(ratio)}
```

`None` when nothing was converted — there is no basis to record for a value that passed through unchanged. Resolve against `known_products`, mirroring `convert`'s deliberate divergence (config.py:178-186), so the two can never disagree about which product they describe. The returned shape matches the example documented at events.py:20-21.

**Threading.** `_resolve_product` already returns a 3-tuple; a fourth element is where this starts to smell. Replace it with a small frozen dataclass local to `decide.py` — `_ResolvedProduct(canon, basis, writes, warning)`. The registered path sets `basis=cfg.basis(...)`; the auto-register path sets `None` (an auto-registered product's canonical unit *is* the unit just used, so nothing was converted). `_build_event` takes `nominal_basis` as a parameter and forwards it to all six constructors.

The synthesized shortfall `Counted` keeps `None` — it is derived in canonical units, not user input. So does `cli.py`'s `snapshot` (cli.py:282), whose entries are parsed as raw `amount/unit` with no conversion.

**Risk.**

- **No schema change, no version bump.** `nominal_basis` is already emitted in every payload (currently always `None`) and already accepted as `NominalBasisDict` (schemas.py:37). Populating it turns a `null` into an object in new records; existing records are untouched and still validate.
- **Every value must be `str`.** The ingest type is `dict[str, str]`; a `Decimal` would serialize in some cases and then fail `model_validate` on read back. This is the gate-soundness trap the prior journal keeps flagging — `missing_reason` was added for exactly this class of bug: never write something the reader would refuse. Add a round-trip test (build with basis → serialize → `RecordSchema.model_validate` → `to_domain` → assert equal).
- **The fold must keep ignoring it** (events.py:20-21, "never read by the fold"). Model-agreement and fold-determinism property tests are the check.
- No backfill is possible: the raw input was never stored. Records written before this lands have `None` indistinguishably from records where nothing was converted. That asymmetry is permanent, which argues for doing this sooner — every day it is absent produces more irreversibly lossy records.

### 5.4 `seq`-based ordering per §3.7

§3.7 specifies merge order as `sort by (occurred_at, actor, seq)`. The implemented fold sorts by `(ts, actor, id)` (ledger.py:96, ledger.py:290); `seq` is written by `store.append` (store.py:64) and then used *only* for gap/duplicate diagnostics. The microsecond offset is a workaround for a designed mechanism that was never wired into the sort. Split into three commits.

**5.4a — carry `seq` into the loaded record (inert).** `_load` already computes the backfill and discards it: `_check_seq` calls `store.assigned_seqs(objs)` internally (ledger.py:46, ledger.py:82) and the result dies there. Hoist it, pass it in, and zip it against `objs` *before* the `schema_too_new` / `invalid_record` filtering so positions stay aligned with what `assigned_seqs` computed. Attach via `dataclasses.replace(record, seq=assigned)`.

Always take the assigned value rather than only filling when `seq is None` — `assigned_seqs` already prefers a stored `seq` over position (store.py:49), so this keeps `_load` and `_check_seq` looking at identical numbers.

Add `seq: int = -1` to `ledger._EventRecord` (ledger.py:117-124) and populate it in `_load_v2`. Default it, and make it `int` not `int | None`: `test_model_properties` constructs `_EventRecord` positionally in three places (test_model_properties.py:136, 270, 307), so a required field breaks all of them, and an optional `None` makes the sort key below unsortable when two hand-built records tie.

This commit changes no behaviour — nothing reads the field yet. That is the point: it isolates plumbing from semantics.

**5.4b — change the sort key.** In both `_load` and `_fold`:

```python
key = lambda r: (r.ts, r.actor, r.seq, r.id)
```

**Keeping `id` as the final tiebreak is not optional.** `test_fold_determinism` (test_model_properties.py:131) asserts `_fold(shuffled) == _fold(records)` over Hypothesis-drawn records sharing a tiny pool of timestamps and actors, all built with the default `seq=-1`. Drop `id` and shuffle-invariance dies — the sort stops being a total order on content and starts depending on input order. With `(ts, actor, seq, id)`, `seq` becomes the *primary* tiebreak (what §3.7 wants, and what makes the shortfall ordering deterministic) while `id` still guarantees totality when `seq` is absent or duplicated. `seq` is only ever compared within one actor, since `actor` sorts first, so per-segment numbering is never compared across streams.

Then add the deterministic ordering test that test_ledger.py:749 currently approximates: assert directly that the reloaded `Counted` sorts before its event, rather than inferring it from the end state.

**5.4c — drop the microsecond offset.** Once 5.4b lands, `counted_at = occurred_at - timedelta(microseconds=1)` is no longer load-bearing. Remove it, for a reason beyond tidiness: it fabricates a timestamp that never happened, and `build_inventory(as_of=...)` filters on `r.ts <= as_of` (ledger.py:394). An `as_of` landing inside that one-microsecond window includes the `Counted` and excludes the event it corrects, reporting an intermediate state that never physically existed. With a shared `occurred_at`, `as_of` takes both or neither. Its own commit, so a regression is one line to revert without losing 5.4a/b.

**Risk.**

- **No wire-format change whatsoever.** Entirely read-path: nothing new is written, no version moves, old records stay readable. `seq` has been optional-forever in both `models.Record` (models.py:133) and `RecordSchema` (schemas.py:357) since Phase 7. This is the redeeming feature that makes the change tractable.
- **It can change computed inventory for existing logs.** Any two records tying on `(ts, actor)` currently order by random uuid; afterwards they order by append position. Strictly more correct, but a *different answer* for historical data. The blast radius is exactly the same-timestamp same-actor pairs — including every pre-fix `Counted`+event pair written before the offset landed. Verify against `tests/fixtures/golden_log` across 5.4a→5.4b; test_model_properties.py:48 pins only the record *count*, so add an assertion on the full folded inventory to make any shift visible and deliberate.
- **`sumac log` display order changes** too, sharing `_load`'s sort. Harmless, but expect churn in CLI tests asserting log output order.
- Duplicate `seq` (a bad merge, already reported as `seq_duplicate`) ties and falls through to `id` — unchanged from today, which is right: the anomaly channel reports it, the fold stays total.

---

## 6. Notes for the implementing agent

- 5.1 and 5.2 are the cheap wins and should land first; neither can change a stored byte or a folded answer.
- Do not attempt 5.4b without the golden-log inventory assertion in place first. It is the only safety net for a change that alters historical results.
- Resist the temptation to fold 5.4a and 5.4b into one commit. The inert-plumbing / behaviour-change split is what makes a bisect meaningful if the fold output moves.
- The broader lesson from §1: this module's rationale does not survive being read in isolation. Three of the assessment's recommendations would have re-broken §3.1's totality rule, and it had no way to know. Where a decision in `decide.py` is load-bearing and non-obvious, the comment explaining it is doing real work — that is a reason to keep them, not to trim them as noise.

---

The section below records repository state in the format specified by `docs/JOURNAL.md`. Sections §1-§6 above predate the adoption of that format and keep the sumac entry conventions.

---

# 2026-08-31: Journal Format Adoption

## Current State

- `docs/JOURNAL.md` carries a verbatim copy of the journal format published at https://github.com/lmmx/giacometti/docs/JOURNAL.md, copied on 2026-08-31 under the instruction the document gives to repositories that adopt the format (docs/JOURNAL.md:3-6).
- `docs/JOURNAL.md` governs code comments, docstrings, commit messages, and journal entries written in `sumac-home` from 2026-08-31 onward.
- `docs/JOURNAL.md` names journal entries `docs/journal/YYYY-MM-DD-title.md` (docs/JOURNAL.md:10).
- `docs/JOURNAL.md` defines four entry sections — Current State, Stubbed, Missing, Divergence — and requires Current State alone (docs/JOURNAL.md:11-16).
- `docs/JOURNAL.md` separates "stubbed", denoting code that runs and returns an error, from "missing", denoting absent code (docs/JOURNAL.md:20).
- `docs/JOURNAL.md` scopes the Divergence section to claims made in project documentation rather than to aspirations (docs/JOURNAL.md:21).
- `docs/JOURNAL.md` restricts entries to factual present tense (docs/JOURNAL.md:33).
- `docs/JOURNAL.md` forbids recommendations and suggestions (docs/JOURNAL.md:31).
- `docs/JOURNAL.md` forbids evaluation words including "better", "cleaner", and "should" (docs/JOURNAL.md:32).
- `docs/JOURNAL.md` forbids bare constative verbs such as "exists" and "is", and calls for a description of what a component does or how the implementation works in place of one (docs/JOURNAL.md:30).
- `docs/JOURNAL.md` limits each bullet to one statement covering a single component such as a backend, a CLI command, or a workflow (docs/JOURNAL.md:37-38).
- `docs/JOURNAL.md` lists related files inline separated by commas when the files implement one component (docs/JOURNAL.md:39).
- `docs/JOURNAL.md` forbids pronouns that depend on a neighbouring bullet, and calls for each bullet to carry enough context for independent verification (docs/JOURNAL.md:40).
- `docs/JOURNAL.md` joins a cause and the effect within one bullet using a dash rather than splitting the pair across two bullets (docs/JOURNAL.md:46).
- `docs/JOURNAL.md` orders bullets chronologically within a section to imply sequence (docs/JOURNAL.md:45).
- `docs/JOURNAL.md` permits bullets to reference each other through shared artifacts such as filenames and line counts (docs/JOURNAL.md:44).
- `docs/JOURNAL.md` places file paths in parentheses after statements (docs/JOURNAL.md:59).
- `docs/JOURNAL.md` appends line numbers after a file path when a statement references specific code, in the form `file.rs:45-52` (docs/JOURNAL.md:60).
- The change plan at §5 above covers four items — routing `decide_correct` through `serialize_event`, extracting `_reconcile_shortfall`, adding a `Config.basis()` producer, and replacing the microsecond-offset ordering with a `seq`-based sort key (docs/journal/2026-08-31-decide-simplification-review.md:75-201).
- The change plan at §5 above modifies three modules — `decide.py`, `config.py`, `ledger.py` (docs/journal/2026-08-31-decide-simplification-review.md:5).
- `decide.py` carries comments phrased in evaluative terms, including "not worth solving now" on the auto-registration canonical-unit fallback (decide.py:170-171).

## Stubbed

- `serialize_event` accepts any member of the `events.Event` union and raises `TypeError` for `events.Correction`, the one member with no `match` arm — the fallback arm carries the annotation "exhaustive given events.Event" (decide.py:245-316, events.py:110).
- `serialize_event` writes a `nominal_basis` key into the `acquired`, `consumed`, `discarded`, `moved`, and `counted` payloads and into each `snapshot` entry, and `Config` defines no `basis()` method to populate the key — every record written to date carries `nominal_basis: null` (decide.py:245-316, config.py:172).

## Missing

- No pre-commit hook or CI check validates journal entry filenames or section structure (.pre-commit-config.yaml).
- `ledger._load` and `ledger._fold` sort on `(ts, actor, id)` and omit `seq` from both sort keys, where docs/journal/2026-08-30_decide-pattern-data-integrity-upgrade.md:494-504 specifies `(occurred_at, actor, seq)` (ledger.py:96, ledger.py:290).
- `ledger._load` computes `store.assigned_seqs(objs)` inside `_check_seq` for gap and duplicate diagnostics and discards the returned positions, leaving `Record.seq` at `None` on every record written before Phase 7 (ledger.py:46, ledger.py:82).
- No journal entry in the four-section format records repository state after any of the four change-plan items (docs/journal/).

## Divergence

- `docs/JOURNAL.md` names entries `YYYY-MM-DD-title.md` with hyphen separators, and docs/journal/2026-08-30_decide-pattern-data-integrity-upgrade.md uses an underscore before a slug (docs/JOURNAL.md:10).
- `docs/JOURNAL.md` defines four `##` sections headed Current State, Stubbed, Missing, and Divergence, and sections §1-§6 of this entry open with a Status/Author/Scope header block followed by numbered `##` sections (docs/journal/2026-08-31-decide-simplification-review.md:1-6).
- `docs/JOURNAL.md` forbids recommendations, and §5 of this entry sequences four prospective changes and ranks the four by risk in a summary table (docs/journal/2026-08-31-decide-simplification-review.md:75-85).
- `docs/JOURNAL.md` forbids evaluation words, and §6 of this entry uses "should" and "cheap wins" (docs/journal/2026-08-31-decide-simplification-review.md:207).
- `docs/JOURNAL.md` limits each bullet to one statement, and §5.3 and §5.4 of this entry carry multi-sentence bullets and fenced code blocks (docs/journal/2026-08-31-decide-simplification-review.md:165-170, docs/journal/2026-08-31-decide-simplification-review.md:196-201).

---

# 2026-08-31: decide.py Simplification — 5.1/5.2/5.3 Landed

## Current State

- `serialize_event` accepts `events.Correction` through a `case events.Correction(reason=r):` arm (decide.py:390-392) — the "Journal Format Adoption" entry's Stubbed section above records the prior state, where this case fell through to `case _` and raised `TypeError` (decide.py:245-316 in that entry's citation).
- `serialize_event`'s fallback arm carries the pragma "defensive; every events.Event member is handled above" (decide.py:393), replacing the prior "exhaustive given events.Event" annotation the same Stubbed bullet names as false.
- `serialize_event` takes two additional keyword parameters, `supersedes: str | None = None` and `record_id: str | None = None` (decide.py:292-300) — every call site that predates these params (`decide_change`'s two internal calls, `cli.py:282`'s `snapshot` command) passes neither, and receives the same `supersedes=None`/freshly-generated-`id` behavior those call sites had before this change.
- `decide_correct` builds its return `Write` by calling `serialize_event(events.Correction(reason=reason), ..., supersedes=target_id, record_id=record_id)` (decide.py:560-570) rather than constructing the wire dict inline — `record_id` is generated in `decide_correct` (decide.py:556) ahead of that call, so the self-supersede check (decide.py:557-558) compares against the id before the record is serialized, not after.
- `decide.py` defines `_reconcile_shortfall(event, inventory, *, actor, occurred_at, cmd_id) -> tuple[list[Write], list[str]]` (decide.py:408-475) as a module-level function — `decide_change` (decide.py:476-528) calls it once (decide.py:516-520) and extends its own `writes`/`messages` lists with the two lists it returns, in place of the block that previously sat inline in `decide_change`'s body.
- `_reconcile_shortfall` resolves the delta event's `product_id`, `frm` side, `amount`, and `unit` through a `match` over `events.Consumed | events.Discarded | events.Moved` (decide.py:427-436), in place of the prior `getattr(event, "frm", None)` probe across the full `events.Event` union.
- `Config` defines `convert_with_basis(product_id, amount, unit) -> tuple[models.Quantity, dict[str, str] | None] | None` (config.py:172-208) as the single lookup both a converted quantity and its audit trail are derived from; `Config.convert` (config.py:210-217) returns the first element of that pair, unchanged in its own signature and return value from before this entry.
- `convert_with_basis`'s second return element is `None` when the input `unit` already equals the product's canonical unit (config.py:202-203) — no `nominal_basis` is recorded for that case, since the resulting event's own `amount`/`unit` fields already carry the same value the input asserted.
- `decide.py` defines `_ResolvedProduct` (decide.py:126-137), a frozen dataclass whose `writes` field is typed `tuple[Write, ...]`, not `list[Write]` — `decide_change` converts it once at the call site, `writes = list(resolved.writes)` (decide.py:507), before appending further writes to that local list.
- `_resolve_product` returns a `_ResolvedProduct` (decide.py:140-206) carrying `basis` from `cfg.convert_with_basis(...)` on the registered-product path (decide.py:151-158) and `basis=None` on the auto-register path (decide.py:160-206).
- `_build_event` takes a `nominal_basis: dict[str, str] | None` parameter (decide.py:209-216) and passes it into whichever of the six event constructors it builds (decide.py:217-289); `decide_change` supplies `resolved.basis` at the one call site (decide.py:514).
- The `Counted` event `_reconcile_shortfall` synthesizes on an insufficient-stock adjustment (decide.py:448-454) receives no `nominal_basis` argument and keeps the field's `None` default — that value comes from current holdings, not from anything the user typed.
- `cli.py`'s `snapshot` command (cli.py:270-286) constructs `events.SnapshotEntry` from parsed `PRODUCT=AMOUNT/UNIT` text with no call through `Config.convert`/`convert_with_basis` anywhere in that path (cli.py:65-71) — its entries carry `nominal_basis=None` both before and after this entry's changes, unmodified by them.
- `render.py`'s `_describe_payload` (render.py:193-234, the function `print_log` (render.py:237-248) calls for `sumac log`) and `print_unit_check` (render.py:73-140) read no `nominal_basis` field on any event or snapshot-entry type — no `render.py` line changed as part of this entry.
- `tests/fixtures/generate_golden_log.py`'s three `decide.decide_change` calls (generate_golden_log.py:125-172) each pass the unit the target product already declares as canonical (milk in `l`, flour in `kg`, both set at generate_golden_log.py:95-100) — regenerating `tests/fixtures/golden_log` under this entry's code leaves every record's `nominal_basis` at `None`, the same as before; `tests/test_decide.py`'s `test_nominal_basis_round_trips_through_record_schema` and the extended `test_registered_product_applies_conversion` are the only coverage of a populated `nominal_basis` surviving `RecordSchema.model_validate(...).to_domain()`.
- As of this entry, `nominal_basis: null` on a stored record has three distinct causes with no field-level marker distinguishing them: a v1 record upcast by `upcast.py`, whose six event constructors (upcast.py:33-88) never pass a `nominal_basis` argument; a v2 record written before this entry's commit, when `_build_event` had no `nominal_basis` parameter; and a v2 record written after this entry's commit whose input unit already equalled the product's canonical unit. Telling the second case from the third requires comparing the record's `ts` against this entry's commit, external to the record itself.
- `docs/journal/2026-08-31-decide-simplification-review.md` §5's summary table names 5.1-5.3's "Wire format"/"Read path" columns as "unchanged"/"untouched" except 5.3's wire format ("new values in an existing field") — the implementation lands with no `RecordSchema`/`ConfigRecordSchema` field additions or removals and no `SCHEMA_VERSION` change (schemas.py, config.py:172-217), consistent with that table.
- `tests/fixtures/golden_log`, loaded via `ledger.load_all_records`/`ledger.load_records`/`ledger._load_v2`, contains zero record pairs sharing `(ts, actor)` in any of the three views (15, 14, and 14 records respectively) — measured directly against the checked-in fixture, not estimated, ahead of this entry's implementation work.

## Missing

- `ledger._load` and `ledger._fold` sort on `(ts, actor, id)` and omit `seq` from both sort keys, unchanged by this entry (ledger.py:96, ledger.py:290) — §5.4 of this file's change plan was not implemented in this pass.
- `tests/fixtures/golden_log` contains no record pair sharing `(ts, actor)` (see the Current State bullet above), so no fixture in the repository exercises what `_fold`'s sort does when such a pair exists, independent of whether §5.4 is implemented.
