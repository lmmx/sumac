# sumac: Write-Time Validation and Log Remediation

**Status:** proposal (rev 2, incorporating review feedback)
**Author:** drafted with Claude, 2026-08-30
**Scope:** `sumac` event log, config projection, CLI write path

---

## 1. Motivation

`sumac status` and `sumac find` are unreliable because the log contains events that cannot be folded.

The triggering case: a movement was recorded to `hob-right-below-bottom`, a location ID that does not exist in config. The CLI accepted it — it validated that `kind` was a valid `ChangeKind`, that `amount` parsed as a decimal, and that `unit` and `product_id` were strings. Nothing checked that the location existed. The event was encrypted, signed, appended, committed, and pushed.

The damage surfaces at read time, in three distinct ways:

| Failure | Cause | Symptom |
|---|---|---|
| Fold crashes partway | Unit mismatch (`1 jar` moved as `1 unit`) | Partially-built inventory, exception |
| Event silently vanishes | Movement missing `from` or `to` | Quantities quietly wrong, no error |
| Config unreadable | Circular parent reference | Nothing works at all |

Confirmed in the current code: `InventoryChange.__post_init__` (models.py:73-82) raises on a malformed movement, and `RecordSchema.to_domain` (schemas.py:148) calls it while replaying every stored record — so the validation fires at read time, not just write time. Likewise `Quantity.__add__` (models.py:53-56) raises on unit mismatch and is called directly from `ledger._apply_delta`. `build_inventory` is therefore validating during the fold, which is precisely the anti-pattern §3.1 names.

Two separate faults are compounding, and it matters to name them apart:

1. **There is no write-time gate.** Semantic validity — does this location exist, is this unit convertible for this product, does this movement have both endpoints — is never checked before append.
2. **The fold is not total.** A single malformed event takes down the entire replay rather than being isolated. This is why one typo in one location ID corrupts *everything*, instead of corrupting one line item.

Existing read-time validation (`verify_all()`) checks signatures. It detects tampering. It has nothing to say about whether `hob-right-below-bottom` is a real place.

The consequence is that the log is append-only and now contains permanently-bad data. The apparent options are to hand-edit the encrypted log — destructive, breaks signatures, and requires a coordinated history rewrite across two git clones — or to accept a broken inventory.

**Neither is necessary.** The thesis of this document:

> You cannot undo an append, and you should not want to. Make the fold total so bad events are survivable, correct them by appending corrections, and add a validation gate so it stops happening.

Fault 2 is fixed *first*, because it is the remediation path for damage already committed.

---

## 2. Routes considered

**A. Tighten CLI validation.** Add existence checks where the commands are parsed.

Rejected. It does nothing for the events already in the log — `sumac status` still crashes tomorrow. Validation logic living in the CLI drifts from the fold's assumptions over time, so the two disagree and you get a new class of bug. And it leaves the fold non-total, so any future gap re-creates the same total failure.

**B. Repair the log in place.** Decrypt, edit the offending lines, re-encrypt, re-sign.

Rejected. Re-signing your own corrections defeats the purpose of signing. It breaks the append-only property that the git-sync design rests on: both clones would need a coordinated history rewrite, and a stale clone silently reintroduces the bad line on the next push. It is also purely remedial — nothing prevents recurrence, and this will recur.

**C. Snapshot and truncate.** Fold to current state, write a clean snapshot, discard the log.

Rejected. It discards the history the whole design exists to keep, and it freezes whatever wrong state the bad events already produced into a snapshot that now *is* the truth, with no way to trace where it came from.

**D. Migrate to a validating store.** SQLite with constraints, or Pydantic models on every path.

Rejected. The storage layer is not the problem. Per-writer encrypted JSONL under git is correct for the topology — single writer per file means git never merges log content, and line-independent encryption keeps diffs as clean appends. SQLite over a git-synced directory would be strictly worse. Separately, neither DB constraints nor schema validation can express the rules that actually matter here: *this unit is convertible for this product*, *this parent chain is acyclic*, *this location existed when the event was written*. Those are domain rules, not schema rules.

**E. Decider split, total fold, correction events.** ← **chosen**

Three changes, in dependency order:

- **`decide(command, state, config) -> [Event] | Rejection`** — the single place semantic validation lives. Rejects before append.
- **`evolve(state, event) -> state`** — total. Accepts every event it could ever see, including malformed legacy ones. Never raises. Never rejects.
- **Correction events** — `MovementVoided(target_event_id, reason)` appended as new events. The log stays byte-intact, signatures stay valid, the fold produces correct state.

Costs: an event schema change with upcasters, a real config projection, and ongoing discipline about which layer validates. Buys: prevention, remediation without touching a single stored byte, and a readable log on day one.

### Explicitly out of scope

Earlier discussion raised optimistic concurrency control, vector clocks, and CRDT-style convergent counters for the two-writer case. These are dropped. Two people sharing a pantry via git pull do not have a concurrency problem worth that machinery. Deterministic merge order — sort all events across both segments by `(occurred_at, actor, seq)` before folding — is sufficient and is the only ordering change required.

---

## 3. Design

### 3.1 The rule that must not be broken

> **`decide` validates. `evolve` never validates.**

`evolve` must be total over every event it could encounter, including events written by older code under older rules. An event is a fact about the past. A rule added in 2026 cannot make a 2025 event invalid — it can only make the *state* those events produce something you want to flag.

Concretely: `evolve` never raises, never assumes a key exists, and **never converts units**. Where it cannot apply an event, it records an anomaly and moves on.

```python
@dataclass(frozen=True, slots=True)
class Anomaly:
    event_id: str
    reason: str  # "unknown_location" | "unit_unconvertible" | ...
    detail: dict


@dataclass(frozen=True, slots=True)
class Inventory:
    holdings: Map[tuple[str, str], Decimal]  # (product_id, location_id) -> canonical qty
    voided: frozenset[str]  # record_ids superseded by later records
    anomalies: tuple[Anomaly, ...] = ()


def evolve(s: Inventory, e: Event, cfg: Config) -> Inventory:
    if e.event_id in s.voided:
        return s

    match e:
        case Moved(product_id=p, frm=a, to=b, amount=q):
            if a not in cfg.known_locations or b not in cfg.known_locations:
                return flag(s, e, "unknown_location", {"from": a, "to": b})
            return apply_move(s, p, a, b, q)  # q is already canonical

        case Counted(product_id=p, at=loc, amount=q):
            if loc not in cfg.known_locations:
                return flag(s, e, "unknown_location", {"at": loc})
            return replace(s, holdings=s.holdings.set((p, loc), q))

        case LegacyChange():  # v1, raw units, may not convert
            conv = cfg.convert_legacy(e)
            if conv is None:
                return flag(s, e, "unit_unconvertible", {"unit": e.unit})
            return evolve(s, conv, cfg)

        case _:
            return flag(s, e, "unknown_event_type", {"type": type(e).__name__})
```

`flag()` appends an `Anomaly` and returns unchanged holdings. Supersede resolution is handled by `ledger.load_records` before the fold begins.

`cfg` is consulted only for **existence** (`known_*`), never for conversion. See §3.4.

### 3.2 Anomalies must be loud

Totality is only safe if quarantined events are visible. Silent tolerance is how the "movement without both endpoints vanishes" bug happened in the first place.

- `sumac status` prints a banner when `anomalies` is non-empty: `⚠ 3 events could not be applied — run 'sumac doctor'`
- `sumac doctor` lists each anomaly with its event ID, reason, decrypted payload, and a suggested `sumac correct` invocation.
- Exit code is non-zero when anomalies exist, so this is catchable in scripts.

`sumac doctor` output contains decrypted payloads. Never paste it into a commit message, issue, or PR.

### 3.3 Making illegal states unrepresentable

The "movement with only one endpoint" bug should not be fixable by validation — it should be untypeable. Replace the single `Change` record carrying a `ChangeKind` enum plus optional `from`/`to` fields with distinct event types:

```python
Acquired(product_id, to, amount)  # enters the system
Consumed(product_id, frm, amount)  # leaves, used
Discarded(product_id, frm, amount)  # leaves, binned
Moved(product_id, frm, to, amount)  # relocates
Counted(product_id, at, amount, reason=None)  # observed truth, absolute
Retired(entity_type, entity_id)  # location or product, see §3.4
```

An `Acquired` has no `frm` field to forget. A `Moved` cannot be constructed without both. All `amount` fields are **canonical units, resolved at decide-time**; the raw user input and the conversion basis are recorded alongside for audit but are not used by the fold.

This is a schema change (§5, phase 4) requiring an upcaster, and it eliminates an entire bug class permanently.

### 3.4 Config is a validated projection

Config is a stream like any other, but its fold has invariants of its own and produces the read model that `decide` validates against.

```python
def build_config(events) -> tuple[Config, list[Anomaly]]:
    cfg, anomalies = fold_config(events)
    for loc in cfg.known_locations:
        chain, seen, node = [], set(), loc
        while node is not None:
            if node in seen:
                anomalies.append(Anomaly(loc, "circular_parent", {"chain": chain}))
                break
            seen.add(node)
            chain.append(node)
            node = cfg.known_locations[node].parent if node in cfg.known_locations else None
    return cfg, anomalies
```

Iterative with a visited set — never recursive, never unbounded. A cycle produces a diagnostic naming the chain, not an unreadable config.

**Three rules make cross-stream referential integrity tractable:**

**(a) Nothing is deleted, only retired.** `Retired(entity_type, entity_id)` applies to **both locations and products**. This yields two views on config:

| Accessor | Consumer | Contents |
|---|---|---|
| `cfg.active_locations` / `cfg.active_products` | `decide` | Not retired — valid targets for new commands |
| `cfg.known_locations` / `cfg.known_products` | `evolve` | Everything ever defined — resolvable forever |

Products are not currently config entities — `Product` exists as a dataclass but has no registry, storage stream, or `add_product` path. A product registry must be built from scratch, mirroring the existing location handling in `config.py`, before `active_products` / `known_products` or unit conversion can exist. See Phase 2b.

Referential integrity is therefore **monotone**: once an ID is valid it is valid for folding forever. If Alice retires Homemade Rice Pudding, Bob's historical `Counted` for it still folds normally; Alice simply cannot record new movements of it. Retiring a location that still holds stock is rejected (`retire_nonempty`, with holdings listed). Retiring a product is permitted at any time.

**(b) Config is resolved as-of-now, not as-of-event-time.** `decide` uses the full fold of the config stream. Do not attempt to reconstruct config as it stood when a historical event was written — it doubles the fold cost and buys nothing.

**(c) Unit conversions are nominal defaults, resolved once, and frozen into events.** Every product declares a canonical unit plus permitted conversions (`1 jar = 340 g`). These exist to make units *total*, not to be accurate — jars vary, and `Counted` is the mechanism for correcting reality.

The critical consequence: **`decide` resolves the conversion and stores the canonical amount; `evolve` never converts.** If both layers converted, editing the jar definition in config would silently change every historical number and your inventory would move because you corrected a definition. Events optionally record `nominal_basis: {"jar": 340}` so `sumac doctor` can explain where a figure came from.

`unit_unconvertible` is therefore a **decide-time rejection**, surviving as an anomaly reason only for v1 legacy events, which stored raw units.

### 3.5 `decide`

```python
def decide(cmd: Command, s: Inventory, cfg: Config) -> list[Event]:
    match cmd:
        case MoveCmd(product_id=p, frm=a, to=b, amount=q, unit=u):
            if a not in cfg.active_locations:
                raise Rejected(
                    "unknown_location",
                    field="from",
                    value=a,
                    suggestions=near_matches(a, cfg.active_locations),
                )
            if b not in cfg.active_locations:
                raise Rejected(
                    "unknown_location",
                    field="to",
                    value=b,
                    suggestions=near_matches(b, cfg.active_locations),
                )
            if a == b:
                raise Rejected("noop_move", value=a)
            if p not in cfg.active_products:
                raise Rejected("unknown_product", value=p)
            if q <= 0:
                raise Rejected("non_positive_amount", value=q)

            canon = cfg.convert(p, q, u)
            if canon is None:
                raise Rejected("unit_unconvertible", value=u, expected=cfg.unit_of(p))

            events = []
            if s.at(p, a) < canon:  # shelf is authoritative, not the log
                events.append(
                    Counted(
                        product_id=p,
                        at=a,
                        amount=canon,
                        reason="implied_by_movement",
                        actor=cmd.actor,
                        occurred_at=cmd.at,
                    )
                )
            events.append(
                Moved(
                    product_id=p,
                    frm=a,
                    to=b,
                    amount=canon,
                    nominal_basis=cfg.basis(p, u),
                    actor=cmd.actor,
                    occurred_at=cmd.at,
                )
            )
            return events
```

Two things to note.

`near_matches` on a rejection is worth the twenty lines. `hob-right-below-bottom` would have been caught with `did you mean: hob-right-bottom?` and the whole incident would have been a typo fixed in three seconds.

**Insufficient stock is not a rejection.** For a pantry, the log is a *model* of the physical shelf and the shelf is authoritative — you will eat things without logging them. `decide` returns a `Counted` correcting the record, then the movement. An informational line is printed (`note: hob-right-bottom held 200g, recorded 340g — adjusted`); there is no flag and no prompt. A warning requiring `--force` would train exactly the habit of ignoring warnings, or of lying to the tool. This is why `decide` returns a *list*.

`decide` is pure: no I/O, no clock, no filesystem. State and config are arguments. This is what makes it testable and what keeps it from drifting away from the fold.

### 3.6 Corrections via supersedes

`Record.supersedes` already implements most of this: `ledger.load_records` drops any record targeted by another record's `supersedes` field before folding, which is append-only, byte-intact, and cross-actor-safe. Extend that mechanism rather than introducing a parallel `MovementVoided` event type — less schema surface, identical guarantees.

**Semantics, decided:**

1. **`supersedes` means cancel, not replace.** The targeted record is excluded from the fold — nothing more. The superseding record's own payload, if it has one, folds normally on its own merits. This gives cancel-only and replace-with-correction without a second concept: cancel-only is a record with no change payload, replace is a record that cancels and carries a correction.

2. **A content-free record gets its own payload type.** Don't make `payload` nullable — that forces a `None` branch into every `match` site forever. Add a third payload variant alongside `InventoryChange` and `InventorySnapshot`:

   ```python
   @dataclass(frozen=True, slots=True)
   class Correction:
       reason: str
       actor: str
   ```

   `evolve` treats it as a no-op on holdings. Note `Record.actor` already exists on the envelope for every record — check before implementing whether `Correction.actor` duplicates it or is meant to record something distinct (e.g. who is being overridden vs. who is overriding).

3. **Supersede claims are permanent.** If C supersedes A and A supersedes B, A and B both stay dead — A being superseded does not resurrect B. B was wrong; that's why A killed it. Undoing a correction should not resurrect an error as a side effect. This is monotone (a dead record never comes back), which keeps the fold predictable, and it's what the current one-level filtering in `ledger.load_records` already does — no fold-logic change needed for chains, only the `reason`/`actor` plumbing.

What needs adding: the `Correction` payload above, and the explicit rule that a supersede **may target a record in either segment but is always appended to the author's own segment**. Bob can supersede one of Alice's records by appending to `node.jsonl`; Alice's file is never touched, signatures stay intact, single-writer-per-file is preserved. There is no ownership restriction on *correcting*, only on *writing*.

`sumac correct <record-id> --reason "typo, location does not exist"` appends a `Correction` record with `supersedes` set to the target.

Two guards for the rejection catalogue (§4): a supersede whose target isn't in the log (`supersede_target_missing`), and a record that supersedes itself (`supersede_self`).

Where the truth has drifted rather than a specific event being wrong, `Counted(product_id, at, amount)` sets the holding absolutely rather than relatively. This is the right tool when you don't know which events were wrong, which for a pantry is most of the time.

### 3.7 Envelope

`seq` becomes **explicit envelope data**: a monotone integer per segment, written at append time, not inferred from line position. Two benefits beyond making the sort a pure function of event content:

- **Gap detection.** `seq` must be contiguous per actor. A gap means truncation; a duplicate means a bad merge. Both are silent today. Checked by `sumac doctor` (`seq_gap`).
- The v1 upcaster assigns `seq` from line position, which is correct because those files have never been reordered.

Merge order: concatenate both segments, sort by `(occurred_at, actor, seq)`, fold. Identical on both machines. Anything touching the envelope must preserve namespace and stream_id in the AAD, or old lines stop authenticating.

---

## 4. Rejection catalogue

Every one of these needs a unit test with a named case. This list is the acceptance criteria for phase 3.

| Rejection | Trigger |
|---|---|
| `unknown_location` | `from`/`to`/`at` not in active locations |
| `unknown_product` | `product_id` not in active products |
| `retired_location` | Location exists but is retired |
| `retired_product` | Product exists but is retired |
| `noop_move` | `from == to` |
| `unit_unconvertible` | No conversion path to product's canonical unit |
| `non_positive_amount` | `amount <= 0` |
| `circular_parent` | Config command would create a cycle |
| `unknown_parent` | Config command names a non-existent parent |
| `duplicate_id` | Location or product ID already in use |
| `retire_nonempty` | Retiring a location that still holds stock |
| `supersede_target_missing` | `correct` names an event ID not in the log |
| `supersede_already_applied` | Target already superseded |
| `supersede_self` | Record's `supersedes` names its own id |

Doctor-only diagnostics (not command rejections, deferred to Phase 7 — require `seq`, see §3.7): `seq_gap`, `seq_duplicate`.

---

## 5. Implementation plan

Ordered by dependency. Each phase is independently shippable and leaves the repo working.

**Phase 0 — Assess damage.** Read-only. A tolerant fold behind `sumac doctor` reporting every event it cannot apply, with reason and payload. No schema change, no writes.
*Deliverable:* the actual list of bad events in both segments.
*Acceptance:* runs to completion on the current log without raising.

**Phase 1 — Total fold.** Rewrite `evolve` per §3.1 with the `Anomaly` channel. Remove every raise, every bare dict index, every unit conversion from the read path. Wire the `sumac status` banner and non-zero exit.
*Acceptance:* `sumac status` and `sumac find` complete on any input, including a log of random bytes that fail decryption (those become anomalies too). Property test: `evolve` never raises for any event drawn from a permissive generator.

**Phase 2a — Location config.** Extract `build_config` with iterative cycle detection. Introduce `Retired` for locations; make deletion an alias for retirement. Split `cfg.active_locations` from `cfg.known_locations`.
*Acceptance:* a config containing a deliberate cycle yields a named diagnostic rather than an exception or hang. `known_locations` resolves retired entities; `active_locations` does not.

**Phase 2b — Product registry.** Build from scratch, mirroring `config.py`'s location handling: storage stream, `add_product`, `load_products`, `Retired` for products, `active_products` / `known_products`. This is new code, not an extension.
*Acceptance:* products round-trip through the config stream; retired products resolve in `known_products` only.

**Phase 2c — Canonical units.** Add `canonical_unit` and permitted conversions to product definitions, backfilled from each product's most common observed unit in the existing log.
*Acceptance:* every product in the current log has a canonical unit; `cfg.convert` is total over all observed (product, unit) pairs or reports which are unconvertible.

**Phase 3 — Extract `decide`.** Pure function per §3.5, taking `(cmd, inventory, config)`. The CLI write path becomes: parse → build state → `decide` → append. All of §4 implemented with `near_matches` suggestions. Conversion resolution moves here.
*Acceptance:* every row in §4 has a test asserting the specific rejection code. A grep for `raise` inside `evolve` returns nothing. A grep for `convert` inside `evolve` returns nothing.

Blocks on 2b and 2c — `unknown_product`, `retired_product`, and `unit_unconvertible` are untestable without the product registry.

Also add the **gate soundness** property here rather than deferring to Phase 6: no sequence of commands accepted by `decide` produces an anomaly. Exercising it while `decide` and `evolve` are freshly written catches drift immediately instead of four phases later.

**Phase 4a — Upcaster only, read path, no writes.** Ship `schema_v` on the envelope and the v1→v2 upcaster. **The writer still emits v1.** Nothing observable changes.
*Acceptance:* `fold(v1_events) == fold(upcast(v1_events))` on the actual log, compared as full inventory output, not spot checks. If this holds, the upcaster is correct against every event ever written.

**Phase 4b — Writer emits v2.** Only after 4a has run for a while. Split `Change` into the distinct types of §3.3. Fully reversible: revert the writer, the upcaster keeps handling both.
*Acceptance:* round-trip write-then-fold produces expected state for each new event type.

**Phase 5 — Corrections via supersedes.** Add the `Correction` payload type and wire `Record.supersedes` per §3.6 (semantics decided: cancel not replace, permanent claims). `sumac correct <record-id> --reason ...`. `sumac doctor` emits ready-to-paste correction commands for each anomaly.
*Acceptance:* the `hob-right-below-bottom` event is neutralised, `sumac status` reports zero anomalies, and `verify_all()` still passes on unmodified bytes.

**Phase 6 — Property tests.** Hypothesis, four properties (gate soundness moved to Phase 3):

1. **Totality** — `evolve` raises for no generated event sequence.
2. **Model agreement** — the fold matches an independent naive dict pantry that applies commands directly with no events. This catches shared arithmetic errors that gate soundness cannot, because `decide` and `evolve` can be consistently wrong together.
3. **Fold determinism** — same events, same order, same state, twice.
4. **Upcaster round-trip** — v1 corpus folds to the same state as its v2 translation.

```python
class PantryMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.log: list[Event] = []
        self.model: dict[tuple[str, str], Decimal] = {}  # independent ground truth
        self.cfg = test_config()

    @rule(p=products(), a=locations(), b=locations(), q=amounts())
    def move(self, p, a, b, q):
        try:
            events = decide(MoveCmd(p, a, b, q, "g"), self.fold(), self.cfg)
        except Rejected:
            return  # rejection is a legal outcome
        self.log += events
        for e in events:
            apply_to_model(self.model, e)

    @invariant()
    def fold_matches_model(self):
        assert self.fold().holdings == self.model

    @invariant()
    def no_anomalies_from_accepted_commands(self):
        assert self.fold().anomalies == ()
```

*Acceptance:* all five green at `max_examples=300`, `stateful_step_count=50`.

**Golden log — synthetic only.** The checked-in corpus is generated by the current writer under a fixed test key and fixed test config, covering every event type and every schema version. No production data, no real key in the repo. Separately, a local-only test folds the actual log and asserts the resulting inventory matches a checked-in *hash* — gated on the passphrase being present, skipped in CI. Real-data regression detection without committing real data.

**Phase 7 — Optional.** Two independent envelope additions, add whenever convenient — the value in both is having them in the historical record before you need them:
- **`seq`** (§3.7): explicit monotone integer per segment, written at append time. Enables `sumac doctor` to detect `seq_gap` (truncation) and `seq_duplicate` (bad merge) per segment.
- **`cmd_id`** (UUID per command), deduplicated during the fold. Protects against a pull ever replaying an event.

---

## 6. Notes for the implementing agent

- **Do not put validation in `evolve`.** This is the single most likely mistake. If a check can reject, it belongs in `decide`. If `evolve` encounters something it cannot handle, it records an `Anomaly` and returns unchanged state.
- **Do not convert units in `evolve`.** Conversions are resolved once in `decide` and frozen into the event. If `evolve` converts, editing a config definition retroactively changes historical inventory.
- **Do not modify existing log lines.** Not to fix them, not to re-encrypt them, not to reformat them. All repair is by append.
- **Do not migrate storage.** Per-writer encrypted JSONL under git is deliberate and correct. Do not introduce SQLite.
- **Corrections use `Record.supersedes`, not a new event type.** The mechanism already exists in `ledger.load_records`; extend it.
- **Never write to the other person's segment file.** Correcting another person's *event* is fine; the supersede goes in your own file.
- **Sort before folding.** Merge both segments and sort by `(occurred_at, actor, seq)`. Must be identical on both machines.
- **`decide` takes no I/O.** State and config are arguments. If it needs to read a file, the design is wrong.
- **Preserve the AAD binding.** Anything touching the envelope must keep namespace and stream_id in the associated data, or old lines stop authenticating.
- **Phase 4a must prove out before 4b.** No v2 event is written until the upcaster reproduces the current inventory exactly from the real log.
