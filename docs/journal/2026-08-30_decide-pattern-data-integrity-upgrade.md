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

Also confirmed against the real log, via `sumac doctor` (Phase 0): every anomaly it found was `unknown_location` — no `unit_mismatch`, no malformed records. That breaks down into two distinct shapes, not one, which matters for remediation order (§5): a genuinely-missing location is fixed by registering it; a malformed reference to a location that already exists is not — registering the malformed string as a new location would create an orphan, so that shape can only be fixed once corrections exist (Phase 5). One of the malformed-reference cases is a display-path string pasted where an id belonged — see the `decide` note in §3.5.

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

### 3.3a Phase 4a design: the v1→v2 upcaster

The list above is illustrative, not a spec — no field types, no envelope layout, no v1→v2 mapping. Design pass before any code, per the Phase 3 review: handing an implementation phase an untyped sketch is how an upcaster goes subtly wrong against 107 real records that can't be rewritten.

**Real-data shape, checked before designing the mapping (aggregate counts, no content):**

| | |
|---|---|
| Live records | 107 (75 `snapshot`, 32 `change`) |
| `change` kind counts | `waste`=7, `purchase`=9, `consumption`=5, `movement`=5, `correction`=6, `discovery`=0 |
| Records with `supersedes` set | **0** |
| `correction` shape | 3× to-only (purchase-shaped), 3× from-only (consumption/waste-shaped) |
| Snapshot entry-count range | 0 to 34 entries per record; **19 of 75 have exactly 0 entries** |

Three things this changes about the design:

1. **`discovery` is unused** — real, but doesn't need real-data-driven design; the structural mapping (behaves like `purchase`, per `__post_init__`) is enough.
2. **`correction` never uses `supersedes`.** It isn't an early form of §3.6's correction mechanism — it's a plain delta, used both to add stock (3×) and remove it (3×), for accounting reasons rather than a real transaction. It has no home in the six-type list as written; see below.
3. **Snapshots are the dominant record type, not an edge case, and their shape varies enormously — including a legitimate zero-entry "this location is empty" assertion, 19 times.** This is the "one v1 record becomes N v2 events" case flagged for review, and the real distribution makes it the highest-risk part of this design, not a footnote.

**Snapshot → `Counted`: recommend against decomposing it.** The naive reading of the six-type list turns one `InventorySnapshot` (location, N entries) into N `Counted` events (product, location, amount). This has a real correctness problem: a 0-entry snapshot — 19 of them, real, not hypothetical — would upcast to **zero** events, silently losing the "this shelf was checked and is empty" fact. `InventorySnapshot`'s reset semantics ("resets rather than merges") mean an empty shelf must *clear* whatever the fold currently believes is there; an upcaster that emits nothing for a 0-entry record can't do that.

Making it correct requires the upcaster to know, for every non-empty product previously believed present at that location, that it's now absent — i.e. it needs *running fold state*, not just the one record being transformed. That turns the upcaster from a stateless per-record function into something that re-implements a chunk of the fold internally, for the single largest and most-varied record shape in the real log.

**Decided: don't decompose. `Snapshot` stays a location-wide v2 event; `Counted` is added as a new, separate, per-product primitive.** This supersedes §3.3's sketch (which implied decomposing a snapshot into one `Counted` per entry) — the empty-snapshot data-loss finding above is decisive on its own, and the 1:1 mapping that falls out of it (no synthesized ids, no supersedes-across-the-version-boundary question — see below) is worth more than the tidiness of a six-type list. `Snapshot` is "this is everything at this location, full stop" (what 70% of the real log actually is); `Counted` is "adjust this one product's count" (what §3.5's insufficient-stock behavior in Phase 4b needs, and what the real log's `correction` records — see below — turn out to actually be).

**`correction` → no clean fit in the six types; decided: a `reason` field.** A delta-shaped accounting adjustment isn't `Acquired`/`Consumed`/`Discarded` (those are real transactions) and isn't `Counted` (that's an absolute value, and `correction` never recorded one — computing an absolute count backwards from a historical running balance risks fabricating a number the original record never asserted, especially since the *reason* someone corrects the record is often that the running balance was already wrong). Decided: add `reason: str | None = None` to `Acquired` and `Consumed`, and map `correction` directionally — to-only → `Acquired(reason="correction")`, from-only → `Consumed(reason="correction")` — preserving "this wasn't a real purchase/use" without inventing data. `discovery` gets the same treatment: `Acquired(reason="discovery")`.

**Decided: `Consumed`, not `Discarded`, for a from-only `correction`.** `reason="correction"` is what actually carries the truth either way — this was cosmetic — but `Discarded` asserts the food was binned, a claim about the world the record doesn't support; `Consumed` is the weaker claim.

**v1 → v2 mapping table:**

| v1 `kind` / type | Shape | v2 event | Notes |
|---|---|---|---|
| `purchase` | to-only | `Acquired` | `reason=None` |
| `discovery` | to-only | `Acquired` | `reason="discovery"` |
| `correction`, to-only | to-only | `Acquired` | `reason="correction"` |
| `consumption` | from-only | `Consumed` | `reason=None` |
| `correction`, from-only | from-only | `Consumed` | `reason="correction"` |
| `waste` | from-only | `Discarded` | — |
| `movement` | from+to | `Moved` | — |
| `InventorySnapshot` | location + N entries | `Snapshot` | 1:1, entries carry over unchanged in shape (see `nominal_basis` below) |

`Counted` and `Retired` have no v1 source — both are v2-only, produced by Phase 4b's `decide`, never by this upcaster.

**v2 payload schemas, field by field:**

```python
@dataclass(frozen=True, slots=True)
class Acquired:
    product_id: str
    to: str  # location id
    amount: Decimal  # canonical units
    unit: str  # canonical unit, frozen at event time —
    # never re-derived from current config
    reason: str | None = None  # "correction" | "discovery" | None (ordinary purchase)
    nominal_basis: dict[str, str] | None = None
    # audit only, never read by the fold — e.g. {"raw_amount": "2", "raw_unit": "jar", "ratio": "340"}


@dataclass(frozen=True, slots=True)
class Consumed:
    product_id: str
    frm: str
    amount: Decimal
    unit: str
    reason: str | None = None  # "correction" | None (ordinary consumption)


@dataclass(frozen=True, slots=True)
class Discarded:
    product_id: str
    frm: str
    amount: Decimal
    unit: str


@dataclass(frozen=True, slots=True)
class Moved:
    product_id: str
    frm: str
    to: str
    amount: Decimal
    unit: str
    nominal_basis: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class Counted:
    product_id: str
    at: str
    amount: Decimal
    unit: str
    reason: str | None = None  # e.g. "implied_by_movement" (§3.5)
    # New in v2, not produced by the v1 upcaster — v1 never recorded a
    # single-product absolute count. Phase 4b's decide is the only producer.


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    product_id: str
    amount: Decimal
    unit: str
    nominal_basis: dict[str, str] | None = None
    # Decided: yes, entries carry this too — a physical count is often taken
    # in a countable unit ("4 jars"), so conversion is exactly as relevant
    # here as for a purchase, and it's free to add now vs. another schema
    # change once 4b is writing v2 for real.


@dataclass(frozen=True, slots=True)
class Snapshot:
    location_id: str
    entries: tuple[SnapshotEntry, ...]  # empty tuple is valid and means "empty"
    # Kept as its own v2 event rather than decomposed — see above.


@dataclass(frozen=True, slots=True)
class Retired:
    entity_type: Literal["location", "product"]
    entity_id: str
```

Envelope: no new field beyond the existing `schema_version` — "ship `schema_v`" in the original Phase 4a line reads, on reflection, as "make `schema_version` discrimination drive the upcast decision," not a second version field. `type` gains new values once the writer emits v2 in Phase 4b (`"acquired"`, `"consumed"`, …, replacing today's coarse `"change"`/`"snapshot"`); 4a's writer still only ever produces `"change"`/`"snapshot"` on disk, so this only matters for the in-memory v2 objects until 4b.

**Where the upcaster runs.** In `ledger._load`, after `RecordSchema.model_validate(obj).to_domain()` succeeds — operating on the already-validated **v1 domain object** (`models.Record`), not the raw JSON. No new pydantic ingest schema is needed for 4a: nothing on disk is v2-shaped yet (the writer still emits v1), so there's nothing to validate on ingest as v2 until Phase 4b. `upcast(v1_record: models.Record) -> list[V2Record]` is a pure domain-to-domain transform; `_load`'s `parsed.append(...)` becomes `parsed.extend(upcast(v1_record))`, since one v1 record can become more than one v2 record for... actually, given the `Snapshot`-stays-together decision above, every mapping in this design is 1:1 except none are 1:N anymore — the one case that would have been 1:N (`InventorySnapshot`) is exactly the case being kept together. Confirms the "don't decompose" call is also the simpler implementation, not just the safer one.

Record ids for upcast events: since every real mapping is 1:1, the v1 record's own id carries over unchanged — no synthesized ids, no id-collision bookkeeping, no special-casing for how `supersedes` resolves across the v1/v2 boundary. Another reason the "keep Snapshot together" call is worth taking: the alternative (1:N) would have needed a resolution story for exactly this, and the real data shows 0 records currently use `supersedes` at all — so this is untested territory either way, but 1:1 keeps it simple rather than open.

**Consequence for `evolve`.** Since every v1 record upcasts 1:1, `ledger._load` can upcast unconditionally and `evolve`/`build_inventory` only ever needs to fold v2 events — no permanent dual-shape fold. But *proving* the upcaster correct (the acceptance criterion below) needs both versions running side by side: today's fold (operating on `InventoryChange`/`InventorySnapshot` directly) stays as the reference implementation until the new v2 fold (operating on the types above via `match`) is shown to agree with it on the entire real log. Once proven, the v1-direct fold path is dead code and comes out — kept only long enough to be the thing being checked against.

**Acceptance criterion, clarified.** "`fold(v1_events) == fold(upcast(v1_events))`" as originally stated compares holdings only, which isn't enough: an upcaster that silently turns one of the real log's 24 current `unknown_location` anomalies into a clean fold — by, say, dropping an unresolvable location reference instead of preserving it — would still show correct *holdings* everywhere it succeeded, while quietly disappearing the anomaly. The comparison must cover **both** `Inventory.by_location` and `Inventory.anomalies` (by `(record_id, reason)`, not exact `detail` text — the detail string is allowed to read differently once records carry `frm`/`to` instead of `from_location`/`to_location`, as long as the *set* of anomalous records and their reasons is unchanged). Confirmed as part of this design pass, not left for the implementation to discover.

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
def decide(cmd: Command, s: Inventory, cfg: Config) -> list[Write]:
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
            if q <= 0:
                raise Rejected("non_positive_amount", value=q)

            writes = []
            product = cfg.active_products.get(p)
            if product is None:
                near = near_matches(p, cfg.active_products)
                if near:
                    warn(
                        f"{p!r} is not a registered product — did you mean {near[0]!r}? "
                        f"Registering {p!r} instead; correct it with `sumac correct` if it was a typo."
                    )
                # canonical unit = whatever this first command used — see §3.5a's
                # first-use unit trap; add-product redefines it later if wrong.
                writes.append(
                    Write(
                        "config",
                        Registered(
                            product_id=p,
                            unit=u,
                            metadata={"auto": True},
                            actor=cmd.actor,
                        ),
                    )
                )
                canon = q
            elif product.retired:
                raise Rejected("retired_product", value=p)
            else:
                canon = cfg.convert(p, q, u)
                if canon is None:
                    raise Rejected("unit_unconvertible", value=u, expected=cfg.unit_of(p))

            if s.at(p, a) < canon:  # shelf is authoritative, not the log
                writes.append(
                    Write(
                        f"log:{cmd.actor}",
                        Counted(
                            product_id=p,
                            at=a,
                            amount=canon,
                            reason="implied_by_movement",
                            actor=cmd.actor,
                            occurred_at=cmd.at,
                        ),
                    )
                )
            writes.append(
                Write(
                    f"log:{cmd.actor}",
                    Moved(
                        product_id=p,
                        frm=a,
                        to=b,
                        amount=canon,
                        nominal_basis=cfg.basis(p, u),
                        actor=cmd.actor,
                        occurred_at=cmd.at,
                    ),
                )
            )
            return writes
```

`Write(stream, payload)` targets `"config"` or `f"log:{actor}"` — the same `stream_id` shapes `store.py` already uses. Config writes are ordered first so a registration lands before the change that depends on it; the CLI write path appends the whole list, in order, to whichever stream each entry names.

Three things to note.

`near_matches` on a rejection is worth the twenty lines. `hob-right-below-bottom` would have been caught with `did you mean: hob-right-shelf-bottom?` and the whole incident would have been a typo fixed in three seconds. For an unknown *product*, the same lookup runs but only ever warns — see §3.5a for why this one auto-registers instead of rejecting.

**Display paths should resolve, not reject.** The real-data audit (§1) turned up a second, distinct failure shape alongside the typo: a value like `"Pantry > White Unit R1C3"` — exactly `config.location_path`'s output format — written into `--to`. That's not a misspelling, it's a display string pasted from `sumac status` where a raw id belonged, and `near_matches` would only ever offer a weak fuzzy guess at it. Since `location_path` is a pure function from id to string, `decide` can check for an exact reverse match first — cheap, because the format is already generated by that function — and resolve the write to the real id instead of rejecting it. Check this *before* `near_matches`, since a display-path match is exact where a typo match is fuzzy, and the two failure modes shouldn't be conflated.

**Insufficient stock is not a rejection.** For a pantry, the log is a *model* of the physical shelf and the shelf is authoritative — you will eat things without logging them. `decide` returns a `Counted` correcting the record, then the movement. An informational line is printed (`note: hob-right-bottom held 200g, recorded 340g — adjusted`); there is no flag and no prompt. A warning requiring `--force` would train exactly the habit of ignoring warnings, or of lying to the tool. This is why `decide` returns a *list*.

**Not implemented in Phase 3 as shipped: this specific behavior needs `Counted`, and `Counted` doesn't exist yet.** Today's schema only has `InventorySnapshot` (resets a *whole location*, every product on it) — there's no per-product absolute-count event to correct just the one mismatched item without also having to know and restate everything else on that shelf. `Counted` is part of the distinct-event-types schema change in §3.3, which is Phase 4b. `decide_change` (`src/sumac/decide.py`) covers everything else in this section — location/product resolution, `near_matches`, display-path resolution, auto-register — against today's `InventoryChange` shape; the shelf-is-authoritative auto-correction is deferred until Phase 4b lands, not silently dropped.

`decide` is pure: no I/O, no clock, no filesystem. State and config are arguments. This is what makes it testable and what keeps it from drifting away from the fold.

### 3.5a Bootstrapping the product registry

§4's original rejection catalogue made `unknown_product` a hard rejection against `active_products`, mirroring `unknown_location`. That set is currently empty against the real log — Phase 2c's `check-units` (report-only, see the note on Phase 2c's acceptance below) found **472 distinct product ids** with no registry entry. Shipping that as specified, unmodified, means every `sumac add` fails the moment the gate turns on, until all 472 are registered by hand. **This is a bigger version of the location problem** (§1's 24 `unknown_location` anomalies) and needed a decision here, not a surprise during Phase 3 — the decision below is why `unknown_product` no longer appears in §4's table, and why the `decide` sketch in §3.5 auto-registers instead of rejecting.

**First: is it really 472 things, or ID drift?** Free-text product ids across a year of entries could mean `milk` / `Milk` / `whole-milk` are recorded as three unrelated products when fewer real ones are involved. Checked against the real log (aggregate only — counts, no ids):

- 472 distinct raw product ids observed.
- Only **3 pairs** collapse under normalization (case-fold, strip `-`/`_`/space, crude trailing-`s` fold) — 6 raw ids out of 472.
- **353 of 472 (75%) appear exactly once** in the entire log.

So it is not primarily an ID-hygiene problem — normalization barely moves the number (472 → 469 distinct). This is a household that genuinely buys ~469 distinct things, three-quarters of which were purchased once. That shape matters for which bootstrap strategy fits: a one-time cleanup solves a backlog; it does not solve a pattern that keeps generating new one-off product ids indefinitely, and this household's actual usage is the latter.

**Three strategies:**

1. **Auto-register on first use.** `decide` resolves an unknown product by emitting a registration alongside the change event, canonical unit = the unit used in the command. Self-sustaining — matches the observed long-tail-of-one-offs shape exactly, since new one-off products will keep appearing and each needs to clear the gate the moment it's bought, not in a batch later. Downside: a typo becomes a permanently registered "product" unless caught, so it needs a `near_matches` *warning* alongside the auto-registration (not a rejection — the point is the write still succeeds) so the household can catch and `sumac correct` a typo shortly after, rather than being blocked by one.

2. **`sumac config check-units --write`: bulk backfill.** Clears today's 472 in one pass. Does not address that 75% of products are one-off — the backlog re-forms the moment a new item is bought, so this becomes a repeating chore rather than a one-time migration, without `near_matches` typo protection at write time either. Would be the better fit if the shape were "12 staples with spelling drift"; it isn't.

3. **Demote `unknown_product` to a warning, promote later.** Defers the pain rather than resolving it, and needs a manual "flag day" to turn the gate back on. Given the registry never naturally reaches "populated" under this usage pattern, there's no natural moment to promote it.

**Decided: option 1, auto-register with a `near_matches` warning.** The real-data shape — long tail of one-offs, not spelling drift — is what makes "self-sustaining" the deciding property rather than "one-time cleanup."

**Architecture: `decide` returns writes for both streams, not just the log's.** `decide` today returns `list[Event]` implicitly meaning "log stream." Two ways to add a config-stream write for the registration: (a) widen `decide`'s return type so it can carry either kind, still emitted from the one pure call; (b) have the CLI write path call `config.add_product` itself when `decide` reports `unknown_product` as auto-registerable, then retry `decide`. Rejected (b): it moves the "safe to auto-register" decision outside `decide`, and opens a partial-failure window — the product registers but the process dies before the change appends, leaving a registered product with no corresponding movement and no way to tell that happened from a crash log alone. Taking (a): `decide` returns `list[Write]`, where a `Write` carries a stream target (`"config"` or `"log:<actor>"`) and a payload; config writes are ordered first in the list so a registration lands before the change that depends on it, and the CLI write path appends the whole list in order. `decide` still does no I/O — only its return type got richer.

Two things worth writing down now so neither gets rediscovered as a bug later:

- **The first-use unit trap.** Auto-register takes canonical unit from whichever command hits it first. Buying "1 jar of rice pudding" before ever recording grams registers `jar` as canonical, permanently, until someone notices. Not worth solving now: `sumac config add-product` redefines it (latest-revision-wins already handles superseding a canonical-unit choice), and `Counted` corrects whatever quantities drifted from the wrong initial conversion basis in the meantime.
- **Auto-registrations must be marked, not indistinguishable from deliberate ones.** Every auto-registered product carries `metadata: {"auto": true}` (and the usual `actor` on the envelope, naming who triggered it, not who confirmed it). This is the typo backstop `near_matches`'s *warning* (rather than rejection) needs: `check-units` can later report "N products auto-registered and never confirmed," giving a household a list to review instead of relying on catching every warning in the moment it's printed.

### 3.6 Corrections via supersedes

`Record.supersedes` already implements most of this: `ledger.load_records` drops any record targeted by another record's `supersedes` field before folding, which is append-only, byte-intact, and cross-actor-safe. Extend that mechanism rather than introducing a parallel `MovementVoided` event type — less schema surface, identical guarantees.

**Semantics, decided:**

1. **`supersedes` means cancel, not replace.** The targeted record is excluded from the fold — nothing more. The superseding record's own payload, if it has one, folds normally on its own merits. This gives cancel-only and replace-with-correction without a second concept: cancel-only is a record with no change payload, replace is a record that cancels and carries a correction.

2. **A content-free record gets its own payload type.** Don't make `payload` nullable — that forces a `None` branch into every `match` site forever. Add a third payload variant alongside `InventoryChange` and `InventorySnapshot`:

   ```python
   @dataclass(frozen=True, slots=True)
   class Correction:
       reason: str
   ```

   `evolve` treats it as a no-op on holdings. `actor` is deliberately not here — `Record.actor` already carries it on every envelope, and duplicating it invites the two drifting apart.

3. **Supersede claims are permanent.** If C supersedes A and A supersedes B, A and B both stay dead — A being superseded does not resurrect B. B was wrong; that's why A killed it. Undoing a correction should not resurrect an error as a side effect. This is monotone (a dead record never comes back), which keeps the fold predictable, and it's what the current one-level filtering in `ledger.load_records` already does — no fold-logic change needed for chains, only the `reason` plumbing.

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

Every one of these needs a unit test with a named case. This list is the acceptance criteria for phase 3, except `retire_nonempty`, which shipped in Phase 2a (see the note there) — its test lives in `test_cli.py`, not the `decide` suite.

`unknown_product` is deliberately **not** in this table — §3.5a decided auto-register-with-a-warning over rejection, once the real-data audit showed a long tail of one-off products rather than a fixable backlog. `retired_product` stays a hard rejection: retiring is a deliberate signal to stop tracking something, and auto-registering around it would defeat that on the first purchase after.

| Rejection | Trigger |
|---|---|
| `unknown_location` | `from`/`to`/`at` not in active locations |
| `retired_location` | Location exists but is retired |
| `retired_product` | Product exists but is retired |
| `noop_move` | `from == to` |
| `missing_endpoint` | `from`/`to` absent for what `kind` requires — not in the original catalogue, found by the gate soundness property test (`InventoryChange.__post_init__` raises `ValueError`, not `Rejected`, for this; had to be caught and converted) |
| `unit_unconvertible` | No conversion path to product's canonical unit |
| `non_positive_amount` | `amount <= 0` |
| `circular_parent` | Config command would create a cycle |
| `unknown_parent` | Config command names a non-existent parent |
| `duplicate_id` | Location or product ID already in use |
| `retire_nonempty` | Retiring a location that still holds stock (Phase 2a, ahead of the rest of this catalogue) |
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

`retire_nonempty` (§4) shipped here rather than waiting for Phase 3: it doesn't need `decide`'s Command/Event/`Rejected`/`near_matches` machinery, just a holdings check ahead of the write, so there was no reason to leave the CLI able to retire a location that still holds stock in the interim. It checks the named location's own holdings only — a parent with empty holdings of its own retires fine even if a sub-location still holds stock, since each sub-location is retired (and checked) independently. Lives in `cli.py`, not `config.py`: `config` must not depend on `ledger` (the reverse dependency already exists), and this check needs a fold.

**Phase 2b — Product registry.** Build from scratch, mirroring `config.py`'s location handling: storage stream, `add_product`, `load_products`, `Retired` for products, `active_products` / `known_products`. This is new code, not an extension.
*Acceptance:* products round-trip through the config stream; retired products resolve in `known_products` only.

**Phase 2c — Canonical units.** `Product.conversions` and `Config.convert`/`can_convert` shipped. `sumac config check-units` reports, for every (product, unit) pair observed in the log, whether it converts — a suggested `add-product` command for an unregistered product, a named gap for a registered one whose observed unit doesn't convert.
*Acceptance, as shipped:* `cfg.convert` is total over all observed (product, unit) pairs *or* `check-units` reports which are unconvertible — **met**. "Every product in the current log has a canonical unit" — **not met**: `check-units` is a report, not a backfill, and zero of the 472 products it found are registered. Registering them is deliberately not done yet; it's blocked on the bootstrap-strategy decision in §3.5a, which affects *how* they get registered (auto vs. bulk vs. deferred) and shouldn't be pre-empted by backfilling ad hoc.

**Update after Phase 3:** Phase 3's auto-register means `check-units`'s unregistered/unconvertible sections are now legacy-data-only — `sumac add` can no longer produce either, so the only way to see them is data written before `decide` existed (the real 472, still unregistered). `check-units` gained a third, now-primary job: reporting `{"auto": true}` products that haven't since been confirmed by a deliberate `add-product` (which clears the flag) — this is the `near_matches`-warning backstop §3.5a called for, and until this landed nothing implemented it.

**Phase 3 — Extract `decide`.** Pure function, `decide_change` in `src/sumac/decide.py`, taking `(kind, product_id, amount, unit, from_location, to_location, actor, occurred_at, cfg)` — no `inventory` argument (see the note under §3.5's "insufficient stock" paragraph: the shelf-is-authoritative auto-`Counted` behavior needs a per-product count event that doesn't exist until Phase 4b, so nothing here reads current holdings yet). Returns `list[Write]` per §3.5a's shape-(a) decision — `decide` still does no I/O. Wired into `sumac add`, replacing its direct `InventoryChange` construction; `cli.py`'s `_change_to_obj` helper is gone, `decide_change` builds the record body itself.
*Acceptance, as shipped:* every row in §4 *except* the config-command rejections (`circular_parent`, `unknown_parent`, `duplicate_id` — `add-location`/`add-product` don't validate against `decide` yet, left for a follow-up; lower-stakes than the `add` gate this phase exists for) has a test asserting the specific rejection code. `unknown_product` isn't in §4 (§3.5a) but is covered — auto-register, near-match warning, ordering. Gate soundness property test added (below), including a real bug it found. A grep for `raise` inside `evolve` still returns nothing — `decide` didn't touch `ledger.py`. A grep for `convert` inside `evolve` still returns nothing.

**Gate soundness, pulled forward from Phase 6 per the earlier review:** no sequence of commands accepted by `decide` produces an anomaly — `tests/test_decide_properties.py`, Hypothesis, in-memory (decide is pure, no files/crypto needed). It found a real bug on first run: `InventoryChange.__post_init__` raises a bare `ValueError` for a missing endpoint (e.g. `purchase` with no `--to`), which isn't a `SumacError` — uncaught, `cli.main()`'s handler wouldn't have caught it, so that command would have crashed with a raw traceback instead of a clean rejection. Fixed by catching it and raising `Rejected("missing_endpoint", ...)`; now in §4's table.

**Phase 4a — shipped.** `sumac/events.py` (the v2 types), `sumac/upcast.py` (the v1→v2 mapping), `ledger.py`'s fold rewritten to operate on `events.Event` exclusively, fed by an upcast pass in `_load_v2`. **The writer still emits v1**; `sumac add`/`decide.py` untouched. `load_records` (used by `sumac log`) still returns v1 records unchanged — only `build_inventory` upcasts.
*Acceptance, per §3.3a — met:* compared holdings and the anomaly set (by `(record_id, reason)`) between the pre-4a fold (operating on v1 records directly, extracted from git history rather than kept live in the tree) and the new fold (operating on upcast v2 events), on the real log. Holdings matched exactly (429 `(location, product)` pairs both sides); anomalies matched exactly (24 anomalies, 23 distinct `(record_id, reason)` keys both sides — the one repeat is the same movement flagged for both a bad `from` and a bad `to`, already known from §1). One intentional, real difference found and confirmed harmless by the same real-data audit that shaped §3.3a: a `correction` with *both* endpoints set is a structurally-possible-but-unmapped shape (`InventoryChange.__post_init__` doesn't constrain `correction` the way it does every other kind) — the old fold would have silently applied it as a two-sided movement, the new one quarantines it as `upcast_failed`. Zero real records hit this; caught by a synthetic negative-control test before running against the real log, not by the real log itself.

**Phase 4b — Writer emits v2.** Only after 4a has run for a while. `decide` constructs the v2 types directly (§3.3a already defines them; this phase wires the write path to them and adds the pydantic ingest schema v2 JSON needs once it's actually on disk). Fully reversible: revert the writer, the upcaster keeps handling both.
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
