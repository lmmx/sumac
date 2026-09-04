# sumac: Basmati Rice Unit-Mismatch Failure — Handoff

**Status:** diagnosis only — nothing implemented. Written as a handoff for a future session to
pick up and build, per explicit request not to build it alongside the trace/verdict redesign work
it was found through. See `docs/journal/2026-09-04-trace-and-verdict-redesign.md` for that
redesign and the verification run this diagnosis came from.

## Context

`evals/test_add.py::test_basmati_rice_in_different_unit` (`_CATEGORY = "add"`,
`evals/test_add.py:91-105`) already documents this as a known, deliberately-left-failing scenario
— its own docstring (`evals/test_add.py:92-98`) states the product/unit mismatch fails today
because `decide._resolve_product` has no registered bag-to-jug conversion, names
"accept-with-confirmation" as the preferable fix over a flat reject, and explicitly rules out
weakening the assertion or accepting a fabricated new product identity as ways to make it "pass."
None of that was previously visible in a run's own output — `scripts/verify-trace-redesign.sh`'s
10-epoch `qwen3.5-9b`/`default` run (`runs/epochs/verify-qwen3.5-9b-default/`) is the first time
the actual mechanism producing the 0/10 failure was inspectable in the trace itself, via this
session's `AgentRunner.messages` addition, rather than inferred from `checks`/`failures` alone.

## Diagnosis: the domain-layer rejection is correct, given the current schema

- `models.Product.conversions` (`src/sumac/models.py:53-56`) is an alt-unit-to-canonical-unit
  ratio, e.g. `{"jar": 340}` for a canonical unit of `"g"` — a fixed ratio, supplied by a human,
  per product.
- `Config.convert_with_basis` (`src/sumac/config.py:172-208`) returns `None` when the requested
  unit is neither the product's canonical unit nor a key in `conversions` — `decide._resolve_product`
  (`src/sumac/decide.py:151-156`) turns that `None` into `Rejected("unit_unconvertible", value=unit,
  expected=<canonical unit>)`.
- "Basmati Rice"'s canonical unit is `"jug"` — whatever unit it was first ever registered with
  ("the first-use unit trap, §3.5a — not worth solving now," `src/sumac/decide.py:184-188`), not a
  physical measure like grams. No fixed ratio from "bag" to "jug" has ever been supplied, and none
  could be pre-configured generically — a bag and a jug of rice aren't a fixed ratio the way jar-to-
  grams is.
- Confirmed against the real trace (`runs/epochs/verify-qwen3.5-9b-default/epoch-01.json`,
  `add.basmati_rice_in_different_unit`): the model calls `sumac_find_inventory("Basmati Rice")`,
  finds the existing stock at unit `"jug"`, then calls `sumac_discover_inventory(unit="bag")` —
  exactly what the prompt asked for — and gets `unit_unconvertible` back.

## Diagnosis: the orchestration layer turns one wrong rejection into a fabricated product

Reconstructed round by round from `log.messages` in the same trace, cross-referenced to the code
paths that produced each step:

1. `sumac_discover_inventory(unit="bag")` is rejected `unit_unconvertible` (above). `_rejected()`'s
   own hint text (`_REJECTION_HINT`, `src/sumac/llm.py:587-591`) already tells the model: "If a fix
   is obvious from this, retry with the correction... Otherwise explain why in plain text." The
   model's next round is plain text — three options laid out for the human, no further tool call —
   correctly following that hint. `_run_loop` (`src/sumac/llm.py:1032`) returns with `writes=()`.
2. `_maybe_force_action` (`src/sumac/llm.py:1097-1114`) only checks `plan.writes == ()` and
   `self._kind in (ADD, REMOVE)` — it has no visibility into *why* the round produced no writes, so
   it can't distinguish "the model is stalling" from "the model just followed `_REJECTION_HINT`'s
   own instruction to stop and explain." It appends `_EMPTY_PLAN_NUDGE` and forces another
   `_run_loop()` round regardless.
3. Forced to act, the model calls `sumac_discover_inventory(unit="jug")` — satisfies the schema
   constraint, silently drops what was actually requested. This succeeds
   (`status: "proposed"`), so `self._pending` now holds one write; the loop continues (a successful
   tool call doesn't end a round) and the next round is plain text, flagging the jug/bag mismatch
   itself. `_run_loop` returns `writes=(<the jug write>,)` — this is `_maybe_force_action`'s own
   `return self._run_loop()` (`src/sumac/llm.py:1114`).
4. `_maybe_self_review` (`src/sumac/llm.py:1116-1129`, `SELF_REVIEW_ROUNDS = 1`,
   `src/sumac/llm.py:152`) only checks `plan.writes` — non-empty now, so it fires. It appends
   `_SELF_REVIEW_MESSAGE` ("Check the plan above against the original request...") and calls
   `_run_loop()` a third time. `_run_loop`'s `self._pending = []` reset at its own top
   (`src/sumac/llm.py:1042`) discards the jug write before this round's own tool call can add a new
   one — no duplicate/conflicting write is ever at risk.
5. Reviewing the jug-substituted plan against the original "a bag... next to the existing jug"
   wording, the model resolves the conflict itself by calling
   `sumac_discover_inventory(product_id="Basmati Rice Bag", unit="bag")` — a fabricated new product
   identity that satisfies the schema (unit now matches a nonexistent product's own canonical unit)
   and the original wording (bag) at once. This succeeds, with `decide._resolve_product`'s
   auto-registration warning ("not a registered product — did you mean 'Basmati Rice'?",
   `src/sumac/decide.py:176-183`) attached. One more plain-text round confirms the plan; self-review
   exits (`SELF_REVIEW_ROUNDS = 1`, one iteration only).

Two separate orchestration mechanisms each push once — `_maybe_force_action` first,
`_maybe_self_review` second — neither individually produces the fabricated-product outcome; stacked,
they turn a correct clarifying question into a silent near-duplicate product.

## Recommended fix

**Primary, per the scenario's own docstring:** extend `decide._resolve_product`
(`src/sumac/decide.py:140-206`) so a known product's write in a unit that has no canonical match or
`conversions` entry is accepted with a confirmation-style warning — the same shape
`decide._resolve_product`'s existing auto-registration path already uses for an unknown product
(`src/sumac/decide.py:176-183`) — rather than raising `Rejected("unit_unconvertible", ...)`.
Product identity stays "Basmati Rice" in both units; what changes is how a second, non-convertible
unit's stock is tracked for one product_id, which needs its own design decision (a second
independent `Quantity` per (product_id, unit) pair is the shape the docstring implies — "the same
'Basmati Rice' can legitimately have both a jug and a bag registered" — but the storage/`ledger`
side of that is not scoped here). Doing this removes the failure at its source: the model's first,
unforced `sumac_discover_inventory(unit="bag")` call would simply succeed, and none of the
orchestration cascade above would ever trigger.

**Secondary, general-purpose, independent of the above:** `_maybe_force_action` forcing a retry
without knowing why the previous round produced no writes is a real gap beyond this one scenario —
any rejection reason that represents a genuine blocking constraint rather than a mechanically-
fixable mistake gets the same treatment. `src/sumac/decide.py`'s `Rejected` reasons split roughly
into "the model can just fix this and retry" (`missing_endpoint`, `non_positive_amount`,
`missing_reason`, `missing_required_argument`/`invalid_amount`/`tool_not_available` from
`src/sumac/llm.py`) and "this needs a human decision, not a retry" (`unit_unconvertible`,
`retired_product`, `retired_location`, plausibly `unknown_location`) — exact categorization is a
judgment call for the implementing session, not dictated here. Worth building regardless of whether
the primary fix lands, since it generalizes past this one scenario; not required to fix this
specific one if the primary fix removes the rejection entirely.

## Missing

- Nothing implemented this session — diagnosis and recommended fix only, by explicit request, kept
  separate from this session's trace/verdict redesign work.
- The storage-layer shape for "one product, two independently-tracked units" (primary fix) is
  unresolved — `ledger`'s inventory-by-location model was not audited for this entry.
- Whether to also build the secondary (`_maybe_force_action` rejection-awareness) fix, and the exact
  reason-category split if so, is unresolved.
