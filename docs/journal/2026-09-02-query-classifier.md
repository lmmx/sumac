# sumac: Query Classifier for `sumac ask` — Routing to Task-Specific Prompts

**Status:** implemented, untested against a real model
**Scope:** `sumac/llm.py` only — no change to `decide.py`, `ledger.py`, `store.py`, or `cli.py`

---

## Motivation

A real run on 2026-09-01 (`sumac ask --dry-run "Add 6 Moma pistachio milk cartons to the same
pantry cupboard as existing stock"`, `LiquidAI/LFM2.5-2.6B-GGUF`) made four `sumac_find_inventory`
calls — `"Moma pistachio milk cartons"`, `"Moma milk cartons"`, `"Moma"`, `"milk"` — every one a
reasonable narrowing of the request, and then gave up entirely: "Since 'Moma pistachio milk
cartons' are not currently in the inventory, I cannot add them," followed by three offered
next steps for the human to pick from, no tool call proposed. The `"milk"` search had already
turned up `Pistachio Oat Milk`, a plausible match for what was being asked for even though the
brand name didn't match — the model never acted on it, in either direction (proposing it as a
close match, or proceeding to record a new product under a new name via `sumac_discover_inventory`,
which does not require a prior match at all).

The previous session's response treated this as a prompt-content gap — `SYSTEM_PROMPT` never
stated that a `sumac_discover_inventory` call doesn't need a prior `sumac_find_inventory` match —
and shipped a short addition saying so. That fix is real but incomplete as a diagnosis: every
real-model failure recorded in `docs/journal/2026-09-01-ask-agent-design.md` (a duplicate search,
a wrong-product answer from unranked results, a model skipping search on a plain question, and now
this) traces back to the same root cause — one `SYSTEM_PROMPT` asking a 1-3B model to simultaneously
hold search-relevance judgment, new-vs-existing-product reasoning, and consume-vs-move
disambiguation in mind for every request, regardless of which of those the request actually needed.
Longer, more heavily qualified prompts on a model this size have degraded tool-calling reliability
directly in every real run so far; adding another qualifying paragraph for the Moma case would
have made the same prompt longer again.

The alternative, and what this entry implements: split handling into two calls. A first, cheap
call decides which of a small set of task shapes the request is — find, add, remove, or reject —
using a prompt short enough that classification is close to free. A second call then runs against
a prompt and tool set built for that one task only, with no competing instructions left over from
the other two. Not a single zero-shot call asked to do everything at once; a small router in front
of small specialists.

## Design

**`QueryKind`** (`llm.py`): `FIND`, `ADD`, `REMOVE`, `REJECT`.

**Classification is a tool call, not parsed free text.** `classify_request(kind: enum)` is a
`strict: true` schema with `kind` constrained to the four values above, dispatched through the
same client-side tool-calling mechanism the domain tools already use — reusing strict JSON-schema
decoding rather than asking the model to say "find" in plain text and parsing whatever comes back.

**Each kind gets its own system prompt and its own tool schema subset**, sent on the request
instead of the full four-tool set every request used to carry:

- `find` → `sumac_find_inventory` only. Keeps the search-relevance judgment prose from the old
  single prompt (exact match is strong evidence but doesn't rule out a modified variant; a shared
  word in an unrelated product's name usually does rule it out) — this is specific to search and
  doesn't distract a request that will never call consume/move/discover.
- `add` → `sumac_find_inventory` + `sumac_discover_inventory`. States directly that a `find`
  turning up nothing is expected and fine for a new product, and that `find` is only needed at all
  to resolve an indirect location reference. This is the fix the previous session made, now
  unencumbered by remove-specific and find-specific reasoning competing for the same tokens.
- `remove` → `sumac_find_inventory` + `sumac_consume_inventory` + `sumac_move_inventory`. Must
  find first (both consume and move need the same product/location/amount/unit resolved that way);
  then chooses consume vs. move by whether the person names a destination.
- `reject` → a fixed reply, no domain tool call made and no further model call at all. The point of
  classifying first is that a request outside the three supported shapes never reaches, and never
  spends tokens on, a prompt built for a task it isn't.

**Boundary decision made in this entry, not dictated:** "move" is folded into the `remove` prompt
rather than made a fourth top-level kind. The classification the user described for `remove` —
"which will only work if we can find the item" — applies identically to consume and move; both
need the same `find` step first, and choosing between them only matters after that search result
is in hand. Putting that choice inside `_REMOVE_PROMPT` keeps the classifier itself simpler (three
real kinds plus reject, not four) at the cost of one more decision inside the remove-scoped prompt.
Flagged explicitly since a real move-shaped request hasn't been run against this yet (see Missing,
below) — worth revisiting as its own kind if that decision turns out to need more than the current
one sentence gives it.

**Tool dispatch checks the current kind's allowed names before calling a domain callback.** A
`find`-classified request only ever has `sumac_find_inventory` on the request, but a small model
can still emit a call for a tool it wasn't given a schema for. `AgentRunner._run_loop` now returns
a `{"status": "rejected", "reason": "tool_not_available", ...}` tool result for that case instead
of crashing (`KeyError`) or dispatching a domain write outside the classified kind's scope.

**`revise()` does not reclassify.** Feedback on an already-routed plan continues within the same
kind, using the same scoped tools — only `propose()` picks a kind, once, at the start of a request.

**The classification call shows up in `AgentPlan.trace` like any other tool call**, so
`sumac ask --debug` and the trace table already built for domain calls show which kind a request
was routed to, for free — no new rendering code needed.

## What this does not change

The domain layer (`decide.py`, `ledger.py`, `store.py`) is untouched: no new `ChangeKind`, no new
validation path, the same four `_sumac_*` tool callbacks. The propose → self-review → revise →
commit lifecycle and the dry-run mechanism from the original design are unchanged — routing happens
once, at the very start of `propose()`, and everything downstream of it is exactly what it was.

## Not yet done / open

- Untested against a real model, same caveat as every claim in the previous entry. The whole
  premise — that three small, focused prompts route and perform better than one large one — is a
  hypothesis until run against the Moma pistachio milk request and the earlier documented failures
  (the butter-ranking bug, the duplicate-call bug, the "where is the water?" non-search) that
  motivated this redesign.
- Nothing resolves a misclassified request without the human re-issuing `sumac ask` from scratch —
  `revise()` cannot change kind once `propose()` has picked one.
- The `move`-vs-`remove` boundary above is a design decision made in this entry, not requested
  explicitly by the user — unconfirmed against a real move-shaped request ("move the ragu to the
  fridge").
- `README.md` documents `sumac ask` only in terms of the single-prompt design and does not mention
  query classification or per-kind prompts.

## A note on citation style

This entry, and this session's edits to `sumac/llm.py`, refer back to
`docs/journal/2026-09-01-ask-agent-design.md` by filename rather than by section number (`§N`).
That entry's dense inline section citations made ordinary code comments read like citations to
scripture rather than pointers to documentation — a filename, and a section heading where it
matters which part, is enough for a reader to find the source without a parallel numbering scheme
to hold in mind. This entry doesn't introduce its own `§N` scheme either. Existing `§N` references
elsewhere in the repository are left as they are — retrofitting them is a separate cleanup, not
part of this change.

---

# 2026-09-02: Query Classifier Implementation State

## Current State

- `sumac ask` routes every request through `QueryKind` classification (`AgentRunner._classify`,
  `llm.py`) before running the domain tool-calling loop. `AgentRunner.propose` sets `self._kind`,
  `self._schemas` (`_SCHEMAS_BY_KIND[kind]`), and `self._allowed` (`_TOOL_NAMES_BY_KIND[kind]`)
  from the classification result before building the first domain request.
- Three per-kind system prompts (`_FIND_PROMPT`, `_ADD_PROMPT`, `_REMOVE_PROMPT`, `llm.py`) replace
  the single `SYSTEM_PROMPT` the previous design used; each is shorter than that single prompt was.
- `classify_request` (`_CLASSIFY_SCHEMA`, `llm.py`) is a `strict: true` tool schema with a
  four-value enum (`find`/`add`/`remove`/`reject`), dispatched through the same client-side loop
  mechanism as the domain tools.
- A `reject`-classified request returns a fixed reply (`_REJECT_REPLY`) with no domain tool call
  and no further model call; `AgentRunner.revise` raises if called after a rejected `propose`
  (`self._messages` is `None` in that case, same as before `propose` was ever called).
- `AgentRunner._run_loop` rejects, as a tool result rather than a crash, a tool call outside the
  current kind's allowed set (`self._allowed`) — new in this entry; the single-prompt design always
  had every tool available, so this case could not previously arise.
- `tests/test_llm.py` (25 tests) covers the classifier and the routing it drives: reject
  short-circuiting the domain loop entirely, `revise` after a rejection raising, each kind's tool
  scoping (`agent._allowed`), a same-kind self-review round-trip, and a tool call naming something
  outside the current kind's scope being rejected rather than dispatched. Every existing test that
  drives `propose()` now scripts a leading `classify_request` round via a new `_classify_round(kind)`
  helper. `tests/test_cli.py` (54 tests) is unaffected — its `_FakeAgentRunner` stands in for the
  whole `AgentRunner`, not `mistralrs.Runner`, so it has no visibility into kinds at all.
- `pytest` reports 251 passing tests repo-wide; `ruff format --check .`, `ruff check .`, and
  `ty check` each report no findings.

## Stubbed

- None found.

## Missing

- No automated test exercises classification against a real model — `tests/test_llm.py`'s classify
  tests all script the `classify_request` tool call directly through `FakeRunner`, the same
  limitation the previous entry recorded for the rest of the agent loop.
- No real-model run of any kind (find/add/remove) has been made against this routed design — the
  Moma pistachio milk request that motivated it has not yet been re-run.
- Nothing resolves a misclassified request without the human re-issuing `sumac ask` from scratch.
- No test or real-model run has exercised a move-shaped request against `_REMOVE_PROMPT`
  specifically, as opposed to a consume-shaped one — the consume-vs-move disambiguation this entry
  places inside that one prompt is unconfirmed.
- `README.md` still documents `sumac ask` only in terms of the single-prompt design and does not
  mention query classification or per-kind prompts.

## Divergence

- None found against `README.md`, which makes no specific claims about the prompt/tool-scoping
  mechanism to diverge from.
