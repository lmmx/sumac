"""Canonicalisation, write-set F1, and trace/reply assertions — see
docs/journal/2026-09-02-eval-suite.md, "Scoring". Expectation types
(`NoWrites`, `Writes`, `AskOrAct`, `TraceExpectation`) live in `cases.py`;
this module is the pure functions that consume them, importing no pytest
name and no `sumac.llm` name beyond the `ProposedWrite`/`AgentPlan` shapes
it scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from sumac import config as sumac_config

# Explicit synonym table, not a stemming rule — stripping a trailing "s"
# maps "boxes" to "boxe", which fails to compare equal to "box" (the unit
# the new-product template uses). A unit outside this table compares
# verbatim.
UNIT_SYNONYMS: dict[str, str] = {
    "can": "can", "cans": "can",
    "jar": "jar", "jars": "jar",
    "carton": "carton", "cartons": "carton",
    "pack": "pack", "packs": "pack",
    "tub": "tub", "tubs": "tub",
    "box": "box", "boxes": "box",
    "bag": "bag", "bags": "bag",
    "bottle": "bottle", "bottles": "bottle",
    "tin": "tin", "tins": "tin",
    "jug": "jug", "jugs": "jug",
    "packet": "packet", "packets": "packet",
    "block": "block", "blocks": "block",
    "pouch": "pouch", "pouches": "pouch",
    "kg": "kg", "g": "g", "l": "l", "ml": "ml", "ct": "ct",
}  # fmt: skip


def canonical_unit(value: str) -> str:
    return UNIT_SYNONYMS.get(value.strip().lower(), value.strip().lower())


def canonical_location(cfg: sumac_config.Config, value: str | None) -> str | None:
    """Resolves a display path to its location id, passes an id through
    unchanged, and returns any other value unchanged — an unresolvable
    location then scores as a mismatch rather than raising, matching how
    `ProposedWrite.from_location`/`to_location` hold the model's raw,
    unresolved string (`src/sumac/llm.py:651-751`)."""
    if value is None:
        return None
    if value in cfg.known_locations:
        return value
    for loc_id in cfg.known_locations:
        if sumac_config.location_path(cfg.known_locations, loc_id) == value:
            return loc_id
    return value


@dataclass(frozen=True, slots=True)
class CanonicalWrite:
    kind: str
    product_id: str
    amount: Decimal
    unit: str
    from_location: str | None
    to_location: str | None


def canonicalize_proposed_write(pw: object, cfg: sumac_config.Config) -> CanonicalWrite:
    """`pw` is an `llm.ProposedWrite`; typed as `object` here so this module
    never needs to import `sumac.llm` (and therefore `mistralrs`) itself —
    every caller already has one in hand."""
    return CanonicalWrite(
        kind=str(pw.kind.value),  # ty: ignore[unresolved-attribute]
        product_id=str(pw.product_id).strip().lower(),  # ty: ignore[unresolved-attribute]
        amount=pw.amount,  # ty: ignore[unresolved-attribute]
        unit=canonical_unit(pw.unit),  # ty: ignore[unresolved-attribute]
        from_location=canonical_location(cfg, pw.from_location),  # ty: ignore[unresolved-attribute]
        to_location=canonical_location(cfg, pw.to_location),  # ty: ignore[unresolved-attribute]
    )


def canonicalize_write_spec(spec: object, cfg: sumac_config.Config) -> CanonicalWrite:
    """`spec` is a `cases.WriteSpec` — the hand-authored gold. Typed as
    `object` for the same reason as `canonicalize_proposed_write`."""
    return CanonicalWrite(
        kind=str(spec.kind.value),  # ty: ignore[unresolved-attribute]
        product_id=str(spec.product_id).strip().lower(),  # ty: ignore[unresolved-attribute]
        amount=Decimal(str(spec.amount)),  # ty: ignore[unresolved-attribute]
        unit=canonical_unit(spec.unit),  # ty: ignore[unresolved-attribute]
        from_location=canonical_location(cfg, spec.from_location),  # ty: ignore[unresolved-attribute]
        to_location=canonical_location(cfg, spec.to_location),  # ty: ignore[unresolved-attribute]
    )


@dataclass(frozen=True, slots=True)
class WriteScore:
    exact_match: bool
    f1: float
    actual: tuple[CanonicalWrite, ...]
    expected: tuple[CanonicalWrite, ...]


def score_no_writes(actual: tuple[object, ...]) -> WriteScore:
    ok = len(actual) == 0
    return WriteScore(exact_match=ok, f1=1.0 if ok else 0.0, actual=(), expected=())


def score_writes(
    actual: tuple[object, ...], expected_specs: tuple[object, ...], cfg: sumac_config.Config
) -> WriteScore:
    actual_c = tuple(canonicalize_proposed_write(w, cfg) for w in actual)
    expected_c = tuple(canonicalize_write_spec(s, cfg) for s in expected_specs)
    actual_set = set(actual_c)
    expected_set = set(expected_c)
    exact_match = actual_set == expected_set
    if not actual_set and not expected_set:
        f1 = 1.0
    elif not actual_set or not expected_set:
        f1 = 0.0
    else:
        tp = len(actual_set & expected_set)
        precision = tp / len(actual_set)
        recall = tp / len(expected_set)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return WriteScore(exact_match=exact_match, f1=f1, actual=actual_c, expected=expected_c)


# --- trace and reply assertions ----------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceCheckResult:
    ok: bool
    failures: tuple[str, ...]


def check_trace(
    trace: tuple[object, ...], reply_text: str, expectation: object
) -> TraceCheckResult:
    """`trace` is `AgentPlan.trace` (a tuple of `llm.ToolCallRecord`);
    `expectation` is a `cases.TraceExpectation`. Every check here is a
    deterministic assertion over `ToolCallRecord.name`/`.arguments`/
    `.result` and `AgentPlan.reply_text` — no second model call."""
    failures: list[str] = []
    call_names = [t.name for t in trace]  # ty: ignore[unresolved-attribute]

    for name in expectation.called:  # ty: ignore[unresolved-attribute]
        if name not in call_names:
            failures.append(f"expected {name!r} to be called, was not")

    for name in expectation.not_called:  # ty: ignore[unresolved-attribute]
        if name in call_names:
            failures.append(f"expected {name!r} not to be called, was")

    for name, limit in expectation.max_calls.items():  # ty: ignore[unresolved-attribute]
        count = call_names.count(name)
        if count > limit:
            failures.append(f"{name!r} called {count} times, expected at most {limit}")

    folded_reply = reply_text.casefold()

    for phrase in expectation.reply_mentions:  # ty: ignore[unresolved-attribute]
        if phrase.casefold() not in folded_reply:
            failures.append(f"reply does not mention {phrase!r}")

    for intended, decoy in expectation.reply_before:  # ty: ignore[unresolved-attribute]
        intended_idx = folded_reply.find(intended.casefold())
        decoy_idx = folded_reply.find(decoy.casefold())
        if intended_idx == -1:
            failures.append(f"reply does not mention {intended!r}")
        elif decoy_idx != -1 and decoy_idx < intended_idx:
            failures.append(f"reply mentions decoy {decoy!r} before intended {intended!r}")

    if expectation.reply_amount is not None:  # ty: ignore[unresolved-attribute]
        amount, unit = expectation.reply_amount  # ty: ignore[unresolved-attribute]
        pattern = re.compile(
            rf"\b{re.escape(str(amount))}\s*{re.escape(canonical_unit(unit))}s?\b",
            re.IGNORECASE,
        )
        if not pattern.search(reply_text):
            failures.append(f"reply does not name {amount} adjacent to a form of {unit!r}")

    return TraceCheckResult(ok=not failures, failures=tuple(failures))


# --- ask-versus-act -----------------------------------------------------------

_ASK_REPLY_MAX_LEN = 400


def classify_ask_or_act(plan: object) -> str:
    """`plan` is an `llm.AgentPlan`. Returns `"act"` (writes proposed),
    `"ask"` (empty writes, a question, no tool call beyond `find`, reply
    under a length bound), or `"inaction"` (empty writes matching neither —
    the one branch that should count as a failure). A bare question mark is
    not enough on its own: it conflates a genuine clarifying question with
    a reply that rambles without acting."""
    writes = plan.writes  # ty: ignore[unresolved-attribute]
    if writes:
        return "act"
    reply = plan.reply_text or ""  # ty: ignore[unresolved-attribute]
    trace = plan.trace  # ty: ignore[unresolved-attribute]
    domain_calls = [t.name for t in trace if t.name != "classify_request"]
    only_find = all(name == "sumac_find_inventory" for name in domain_calls)
    if "?" in reply and only_find and len(reply) <= _ASK_REPLY_MAX_LEN:
        return "ask"
    return "inaction"


# --- whole-case scoring --------------------------------------------------
# Shared by `test_scoring.py`'s null-baseline run, `test_proposals.py`'s
# real-model run, and `report.py` — one definition of what "passed" means,
# so a baseline can't look like it passes a `NoWrites` case by producing no
# writes while still being classified as the wrong kind entirely.


def actual_kind_value(plan: object) -> str | None:
    """The classified kind's raw string, read from `plan.trace[0]` (an
    `llm.ToolCallRecord`) — mirrors `AgentRunner._classify`'s own
    no-tool-call-means-reject fallback (`src/sumac/llm.py:807-824`) using
    only public `AgentPlan` surface, so this module never needs to import
    `sumac.llm` itself. `None` means `plan` didn't come from a real
    `propose()` call (no classify round in its trace at all)."""
    trace = plan.trace  # ty: ignore[unresolved-attribute]
    if not trace or trace[0].name != "classify_request":
        return None
    kind = trace[0].arguments.get("kind")
    return kind if kind is not None else "reject"


@dataclass(frozen=True, slots=True)
class CaseScore:
    kind_ok: bool
    outcome_ok: bool
    trace_ok: bool
    write_f1: float | None
    ask_or_act_branch: str | None
    trace_failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.kind_ok and self.outcome_ok and self.trace_ok


def score_case(case: object, plan: object, cfg: sumac_config.Config) -> CaseScore:
    """`case` is a `cases.EvalCase`. `passed` requires classification AND
    the write/trace outcome to both be correct — an agent that classifies
    `add.unit_collision` as `reject`, and therefore never writes anything,
    has not demonstrated the intended behaviour even though "no writes" is
    separately the right write-check outcome. Scoring write/trace without
    kind is exactly how a reject-everything baseline would otherwise look
    like it passes every `NoWrites` case regardless of kind — the null-floor
    failure this suite exists to avoid."""
    from evals.cases import AskOrAct, NoWrites, Writes

    expect = case.expect  # ty: ignore[unresolved-attribute]
    if isinstance(expect, NoWrites):
        result = score_no_writes(plan.writes)  # ty: ignore[unresolved-attribute]
        outcome_ok, write_f1, branch = result.exact_match, result.f1, None
    elif isinstance(expect, Writes):
        result = score_writes(plan.writes, expect.specs, cfg)  # ty: ignore[unresolved-attribute]
        outcome_ok, write_f1, branch = result.exact_match, result.f1, None
    elif isinstance(expect, AskOrAct):
        branch = classify_ask_or_act(plan)
        outcome_ok, write_f1 = branch in ("ask", "act"), None
    else:  # pragma: no cover - exhaustive over Expectation
        raise TypeError(f"unhandled expectation type: {type(expect)}")

    kind_ok = actual_kind_value(plan) == case.kind.value  # ty: ignore[unresolved-attribute]
    trace_result = check_trace(plan.trace, plan.reply_text, case.trace)  # ty: ignore[unresolved-attribute]

    return CaseScore(
        kind_ok=kind_ok,
        outcome_ok=outcome_ok,
        trace_ok=trace_result.ok,
        write_f1=write_f1,
        ask_or_act_branch=branch,
        trace_failures=trace_result.failures,
    )
