"""A small typed result and a handful of evaluator functions — the
assertions `test_agent.py` used to make inline, turned into named,
reusable checks rather than discarded. See
docs/journal/2026-09-02-eval-suite.md.

Each `evaluate_*` function takes an in-progress `EvalResult` and mutates it
in place via `result.check(name, ok, message)`, so a failure names which
dimension broke (classification, product, amount, unit, location, tools,
reply) rather than just that the scenario did. The `result` fixture in
`conftest.py` creates one per test, from the test's own name, and captures
it for the end-of-session summary regardless of whether the test's final
`assert result.passed` succeeds — a test that fails part way still reports
which checks it got right.

Deliberately not a scorer class hierarchy or a plugin system: six
functions and one dataclass, called directly from each test, in whatever
combination that scenario needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from sumac import config as sumac_config

UNIT_SYNONYMS: dict[str, str] = {
    "can": "can", "cans": "can",
    "jar": "jar", "jars": "jar",
    "carton": "carton", "cartons": "carton",
    "pack": "pack", "packs": "pack",
    "tub": "tub", "tubs": "tub",
    "box": "box", "boxes": "box",
    "bag": "bag", "bags": "bag",
    "bottle": "bottle", "bottles": "bottle",
    "jug": "jug", "jugs": "jug",
    "g": "g", "kg": "kg",
}  # fmt: skip


def _canon_unit(unit: str) -> str:
    return UNIT_SYNONYMS.get(unit.strip().lower(), unit.strip().lower())


def _canon_location(cfg: sumac_config.Config, value: str | None) -> str | None:
    """Resolves a display path to its location id, passes an id through
    unchanged, and returns any other value unchanged — an unresolvable
    location scores as a mismatch rather than raising, matching how
    `ProposedWrite.from_location`/`to_location` hold the model's raw,
    unresolved string."""
    if value is None:
        return None
    if value in cfg.known_locations:
        return value
    for loc_id in cfg.known_locations:
        if sumac_config.location_path(cfg.known_locations, loc_id) == value:
            return loc_id
    return value


@dataclass
class EvalResult:
    scenario: str
    category: str
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    note: str | None = None
    duration_s: float = 0.0
    tokens_per_sec: float | None = None

    @property
    def passed(self) -> bool:
        return not self.failures

    def check(self, name: str, ok: bool, message: str | None = None) -> None:
        self.checks[name] = ok
        if not ok:
            self.failures.append(message or f"{name} check failed")


def evaluate_classification(result: EvalResult, plan, expected) -> None:  # noqa: ANN001
    if not plan.trace or plan.trace[0].name != "classify_request":
        result.check("classification", False, "no classify_request round in the trace")
        return
    actual = plan.trace[0].arguments.get("kind")
    ok = actual == expected.value
    result.check(
        "classification",
        ok,
        None if ok else f"expected classified as {expected.value!r}, got {actual!r}",
    )


def evaluate_no_writes(result: EvalResult, plan) -> None:  # noqa: ANN001
    ok = plan.writes == ()
    result.check("writes", ok, None if ok else f"expected no writes, got {plan.writes!r}")


def evaluate_write(
    result: EvalResult,
    plan,  # noqa: ANN001
    cfg: sumac_config.Config,
    *,
    kind,  # noqa: ANN001
    product_id: str,
    amount: str | None = None,
    unit: str | None = None,
    to_location: str | None = None,
    from_location: str | None = None,
) -> None:
    """Exactly one write in `plan`, matching every field given. `amount`
    and `unit` are optional — omit them where the agent is expected to
    infer a plausible value rather than match one exactly (any positive
    amount and any non-empty unit then pass)."""
    if len(plan.writes) != 1:
        result.check("writes", False, f"expected exactly one write, got {plan.writes!r}")
        return
    result.check("writes", True)
    w = plan.writes[0]

    ok = w.kind == kind
    result.check("kind", ok, None if ok else f"expected kind={kind}, got {w.kind}")

    ok = w.product_id.strip().lower() == product_id.strip().lower()
    result.check(
        "product", ok, None if ok else f"expected product {product_id!r}, got {w.product_id!r}"
    )

    if amount is not None:
        ok = w.amount == Decimal(amount)
        result.check("amount", ok, None if ok else f"expected amount {amount!r}, got {w.amount!r}")
    else:
        result.check("amount", w.amount > 0, f"expected a positive amount, got {w.amount!r}")

    if unit is not None:
        ok = _canon_unit(w.unit) == _canon_unit(unit)
        result.check("unit", ok, None if ok else f"expected unit {unit!r}, got {w.unit!r}")
    else:
        result.check("unit", bool(w.unit.strip()), "expected a non-empty unit")

    if to_location is not None:
        resolved = _canon_location(cfg, w.to_location)
        ok = resolved == to_location
        msg = (
            f"expected to_location {to_location!r}, got {w.to_location!r} (resolved: {resolved!r})"
        )
        result.check("location", ok, None if ok else msg)
    if from_location is not None:
        resolved = _canon_location(cfg, w.from_location)
        ok = resolved == from_location
        msg = (
            f"expected from_location {from_location!r}, got {w.from_location!r} "
            f"(resolved: {resolved!r})"
        )
        result.check("location", ok, None if ok else msg)


def evaluate_tools(
    result: EvalResult,
    plan,  # noqa: ANN001
    *,
    called: tuple[str, ...] = (),
    at_most: dict[str, int] | None = None,
) -> None:
    trace_names = [t.name for t in plan.trace]
    for name in called:
        ok = name in trace_names
        result.check(f"tool:{name}", ok, None if ok else f"expected {name!r} to be called, wasn't")
    for name, limit in (at_most or {}).items():
        count = trace_names.count(name)
        ok = count <= limit
        result.check(
            f"tool:{name}:bound",
            ok,
            None if ok else f"{name!r} called {count} times, expected at most {limit}",
        )


def evaluate_only_tools(result: EvalResult, plan, allowed: set[str]) -> None:  # noqa: ANN001
    used = {t.name for t in plan.trace if t.name != "classify_request"}
    ok = used <= allowed
    msg = f"used tools outside {allowed}: {used - allowed}"
    result.check("tool_scope", ok, None if ok else msg)


def evaluate_reply_mentions(result: EvalResult, plan, phrase: str) -> None:  # noqa: ANN001
    ok = phrase.lower() in (plan.reply_text or "").lower()
    msg = f"reply doesn't mention {phrase!r}: {plan.reply_text!r}"
    result.check("reply", ok, None if ok else msg)


def evaluate_reply_order(result: EvalResult, plan, *, first: str, not_before: str) -> None:  # noqa: ANN001
    reply = (plan.reply_text or "").lower()
    first_idx = reply.find(first.lower())
    other_idx = reply.find(not_before.lower())
    if first_idx == -1:
        result.check("reply", False, f"reply doesn't mention {first!r}: {plan.reply_text!r}")
        return
    ok = other_idx == -1 or other_idx > first_idx
    result.check(
        "reply",
        ok,
        None if ok else f"{not_before!r} mentioned before {first!r}: {plan.reply_text!r}",
    )


_ASK_REPLY_MAX_LEN = 400


def evaluate_ask_or_act(result: EvalResult, plan) -> str:  # noqa: ANN001
    """Records an `"outcome"` check that passes for `"act"` (wrote
    something) or `"ask"` (empty writes, a question, no tool call beyond
    find, a short reply) and fails only for `"inaction"` (empty writes
    matching neither) — both a clarifying question and a correct action
    are acceptable outcomes for these scenarios. Returns the branch name
    and records it on `result.note`, so the summary can show which branch
    fired even on a pass."""
    if plan.writes:
        branch = "act"
    else:
        reply = plan.reply_text or ""
        domain_calls = [t.name for t in plan.trace if t.name != "classify_request"]
        only_find = all(name == "sumac_find_inventory" for name in domain_calls)
        branch = (
            "ask" if "?" in reply and only_find and len(reply) <= _ASK_REPLY_MAX_LEN else "inaction"
        )
    result.note = f"branch={branch}"
    ok = branch in ("ask", "act")
    result.check("outcome", ok, None if ok else f"neither asked nor acted (branch={branch})")
    return branch
