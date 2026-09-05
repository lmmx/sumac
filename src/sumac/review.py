"""Deterministic checks over a proposed `sumac ask` plan, for the person
about to accept it.

Every check here reads only the plan, the current `Config`, and the plan's
own tool-call trace: no model call, and no second opinion from the model
being checked. That has two consequences. It adds no latency to a decision
someone is already waiting on, and it is reproducible, which anything routed
back through the model is not — `docs/journal/2026-09-04-trace-and-verdict-redesign.md`
records why (a `mistralrs.Runner`'s RNG stream position depends on everything
that ran before it in the same session).

The check this module exists for is `ungrounded`.
`docs/journal/2026-09-04-basmati-rice-unit-mismatch.md` traces an eval failure
to a plan whose `product_id` — "Basmati Rice Bag" — was in no search result and
in no config record, produced by the model to satisfy two constraints at once
after `_maybe_force_action` and `_maybe_self_review` each forced a further
round. The information needed to detect that was already in `AgentPlan.trace`
when the plan was displayed; nothing read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sumac import config, decide

if TYPE_CHECKING:
    from sumac.llm import AgentPlan, ProposedWrite, ToolCallRecord

# Tool results the vault supplied, as opposed to ones echoing what the model
# just asked for. `_propose_write`'s result JSON repeats the `product_id` it
# was called with, so a grounding check reading every trace entry would treat
# any invented id as grounded in the record of the call that invented it.
READ_TOOLS = frozenset({"sumac_find_inventory"})


@dataclass(frozen=True, slots=True)
class Finding:
    """One observation about a single proposed write. `label` is the badge
    shown on the write's own row; `detail` is the line under it."""

    code: str
    label: str
    detail: str
    # Whether `detail` is printed under the write, or the badge appears
    # alone. `decide` attaches its own warning text to a write it
    # auto-registered or recorded in an unconvertible unit
    # (`_resolve_product`), and `render.print_plan` prints those, so a finding
    # restating one would duplicate the same sentence. Only findings `decide`
    # cannot make are printed: `decide` never sees the tool-call trace, so
    # `ungrounded` is one it cannot produce.
    explain: bool = False


def _searched_text(trace: tuple[ToolCallRecord, ...]) -> str:
    return "\n".join(t.result for t in trace if t.name in READ_TOOLS).lower()


def review_write(write: ProposedWrite, cfg: config.Config, searched: str) -> tuple[Finding, ...]:
    """`searched` is the concatenated lowercase text of every read-tool
    result in the plan's trace — passed in rather than recomputed per write,
    since one plan's writes all share one trace."""
    findings: list[Finding] = []
    product = cfg.known_products.get(write.product_id)

    if product is None:
        # `ungrounded` reports a product id the model produced without a
        # source. A product id the person typed into the edit menu has a
        # source, so it is excluded; `new-product` below still applies.
        if write.product_id.lower() not in searched and "product_id" not in write.edited_fields:
            findings.append(
                Finding(
                    "ungrounded",
                    "unverified",
                    f"{write.product_id!r} is in no search result and in no config record — "
                    "nothing the agent looked up supplied this name",
                    explain=True,
                )
            )
        findings.append(
            Finding(
                "new-product",
                "new product",
                f"accepting registers {write.product_id!r} as a new product in {write.unit!r}",
            )
        )
        near = decide.near_matches(write.product_id, cfg.known_products)
        if near:
            findings.append(
                Finding(
                    "near-match",
                    "near-duplicate",
                    f"{write.product_id!r} is one edit away from registered {near[0]!r}",
                )
            )
    elif not cfg.can_convert(write.product_id, write.unit):
        findings.append(
            Finding(
                "new-unit",
                "new unit",
                f"{write.product_id!r} is recorded in {product.unit!r}, and no conversion "
                f"from {write.unit!r} is configured",
            )
        )

    for location_id in (write.from_location, write.to_location):
        if location_id is not None and location_id not in cfg.known_locations:
            findings.append(
                Finding(
                    "unknown-location",
                    "new location",
                    f"{location_id!r} is not a configured location",
                    explain=True,
                )
            )

    return tuple(findings)


def review_plan(plan: AgentPlan, cfg: config.Config) -> tuple[tuple[Finding, ...], ...]:
    """One tuple of findings per write, positionally aligned with
    `plan.writes` — an empty tuple where a write raised nothing, so a caller
    can `zip` the two without filtering."""
    searched = _searched_text(plan.trace)
    return tuple(review_write(w, cfg, searched) for w in plan.writes)


_HEADLINE_PHRASE = {
    "ungrounded": ("names a product nothing looked up", "name products nothing looked up"),
    "new-product": ("creates a new product", "create new products"),
    "new-unit": ("uses an unconfigured unit", "use unconfigured units"),
    "unknown-location": ("names a new location", "name new locations"),
}


def headline(findings: tuple[tuple[Finding, ...], ...]) -> str:
    """A one-line summary above the plan: "2 changes · 1 creates a new
    product". Counts writes, not findings, since three findings on one write
    is still one write to check. `near-match` does not appear here, as it
    only accompanies `new-product`."""
    total = len(findings)
    flagged = [per_write for per_write in findings if per_write]
    if total == 1 and not flagged:
        # One change with nothing flagged: the row underneath already states
        # everything a "1 change" line would. A header that repeats the row
        # is skipped over, including in the cases where it carries a
        # finding.
        return ""
    parts = [f"{total} change" + ("" if total == 1 else "s")]
    for code, (singular, plural) in _HEADLINE_PHRASE.items():
        n = sum(1 for per_write in findings if any(f.code == code for f in per_write))
        if n:
            parts.append(f"{n} {singular if n == 1 else plural}")
    return " · ".join(parts)
