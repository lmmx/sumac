"""rich tables/panels, kept out of cli.py so command logic stays testable."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from sumac import config, events, ledger, models, review

if TYPE_CHECKING:
    from sumac.llm import AgentPlan, ProposedWrite, ToolCallRecord

console = Console()
error_console = Console(stderr=True)


def print_success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    error_console.print(f"[red]✗ error:[/red] {message}")


def print_warning(message: str) -> None:
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_locations(locations: dict[str, models.Location]) -> None:
    if not locations:
        console.print("[yellow]no locations configured yet[/yellow]")
        return

    children_of: dict[str | None, list[models.Location]] = {}
    for loc in locations.values():
        children_of.setdefault(loc.parent_id, []).append(loc)
    for siblings in children_of.values():
        siblings.sort(key=lambda location: location.id)

    def add(node: Tree, loc: models.Location) -> None:
        retired = " [dim]\\[retired][/dim]" if loc.retired else ""
        branch = node.add(f"{loc.name} [dim]({loc.id})[/dim]{retired}")
        for child in children_of.get(loc.id, []):
            add(branch, child)

    tree = Tree("Locations")
    known_ids = set(locations)
    roots = [loc for loc in locations.values() if loc.parent_id not in known_ids]
    for root in sorted(roots, key=lambda location: location.id):
        add(tree, root)
    console.print(tree)


def print_products(products: dict[str, models.Product]) -> None:
    if not products:
        console.print("[yellow]no products configured yet[/yellow]")
        return
    table = Table(title="Products")
    table.add_column("product")
    table.add_column("unit")
    table.add_column("category")
    table.add_column("status")
    for p in sorted(products.values(), key=lambda product: product.id):
        table.add_row(
            f"{p.name} [dim]({p.id})[/dim]",
            p.unit,
            p.category or "-",
            "[dim]retired[/dim]" if p.retired else "",
        )
    console.print(table)


def print_unit_check(observed: dict[str, Counter[str]], cfg: config.Config) -> bool:
    """Returns True if every observed (product, unit) pair converts cleanly
    and no auto-registration is still unconfirmed.

    Since Phase 3, `sumac add` auto-registers on first use (docs/journal
    §3.5a) — an unregistered or unconvertible pair can no longer arise
    through it, only through data written before decide existed. The
    unregistered/unconvertible sections below are that legacy-data report.
    The unconfirmed-auto-registrations section is the *current* job: the
    `near_matches` warning at write time is easy to miss in the moment,
    so this is the backstop §3.5a promised — everything auto-registered
    (`metadata: {"auto": true}`) that hasn't since been redefined (which
    clears the flag, since a fresh `add-product` call doesn't carry it
    forward) is still worth a human glance."""
    unregistered: list[tuple[str, str]] = []  # (product_id, suggested add-product command)
    unconvertible: list[tuple[str, str, int]] = []  # (product_id, unit, times observed)

    for product_id, units in sorted(observed.items()):
        product = cfg.known_products.get(product_id)
        if product is None:
            canonical, _count = units.most_common(1)[0]
            cmd = f"sumac config add-product {product_id} {canonical} --id {product_id}"
            unregistered.append((product_id, cmd))
            continue
        for unit, n in sorted(units.items()):
            if not cfg.can_convert(product_id, unit):
                unconvertible.append((product_id, unit, n))

    unconfirmed = [
        (pid, p.unit)
        for pid, p in sorted(cfg.known_products.items())
        if p.metadata.get("auto") is True
    ]

    if not unregistered and not unconvertible and not unconfirmed:
        console.print("[green]✓ every observed product/unit pair converts[/green]")
        return True

    if unregistered:
        table = Table(title=f"{len(unregistered)} unregistered products")
        table.add_column("product")
        table.add_column("suggested command")
        for product_id, cmd in unregistered:
            table.add_row(product_id, cmd)
        console.print(table)

    if unconvertible:
        table = Table(title=f"{len(unconvertible)} (product, unit) pairs need a conversion")
        table.add_column("product")
        table.add_column("unit")
        table.add_column("times observed", justify="right")
        for product_id, unit, n in unconvertible:
            table.add_row(product_id, unit, str(n))
        console.print(table)

    if unconfirmed:
        table = Table(title=f"{len(unconfirmed)} auto-registered, never confirmed")
        table.add_column("product")
        table.add_column("unit")
        for product_id, unit in unconfirmed:
            table.add_row(product_id, unit)
        console.print(table)
        console.print(
            "[dim]Confirm a real one with `sumac config add-product`; "
            "typos with `sumac correct` once that exists.[/dim]"
        )

    return False


def print_anomaly_banner(anomalies: tuple[models.Anomaly, ...]) -> None:
    if anomalies:
        n = len(anomalies)
        event = "event" if n == 1 else "events"
        console.print(f"[yellow]⚠ {n} {event} could not be applied — run 'sumac doctor'[/yellow]")


def print_status(
    inventory: ledger.Inventory,
    locations: dict[str, models.Location],
    scope: set[str] | None,
) -> None:
    loc_ids = sorted(scope & set(inventory.by_location)) if scope else sorted(inventory.by_location)
    if not loc_ids:
        console.print("[yellow]no inventory recorded yet[/yellow]")
        return
    for loc_id in loc_ids:
        entries = inventory.at(loc_id)
        table = Table(title=config.location_path(locations, loc_id))
        table.add_column("product")
        table.add_column("quantity", justify="right")
        for product_id, qty in sorted(entries.items()):
            table.add_row(product_id, f"{qty.amount} {qty.unit}")
        console.print(table)


def _find_table(
    title: str, matches: tuple[ledger.InventoryMatch, ...], locations: dict[str, models.Location]
) -> Table:
    table = Table(title=title)
    table.add_column("product")
    table.add_column("location")
    table.add_column("quantity", justify="right")
    for m in matches:
        table.add_row(
            m.product_id,
            config.location_path(locations, m.location_id),
            f"{m.quantity.amount} {m.quantity.unit}",
        )
    return table


_FIND_SECTIONS = (
    (ledger.MatchKind.EXACT, "Exact matches"),
    (ledger.MatchKind.WHOLE_WORD, "Whole-word matches"),
    (ledger.MatchKind.SUBSTRING, "Substring matches"),
)


def print_find(
    matches: tuple[ledger.InventoryMatch, ...],
    locations: dict[str, models.Location],
    query: str,
) -> None:
    """Renders `ledger.search_inventory`'s classified result as one table per
    tier that actually has rows — exact, then whole-word, then substring —
    rather than collapsing whole-word and substring together under a single
    "related" heading; a tier a caller filtered out entirely (`cli.py`'s
    `find --whole-word`, say) just has no table, not an empty one."""
    if not matches:
        console.print(f"[yellow]{query!r} not found in current inventory[/yellow]")
        return
    for kind, title in _FIND_SECTIONS:
        tier = tuple(m for m in matches if m.match_kind is kind)
        if tier:
            console.print(_find_table(title, tier, locations))


def _describe_payload(
    payload: models.InventoryChange | models.InventorySnapshot | events.Event,
) -> str:
    if isinstance(payload, events.Correction):
        return f"correction: {payload.reason}"
    if isinstance(payload, models.InventoryChange):
        if payload.from_location and payload.to_location:
            arrow = f" {payload.from_location} -> {payload.to_location}"
        elif payload.from_location:
            arrow = f" from {payload.from_location}"
        elif payload.to_location:
            arrow = f" to {payload.to_location}"
        else:
            arrow = ""
        qty = payload.quantity
        return f"{payload.kind.value} {qty.amount} {qty.unit} {payload.product_id}{arrow}"
    if isinstance(payload, models.InventorySnapshot | events.Snapshot):
        return f"snapshot of {payload.location_id} ({len(payload.entries)} entries)"
    if isinstance(payload, events.Acquired):
        reason = f" ({payload.reason})" if payload.reason else ""
        return (
            f"acquired{reason} {payload.amount} {payload.unit} {payload.product_id} -> {payload.to}"
        )
    if isinstance(payload, events.Consumed):
        reason = f" ({payload.reason})" if payload.reason else ""
        return (
            f"consumed{reason} {payload.amount} {payload.unit} {payload.product_id} "
            f"from {payload.frm}"
        )
    if isinstance(payload, events.Discarded):
        return f"discarded {payload.amount} {payload.unit} {payload.product_id} from {payload.frm}"
    if isinstance(payload, events.Moved):
        return (
            f"moved {payload.amount} {payload.unit} {payload.product_id} "
            f"{payload.frm} -> {payload.to}"
        )
    if isinstance(payload, events.Counted):
        reason = f" ({payload.reason})" if payload.reason else ""
        return (
            f"counted{reason} {payload.amount} {payload.unit} {payload.product_id} at {payload.at}"
        )
    raise TypeError(f"unrecognized payload type: {type(payload).__name__}")  # unreachable


def print_log(records: list[models.Record]) -> None:
    table = Table(title="Log")
    table.add_column("ts")
    table.add_column("actor")
    table.add_column("type")
    table.add_column("detail")
    for r in records:
        detail = _describe_payload(r.payload)
        if r.supersedes is not None:
            detail += f" [dim](supersedes {r.supersedes})[/dim]"
        table.add_row(r.ts.isoformat(), r.actor, r.type, detail)
    console.print(table)


def print_doctor(report: ledger.DoctorReport) -> None:
    if not report.anomalies:
        console.print(f"[green]✓ {report.total_lines} lines, no anomalies[/green]")
        return
    table = Table(title=f"{len(report.anomalies)} anomalies out of {report.total_lines} lines")
    table.add_column("reason")
    table.add_column("record")
    table.add_column("detail")
    for a in report.anomalies:
        table.add_row(a.reason, a.record_id or "-", a.detail)
    console.print(table)

    # One suggestion per record, not per anomaly — a record can trip more
    # than one anomaly type at once (e.g. a duplicated line is both
    # seq_duplicate and duplicate_record), and `sumac correct` only accepts
    # a target once; a second identical command would just fail on
    # supersede_already_applied.
    seen_ids: set[str] = set()
    suggestions = []
    for a in report.anomalies:
        if a.record_id is not None and a.record_id not in seen_ids:
            seen_ids.add(a.record_id)
            suggestions.append(a)
    if suggestions:
        console.print(
            "\n[dim]Ready-to-paste corrections (cancels the record, doesn't replace it — "
            "re-add a fixed version separately if needed):[/dim]"
        )
        for a in suggestions:
            reason = f"{a.reason}: {a.detail}".replace('"', "'")
            console.print(f'  sumac correct {a.record_id} --reason "{reason}"')


def print_agent_messages(messages: object, title: str) -> None:
    """The accumulated `AgentRunner._messages` list, at whatever point in
    `_run_loop` it's passed in — same list object `repr()` used to dump
    line-by-line, now one `Pretty` panel labeled by the caller."""
    console.print(Panel(Pretty(messages, expand_all=False), title=title))


def print_agent_request(request: object, round_num: int) -> None:
    console.print(Panel(Pretty(request, expand_all=False), title=f"REQUEST · round {round_num}"))


def print_agent_response(response: object, round_num: int) -> None:
    console.print(
        Panel(Pretty(response, expand_all=False), title=f"RAW RESPONSE · round {round_num}")
    )


def print_agent_message(message: object) -> None:
    console.print(Panel(Pretty(message, expand_all=False), title="MESSAGE"))


def print_agent_content(content: object) -> None:
    console.print(Panel(Pretty(content, expand_all=False), title="CONTENT"))


def print_agent_tool_calls(tool_calls: object) -> None:
    console.print(Panel(Pretty(tool_calls, expand_all=False), title="TOOL CALLS"))


def _trace_summary(record: ToolCallRecord) -> str:
    """One line's worth of what a tool call did — how many products a search
    found, or the status a proposed write came back with. Falls back to the
    head of the raw result for anything unrecognized, so a tool added later
    still says something rather than nothing.

    Defensive about the result's shape on purpose: a tool result is a string
    this module did not build, and a preview that raised while summarizing a
    trace would take down the decision it was drawn for."""
    try:
        parsed = json.loads(record.result)
    except (json.JSONDecodeError, TypeError):
        return record.result[:60]
    if not isinstance(parsed, dict):
        return record.result[:60]
    if "products" in parsed and isinstance(parsed["products"], list):
        products = parsed["products"]
        locations = sum(len(p.get("locations", ())) for p in products if isinstance(p, dict))
        n = len(products)
        return (
            f"{n} product{'' if n == 1 else 's'}, "
            f"{locations} location{'' if locations == 1 else 's'}"
        )
    status = parsed.get("status")
    if status == "rejected":
        return f"rejected: {parsed.get('reason', '?')}"
    if isinstance(status, str):
        return status
    return record.result[:60]


def print_trace(trace: tuple[ToolCallRecord, ...], *, verbose: bool = False) -> None:
    """What an `AgentRunner.propose`/`.revise` call actually looked up, shown
    above the plan it produced — a plain final reply ("the jam is in the
    fridge") otherwise hides the query and the match data behind it, with no
    way to tell a vague answer from a genuinely empty result.

    One line per call by default. The full table — every argument and the
    raw JSON result verbatim — is what `verbose` restores, and what this
    printed unconditionally before: a three-round request put a screen of
    JSON between the person and the panel they were being asked to approve,
    which is a poor default for the common case and the only useful view
    when a tool result is what's actually in question. A no-op for an empty
    trace, matching `print_anomaly_banner`'s "nothing to say" behavior."""
    if not trace:
        return
    if not verbose:
        for t in trace:
            args = ", ".join(f"{k}={v!r}" for k, v in t.arguments.items())
            console.print(f"[dim]· {t.name}({escape(args)}) → {escape(_trace_summary(t))}[/dim]")
        return
    table = Table(title=f"Tool calls ({len(trace)})")
    table.add_column("tool")
    table.add_column("arguments")
    table.add_column("result")
    for t in trace:
        table.add_row(t.name, json.dumps(t.arguments), t.result)
    console.print(table)


_KIND_MARK = {
    "consumption": ("−", "yellow"),
    "waste": ("−", "yellow"),
    "movement": ("→", "cyan"),
    "purchase": ("+", "green"),
    "discovery": ("+", "green"),
    "correction": ("±", "magenta"),
}


def _amount(value: Decimal | None, unit: str) -> str:
    """A holding, or an em dash for none of it. `None` is "the location holds
    none of this product", which reads as nothing rather than as zero — the
    fold drops a zero entry entirely (`ledger._commit`), so there is no
    stored `0` for this to be showing."""
    return f"{value} {unit}" if value is not None else "—"


def _effect_text(write: ProposedWrite) -> str:
    """`before → after` per location touched, from the projection
    `ledger.project` computed off the records `decide_change` itself
    returned. Falls back to `current_amount`'s descriptive "already there"
    when a write carries no projection — a `ProposedWrite` built by hand
    (a test double, a scripted plan) renders as it did before `effects`
    existed, rather than showing a blank column."""
    if write.effects:
        return "  ·  ".join(
            f"{_amount(e.before, e.unit)} → {_amount(e.after, e.unit)}" for e in write.effects
        )
    if write.current_amount is not None:
        return f"{write.current_amount} {write.unit} already there"
    return ""


def _where_text(write: ProposedWrite, locations: dict[str, models.Location]) -> str:
    def name(location_id: str) -> str:
        return config.location_path(locations, location_id) if locations else location_id

    if write.from_location and write.to_location:
        return f"{name(write.from_location)} → {name(write.to_location)}"
    if write.from_location:
        return name(write.from_location)
    if write.to_location:
        return name(write.to_location)
    return ""


def _indented(markup: str) -> None:
    """A detail line under a write, indented four columns — as `Padding` so a
    line long enough to wrap keeps its indent on every wrapped line instead
    of falling back to column zero mid-sentence."""
    console.print(Padding(Text.from_markup(markup), (0, 0, 0, 4), expand=False))


def print_plan(
    plan: AgentPlan,
    *,
    findings: tuple[tuple[review.Finding, ...], ...] = (),
    locations: dict[str, models.Location] | None = None,
    header: str = "",
) -> None:
    """`sumac ask`'s preview: the accumulated, not-yet-committed writes an
    `AgentRunner.propose`/`.revise` call resolved. This is the one point in
    the CLI where a person is about to make a real decision, so each write
    gets two lines — what it does, then where and what changes on the shelf
    — rather than a table row that a narrow terminal would truncate away
    exactly the number being decided on.

    The "after" is not a subtraction. `ledger.project` folds the records
    `decide_change` returned for that write, so a consumption exceeding the
    recorded holding shows the zero its §3.5 reconciling `Counted` actually
    produces, not the negative a current-minus-amount arithmetic would. It
    still describes the moment the plan was proposed: `AgentRunner.commit`
    re-decides against freshly reloaded state, and real time passes while
    someone reads this.

    `findings` (`review.review_plan`) is positional per write: badges on the
    write's own line, one detail line each underneath. A write with nothing
    flagged has an empty tuple and no extra lines."""
    if header:
        console.print(f"[bold]{header}[/bold]")

    locations = locations or {}
    for i, write in enumerate(plan.writes):
        mark, colour = _KIND_MARK.get(write.kind.value, ("·", "white"))
        per_write = findings[i] if i < len(findings) else ()
        badges = "".join(f" [yellow]\\[{escape(f.label)}][/yellow]" for f in per_write)
        subject = escape(f"{write.amount} {write.unit} of {write.product_id}")
        console.print(f"[{colour}]{mark} {write.kind.value}[/{colour}]  {subject}{badges}")

        where = escape(_where_text(write, locations))
        effect = escape(_effect_text(write))
        parts = [p for p in (f"[dim]{where}[/dim]" if where else "", effect) if p]
        if parts:
            _indented("   ".join(parts))
        for f in per_write:
            if f.explain:
                _indented(f"[yellow]⚠ {escape(f.detail)}[/yellow]")
        for warning in write.warnings:
            _indented(f"[yellow]⚠ {escape(warning)}[/yellow]")

    if plan.reply_text:
        console.print(plan.reply_text)


def print_decision_options(options: list[tuple[str, str]]) -> None:
    """The available responses to a plan decision, one per row — key then
    description — instead of cramming them into one dense prompt line the
    person has to parse before they can even see what's on offer. `key` is
    what to type; not every option needs a single-letter shortcut (e.g. the
    free-text feedback fallback), so this takes whatever label the caller
    already decided on rather than assuming one shape."""
    table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
    for key, description in options:
        table.add_row(f"[bold cyan]{key}[/bold cyan]", description)
    console.print(table)


def print_verify(result: ledger.VerifyResult) -> None:
    if result.ok:
        console.print("[green]✓ all lines verified[/green]")
        return
    for f in result.line_failures:
        console.print(f"[red]line failure[/red] {f.path}:{f.lineno}: {f.error}")
    for path, actor, osuser in result.actor_mismatches:
        console.print(
            f"[red]actor mismatch[/red] {path}: record actor={actor!r} owning user={osuser!r}"
        )
