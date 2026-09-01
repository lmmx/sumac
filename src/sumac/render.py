"""rich tables/panels, kept out of cli.py so command logic stays testable."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.tree import Tree

from sumac import config, events, ledger, models

if TYPE_CHECKING:
    from sumac.llm import AgentPlan, ToolCallRecord

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


def print_trace(trace: tuple[ToolCallRecord, ...]) -> None:
    """The tool calls an `AgentRunner.propose`/`.revise` call actually made,
    with their raw JSON results — shown before `print_plan`'s writes table or
    a read-only reply, since neither on its own says what the agent searched
    or found; added after real usage showed a plain final reply hides that
    entirely (see `sumac/llm.py`'s `ToolCallRecord`). A no-op for an empty
    trace, matching `print_anomaly_banner`'s "nothing to say" behavior."""
    if not trace:
        return
    table = Table(title=f"Tool calls ({len(trace)})")
    table.add_column("tool")
    table.add_column("arguments")
    table.add_column("result")
    for t in trace:
        table.add_row(t.name, json.dumps(t.arguments), t.result)
    console.print(table)


def print_plan(plan: AgentPlan) -> None:
    """`sumac ask`'s preview (docs/journal/2026-09-01-ask-agent-design.md §14
    step 4): the accumulated, not-yet-committed writes an `AgentRunner.propose`/
    `.revise` call resolved, in the structured-table style of `print_log`/
    `print_doctor` rather than a single string."""
    table = Table(title=f"Proposed plan ({len(plan.writes)} write(s))")
    table.add_column("action")
    table.add_column("detail")
    for w in plan.writes:
        detail = f"{w.amount} {w.unit} {w.product_id}"
        if w.from_location and w.to_location:
            detail += f" {w.from_location} -> {w.to_location}"
        elif w.from_location:
            detail += f" from {w.from_location}"
        elif w.to_location:
            detail += f" to {w.to_location}"
        table.add_row(w.kind.value, detail)
    console.print(table)
    if plan.reply_text:
        console.print(plan.reply_text)
    for w in plan.writes:
        for warning in w.warnings:
            print_warning(warning)


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
