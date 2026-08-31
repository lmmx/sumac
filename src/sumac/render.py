"""rich tables/panels, kept out of cli.py so command logic stays testable."""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from sumac import config, events, ledger, models

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


def print_find(
    inventory: ledger.Inventory,
    locations: dict[str, models.Location],
    product_id: str,
    exact: bool = False,
) -> None:
    table = Table(title=f"Locations of {product_id!r}")
    table.add_column("product")
    table.add_column("location")
    table.add_column("quantity", justify="right")
    found = False
    for loc_id, entries in sorted(inventory.by_location.items()):
        for pid, qty in sorted(entries.items()):
            matches = (pid == product_id) if exact else (product_id.lower() in pid.lower())
            if matches:
                found = True
                table.add_row(
                    pid, config.location_path(locations, loc_id), f"{qty.amount} {qty.unit}"
                )
    console.print(table)
    if not found:
        console.print(f"[yellow]{product_id!r} not found in current inventory[/yellow]")


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

    suggestions = [a for a in report.anomalies if a.record_id is not None]
    if suggestions:
        console.print(
            "\n[dim]Ready-to-paste corrections (cancels the record, doesn't replace it — "
            "re-add a fixed version separately if needed):[/dim]"
        )
        for a in suggestions:
            reason = f"{a.reason}: {a.detail}".replace('"', "'")
            console.print(f'  sumac correct {a.record_id} --reason "{reason}"')


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
