"""rich tables/panels, kept out of cli.py so command logic stays testable."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from sumac import config, ledger, models

console = Console()
error_console = Console(stderr=True)


def print_success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    error_console.print(f"[red]✗ error:[/red] {message}")


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


def _describe_payload(payload: models.InventoryChange | models.InventorySnapshot) -> str:
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
    return f"snapshot of {payload.location_id} ({len(payload.entries)} entries)"


def print_log(records: list[models.Record]) -> None:
    table = Table(title="Log")
    table.add_column("ts")
    table.add_column("actor")
    table.add_column("type")
    table.add_column("detail")
    for r in records:
        table.add_row(r.ts.isoformat(), r.actor, r.type, _describe_payload(r.payload))
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
