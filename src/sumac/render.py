"""rich tables/panels, kept out of cli.py so command logic stays testable."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from sumac import ledger, models

console = Console()
error_console = Console(stderr=True)


def print_success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    error_console.print(f"[red]✗ error:[/red] {message}")


def print_locations(locations: dict[str, models.Location]) -> None:
    table = Table(title="Locations")
    table.add_column("id")
    table.add_column("name")
    table.add_column("parent")
    for loc in sorted(locations.values(), key=lambda location: location.id):
        table.add_row(loc.id, loc.name, loc.parent_id or "")
    console.print(table)


def _location_name(locations: dict[str, models.Location], location_id: str) -> str:
    loc = locations.get(location_id)
    return loc.name if loc else location_id


def print_status(
    inventory: ledger.Inventory,
    locations: dict[str, models.Location],
    location_id: str | None,
) -> None:
    loc_ids = [location_id] if location_id else sorted(inventory.by_location)
    if not loc_ids:
        console.print("[yellow]no inventory recorded yet[/yellow]")
        return
    for loc_id in loc_ids:
        entries = inventory.at(loc_id)
        table = Table(title=_location_name(locations, loc_id))
        table.add_column("product")
        table.add_column("quantity", justify="right")
        for product_id, qty in sorted(entries.items()):
            table.add_row(product_id, f"{qty.amount} {qty.unit}")
        console.print(table)


def print_find(
    inventory: ledger.Inventory,
    locations: dict[str, models.Location],
    product_id: str,
) -> None:
    table = Table(title=f"Locations of {product_id!r}")
    table.add_column("location")
    table.add_column("quantity", justify="right")
    found = False
    for loc_id, entries in sorted(inventory.by_location.items()):
        qty = entries.get(product_id)
        if qty is not None:
            found = True
            table.add_row(_location_name(locations, loc_id), f"{qty.amount} {qty.unit}")
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
