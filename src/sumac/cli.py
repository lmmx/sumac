"""Typer app: sumac's command-line interface."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from sealedlog import Vault
from sealedlog.errors import SealError

from sumac import (
    FORMAT_VERSION,
    config,
    decide,
    events,
    ledger,
    models,
    paths,
    render,
    store,
)
from sumac import vault as sumac_vault
from sumac.errors import RetireNonemptyError, SumacError, VaultExistsError, VaultNotFoundError
from sumac.models import ChangeKind
from sumac.passphrase import get_key, resolve_passphrase

app = typer.Typer(add_completion=False, no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True, help="Inspect and edit the location layout.")
app.add_typer(config_app, name="config")

DataDirOption = Annotated[
    Path, typer.Option("--data-dir", envvar="SUMAC_DATA_DIR", help="Data directory.")
]


def _load_vault(data_dir: Path) -> Vault:
    vpath = paths.vault_path(data_dir)
    if not vpath.exists():
        raise VaultNotFoundError(f"no vault at {vpath}; run `sumac init` first")
    return Vault.from_dict(json.loads(vpath.read_text(encoding="utf-8")))


def _key(data_dir: Path) -> bytes:
    return get_key(_load_vault(data_dir))


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "location"


def _parse_decimal(raw: str) -> Decimal:
    try:
        return Decimal(raw)
    except InvalidOperation as e:
        raise typer.BadParameter(f"expected a decimal amount, got {raw!r}") from e


def _parse_snapshot_entry(spec: str) -> events.SnapshotEntry:
    try:
        product_id, rest = spec.split("=", 1)
        amount_str, unit = rest.split("/", 1)
        return events.SnapshotEntry(product_id=product_id, amount=Decimal(amount_str), unit=unit)
    except (ValueError, InvalidOperation) as e:
        raise typer.BadParameter(f"expected PRODUCT=AMOUNT/UNIT, got {spec!r}") from e


@app.command()
def init(data_dir: DataDirOption = Path("data")) -> None:
    """Create a new vault in DATA_DIR."""
    vpath = paths.vault_path(data_dir)
    if vpath.exists():
        raise VaultExistsError(f"vault already exists at {vpath}")
    passphrase = resolve_passphrase()
    vault = sumac_vault.create(passphrase)
    data_dir.mkdir(parents=True, exist_ok=True)
    doc = {"format_version": FORMAT_VERSION, **vault.to_dict()}
    vpath.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    paths.log_dir(data_dir).mkdir(parents=True, exist_ok=True)
    render.print_success(f"Initialized sumac vault at {data_dir}")


@config_app.command("show")
def config_show(
    data_dir: DataDirOption = Path("data"),
    locations_only: Annotated[
        bool, typer.Option("--locations-only", help="Only show locations.")
    ] = False,
    products_only: Annotated[
        bool, typer.Option("--products-only", help="Only show products.")
    ] = False,
) -> None:
    """List all locations and products."""
    if locations_only and products_only:
        raise typer.BadParameter("--locations-only and --products-only are mutually exclusive")
    key = _key(data_dir)
    if not products_only:
        render.print_locations(config.load_locations(data_dir, key))
    if not locations_only:
        render.print_products(config.load_products(data_dir, key))


@config_app.command("add-location")
def config_add_location(
    name: str,
    id: Annotated[str | None, typer.Option(help="Location id; defaults to a slug of NAME.")] = None,
    parent: Annotated[str | None, typer.Option(help="Parent location id.")] = None,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Add (or redefine) a location."""
    key = _key(data_dir)
    loc_id = id or _slugify(name)
    location = models.Location(id=loc_id, name=name, parent_id=parent)
    config.add_location(data_dir, key, paths.current_user(), location)
    render.print_success(f"Added location {loc_id!r}")


@config_app.command("retire-location")
def config_retire_location(
    id: str,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Retire a location. Never deletes — historical records naming it still
    resolve; new writes to it are for a later phase to reject. Rejected if the
    location itself currently holds stock (its sub-locations aren't checked —
    each is retired, and checked, on its own)."""
    key = _key(data_dir)
    holdings = ledger.build_inventory(data_dir, key).at(id)
    if holdings:
        listing = ", ".join(f"{q.amount} {q.unit} {p}" for p, q in sorted(holdings.items()))
        raise RetireNonemptyError(f"cannot retire {id!r}: still holds {listing}")
    config.retire_location(data_dir, key, paths.current_user(), id)
    render.print_success(f"Retired location {id!r}")


@config_app.command("add-product")
def config_add_product(
    name: str,
    unit: str,
    id: Annotated[str | None, typer.Option(help="Product id; defaults to a slug of NAME.")] = None,
    category: Annotated[str | None, typer.Option(help="Product category.")] = None,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Add (or redefine) a product."""
    key = _key(data_dir)
    prod_id = id or _slugify(name)
    product = models.Product(id=prod_id, name=name, unit=unit, category=category)
    config.add_product(data_dir, key, paths.current_user(), product)
    render.print_success(f"Added product {prod_id!r}")


@config_app.command("retire-product")
def config_retire_product(
    id: str,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Retire a product. Never deletes — historical records naming it still
    resolve. Unlike a location, permitted at any time regardless of stock."""
    key = _key(data_dir)
    config.retire_product(data_dir, key, paths.current_user(), id)
    render.print_success(f"Retired product {id!r}")


@config_app.command("check-units")
def config_check_units(data_dir: DataDirOption = Path("data")) -> None:
    """Report unregistered products and unconvertible units observed in the
    log (legacy data — `sumac add` auto-registers now, so these can only
    come from before decide existed), and every auto-registered product
    that hasn't since been confirmed by a deliberate `add-product`."""
    key = _key(data_dir)
    observed = ledger.observed_product_units(data_dir, key)
    cfg = config.build_config(data_dir, key)
    ok = render.print_unit_check(observed, cfg)
    if not ok:
        raise typer.Exit(code=1)


def _location_id_prefix(parent: str | None, name: str, id_prefix: str | None) -> str:
    if id_prefix:
        return id_prefix
    base = _slugify(name)
    return f"{parent}-{base}" if parent else base


@config_app.command("add-array")
def config_add_array(
    name: str,
    count: Annotated[int, typer.Option(min=1, help="How many sub-locations to create.")],
    parent: Annotated[str | None, typer.Option(help="Parent location id.")] = None,
    start: Annotated[int, typer.Option(help="Starting index.")] = 1,
    id_prefix: Annotated[str | None, typer.Option(help="Override the generated id prefix.")] = None,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Create a numbered row of sub-locations under --parent, e.g. 4 fridge shelves."""
    key = _key(data_dir)
    actor = paths.current_user()
    prefix = _location_id_prefix(parent, name, id_prefix)
    for i in range(start, start + count):
        location = models.Location(id=f"{prefix}-{i}", name=f"{name} {i}", parent_id=parent)
        config.add_location(data_dir, key, actor, location)
    render.print_success(f"Added {count} locations {prefix}-{start}..{prefix}-{start + count - 1}")


@config_app.command("add-grid")
def config_add_grid(
    name: str,
    rows: Annotated[int, typer.Option(min=1, help="Number of rows.")],
    cols: Annotated[int, typer.Option(min=1, help="Number of columns.")],
    parent: Annotated[str | None, typer.Option(help="Parent location id.")] = None,
    id_prefix: Annotated[str | None, typer.Option(help="Override the generated id prefix.")] = None,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Create a rows x cols grid of sub-locations under --parent, e.g. a pantry shelf grid."""
    key = _key(data_dir)
    actor = paths.current_user()
    prefix = _location_id_prefix(parent, name, id_prefix)
    count = 0
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            location = models.Location(
                id=f"{prefix}-r{r}c{c}", name=f"{name} R{r}C{c}", parent_id=parent
            )
            config.add_location(data_dir, key, actor, location)
            count += 1
    render.print_success(f"Added {count} locations {prefix}-r1c1..{prefix}-r{rows}c{cols}")


@app.command()
def add(
    kind: ChangeKind,
    product_id: str,
    amount: str,
    unit: str,
    from_location: Annotated[str | None, typer.Option("--from")] = None,
    to_location: Annotated[str | None, typer.Option("--to")] = None,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Record an inventory change: purchase, consumption, waste, discovery,
    correction, or movement between locations."""
    key = _key(data_dir)
    actor = paths.current_user()
    cfg = config.build_config(data_dir, key)
    inventory = ledger.build_inventory(data_dir, key)
    writes, messages = decide.decide_change(
        kind=kind,
        product_id=product_id,
        amount=_parse_decimal(amount),
        unit=unit,
        from_location=from_location,
        to_location=to_location,
        actor=actor,
        occurred_at=datetime.now(UTC),
        inventory=inventory,
        cfg=cfg,
    )
    for message in messages:
        render.print_warning(message)
    for w in writes:
        store.append(data_dir, key, w.stream, w.obj)
    render.print_success(f"Recorded {kind.value} of {amount} {unit} {product_id}")


@app.command()
def snapshot(
    location_id: str,
    entries: Annotated[
        list[str] | None, typer.Argument(help="PRODUCT=AMOUNT/UNIT, repeatable")
    ] = None,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Record the full observed state of a location, resetting its products."""
    key = _key(data_dir)
    actor = paths.current_user()
    parsed = tuple(_parse_snapshot_entry(e) for e in (entries or []))
    event = events.Snapshot(location_id=location_id, entries=parsed)
    obj = decide.serialize_event(
        event, actor=actor, occurred_at=datetime.now(UTC), cmd_id=str(uuid4())
    )
    store.append(data_dir, key, f"log:{actor}", obj)
    render.print_success(f"Recorded snapshot of {location_id!r} ({len(parsed)} entries)")


@app.command()
def correct(
    record_id: str,
    reason: Annotated[str, typer.Option("--reason", help="Why this record is being corrected.")],
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Cancel RECORD_ID: appends a correction that supersedes it. Nothing is
    rewritten or deleted — the targeted record stays in the log, permanently
    excluded from the fold (§3.6). To replace rather than just cancel, run
    this and then `sumac add`/`sumac snapshot` with the corrected values."""
    key = _key(data_dir)
    actor = paths.current_user()
    records = ledger.load_all_records(data_dir, key)
    write = decide.decide_correct(
        target_id=record_id,
        reason=reason,
        actor=actor,
        occurred_at=datetime.now(UTC),
        records=records,
    )
    store.append(data_dir, key, write.stream, write.obj)
    render.print_success(f"Corrected {record_id!r}: {reason}")


@app.command()
def status(
    location: Annotated[str | None, typer.Argument()] = None,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Show current inventory. Given a location, includes its sub-locations
    (shelves, doors, grid cells, ...), not just that exact node."""
    key = _key(data_dir)
    inventory = ledger.build_inventory(data_dir, key)
    locations = ledger.load_locations_or_empty(data_dir, key)
    scope = config.descendants(locations, location) if location else None
    render.print_anomaly_banner(inventory.anomalies)
    render.print_status(inventory, locations, scope)
    if inventory.anomalies:
        raise typer.Exit(code=1)


@app.command()
def find(
    product_id: str,
    exact: Annotated[
        bool, typer.Option(help="Exact match only (default is partial/substring match)")
    ] = False,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Show every location currently holding PRODUCT_ID (partial match by default)."""
    key = _key(data_dir)
    inventory = ledger.build_inventory(data_dir, key)
    locations = ledger.load_locations_or_empty(data_dir, key)
    render.print_anomaly_banner(inventory.anomalies)
    render.print_find(inventory, locations, product_id, exact=exact)


@app.command(name="log")
def log_cmd(data_dir: DataDirOption = Path("data")) -> None:
    """Show the full, ordered event log."""
    key = _key(data_dir)
    render.print_log(ledger.load_records(data_dir, key))


@app.command()
def verify(data_dir: DataDirOption = Path("data")) -> None:
    """Re-open every line of every log under its own AAD; report failures."""
    key = _key(data_dir)
    result = ledger.verify_all(data_dir, key)
    render.print_verify(result)
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def doctor(data_dir: DataDirOption = Path("data")) -> None:
    """Tolerant fold: report every record that cannot be applied, without crashing."""
    key = _key(data_dir)
    report = ledger.diagnose(data_dir, key)
    render.print_doctor(report)
    if report.anomalies:
        raise typer.Exit(code=1)


@app.command()
def ask(
    prompt: str,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Compute and show the plan; write nothing."),
    ] = False,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Ask the in-process AI agent to perform an inventory operation.

    Resolves an underspecified sentence into a sequence of consumption,
    movement, and discovery writes against the same `decide_change` gate
    `sumac add` uses, shows the plan, and asks before writing anything. See
    docs/journal/2026-09-01-ask-agent-design.md.

    Examples:
        sumac ask "where is the jam?"
        sumac ask "consume 1 jar of jam"
        sumac ask "move the ragu to the fridge"
    """
    key = _key(data_dir)

    # Import llm here so mistralrs is optional
    try:
        from sumac import llm
    except ImportError as e:
        render.print_error(f"Agent requires mistralrs. Install with: pip install mistralrs\n{e}")
        raise typer.Exit(code=1) from e

    try:
        agent = llm.AgentRunner(data_dir, key)
        plan = agent.propose(prompt)
    except FileNotFoundError as e:
        render.print_error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:
        render.print_error(f"Agent error: {e}")
        raise typer.Exit(code=1) from e

    while True:
        render.print_trace(plan.trace)
        if not plan.writes:
            if plan.reply_text:
                render.console.print(plan.reply_text)
            return

        render.print_plan(plan)
        if dry_run:
            return

        answer = typer.prompt("[a]ccept / [r]eject / or type feedback", default="a").strip()
        choice = answer.lower()
        if choice in ("a", "accept"):
            # Not caught here: a Rejected raised while committing is not a
            # modeled outcome the model can react to (the human already
            # accepted the plan) and should propagate exactly the way `add`
            # already lets `decide_change`'s Rejected reach `main`'s handler.
            for summary in agent.commit(plan):
                render.print_success(summary)
            return
        if choice in ("r", "reject"):
            return

        try:
            plan = agent.revise(answer)
        except Exception as e:
            render.print_error(f"Agent error: {e}")
            raise typer.Exit(code=1) from e


def main() -> None:
    try:
        app()
    except (SumacError, SealError) as e:
        render.print_error(str(e))
        raise SystemExit(1) from None
