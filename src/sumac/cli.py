"""Typer app: sumac's command-line interface."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
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
    prompt_ui,
    queue,
    render,
    review,
    store,
)
from sumac import vault as sumac_vault
from sumac.errors import (
    Rejected,
    RetireNonemptyError,
    SumacError,
    VaultExistsError,
    VaultNotFoundError,
)

if TYPE_CHECKING:
    from sumac.llm import AgentPlan, ModelPreset, ProposedWrite
from sumac.models import ChangeKind
from sumac.passphrase import get_key, resolve_passphrase

app = typer.Typer(add_completion=False, no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True, help="Inspect and edit the location layout.")
app.add_typer(config_app, name="config")
models_app = typer.Typer(
    no_args_is_help=True, help="Inspect and pre-download `sumac ask` model presets."
)
app.add_typer(models_app, name="models")

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


# mistral.rs logs through Rust's `tracing` with an `EnvFilter` built from
# `RUST_LOG` (confirmed against the built extension: it carries the `RUST_LOG`
# string and `tracing_subscriber::filter::env` symbols from `mistralrs_core`).
# Every line it prints on a successful load — the DType, the tokenizer, the
# device map, the version, and the entire GGUF chat template verbatim, a
# screen of Jinja on its own — is INFO, so `warn` drops all of it and still
# shows anything that actually went wrong.
QUIET_RUST_LOG = "warn"
VERBOSE_RUST_LOG = "info"


def _set_rust_log(verbose: bool) -> None:
    """Chooses how much mistral.rs itself prints. Must run before the
    extension module is first imported — the filter is built once, when the
    Rust side installs its subscriber — which is why this sits next to the
    lazy import rather than anywhere the flag is parsed.

    A `RUST_LOG` already in the environment is never overridden: someone who
    set it wants exactly what they asked for, including a per-target filter
    finer than either value here."""
    if "RUST_LOG" not in os.environ:
        os.environ["RUST_LOG"] = VERBOSE_RUST_LOG if verbose else QUIET_RUST_LOG


def _import_llm(*, verbose: bool = False):  # noqa: ANN202
    """`sumac.llm` imports `mistralrs` at module scope, which is an optional
    dependency (the `ask`/`ask-cuda` groups) — every command that touches
    model presets imports it here, lazily, instead of at the top of this
    module, so the rest of the CLI works without it installed."""
    _set_rust_log(verbose)
    try:
        from sumac import llm
    except ImportError as e:
        render.print_error(f"Agent requires mistralrs. Install with: pip install mistralrs\n{e}")
        raise typer.Exit(code=1) from e
    return llm


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
        bool, typer.Option("--exact", help="Include exact product-name matches.")
    ] = False,
    whole_word: Annotated[
        bool, typer.Option("--whole-word", help="Include whole-word matches.")
    ] = False,
    substring: Annotated[
        bool, typer.Option("--substring", help="Include substring matches.")
    ] = False,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Show every location currently holding PRODUCT_ID, one table per match
    kind (exact / whole-word / substring). --exact/--whole-word/--substring
    each restrict the result to that kind, and are combined when more than
    one is given (e.g. --exact --whole-word shows both, not substring);
    with none given, shows every kind."""
    key = _key(data_dir)
    inventory = ledger.build_inventory(data_dir, key)
    locations = ledger.load_locations_or_empty(data_dir, key)
    render.print_anomaly_banner(inventory.anomalies)
    matches = ledger.search_inventory(inventory, product_id)
    selected = {
        kind
        for kind, flag in (
            (ledger.MatchKind.EXACT, exact),
            (ledger.MatchKind.WHOLE_WORD, whole_word),
            (ledger.MatchKind.SUBSTRING, substring),
        )
        if flag
    }
    if selected:
        matches = tuple(m for m in matches if m.match_kind in selected)
    render.print_find(matches, locations, product_id)


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
    prompt: Annotated[str | None, typer.Argument()] = None,
    loop: Annotated[
        bool,
        typer.Option("--loop", help="Keep prompting for new requests instead of taking just one."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Compute and show the plan; write nothing."),
    ] = False,
    trace: Annotated[
        bool,
        typer.Option("--trace", help="Show each tool call's full arguments and raw result."),
    ] = False,
    stats: Annotated[
        bool,
        typer.Option("--stats", help="Show per-round token counts and generation speed."),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Show raw agent request/response diagnostics, and mistral.rs's own load logs.",
        ),
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
        sumac ask --loop

    With no PROMPT, or with --loop, repeatedly prompts for a new request
    instead of taking one from argv and exiting — see
    docs/journal/2026-09-02-query-classifier.md for why (a request can crash
    the process outright, and shouldn't take the rest of the session with
    it). A request that fails, or that you type "d" to defer, goes into a
    local cache (queue.py) instead of being lost; "queue" lists what's
    pending and "retry N" revisits one.
    """
    key = _key(data_dir)
    llm = _import_llm(verbose=debug)
    view = _AskView(trace=trace, stats=stats, debug=debug)

    if prompt is None or loop:
        _ask_loop(data_dir, key, llm, first_prompt=prompt, dry_run=dry_run, view=view)
        return

    _ask_one(data_dir, key, llm, prompt, dry_run=dry_run, view=view)


def _decision_options(
    dry_run: bool, *, defer: bool = False, pick: bool = False
) -> list[prompt_ui.Option]:
    """The responses available to a plan decision. Every option's `key` is
    both what a typed answer must equal and the keystroke that chooses it in
    `prompt_ui.select`'s menu, so the two input paths cannot drift.

    `pick` adds per-write selection, and is passed only for a plan with more
    than one write on an interactive terminal — there is nothing for it to
    do on a single write, and `prompt_ui.multiselect` has no line-typed
    equivalent to offer a pipe or a test."""
    options = [
        prompt_ui.Option(
            "a", "Accept" + (" — nothing will actually be written (--dry-run)" if dry_run else "")
        ),
        prompt_ui.Option("r", 'Reject (or "quit"/"exit") — discard this proposal'),
        prompt_ui.Option("e", "Edit — manually correct a field, no model call"),
    ]
    if pick:
        options.append(prompt_ui.Option("p", "Pick — apply only some of these changes"))
    if defer:
        options.append(
            prompt_ui.Option("d", "Defer — queue this for later, ask something else now")
        )
    options += [
        prompt_ui.Option(
            "g", "Regenerate — same request, a different model, no memory of this attempt"
        ),
        prompt_ui.Option(
            "s", "Start over — reword the request, same model, no memory of this attempt"
        ),
        prompt_ui.Option(
            "(anything else)",
            "Feedback — the model revises this same plan with your note",
            prompt_for_text=True,
        ),
    ]
    return options


@dataclass(frozen=True, slots=True)
class _AskView:
    """Which diagnostics accompany a plan, and the one thing threaded into
    the agent itself.

    All three default off, which is the change: `render.print_trace` and
    `_print_usage` both used to print unconditionally, putting a table of
    raw tool-call JSON and a line per round above the plan a person is
    being asked to approve. The information is still one flag away, and the
    trace still shows a one-line summary per call with no flag at all —
    what changes is which of them is the default."""

    trace: bool = False
    stats: bool = False
    debug: bool = False


def _show(data_dir: Path, key: bytes, plan: AgentPlan, *, view: _AskView) -> None:
    """The preview itself: what the agent looked up, then what it proposes,
    with `review`'s deterministic findings attached. `config.build_config`
    is read here rather than passed in because a feedback or regenerate
    round can land between two calls of this and the second one should
    describe the vault as it is now, not as it was when the request
    started."""
    render.print_trace(plan.trace, verbose=view.trace)
    if not plan.writes:
        return
    cfg = config.build_config(data_dir, key)
    findings = review.review_plan(plan, cfg)
    render.print_plan(
        plan,
        findings=findings,
        locations=cfg.known_locations,
        header=review.headline(findings),
    )


def _pick_writes(data_dir: Path, key: bytes, plan: AgentPlan) -> AgentPlan | None:
    """A plan narrowed to the writes left checked, or `None` if the person
    cancelled or unchecked everything. Deliberately does not commit what was
    picked: the narrowed plan goes back through the same preview and the
    same accept prompt, so the subset is seen before it is written rather
    than applied straight out of a checklist."""
    locations = ledger.load_locations_or_empty(data_dir, key)
    labels = [prompt_ui.Choice(render.write_summary(w, locations)) for w in plan.writes]
    picked = prompt_ui.multiselect(labels, title="Apply which changes?")
    if not picked:
        render.console.print("[dim](nothing picked — the plan is unchanged)[/dim]")
        return None
    return dataclass_replace(plan, writes=tuple(plan.writes[i] for i in picked))


def _build_agent(llm, data_dir: Path, key: bytes, *, model: ModelPreset, view: _AskView):  # noqa: ANN202
    """Every `AgentRunner` this module constructs, in one place — four call
    sites across the two decision loops each had to be updated in step
    whenever the constructor gained an argument (`debug` in §43, `show_usage`
    here), and two of them are inside `except` branches where a missed
    argument surfaces only on a retry."""
    return llm.AgentRunner(
        data_dir,
        key,
        model=model,
        debug=view.debug,
        # `--debug` implies `--stats`: it is the strictly-more-verbose flag,
        # and a session that asked for the raw per-round request/response
        # dumps wanting *fewer* numbers than the default is not a real case.
        show_usage=view.stats or view.debug,
    )


def _decide_prompt(plan: AgentPlan, *, dry_run: bool, defer: bool) -> str:
    """One decision, from the arrow-key menu on a terminal and from the
    printed option table plus a typed line everywhere else — `prompt_ui.select`
    picks between them, and returns the same strings either way."""
    options = _decision_options(
        dry_run,
        defer=defer,
        pick=len(plan.writes) > 1 and prompt_ui.interactive(),
    )
    return prompt_ui.select(options, default="a").strip()


def _print_dry_run_preview(plan: AgentPlan) -> None:
    """`--dry-run` withholds the write, not the accept/reject/feedback
    interface itself — the person still makes and sees a decision, just
    one that never reaches `agent.commit`/`store.append`."""
    for w in plan.writes:
        render.console.print(
            f"[dim](--dry-run) would record {w.kind.value} of {w.amount} {w.unit} "
            f"{w.product_id}[/dim]"
        )


def _prompt_regenerate(llm, current_model: ModelPreset) -> ModelPreset:
    """Same prompt, a fresh `propose()` (no trace or message history
    carried over — the model got a fair, independent attempt), a
    different model — for when the request was fine but this model's
    answer wasn't. Defaults to the model just used, so hitting Enter just
    re-rolls the identical request against the same one."""
    names = ", ".join(p.name for p in llm.MODEL_PRESETS)
    while True:
        model_name = typer.prompt(f"Model ({names})", default=current_model.name)
        try:
            return llm.model_preset(model_name)
        except KeyError:
            render.print_error(f"Unknown model {model_name!r}. Choose from: {names}")


def _prompt_start_over(current_prompt: str) -> str:
    """A new prompt, a fresh `propose()`, the same model — for when the
    request itself needs rewording, not just a different model's attempt
    at the same one."""
    return typer.prompt("Prompt", default=current_prompt)


# (key, column label, `ProposedWrite` field) for every field `e` can change.
# The endpoint rows are dropped for a write that has no such endpoint —
# `decide_change` rejects a purchase carrying a `from_location`, so offering
# to fill one in would only ever produce a rejection.
_EDIT_FIELDS = (
    ("p", "product", "product_id"),
    ("u", "unit", "unit"),
    ("n", "amount", "amount"),
    ("f", "from", "from_location"),
    ("t", "to", "to_location"),
)


def _editable_fields(write: ProposedWrite) -> list[tuple[str, str, str]]:
    return [
        row
        for row in _EDIT_FIELDS
        if row[2] not in ("from_location", "to_location") or getattr(write, row[2]) is not None
    ]


def _choose_write_to_edit(plan: AgentPlan, locations: dict[str, models.Location]) -> int | None:
    """Which write `e` acts on. One write needs no choosing; more than one
    gets the same menu every other decision in this command uses, and the
    numbered list plus a typed index off a terminal."""
    if len(plan.writes) == 1:
        return 0

    if not prompt_ui.interactive():
        for i, w in enumerate(plan.writes):
            render.console.print(f"  [{i}] {render.write_summary(w, locations)}")
        try:
            index = int(typer.prompt("Edit which one? (number)", default="0"))
        except ValueError:
            index = -1
        if not 0 <= index < len(plan.writes):
            render.print_error("Not a valid index — nothing edited.")
            return None
        return index

    options = [
        prompt_ui.Option(str(i), render.write_summary(w, locations))
        for i, w in enumerate(plan.writes)
    ]
    options.append(prompt_ui.Option("c", "Cancel — edit nothing"))
    answer = prompt_ui.select(options, default="0", title="Edit which change?")
    return int(answer) if answer.isdigit() else None


def _location_rows(locations: dict[str, models.Location]) -> list[prompt_ui.Row]:
    """Every active location, as `config show --locations-only` orders them —
    by display path, so a container and everything nested under it stay
    together. Typing filters on the path *and* the id, since either is a
    reasonable thing to half-remember."""
    rows = [
        prompt_ui.Row(
            value=loc.id,
            label=f"{config.location_path(locations, loc.id)}  ({loc.id})",
            search=f"{config.location_path(locations, loc.id)} {loc.id}",
        )
        for loc in locations.values()
        if not loc.retired
    ]
    return sorted(rows, key=lambda row: row.label)


def _choose_location(
    locations: dict[str, models.Location], field: str, current: str | None
) -> str | None:
    """A location picked from the layout, not typed. Locations are a closed
    set — `decide.resolve_location` rejects one that is not configured, and
    unlike a product there is no auto-registration to fall back on — so
    offering the list is both possible and the only way to be sure the answer
    resolves. `None` when cancelled or when there is no terminal to pick on,
    which leaves the field as it was."""
    rows = _location_rows(locations)
    if not rows:
        render.print_error("No locations configured — nothing to pick from.")
        return None
    return prompt_ui.pick(rows, title=f"Which location for {field}?", current=current)


def _edit_fields_by_menu(
    write: ProposedWrite, locations: dict[str, models.Location]
) -> dict[str, str | None] | None:
    """Pick a field, retype that one, repeat until done — rather than walking
    every field in order and pressing Enter through the ones that were
    already right, which is what correcting a single mistyped location used
    to cost. Values are shown on their own rows, so the menu doubles as the
    record of what has been changed so far.

    Returns the edited values, or `None` if cancelled. Nothing is validated
    or applied here; `_apply_edit` does both, once, on the way out."""
    fields = _editable_fields(write)
    # Seeded from every field, not just the editable ones: an endpoint this
    # write does not have still has to reach `_apply_edit` as `None` rather
    # than be missing, since `decide_change` distinguishes the two.
    values: dict[str, str | None] = {
        "product_id": write.product_id,
        "unit": write.unit,
        "amount": str(write.amount),
        "from_location": write.from_location,
        "to_location": write.to_location,
    }

    while True:
        options = [
            prompt_ui.Option(key, f"{label:8s} {values[field]}") for key, label, field in fields
        ]
        options.append(prompt_ui.Option("d", "Done — re-check this change"))
        options.append(prompt_ui.Option("c", "Cancel — discard these edits"))
        answer = prompt_ui.select(
            options,
            default=fields[0][0],
            title=f"Editing: {render.write_summary(write, locations)}",
        )
        if answer == "d":
            return values
        # "r" is what `prompt_ui.select` answers for Escape — cancelling an
        # edit, here, not rejecting the plan: the caller returns the plan
        # unchanged and asks for a decision on it again.
        if answer in ("c", "r"):
            return None
        match = next((row for row in fields if row[0] == answer), None)
        if match is None:
            continue
        _key, label, field = match
        if field in ("from_location", "to_location"):
            picked = _choose_location(locations, label, values[field])
            if picked is not None:
                values[field] = picked
        else:
            values[field] = typer.prompt(label, default=values[field] or "")


def _edit_fields_by_walkthrough(write: ProposedWrite) -> dict[str, str | None]:
    """Every field in order, each defaulting to what it already holds — the
    only shape a piped or scripted answer can take, and unchanged from what
    `e` has always read."""
    values: dict[str, str | None] = {
        "product_id": typer.prompt("product_id", default=write.product_id),
        "unit": typer.prompt("unit", default=write.unit),
        "amount": typer.prompt("amount", default=str(write.amount)),
        "from_location": write.from_location,
        "to_location": write.to_location,
    }
    if write.from_location is not None:
        values["from_location"] = typer.prompt("from_location", default=write.from_location)
    if write.to_location is not None:
        values["to_location"] = typer.prompt("to_location", default=write.to_location)
    return values


def _apply_edit(
    data_dir: Path,
    key: bytes,
    plan: AgentPlan,
    index: int,
    values: dict[str, str | None],
) -> AgentPlan:
    """Re-validates the edited write through the same `decide_change` gate
    every model-proposed write already passes, so a typo in the correction
    itself cannot reach `commit` unchecked."""
    write = plan.writes[index]
    try:
        amount = Decimal(values["amount"] or "")
    except InvalidOperation:
        render.print_error(f"Not a valid amount: {values['amount']!r} — nothing edited.")
        return plan

    edited = dataclass_replace(
        write,
        product_id=values["product_id"] or "",
        amount=amount,
        unit=values["unit"] or "",
        from_location=values["from_location"],
        to_location=values["to_location"],
    )
    try:
        _writes, messages = decide.decide_change(
            kind=edited.kind,
            product_id=edited.product_id,
            amount=edited.amount,
            unit=edited.unit,
            from_location=edited.from_location,
            to_location=edited.to_location,
            actor=paths.current_user(),
            occurred_at=datetime.now(UTC),
            inventory=ledger.build_inventory(data_dir, key),
            cfg=config.build_config(data_dir, key),
        )
    except Rejected as e:
        render.print_error(f"Edit rejected: {e}")
        return plan

    writes = list(plan.writes)
    # `effects` is dropped rather than recomputed: it described the write the
    # model proposed, and this is a different one. `render.print_plan` falls
    # back to `current_amount` for a write without a projection, so the
    # edited row still says what is there — it just stops claiming an
    # "after" computed for the amount that was replaced.
    writes[index] = dataclass_replace(edited, warnings=tuple(messages), effects=())
    render.print_success("Edit applied — nothing written yet, decide again below.")
    return dataclass_replace(plan, writes=tuple(writes))


def _prompt_edit(data_dir: Path, key: bytes, plan: AgentPlan) -> AgentPlan:
    """Manually corrects one proposed write's fields directly — no model
    call at all, and the fastest fix for something the model got mostly
    right (a mistyped product name it had spelled correctly moments
    earlier in a tool result, a location one digit off). "Regenerate" and
    "start over" are the tools for a model reasoning past correcting; this
    is only for its typing."""
    locations = ledger.load_locations_or_empty(data_dir, key)
    index = _choose_write_to_edit(plan, locations)
    if index is None:
        return plan
    write = plan.writes[index]
    values = (
        _edit_fields_by_menu(write, locations)
        if prompt_ui.interactive()
        else _edit_fields_by_walkthrough(write)
    )
    if values is None:
        return plan
    return _apply_edit(data_dir, key, plan, index, values)


def _ask_one(
    data_dir: Path, key: bytes, llm, prompt: str, *, dry_run: bool, view: _AskView
) -> None:
    """The original one-shot flow: exits the process on error or once the
    single request is resolved, since there is nothing else this
    invocation would do afterward."""
    model = llm.DEFAULT_MODEL_PRESET
    try:
        agent = _build_agent(llm, data_dir, key, model=model, view=view)
        plan = agent.propose(prompt)
    except FileNotFoundError as e:
        render.print_error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:
        render.print_error(f"Agent error: {e}")
        raise typer.Exit(code=1) from e

    while True:
        _show(data_dir, key, plan, view=view)
        if not plan.writes:
            if plan.reply_text:
                render.console.print(plan.reply_text)
            if dry_run:
                render.console.print("[dim](--dry-run: this request produced no writes)[/dim]")
            return

        answer = _decide_prompt(plan, dry_run=dry_run, defer=False)
        choice = answer.lower()
        if choice in ("a", "accept"):
            if dry_run:
                _print_dry_run_preview(plan)
            else:
                # Not caught here: a Rejected raised while committing is not
                # a modeled outcome the model can react to (the human
                # already accepted the plan) and should propagate exactly
                # the way `add` already lets `decide_change`'s Rejected
                # reach `main`'s handler.
                for summary in agent.commit(plan):
                    render.print_success(summary)
            return
        if choice in ("r", "reject", "quit", "exit"):
            return
        if choice in ("e", "edit"):
            plan = _prompt_edit(data_dir, key, plan)
            continue
        if choice in ("p", "pick"):
            plan = _pick_writes(data_dir, key, plan) or plan
            continue
        if choice in ("g", "generate", "regenerate"):
            model = _prompt_regenerate(llm, agent.model)
            try:
                agent = _build_agent(llm, data_dir, key, model=model, view=view)
                plan = agent.propose(prompt)
            except Exception as e:
                render.print_error(f"Agent error: {e}")
                raise typer.Exit(code=1) from e
            continue
        if choice in ("s", "start", "start over"):
            prompt = _prompt_start_over(prompt)
            try:
                agent = _build_agent(llm, data_dir, key, model=model, view=view)
                plan = agent.propose(prompt)
            except Exception as e:
                render.print_error(f"Agent error: {e}")
                raise typer.Exit(code=1) from e
            continue

        try:
            plan = agent.revise(answer)
        except Exception as e:
            render.print_error(f"Agent error: {e}")
            raise typer.Exit(code=1) from e


def _ask_loop(
    data_dir: Path, key: bytes, llm, *, first_prompt: str | None, dry_run: bool, view: _AskView
) -> None:
    """Repeatedly prompts for a new request rather than exiting after one.
    Unlike `_ask_one`, a failure here is caught and queued (queue.py)
    instead of ending the session — the specific crash this was built to
    survive is mistral.rs's max-seq-len KV-cache exhaustion, which takes
    the whole process down uncaught (see the design journal), losing
    everything about the request that triggered it if there's nothing
    outside the process remembering it existed."""
    render.console.print(
        '[dim]Type a request, "queue" to list pending ones, "retry N" to revisit '
        'one, or "quit" to exit.[/dim]'
    )
    if first_prompt is not None:
        _ask_loop_request(data_dir, key, llm, first_prompt, dry_run=dry_run, view=view)

    while True:
        pending = queue.load(data_dir)
        if pending:
            render.console.print(
                f'[dim]{len(pending)} request(s) pending — type "queue" to review.[/dim]'
            )
        try:
            answer = typer.prompt(
                'Request (or "queue" / "retry N" / "quit")', default="", show_default=False
            ).strip()
        except (EOFError, KeyboardInterrupt):
            render.console.print()
            return

        if not answer:
            continue
        lowered = answer.lower()
        if lowered in ("quit", "exit"):
            return

        if lowered == "queue":
            if not pending:
                render.console.print("[dim](queue is empty)[/dim]")
            for i, item in enumerate(pending):
                render.console.print(
                    f"  [{i}] {item.prompt}  [dim]({item.reason}, {item.attempts} attempt(s))[/dim]"
                )
            continue

        if lowered.startswith("retry "):
            index_text = answer.split(maxsplit=1)[1]
            try:
                item = queue.dequeue(data_dir, int(index_text))
            except (ValueError, IndexError):
                render.print_error(f"No queued request at index {index_text!r}.")
                continue
            _ask_loop_request(
                data_dir, key, llm, item.prompt, dry_run=dry_run, view=view, retrying=item
            )
            continue

        _ask_loop_request(data_dir, key, llm, answer, dry_run=dry_run, view=view)


def _ask_loop_request(
    data_dir: Path,
    key: bytes,
    llm,
    prompt: str,
    *,
    dry_run: bool,
    view: _AskView,
    retrying: queue.QueuedRequest | None = None,
) -> None:
    """One request within `_ask_loop`. `retrying` is the queue entry this
    came from, if any — its `attempts` count carries forward if the retry
    also fails, so the queue listing shows how many times something has
    already been tried, not just that it's pending."""
    next_attempts = retrying.attempts + 1 if retrying is not None else 0
    model = llm.DEFAULT_MODEL_PRESET

    def _fail(e: Exception) -> None:
        render.print_error(f"Agent error: {e} — queued for later.")
        queue.enqueue(data_dir, prompt, reason=f"error: {e}", attempts=next_attempts)

    try:
        agent = _build_agent(llm, data_dir, key, model=model, view=view)
        plan = agent.propose(prompt)
    except Exception as e:
        _fail(e)
        return

    while True:
        _show(data_dir, key, plan, view=view)
        if not plan.writes:
            if plan.reply_text:
                render.console.print(plan.reply_text)
            return

        answer = _decide_prompt(plan, dry_run=dry_run, defer=True)
        choice = answer.lower()
        if choice in ("a", "accept"):
            if dry_run:
                _print_dry_run_preview(plan)
            else:
                for summary in agent.commit(plan):
                    render.print_success(summary)
            return
        if choice in ("r", "reject", "quit", "exit"):
            return
        if choice in ("d", "defer"):
            queue.enqueue(data_dir, prompt, reason="deferred", attempts=next_attempts)
            render.console.print("[dim](queued for later)[/dim]")
            return
        if choice in ("e", "edit"):
            plan = _prompt_edit(data_dir, key, plan)
            continue
        if choice in ("p", "pick"):
            plan = _pick_writes(data_dir, key, plan) or plan
            continue
        # `prompt` and `model` are reassigned in these two branches, not
        # shadowed — `_fail` closes over the same names, so a later failure
        # or defer on this new attempt enqueues what it actually ran, not
        # the original request.
        if choice in ("g", "generate", "regenerate"):
            model = _prompt_regenerate(llm, agent.model)
            try:
                agent = _build_agent(llm, data_dir, key, model=model, view=view)
                plan = agent.propose(prompt)
            except Exception as e:
                _fail(e)
                return
            continue
        if choice in ("s", "start", "start over"):
            prompt = _prompt_start_over(prompt)
            try:
                agent = _build_agent(llm, data_dir, key, model=model, view=view)
                plan = agent.propose(prompt)
            except Exception as e:
                _fail(e)
                return
            continue

        try:
            plan = agent.revise(answer)
        except Exception as e:
            _fail(e)
            return


@models_app.command("list")
def models_list(
    names_only: Annotated[
        bool,
        typer.Option("--names-only", help="One bare preset name per line — for scripting."),
    ] = False,
) -> None:
    """List every `ModelPreset` in the registry and whether its GGUF is
    already in the local Hugging Face cache."""
    llm = _import_llm()
    if names_only:
        for model in llm.MODEL_PRESETS:
            print(model.name)
        return
    for model in llm.MODEL_PRESETS:
        cached = "cached" if llm.is_cached(model) else "not cached"
        repo = f"{model.quantized_model_id}/{model.quantized_filename}"
        render.console.print(f"  {model.name:24s} {repo}  [dim]({cached})[/dim]")


@models_app.command("pull")
def models_pull(
    names: Annotated[
        list[str] | None,
        typer.Argument(help="Preset names to pull; defaults to every preset in the registry."),
    ] = None,
) -> None:
    """Download every named preset's GGUF into the local Hugging Face
    cache (all of them, by default) by loading each one just long enough
    to trigger `mistralrs`' own download-on-load — the same mechanism
    `sumac ask` relies on, without needing to run `sumac ask` once per
    model and pick it from the "g" regenerate prompt. Already-cached
    presets are skipped."""
    # The one command whose whole job is a long model load: mistral.rs's own
    # progress is the only sign it is working, so this keeps it.
    llm = _import_llm(verbose=True)
    targets = list(llm.MODEL_PRESETS)
    if names:
        try:
            targets = [llm.model_preset(name) for name in names]
        except KeyError as e:
            valid = ", ".join(p.name for p in llm.MODEL_PRESETS)
            render.print_error(f"Unknown model {e.args[0]!r}. Choose from: {valid}")
            raise typer.Exit(code=1) from e

    failed: list[str] = []
    for model in targets:
        if llm.is_cached(model):
            render.console.print(f"[dim]{model.name}: already cached, skipping[/dim]")
            continue
        try:
            llm._build_runner(model)
        except Exception as e:  # noqa: BLE001 - report and keep pulling the rest
            render.print_error(f"{model.name}: {e}")
            failed.append(model.name)
        else:
            render.print_success(f"{model.name}: pulled")
    if failed:
        raise typer.Exit(code=1)


def main() -> None:
    try:
        app()
    except (SumacError, SealError) as e:
        render.print_error(str(e))
        raise SystemExit(1) from None
