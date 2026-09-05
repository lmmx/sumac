#!/usr/bin/env python3
"""Render every `sumac ask` review screen against hand-built plans.

    uv run scripts/preview-ask-ui.py             # print every scene
    uv run scripts/preview-ask-ui.py --scene compound
    uv run scripts/preview-ask-ui.py --svg out/  # one SVG per scene

No vault, no model, no `mistralrs` import: the scenes are literal
`AgentPlan`s, so a rendering change is compared against fixed input.
`docs/journal/2026-09-04-trace-and-verdict-redesign.md` records why the model
side cannot be compared that way: a `mistralrs.Runner`'s RNG stream position
depends on everything that ran before it in the same session, so two sessions
differing only in wording are not a controlled comparison. Rendering has no
such dependency.

The plans below are ones the journal already discusses: the ragu/tomatoes
compound request from the ask design entry's §2, and the fabricated "Basmati
Rice Bag" from docs/journal/2026-09-04-basmati-rice-unit-mismatch.md.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sumac import models, prompt_ui, render, review  # noqa: E402
from sumac.cli import _decision_options  # noqa: E402
from sumac.config import Config  # noqa: E402
from sumac.llm import AgentPlan, LocationEffect, ProposedWrite, ToolCallRecord  # noqa: E402
from sumac.models import ChangeKind  # noqa: E402

LOCATIONS = {
    "fridge": models.Location(id="fridge", name="Fridge"),
    "fridge-shelf-2": models.Location(id="fridge-shelf-2", name="Shelf 2", parent_id="fridge"),
    "fridge-door": models.Location(id="fridge-door", name="Door", parent_id="fridge"),
    "freezer": models.Location(id="freezer", name="Freezer"),
    "pantry": models.Location(id="pantry", name="Pantry"),
}
PRODUCTS = {
    "Strawberry Jam": models.Product(id="Strawberry Jam", name="Strawberry Jam", unit="jar"),
    "Homemade Ragu": models.Product(id="Homemade Ragu", name="Homemade Ragu", unit="tub"),
    "Tinned Tomatoes": models.Product(id="Tinned Tomatoes", name="Tinned Tomatoes", unit="tin"),
    "Basmati Rice": models.Product(id="Basmati Rice", name="Basmati Rice", unit="jug"),
}
CFG = Config(
    known_locations=LOCATIONS,
    active_locations=LOCATIONS,
    known_products=PRODUCTS,
    active_products=PRODUCTS,
    anomalies=(),
)

FIND_RESULT = (
    '{"products": [{"product_id": "Homemade Ragu", "is_exact_match": true, '
    '"locations": [{"location_id": "freezer", "location_path": "Freezer", '
    '"amount": "2", "unit": "tub"}]}]}'
)


def _consumption() -> AgentPlan:
    return AgentPlan(
        reply_text="",
        writes=(
            ProposedWrite(
                kind=ChangeKind.CONSUMPTION,
                product_id="Strawberry Jam",
                amount=Decimal(1),
                unit="jar",
                from_location="fridge-door",
                to_location=None,
                effects=(
                    LocationEffect("fridge-door", "Strawberry Jam", "jar", Decimal(3), Decimal(2)),
                ),
            ),
        ),
        trace=(
            ToolCallRecord(
                "sumac_find_inventory",
                {"query": "jam"},
                '{"products": [{"product_id": "Strawberry Jam", "is_exact_match": true, '
                '"locations": [{"location_id": "fridge-door", "location_path": "Fridge > Door", '
                '"amount": "3", "unit": "jar"}]}]}',
            ),
        ),
    )


def _compound() -> AgentPlan:
    """The ask design entry's §2 worked example: one physical action, four
    writes, and thirteen `sumac add` invocations before `ask` existed."""
    return AgentPlan(
        reply_text="",
        writes=(
            ProposedWrite(
                kind=ChangeKind.MOVEMENT,
                product_id="Homemade Ragu",
                amount=Decimal(1),
                unit="tub",
                from_location="freezer",
                to_location="fridge-shelf-2",
                effects=(
                    LocationEffect("freezer", "Homemade Ragu", "tub", Decimal(2), Decimal(1)),
                    LocationEffect("fridge-shelf-2", "Homemade Ragu", "tub", None, Decimal(1)),
                ),
            ),
            ProposedWrite(
                kind=ChangeKind.MOVEMENT,
                product_id="Tinned Tomatoes",
                amount=Decimal(1),
                unit="tin",
                from_location="pantry",
                to_location="fridge-shelf-2",
                effects=(
                    LocationEffect("pantry", "Tinned Tomatoes", "tin", Decimal(8), Decimal(7)),
                    LocationEffect("fridge-shelf-2", "Tinned Tomatoes", "tin", None, Decimal(1)),
                ),
            ),
            ProposedWrite(
                kind=ChangeKind.CONSUMPTION,
                product_id="Strawberry Jam",
                amount=Decimal(5),
                unit="jar",
                from_location="fridge-door",
                to_location=None,
                warnings=("note: fridge-door held 3 jar, recorded 5 jar — adjusted",),
                effects=(LocationEffect("fridge-door", "Strawberry Jam", "jar", Decimal(3), None),),
            ),
        ),
        trace=(
            ToolCallRecord("sumac_find_inventory", {"query": "ragu"}, FIND_RESULT),
            ToolCallRecord(
                "sumac_move_inventory",
                {"product_id": "Homemade Ragu", "amount": "1"},
                '{"status": "proposed", "product_id": "Homemade Ragu"}',
            ),
        ),
    )


def _fabricated() -> AgentPlan:
    """`docs/journal/2026-09-04-basmati-rice-unit-mismatch.md`'s outcome: a
    product id the vault never held and the agent's own search did not
    return."""
    return AgentPlan(
        reply_text="",
        writes=(
            ProposedWrite(
                kind=ChangeKind.DISCOVERY,
                product_id="Basmati Rice Bag",
                amount=Decimal(1),
                unit="bag",
                from_location=None,
                to_location="pantry",
                warnings=(
                    "'Basmati Rice Bag' is not a registered product — did you mean "
                    "'Basmati Rice'? Registering 'Basmati Rice Bag' instead; "
                    "run `sumac correct` if it was a typo.",
                ),
                effects=(LocationEffect("pantry", "Basmati Rice Bag", "bag", None, Decimal(1)),),
            ),
        ),
        trace=(
            ToolCallRecord(
                "sumac_find_inventory",
                {"query": "Basmati Rice"},
                '{"products": [{"product_id": "Basmati Rice", "is_exact_match": true, '
                '"locations": [{"location_id": "pantry", "location_path": "Pantry", '
                '"amount": "1", "unit": "jug"}]}]}',
            ),
        ),
    )


def _read_only() -> AgentPlan:
    return AgentPlan(
        reply_text="The Homemade Ragu is in the Freezer — 2 tubs.",
        writes=(),
        trace=(ToolCallRecord("sumac_find_inventory", {"query": "ragu"}, FIND_RESULT),),
    )


def _show_plan(plan: AgentPlan, *, verbose_trace: bool = False) -> None:
    render.print_trace(plan.trace, verbose=verbose_trace)
    if not plan.writes:
        render.console.print(plan.reply_text)
        return
    findings = review.review_plan(plan, CFG)
    render.print_plan(
        plan,
        findings=findings,
        locations=LOCATIONS,
        header=review.headline(findings),
    )


def scene_single() -> None:
    _show_plan(_consumption())
    _menu(_decision_options(dry_run=False), cursor=0)


def scene_compound() -> None:
    _show_plan(_compound())
    _menu(_decision_options(dry_run=False, defer=True, pick=True), cursor=3)


def scene_fabricated() -> None:
    _show_plan(_fabricated())
    _menu(_decision_options(dry_run=False), cursor=1)


def scene_read_only() -> None:
    _show_plan(_read_only())


def scene_trace() -> None:
    """What `--trace` restores, against the same plan `single` renders."""
    _show_plan(_consumption(), verbose_trace=True)


def scene_typed() -> None:
    """The non-TTY path: the printed option table plus a typed line, which is
    what a pipe, a test, and `evals/` receive."""
    render.print_decision_options(
        [(o.key, o.description) for o in _decision_options(dry_run=False)]
    )
    render.console.print(r"Choice \[a]: ")


def scene_checklist() -> None:
    """`p`'s per-write picker, drawn at one cursor position. The live version
    redraws on every keypress."""
    plan = _compound()
    choices = [
        prompt_ui.Choice(render.write_summary(w, LOCATIONS), checked=i != 2)
        for i, w in enumerate(plan.writes)
    ]
    render.console.print(
        prompt_ui._checklist(choices, [c.checked for c in choices], 2, "Apply which changes?")
    )


def scene_edit() -> None:
    """`e`'s two menus, which change and then which field, each drawn at one
    cursor position. The live versions redraw on every keypress."""
    plan = _compound()
    write = plan.writes[2]
    picker = [
        prompt_ui.Option(str(i), render.write_summary(w, LOCATIONS))
        for i, w in enumerate(plan.writes)
    ]
    picker.append(prompt_ui.Option("c", "Cancel — edit nothing"))
    render.console.print(
        prompt_ui._menu(picker, 2, "Edit which change?", "↑/↓ move · enter choose · esc cancel")
    )

    fields = [
        prompt_ui.Option(key, f"{label:8s} {value}")
        for key, label, value in (
            ("p", "product", write.product_id),
            ("u", "unit", write.unit),
            ("n", "amount", write.amount),
            ("f", "from", write.from_location),
        )
    ]
    fields.append(prompt_ui.Option("d", "Done — re-check this change"))
    fields.append(prompt_ui.Option("c", "Cancel — discard these edits"))
    render.console.print()
    render.console.print(
        prompt_ui._menu(
            fields,
            2,
            f"Editing: {render.write_summary(write, LOCATIONS)}",
            "↑/↓ move · enter choose · esc cancel",
        )
    )


def scene_location_picker() -> None:
    """`e`'s location field: the layout, filtered as you type, rather than a
    free-text field that accepts an unconfigured location."""
    from sumac.cli import _location_rows

    rows = _location_rows(LOCATIONS)
    render.console.print(prompt_ui._pick_view(rows, 2, "", "Which location for to?", len(rows)))
    render.console.print()
    filtered = [row for row in rows if row.matches("fri")]
    render.console.print(
        prompt_ui._pick_view(filtered, 1, "fri", "Which location for to?", len(rows))
    )


def scene_value_pickers() -> None:
    """The unit and product fields: an existing value to reuse, or the typed
    text as a new one. Unlike a location, either may legitimately be new."""
    from collections import Counter

    from sumac.cli import _product_rows, _unit_rows

    observed = {
        "Strawberry Jam": Counter({"jar": 12}),
        "Homemade Ragu": Counter({"tub": 8}),
        "Tinned Tomatoes": Counter({"tin": 31, "can": 2}),
    }
    rows = _unit_rows(observed, CFG, "Strawberry Jam")
    render.console.print(prompt_ui._pick_view(rows, 0, "", "Which unit?", len(rows)))

    render.console.print()
    typed = "sachet"
    visible = prompt_ui._visible_rows(rows, typed, True, "new unit")
    render.console.print(
        prompt_ui._pick_view(
            visible,
            len(visible) - 1,
            typed,
            "Which unit?",
            len(rows),
            "↑/↓ move · type to filter or add · enter choose · esc cancel",
        )
    )

    render.console.print()
    products = _product_rows(CFG)
    visible = prompt_ui._visible_rows(products, "tom", True, "new product")
    render.console.print(
        prompt_ui._pick_view(
            visible,
            0,
            "tom",
            "Which product?",
            len(products),
            "↑/↓ move · type to filter or add · enter choose · esc cancel",
        )
    )


def scene_amount() -> None:
    """The amount field: numeric only, stepped with the arrow keys."""
    hint = "↑/↓ adjust · digits to type · enter accept · esc cancel"
    render.console.print(prompt_ui._number_view("1", "Amount?", hint, True))
    render.console.print()
    render.console.print(prompt_ui._number_view("2.5", "Amount?", hint, True))
    render.console.print()
    render.console.print(prompt_ui._number_view(".", "Amount?", hint, False))


def _menu(options: list[prompt_ui.Option], cursor: int) -> None:
    """The arrow-key menu drawn at one cursor position."""
    render.console.print()
    render.console.print(
        prompt_ui._menu(options, cursor, None, "↑/↓ move · enter choose · esc reject")
    )


SCENES = {
    "single": scene_single,
    "compound": scene_compound,
    "fabricated": scene_fabricated,
    "read-only": scene_read_only,
    "trace": scene_trace,
    "typed": scene_typed,
    "checklist": scene_checklist,
    "edit": scene_edit,
    "location-picker": scene_location_picker,
    "value-pickers": scene_value_pickers,
    "amount": scene_amount,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(SCENES), action="append")
    parser.add_argument("--svg", type=Path, help="Write one SVG per scene into this directory.")
    parser.add_argument("--width", type=int, default=96)
    args = parser.parse_args()

    names = args.scene or list(SCENES)
    for name in names:
        console = Console(record=bool(args.svg), width=args.width)
        render.console = console
        prompt_ui.console = console
        if not args.svg:
            console.rule(f"[bold]{name}")
        SCENES[name]()
        if args.svg:
            args.svg.mkdir(parents=True, exist_ok=True)
            path = args.svg / f"{name}.svg"
            path.write_text(console.export_svg(title=f"sumac ask · {name}"), encoding="utf-8")
            print(f"wrote {path}")
        else:
            console.print()


if __name__ == "__main__":
    main()
