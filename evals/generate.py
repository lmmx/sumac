"""Templated case generation — see docs/journal/2026-09-02-eval-suite.md,
"Case generation". Each template takes one `FamilyVocab` and returns one
`EvalCase`, deriving the gold write from the template's own parameters
rather than from a hand annotation, so a case's expectation cannot drift
from its prompt. `generate_cases(family)` runs every template once against
`family`; `cases.all_cases()` runs it over every family and appends the
hand-written `hard` table."""

from __future__ import annotations

from collections.abc import Callable

from evals import seed
from evals.cases import AskOrAct, EvalCase, NoWrites, TraceExpectation, Writes, WriteSpec
from evals.vocab import FamilyVocab
from sumac.llm import QueryKind
from sumac.models import ChangeKind

Template = Callable[[FamilyVocab], EvalCase]

_NEW_PRODUCT_AMOUNT = "2"
_NEW_PRODUCT_UNIT = "bottles"


def _add_location_path(f: FamilyVocab) -> EvalCase:
    p = f.near_miss_brand
    loc = seed.ROLE_LOCATIONS["near_miss_brand"]
    path = seed.location_path(loc)
    return EvalCase(
        id=f"{f.id}-add-location-path",
        family_id=f.id,
        prompt=f"Add {p.amount} {p.unit} of {p.id} to {path}",
        kind=QueryKind.ADD,
        expect=Writes((WriteSpec(ChangeKind.DISCOVERY, p.id, p.amount, p.unit, to_location=loc),)),
        template="add.location_path",
    )


def _add_indirect_stock(f: FamilyVocab) -> EvalCase:
    p = f.near_miss_brand
    loc = seed.ROLE_LOCATIONS["near_miss_brand"]
    return EvalCase(
        id=f"{f.id}-add-indirect-stock",
        family_id=f.id,
        prompt=f"Add {p.amount} {p.unit} of {p.id} to the pantry, same spot as existing stock",
        kind=QueryKind.ADD,
        expect=Writes((WriteSpec(ChangeKind.DISCOVERY, p.id, p.amount, p.unit, to_location=loc),)),
        trace=TraceExpectation(called=("sumac_find_inventory",)),
        template="add.indirect_stock",
    )


def _add_new_product(f: FamilyVocab) -> EvalCase:
    loc = "storage"
    path = seed.location_path(loc)
    return EvalCase(
        id=f"{f.id}-add-new-product",
        family_id=f.id,
        prompt=f"Add {_NEW_PRODUCT_AMOUNT} {_NEW_PRODUCT_UNIT} of {f.absent_product} to {path}",
        kind=QueryKind.ADD,
        expect=Writes(
            (
                WriteSpec(
                    ChangeKind.DISCOVERY,
                    f.absent_product,
                    _NEW_PRODUCT_AMOUNT,
                    _NEW_PRODUCT_UNIT,
                    to_location=loc,
                ),
            )
        ),
        trace=TraceExpectation(called=("sumac_find_inventory",)),
        template="add.new_product",
    )


def _add_discriminator(f: FamilyVocab) -> EvalCase:
    b = f.discriminator_b
    loc = seed.ROLE_LOCATIONS["discriminator_b"]
    path = seed.location_path(loc)
    return EvalCase(
        id=f"{f.id}-add-discriminator",
        family_id=f.id,
        prompt=f"Add {b.amount} {b.unit} of {b.id} to {path}, with the existing stock",
        kind=QueryKind.ADD,
        expect=Writes((WriteSpec(ChangeKind.DISCOVERY, b.id, b.amount, b.unit, to_location=loc),)),
        template="add.discriminator",
    )


def _add_unit_collision(f: FamilyVocab) -> EvalCase:
    rice = f.rice
    return EvalCase(
        id=f"{f.id}-add-unit-collision",
        family_id=f.id,
        prompt=(
            f"Add 1 {f.rice_new_unit} of {rice.id} ({f.rice_size}) next to the "
            f"existing {rice.unit} of {rice.id}"
        ),
        kind=QueryKind.ADD,
        expect=NoWrites(),
        template="add.unit_collision",
    )


def _add_missing_amount(f: FamilyVocab) -> EvalCase:
    return EvalCase(
        id=f"{f.id}-add-missing-amount",
        family_id=f.id,
        prompt=f"Add {f.category_new_name} to the pantry, with the other {f.category_word}",
        kind=QueryKind.ADD,
        expect=AskOrAct(),
        template="add.missing_amount",
    )


def _add_positional(f: FamilyVocab) -> EvalCase:
    # Blocked: `decide._resolve_location` accepts only an id or an exact
    # display path (`src/sumac/decide.py:100-123`); nothing in the harness
    # lets a model turn "3rd position along, 3rd row down" into either.
    # See docs/journal/2026-09-02-eval-suite.md, Missing. The gold below is
    # symbolic — recorded so the case flips to headline-scored the moment
    # that capability exists, without redesigning the case.
    return EvalCase(
        id=f"{f.id}-add-positional",
        family_id=f.id,
        prompt=(
            f"Add 3 cartons of {f.absent_product} to the {seed.POSITIONAL_GRID_COLOUR} "
            "pantry, 3rd position along, 3rd row down"
        ),
        kind=QueryKind.ADD,
        expect=Writes(
            (
                WriteSpec(
                    ChangeKind.DISCOVERY,
                    f.absent_product,
                    "3",
                    "cartons",
                    to_location=seed.POSITIONAL_TARGET,
                ),
            )
        ),
        tags=frozenset({"blocked"}),
        template="add.positional",
    )


def _add_absent_spot(f: FamilyVocab) -> EvalCase:
    # Blocked for the same reason as `_add_positional` — "the now-empty spot
    # where the old stock was" names no id and no exact display path.
    return EvalCase(
        id=f"{f.id}-add-absent-spot",
        family_id=f.id,
        prompt=(
            f"Add 1 can of {f.absent_product} to the pantry, in the now-empty "
            "spot where the old stock was"
        ),
        kind=QueryKind.ADD,
        expect=Writes(
            (WriteSpec(ChangeKind.DISCOVERY, f.absent_product, "1", "can", to_location="pantry"),)
        ),
        tags=frozenset({"blocked"}),
        template="add.absent_spot",
    )


def _remove_partial(f: FamilyVocab) -> EvalCase:
    ms = f.movement_source
    loc = seed.ROLE_LOCATIONS["movement_source"]
    return EvalCase(
        id=f"{f.id}-remove-partial",
        family_id=f.id,
        prompt=f"I used 1 tub of {ms.id}",
        kind=QueryKind.REMOVE,
        expect=Writes((WriteSpec(ChangeKind.CONSUMPTION, ms.id, "1", "tub", from_location=loc),)),
        trace=TraceExpectation(called=("sumac_find_inventory",)),
        template="remove.partial",
    )


def _remove_all(f: FamilyVocab) -> EvalCase:
    target = f.consumption_target
    loc = seed.ROLE_LOCATIONS["consumption_target"]
    return EvalCase(
        id=f"{f.id}-remove-all",
        family_id=f.id,
        prompt=f"we finished the {target.id.lower()}",
        kind=QueryKind.REMOVE,
        expect=Writes(
            (
                WriteSpec(
                    ChangeKind.CONSUMPTION, target.id, target.amount, target.unit, from_location=loc
                ),
            )
        ),
        trace=TraceExpectation(called=("sumac_find_inventory",)),
        template="remove.all",
    )


def _move_explicit(f: FamilyVocab) -> EvalCase:
    ms = f.movement_source
    src = seed.ROLE_LOCATIONS["movement_source"]
    dst = seed.MOVE_DESTINATION
    src_path = seed.location_path(src)
    dst_path = seed.location_path(dst)
    return EvalCase(
        id=f"{f.id}-move-explicit",
        family_id=f.id,
        prompt=f"move 1 tub of {ms.id} from {src_path} to {dst_path}",
        kind=QueryKind.REMOVE,
        expect=Writes(
            (WriteSpec(ChangeKind.MOVEMENT, ms.id, "1", "tub", from_location=src, to_location=dst),)
        ),
        template="move.explicit",
    )


def _find_where(f: FamilyVocab) -> EvalCase:
    target = f.consumption_target
    loc = seed.ROLE_LOCATIONS["consumption_target"]
    location_name = seed.LOCATIONS_BY_ID[loc].name
    return EvalCase(
        id=f"{f.id}-find-where",
        family_id=f.id,
        prompt=f"where is the {target.id.lower()}?",
        kind=QueryKind.FIND,
        expect=NoWrites(),
        trace=TraceExpectation(called=("sumac_find_inventory",), reply_mentions=(location_name,)),
        template="find.where",
    )


def _find_quantity(f: FamilyVocab) -> EvalCase:
    rice = f.rice
    return EvalCase(
        id=f"{f.id}-find-quantity",
        family_id=f.id,
        prompt=f"how much {rice.id.lower()} do we have?",
        kind=QueryKind.FIND,
        expect=NoWrites(),
        trace=TraceExpectation(
            called=("sumac_find_inventory",), reply_amount=(rice.amount, rice.unit)
        ),
        template="find.quantity",
    )


def _find_shared_word(f: FamilyVocab) -> EvalCase:
    return EvalCase(
        id=f"{f.id}-find-shared-word",
        family_id=f.id,
        prompt=f"do we have any {f.shared_word}?",
        kind=QueryKind.FIND,
        expect=NoWrites(),
        trace=TraceExpectation(
            called=("sumac_find_inventory",),
            reply_before=((f.discriminator_a.id, f.shared_word_decoy.id),),
        ),
        template="find.shared_word",
    )


_SINGLE_TEMPLATES: tuple[Template, ...] = (
    _add_location_path,
    _add_indirect_stock,
    _add_new_product,
    _add_discriminator,
    _add_unit_collision,
    _add_missing_amount,
    _add_positional,
    _add_absent_spot,
    _remove_partial,
    _remove_all,
    _move_explicit,
    _find_where,
    _find_quantity,
    _find_shared_word,
)


def _reject_out_of_domain(f: FamilyVocab) -> tuple[EvalCase, ...]:
    return tuple(
        EvalCase(
            id=f"{f.id}-reject-out-of-domain-{i + 1}",
            family_id=f.id,
            prompt=prompt,
            kind=QueryKind.REJECT,
            expect=NoWrites(),
            template="reject.out_of_domain",
        )
        for i, prompt in enumerate(f.reject_prompts)
    )


def generate_cases(family: FamilyVocab) -> tuple[EvalCase, ...]:
    singles = tuple(template(family) for template in _SINGLE_TEMPLATES)
    return singles + _reject_out_of_domain(family)
