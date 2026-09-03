"""Hand-written per-family vocabulary for the eval suite's seeded inventory.

Ten structurally-equivalent families (`FAMILIES`, ids `fam-01`..`fam-10`), each
carrying the same roles with different words — see
docs/journal/2026-09-02-eval-suite.md, "Fixture families". `fam-01` reuses the
vocabulary of the real runs recorded in
docs/journal/2026-09-01-ask-agent-design.md, so those transcripts stay
directly comparable.

Location *structure* (ids, nesting, grid shape) is shared across every
family and built once by `evals/seed.py` — it is not vocabulary that needs
varying to prevent memorization, since the model has no pretrained bias
toward a particular grid-cell id the way it might toward a common brand
name. Only product/brand names and reject-prompt wording vary per family.

Every product, brand, and absent-product name here is checked against
`sumac.llm`'s four prompt constants by
`test_no_vocab_name_leaks_into_prompt_constants` in `test_scoring.py` — a
name that appears in a prompt would let a case measure recall of the
prompt's own wording rather than the behaviour the case is meant to test.
`_ADD_PROMPT`'s worked example names "Heinz" and "Baked Beans" explicitly;
neither appears anywhere below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedProduct:
    id: str
    amount: str
    unit: str


@dataclass(frozen=True, slots=True)
class FamilyVocab:
    id: str

    # Seeded, registered under `unit`; only `hard-unit-conflict` (fam-01
    # only) requests it in a different, unconvertible unit. Seeded in every
    # family regardless, since its presence alongside `near_miss_brand` is
    # what makes a search for the shared base name return two candidates —
    # the same realistic ambiguity the real Chopped Tomatoes / Ocado Italian
    # Chopped Tomatoes transcript shows, not just a fam-01 special case.
    unit_collision: SeedProduct

    # A different brand of the same basic product as `unit_collision` — its
    # id contains `unit_collision.id` as a substring. Target of
    # `add.location_path` and `add.indirect_stock`.
    near_miss_brand: SeedProduct

    # A pair separated by exactly one leading word (Salted/Unsalted,
    # Smoked/Unsmoked, ...). Target of `add.discriminator` is
    # `discriminator_b`; `find.shared_word` expects `discriminator_a` named
    # ahead of `shared_word_decoy`.
    discriminator_a: SeedProduct
    discriminator_b: SeedProduct
    shared_word: str
    shared_word_decoy: SeedProduct

    # Registered under `unit` (a jug/tub-shaped container); `add.unit_collision`
    # requests `rice_new_unit` (a bag) instead, which `decide` cannot convert.
    rice: SeedProduct
    rice_new_unit: str
    rice_size: str

    # `remove.all` target — consumed in full.
    consumption_target: SeedProduct

    # `remove.partial` and `move.explicit` source.
    movement_source: SeedProduct

    # `add.missing_amount`: a product already in stock under the same
    # `category_word`, and a new product name a request adds with no
    # amount or unit given at all.
    category_stocked: SeedProduct
    category_new_name: str
    category_word: str

    # Never seeded. `add.new_product` target.
    absent_product: str

    # Exactly three, out-of-domain, distinct per family so `reject.out_of_domain`
    # rows are independent items rather than one list of prompts replicated
    # across all ten families.
    reject_prompts: tuple[str, str, str]


def _family(
    id: str,
    *,
    unit_collision: tuple[str, str, str],
    near_miss_brand: tuple[str, str, str],
    discriminator_a: tuple[str, str, str],
    discriminator_b: tuple[str, str, str],
    shared_word: str,
    shared_word_decoy: tuple[str, str, str],
    rice: tuple[str, str, str],
    rice_new_unit: str,
    rice_size: str,
    consumption_target: tuple[str, str, str],
    movement_source: tuple[str, str, str],
    category_stocked: tuple[str, str, str],
    category_new_name: str,
    category_word: str,
    absent_product: str,
    reject_prompts: tuple[str, str, str],
) -> FamilyVocab:
    return FamilyVocab(
        id=id,
        unit_collision=SeedProduct(*unit_collision),
        near_miss_brand=SeedProduct(*near_miss_brand),
        discriminator_a=SeedProduct(*discriminator_a),
        discriminator_b=SeedProduct(*discriminator_b),
        shared_word=shared_word,
        shared_word_decoy=SeedProduct(*shared_word_decoy),
        rice=SeedProduct(*rice),
        rice_new_unit=rice_new_unit,
        rice_size=rice_size,
        consumption_target=SeedProduct(*consumption_target),
        movement_source=SeedProduct(*movement_source),
        category_stocked=SeedProduct(*category_stocked),
        category_new_name=category_new_name,
        category_word=category_word,
        absent_product=absent_product,
        reject_prompts=reject_prompts,
    )


FAMILIES: tuple[FamilyVocab, ...] = (
    _family(
        "fam-01",
        unit_collision=("Chopped Tomatoes", "1", "jar"),
        near_miss_brand=("Ocado Italian Chopped Tomatoes", "3", "cans"),
        discriminator_a=("Salted Butter", "1", "pack"),
        discriminator_b=("Unsalted Butter", "2", "packs"),
        shared_word="butter",
        shared_word_decoy=("Butter Beans", "2", "cans"),
        rice=("Basmati Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="1kg",
        consumption_target=("Strawberry Jam", "1", "jar"),
        movement_source=("Ragu", "2", "tubs"),
        category_stocked=("Fusilli Pasta", "500", "g"),
        category_new_name="Barilla Rigatoni",
        category_word="pasta",
        absent_product="Irn-Bru Zero",
        reject_prompts=(
            "How many days have the letter d in them?",
            "What's the weather in Edinburgh?",
            "Recommend me a good book to read",
        ),
    ),
    _family(
        "fam-02",
        unit_collision=("Chicken Soup", "1", "tin"),
        near_miss_brand=("Waitrose Chicken Soup", "4", "packs"),
        discriminator_a=("Semi-Skimmed Milk", "2", "cartons"),
        discriminator_b=("Whole Milk", "1", "carton"),
        shared_word="milk",
        shared_word_decoy=("Milk Chocolate", "3", "bars"),
        rice=("Long Grain Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="500g",
        consumption_target=("Blackberry Jam", "1", "jar"),
        movement_source=("Bolognese Sauce", "2", "tubs"),
        category_stocked=("Penne Pasta", "400", "g"),
        category_new_name="De Cecco Fusilli",
        category_word="pasta",
        absent_product="Ginger Beer",
        reject_prompts=(
            "What time is it in Tokyo?",
            "Explain how photosynthesis works",
            "What's your favourite colour?",
        ),
    ),
    _family(
        "fam-03",
        unit_collision=("Rice Cakes", "1", "packet"),
        near_miss_brand=("Kallo Rice Cakes", "2", "bags"),
        discriminator_a=("Orange Juice", "1", "carton"),
        discriminator_b=("Apple Juice", "2", "cartons"),
        shared_word="juice",
        shared_word_decoy=("Juice Extractor", "1", "appliance"),
        rice=("Wild Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="500g",
        consumption_target=("Raspberry Jam", "1", "jar"),
        movement_source=("Pesto", "2", "tubs"),
        category_stocked=("Linguine Pasta", "500", "g"),
        category_new_name="Napolina Tagliatelle",
        category_word="pasta",
        absent_product="Elderflower Cordial",
        reject_prompts=(
            "Write me a haiku about clouds",
            "What's 17 times 23?",
            "Who won the world cup in 1998?",
        ),
    ),
    _family(
        "fam-04",
        unit_collision=("Sweetcorn", "1", "tin"),
        near_miss_brand=("Green Giant Sweetcorn", "2", "tins"),
        discriminator_a=("Smoked Bacon", "1", "pack"),
        discriminator_b=("Unsmoked Bacon", "2", "packs"),
        shared_word="bacon",
        shared_word_decoy=("Bacon Flavour Crisps", "3", "bags"),
        rice=("Jasmine Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="500g",
        consumption_target=("Onion Marmalade", "1", "jar"),
        movement_source=("Chilli Con Carne", "2", "tubs"),
        category_stocked=("Tagliatelle Pasta", "500", "g"),
        category_new_name="Garofalo Orecchiette",
        category_word="pasta",
        absent_product="Dandelion and Burdock",
        reject_prompts=(
            "What's the capital of Peru?",
            "Tell me about the history of jazz",
            "Give me a random number between 1 and 100",
        ),
    ),
    _family(
        "fam-05",
        unit_collision=("Curry Sauce", "1", "jar"),
        near_miss_brand=("Sharwoods Curry Sauce", "3", "pouches"),
        discriminator_a=("Salted Peanuts", "1", "bag"),
        discriminator_b=("Unsalted Peanuts", "2", "bags"),
        shared_word="peanuts",
        shared_word_decoy=("Peanut Butter", "1", "jar"),
        rice=("Brown Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="1kg",
        consumption_target=("Fig Jam", "1", "jar"),
        movement_source=("Chicken Stock", "2", "tubs"),
        category_stocked=("Spaghetti Pasta", "500", "g"),
        category_new_name="De Cecco Linguine",
        category_word="pasta",
        absent_product="Dark Chocolate Digestives",
        reject_prompts=(
            "What's the meaning of life?",
            "How do I get a UK passport?",
            "Tell me a fun fact about space",
        ),
    ),
    _family(
        "fam-06",
        unit_collision=("Fish Fingers", "1", "box"),
        near_miss_brand=("Captain's Choice Fish Fingers", "2", "packs"),
        discriminator_a=("Sweet Chilli Sauce", "1", "bottle"),
        discriminator_b=("Hot Chilli Sauce", "2", "bottles"),
        shared_word="chilli",
        shared_word_decoy=("Chilli Flakes", "1", "jar"),
        rice=("Arborio Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="500g",
        consumption_target=("Marmalade", "1", "jar"),
        movement_source=("Beef Casserole", "2", "tubs"),
        category_stocked=("Farfalle Pasta", "500", "g"),
        category_new_name="Barilla Conchiglie",
        category_word="pasta",
        absent_product="Sparkling Elderflower",
        reject_prompts=(
            "What language do they speak in Brazil?",
            "Give me advice on learning guitar",
            "What's the tallest mountain in the world?",
        ),
    ),
    _family(
        "fam-07",
        unit_collision=("Baked Salmon Fillets", "1", "pack"),
        near_miss_brand=("Waitrose Baked Salmon Fillets", "2", "packs"),
        discriminator_a=("Green Pesto", "1", "jar"),
        discriminator_b=("Red Pesto", "1", "jar"),
        shared_word="pesto",
        shared_word_decoy=("Pesto Crisps", "2", "bags"),
        rice=("Sushi Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="1kg",
        consumption_target=("Apricot Jam", "1", "jar"),
        movement_source=("Vegetable Soup", "2", "tubs"),
        category_stocked=("Rigatoni Pasta", "500", "g"),
        category_new_name="De Cecco Penne",
        category_word="pasta",
        absent_product="Ginger Ale",
        reject_prompts=(
            "What's the square root of 144?",
            "Recommend a good sci-fi film",
            "How far away is the moon?",
        ),
    ),
    _family(
        "fam-08",
        unit_collision=("Vegetable Stock Cubes", "1", "box"),
        near_miss_brand=("Knorr Vegetable Stock Cubes", "2", "boxes"),
        discriminator_a=("Mild Cheddar", "1", "block"),
        discriminator_b=("Mature Cheddar", "2", "blocks"),
        shared_word="cheddar",
        shared_word_decoy=("Cheddar Crackers", "1", "box"),
        rice=("Risotto Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="500g",
        consumption_target=("Lime Marmalade", "1", "jar"),
        movement_source=("Minestrone Soup", "2", "tubs"),
        category_stocked=("Orzo Pasta", "500", "g"),
        category_new_name="Garofalo Trofie",
        category_word="pasta",
        absent_product="Blackcurrant Squash",
        reject_prompts=(
            "What's a good name for a cat?",
            "Explain how a rainbow forms",
            "Tell me about the Roman Empire",
        ),
    ),
    _family(
        "fam-09",
        unit_collision=("Chicken Stock Cubes", "1", "box"),
        near_miss_brand=("Oxo Chicken Stock Cubes", "2", "boxes"),
        discriminator_a=("Brown Onions", "1", "bag"),
        discriminator_b=("Red Onions", "1", "bag"),
        shared_word="onions",
        shared_word_decoy=("Pickled Onions", "1", "jar"),
        rice=("Paella Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="500g",
        consumption_target=("Cherry Jam", "1", "jar"),
        movement_source=("Leek and Potato Soup", "2", "tubs"),
        category_stocked=("Vermicelli Pasta", "500", "g"),
        category_new_name="De Cecco Bucatini",
        category_word="pasta",
        absent_product="Rhubarb Cordial",
        reject_prompts=(
            "What's the boiling point of water?",
            "Tell me a riddle",
            "What's a good workout routine?",
        ),
    ),
    _family(
        "fam-10",
        unit_collision=("Digestive Biscuits", "1", "packet"),
        near_miss_brand=("McVeigh's Digestive Biscuits", "2", "packets"),
        discriminator_a=("Dark Soy Sauce", "1", "bottle"),
        discriminator_b=("Light Soy Sauce", "1", "bottle"),
        shared_word="soy",
        shared_word_decoy=("Soy Milk", "2", "cartons"),
        rice=("Pudding Rice", "1", "jug"),
        rice_new_unit="bag",
        rice_size="500g",
        consumption_target=("Gooseberry Jam", "1", "jar"),
        movement_source=("Tomato Soup", "2", "tubs"),
        category_stocked=("Macaroni Pasta", "500", "g"),
        category_new_name="Napolina Casarecce",
        category_word="pasta",
        absent_product="Cloudy Lemonade",
        reject_prompts=(
            "What's your opinion on modern art?",
            "How do airplanes stay in the air?",
            "Tell me a proverb",
        ),
    ),
)

FAMILIES_BY_ID: dict[str, FamilyVocab] = {f.id: f for f in FAMILIES}
