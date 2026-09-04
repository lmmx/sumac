"""Two cheap, deterministic checks on `evals/fixtures.py` itself — no
model, no GPU. Both were part of the original (larger) suite, dropped in
the reduction pass, and restored after a review found the location-path
one would have caught a real bug (two ADD scenarios built on a
shelf/cupboard naming convention that exists nowhere in the actual system)
before a real-model run ever needed to. See
docs/journal/2026-09-02-eval-suite.md.
"""

from __future__ import annotations

from evals import fixtures as eval_fixtures
from sumac import config as sumac_config
from sumac import llm


def test_no_product_name_leaks_into_prompt_constants() -> None:
    """`_ADD_PROMPT`'s own worked example names "Heinz" and "Baked Beans"
    explicitly — a fixture product sharing either name would let a
    scenario measure recall of the prompt's own wording rather than the
    behaviour under test."""
    prompt_text = " ".join(
        [llm.CLASSIFIER_PROMPT, llm._FIND_PROMPT, llm._ADD_PROMPT, llm._REMOVE_PROMPT]
    ).casefold()

    names = [product_id for product_id, _amount, _unit, _location in eval_fixtures.PRODUCTS]
    names.append(eval_fixtures.ABSENT_PRODUCT)

    leaked = [name for name in names if name.casefold() in prompt_text]
    assert leaked == [], f"fixture product names leak into a prompt constant: {leaked}"


def test_location_path_matches_real_config(inventory: tuple) -> None:
    """`fixtures.location_path()` is what every ADD/MOVE prompt in this
    suite is built from — checked here against what a real seeded `Config`
    actually resolves, so the two can't silently drift apart. Uses the
    real `inventory` fixture (session-scoped, already built by any prior
    test in the session) rather than seeding a second copy."""
    data_dir, key = inventory
    cfg = sumac_config.build_config(data_dir, key)
    for location_id in eval_fixtures.LOCATIONS_BY_ID:
        expected = sumac_config.location_path(cfg.known_locations, location_id)
        actual = eval_fixtures.location_path(location_id)
        assert actual == expected, (
            f"{location_id}: fixtures.location_path()={actual!r}, real Config={expected!r}"
        )
