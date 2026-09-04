"""`evals/conftest.py`'s scenario-order shuffle (`_shuffle_model_scenarios`)
— the fix for the RNG-cascade confound the trace/verdict redesign surfaced:
`mistralrs.Runner` shares one RNG stream across every scenario in a
session, so a fixed collection order is a fixed per-scenario bias. See
docs/journal/2026-09-04-trace-and-verdict-redesign.md's "Open question,
not decided" (now decided — measured 25 loads/epoch at ~2.7s each against
an ~84s `qwen3.5-9b` epoch, ~79% overhead, ruling out a per-scenario
`Runner` for the main benchmark).

Plain `SimpleNamespace` stand-ins in place of real `pytest.Item`s —
`_shuffle_model_scenarios` only ever calls `get_closest_marker`/reads
`nodeid`, so a nested pytest session isn't needed to exercise it.
"""

from __future__ import annotations

from types import SimpleNamespace

from evals.conftest import _shuffle_model_scenarios


def _item(nodeid: str, *, model: bool) -> SimpleNamespace:
    marker = SimpleNamespace(name="model") if model else None
    return SimpleNamespace(nodeid=nodeid, get_closest_marker=lambda name: marker)


def test_shuffle_only_reorders_model_marked_items() -> None:
    items = [
        _item("evals/test_fixtures.py::test_a", model=False),
        _item("evals/test_add.py::test_x", model=True),
        _item("evals/test_add.py::test_y", model=True),
        _item("evals/test_fixtures.py::test_b", model=False),
    ]

    reordered, order = _shuffle_model_scenarios(items, seed=1)

    assert order == ["evals/test_add.py::test_x", "evals/test_add.py::test_y"] or order == [
        "evals/test_add.py::test_y",
        "evals/test_add.py::test_x",
    ]
    assert {i.nodeid for i in reordered} == {i.nodeid for i in items}
    # Every model-marked item comes before every non-model one — the
    # shuffle only permutes within the model group, per the docstring.
    model_positions = [n for n, i in enumerate(reordered) if i.get_closest_marker("model")]
    other_positions = [n for n, i in enumerate(reordered) if not i.get_closest_marker("model")]
    assert max(model_positions) < min(other_positions)


def test_shuffle_is_reproducible_for_the_same_seed() -> None:
    items = [_item(f"evals/test_add.py::test_{i}", model=True) for i in range(10)]

    _, order_a = _shuffle_model_scenarios(list(items), seed=7)
    _, order_b = _shuffle_model_scenarios(list(items), seed=7)

    assert order_a == order_b


def test_different_seeds_produce_different_orders() -> None:
    """Not a mathematical guarantee for every seed pair, but true for these
    two and cheap to pin down — a shuffle that silently ignored `seed`
    entirely would fail this."""
    items = [_item(f"evals/test_add.py::test_{i}", model=True) for i in range(10)]

    _, order_a = _shuffle_model_scenarios(list(items), seed=1)
    _, order_b = _shuffle_model_scenarios(list(items), seed=2)

    assert order_a != order_b
