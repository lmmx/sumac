"""`evals/conftest.py`'s scenario-order shuffle (`_shuffle_model_scenarios`)
and `--eval-json`'s verdict/log split (`_build_eval_payload`/`_log_lines`)
— both fixes docs/journal/2026-09-04-trace-and-verdict-redesign.md hands
off. The shuffle answers the RNG-cascade confound: `mistralrs.Runner`
shares one RNG stream across every scenario in a session, so a fixed
collection order is a fixed per-scenario bias (measured 25 loads/epoch at
~2.7s each against an ~84s `qwen3.5-9b` epoch, ~79% overhead, ruling out a
per-scenario `Runner` for the main benchmark). The verdict/log split
answers the file-size complaint that came after: a single scenario's
`messages` conversation can run to hundreds of lines, and bundling every
scenario's into one `--eval-json` file ran a 25-scenario epoch to 2,700+
lines dominated by content nobody reads on the same pass as `verdict`.

Plain `SimpleNamespace` stand-ins in place of real `pytest.Item`s for the
shuffle tests — `_shuffle_model_scenarios` only ever calls
`get_closest_marker`/reads `nodeid`, so a nested pytest session isn't
needed to exercise it. The payload tests use real `EvalResult` instances
and real temp files — cheap enough here that a stand-in would only cost
fidelity.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from evals.conftest import _build_eval_payload, _log_lines, _shuffle_model_scenarios
from evals.evaluators import EvalResult


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


def _fake_results() -> list[EvalResult]:
    passing = EvalResult(scenario="find.existing_item", category="find")
    passing.checks = {"classification": True}
    passing.duration_s = 1.5
    passing.tokens_per_sec = 40.0
    passing.trace = [{"name": "classify_request", "arguments": {}, "result": "{}"}]
    passing.messages = [{"role": "system", "content": "sys"}]
    passing.terminal = "reply"

    failing = EvalResult(scenario="add.basmati_rice_in_different_unit", category="add")
    failing.checks = {"classification": True, "product": False}
    failing.failures = ["expected product 'Basmati Rice', got 'Basmati Rice Bag'"]
    failing.duration_s = 4.2
    failing.tokens_per_sec = 90.0
    failing.trace = [{"name": "classify_request", "arguments": {}, "result": "{}"}]
    failing.messages = [{"role": "user", "content": "Add 1 bag of Basmati Rice"}]
    failing.classify_messages = [{"role": "system", "content": "classify"}]
    failing.usage_history = [{"round": 1, "prompt_tokens": 50, "completion_tokens": 10}]
    failing.terminal = "reply"
    failing.nudge_fired = True

    return [passing, failing]


def test_build_eval_payload_carries_verdict_and_metrics_not_log() -> None:
    results = _fake_results()

    payload = _build_eval_payload(
        model_name="qwen3.5-9b",
        variant_name="default",
        seed=3,
        scenario_order=["evals/test_find.py::test_existing_item"],
        log_file_name="epoch-01.log.jsonl",
        results=results,
    )

    assert payload["model"] == "qwen3.5-9b"
    assert payload["seed"] == 3
    assert payload["log_file"] == "epoch-01.log.jsonl"
    assert payload["scenario_order"] == ["evals/test_find.py::test_existing_item"]
    assert payload["mean_tokens_per_sec"] == (40.0 + 90.0) / 2
    entries = {r["scenario"]: r for r in payload["results"]}
    assert entries["find.existing_item"]["verdict"] == {
        "passed": True,
        "checks": {"classification": True},
        "failures": [],
    }
    assert entries["add.basmati_rice_in_different_unit"]["verdict"]["passed"] is False
    assert entries["add.basmati_rice_in_different_unit"]["metrics"]["duration_s"] == 4.2
    # None of the execution-record fields belong in this half at all.
    for entry in payload["results"]:
        assert "trace" not in entry
        assert "messages" not in entry
        assert "log" not in entry


def test_log_lines_carry_the_execution_record_joined_by_scenario() -> None:
    results = _fake_results()

    lines = _log_lines(results)

    by_scenario = {line["scenario"] for line in lines}
    assert by_scenario == {"find.existing_item", "add.basmati_rice_in_different_unit"}
    basmati = next(
        line for line in lines if line["scenario"] == "add.basmati_rice_in_different_unit"
    )
    assert basmati["nudge_fired"] is True
    assert basmati["terminal"] == "reply"
    assert basmati["trace"][0]["name"] == "classify_request"
    assert basmati["classify_messages"][0]["content"] == "classify"
    # No verdict/metrics field leaked into the log half either.
    assert "checks" not in basmati
    assert "passed" not in basmati
    assert "duration_s" not in basmati


def test_payload_and_log_lines_round_trip_through_jsonl(tmp_path) -> None:  # noqa: ANN001
    """The actual on-disk shape `pytest_sessionfinish` produces — one
    `.json` file readable whole, one `.jsonl` sidecar readable line by
    line, joined by `scenario` — written and read back exactly like a real
    consumer (`report.jq`, `epoch_report.py`, a human's `jq` one-liner)
    would."""
    results = _fake_results()
    out = tmp_path / "epoch-01.json"
    log_path = tmp_path / "epoch-01.log.jsonl"

    payload = _build_eval_payload(
        model_name="qwen3.5-9b",
        variant_name="default",
        seed=1,
        scenario_order=None,
        log_file_name=log_path.name,
        results=results,
    )
    out.write_text(json.dumps(payload))
    with log_path.open("w") as f:
        for line in _log_lines(results):
            f.write(json.dumps(line) + "\n")

    read_payload = json.loads(out.read_text())
    read_lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(read_lines) == 2
    scenario_to_verdict = {r["scenario"]: r["verdict"]["passed"] for r in read_payload["results"]}
    scenario_to_log = {line["scenario"]: line for line in read_lines}
    assert scenario_to_verdict.keys() == scenario_to_log.keys()
    assert scenario_to_verdict["add.basmati_rice_in_different_unit"] is False
    assert scenario_to_log["add.basmati_rice_in_different_unit"]["nudge_fired"] is True
