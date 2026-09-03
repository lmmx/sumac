"""Safety rails, the one seeded inventory fixture, and per-scenario result
collection for the eval suite — see docs/journal/2026-09-02-eval-suite.md.
The real household inventory lives outside this repository
(`chez/sumac_data`) and `sumac` resolves its data directory from
`SUMAC_DATA_DIR`, defaulting to `./data` — every fixture here exists to
make it structurally impossible for this suite to read or write it, not
merely unlikely.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import pytest

from evals import fixtures as eval_fixtures
from evals.evaluators import EvalResult
from sumac import config as sumac_config
from sumac import store as sumac_store


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--eval-model",
        action="store",
        type=str,
        default=None,
        help="ModelPreset name (default: llm.DEFAULT_MODEL_PRESET).",
    )
    parser.addoption(
        "--eval-seed",
        action="store",
        type=int,
        default=None,
        help="Optional seed for the model Runner, for reproducing one run.",
    )
    parser.addoption(
        "--eval-debug",
        action="store_true",
        default=False,
        help="Print raw agent request/response diagnostics (AgentRunner(debug=True)).",
    )
    parser.addoption(
        "--eval-json",
        action="store",
        type=str,
        default=None,
        help="Write this session's scenario results to this JSON path, for a later "
        "comparison against a different model/prompt run (no comparison tool exists "
        "yet — see docs/journal/2026-09-02-eval-suite.md).",
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@pytest.fixture(scope="session")
def eval_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("sumac-eval-root")


@pytest.fixture(scope="session", autouse=True)
def _eval_environment(eval_root: Path):
    """Rail 1: overrides `SUMAC_DATA_DIR`/`SUMAC_PASSPHRASE` for the whole
    session so an ambient value is never read. `getpass.getuser` is pinned
    too, so the inventory's `log:<osuser>` stream (`docs/LAYOUT.md`) is
    consistent regardless of which OS user actually runs the suite."""
    mp = pytest.MonkeyPatch()
    mp.setenv("SUMAC_PASSPHRASE", eval_fixtures.EVAL_PASSPHRASE)
    mp.setenv("SUMAC_DATA_DIR", str(eval_root / "must-not-be-used"))
    mp.setattr("getpass.getuser", lambda: eval_fixtures.EVAL_OSUSER)
    yield
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def _guard_store_append(eval_root: Path, _eval_environment: None):
    """Rail 3: `sumac.store.append` refuses to write outside `eval_root`,
    checked at the point of the write rather than the configuration that
    produced it — the rail that holds if rails 1 and 2 are ever edited
    wrongly."""
    original_append = sumac_store.append

    def guarded_append(data_dir: Path, key: bytes, stream_id: str, obj: dict) -> None:
        if not _is_within(Path(data_dir), eval_root):
            raise RuntimeError(
                f"refusing sumac.store.append outside the eval temp root: {data_dir}"
            )
        return original_append(data_dir, key, stream_id, obj)

    mp = pytest.MonkeyPatch()
    mp.setattr(sumac_store, "append", guarded_append)
    yield
    mp.undo()


@pytest.fixture(scope="session")
def inventory(eval_root: Path, _guard_store_append: None) -> tuple[Path, bytes]:
    """Rail 2: the data directory is asserted inside `eval_root` before any
    test sees it."""
    data_dir, key = eval_fixtures.build(eval_root)
    if not _is_within(data_dir, eval_root):
        raise pytest.UsageError(f"inventory data_dir escaped eval_root: {data_dir}")
    return data_dir, key


@pytest.fixture(scope="session")
def cfg(inventory: tuple[Path, bytes]) -> sumac_config.Config:
    data_dir, key = inventory
    return sumac_config.build_config(data_dir, key)


@pytest.fixture(scope="session")
def agent_runner_factory(request: pytest.FixtureRequest, inventory: tuple[Path, bytes]):
    """Model-gated tests only (`pytest.mark.model`). Returns a zero-arg
    factory building a fresh `AgentRunner` over one shared real
    `mistralrs.Runner` — a fresh wrapper per test (no leaked conversation
    state) over one loaded model (expensive to reload). Skips — never
    errors, never attempts a network download — when the GGUF isn't
    already in the local cache; see `llm.is_cached`."""
    pytest.importorskip("mistralrs")
    from typing import cast

    from sumac import llm

    model_name = request.config.getoption("--eval-model") or llm.DEFAULT_MODEL_PRESET.name
    model = llm.model_preset(model_name)
    if not llm.is_cached(model):
        pytest.skip(
            f"{model.quantized_model_id}/{model.quantized_filename} not in the local "
            "Hugging Face cache — refusing to trigger a network download from a test fixture"
        )
    seed_value = request.config.getoption("--eval-seed")
    try:
        base_runner = llm._build_runner(model, seed=seed_value)
    except Exception as e:  # noqa: BLE001 - last-resort guard; the cache check above is primary
        pytest.skip(f"could not load {model.quantized_model_id}: {e}")

    data_dir, key = inventory
    debug = request.config.getoption("--eval-debug")

    def make() -> llm.AgentRunner:
        # `_build_runner`'s own return type covers streaming too, which
        # `AgentRunner` never requests — same narrowing `llm.py` itself
        # does at its one `_build_runner` call site.
        return llm.AgentRunner(
            data_dir, key, model=model, runner=cast(llm.SendsCompletions, base_runner), debug=debug
        )

    return make


@pytest.fixture
def agent(agent_runner_factory, result):
    """A fresh `AgentRunner` per test (via `agent_runner_factory`), with
    its `tokens_per_sec` folded into this test's `result` on teardown —
    centralized here instead of the identical one-liner every
    `test_*.py` file used to define locally. Depending on `result` means
    this fixture tears down *before* `result`'s own teardown (pytest tears
    down in reverse dependency order), so the write below always lands
    before `result` is captured into the session's list."""
    a = agent_runner_factory()
    yield a
    result.tokens_per_sec = a.tokens_per_sec


# --- per-scenario results ---------------------------------------------------
# `pytest_configure` stashes a plain list on `config` that both the
# `result` fixture and the `pytest_sessionfinish` hook can reach (hooks
# have no fixture access) — one `EvalResult` per test, captured on
# teardown regardless of whether the test's own final `assert` passed, so
# a failing scenario still reports which of its checks were right.


def pytest_configure(config: pytest.Config) -> None:
    config._eval_results = []  # ty: ignore[unresolved-attribute]


@pytest.fixture
def result(request: pytest.FixtureRequest):
    """One `EvalResult` per test. `scenario` is derived from the test's own
    name (`test_add_discriminator_variant_not_confused` ->
    `add.discriminator_variant_not_confused`) and `category` from the
    test module's `_CATEGORY` constant — no separate id has to be typed
    per test, and a scenario can't drift from the function that runs it."""
    category = getattr(request.module, "_CATEGORY", "uncategorised")
    scenario = f"{category}.{request.node.name.removeprefix('test_')}"
    r = EvalResult(scenario=scenario, category=category)
    started = time.perf_counter()
    yield r
    r.duration_s = time.perf_counter() - started
    request.config._eval_results.append(r)  # ty: ignore[unresolved-attribute]


def _print_summary(results: list[EvalResult]) -> None:
    by_category: dict[str, list[EvalResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    print("\nSUMAC AGENT EVALUATION")
    total_passed = sum(1 for r in results if r.passed)
    for category, rs in sorted(by_category.items()):
        passed = sum(1 for r in rs if r.passed)
        print(f"  {category:10s} {passed}/{len(rs)}")
    print(f"  {'overall':10s} {total_passed}/{len(results)}")
    total_duration = sum(r.duration_s for r in results)
    print(f"  {'time':10s} {total_duration:.1f}s")
    rates = [r.tokens_per_sec for r in results if r.tokens_per_sec is not None]
    if rates:
        print(f"  {'tok/s':10s} {sum(rates) / len(rates):.1f} (mean across {len(rates)} scenarios)")

    failing = [r for r in results if not r.passed]
    if failing:
        print("\nFAILURES")
        for r in failing:
            print(f"  {r.scenario}")
            for f in r.failures:
                print(f"    - {f}")

    branches = Counter(r.note for r in results if r.note)
    if branches:
        print(f"\nask-vs-act branches: {dict(branches)}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    results: list[EvalResult] = getattr(config, "_eval_results", [])
    if not results:
        return
    _print_summary(results)

    json_path = config.getoption("--eval-json")
    if json_path:
        from sumac import llm

        model_name = config.getoption("--eval-model") or llm.DEFAULT_MODEL_PRESET.name
        rates = [r.tokens_per_sec for r in results if r.tokens_per_sec is not None]
        payload = {
            "model": model_name,
            "seed": config.getoption("--eval-seed"),
            "total_duration_s": sum(r.duration_s for r in results),
            "mean_tokens_per_sec": sum(rates) / len(rates) if rates else None,
            "results": [
                {
                    "scenario": r.scenario,
                    "category": r.category,
                    "passed": r.passed,
                    "checks": r.checks,
                    "failures": r.failures,
                    "note": r.note,
                    "duration_s": r.duration_s,
                    "tokens_per_sec": r.tokens_per_sec,
                }
                for r in results
            ],
        }
        out = Path(json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
