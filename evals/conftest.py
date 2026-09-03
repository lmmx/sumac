"""Safety rails, the one seeded inventory fixture, and small assertion
helpers for the eval suite — see docs/journal/2026-09-02-eval-suite.md.
The real household inventory lives outside this repository
(`chez/sumac_data`) and `sumac` resolves its data directory from
`SUMAC_DATA_DIR`, defaulting to `./data` — every fixture here exists to
make it structurally impossible for this suite to read or write it, not
merely unlikely.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from evals import fixtures as eval_fixtures
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


def _gguf_cached_locally(model) -> bool:  # noqa: ANN001
    """Whether `model.quantized_filename` is already present in the local
    Hugging Face Hub cache, checked *before* ever constructing a real
    `mistralrs.Runner`. A first run of `sumac ask` against an uncached
    preset downloads a multi-gigabyte GGUF file over the network
    (`llm._build_runner`'s own log line: "first run downloads it; may take
    a while"), and a try/except around that construction only catches an
    immediate error, not a slow download that never raises. This check is
    what makes `agent_runner` skip cleanly with no network attempt when
    nothing is cached, rather than hang or spend bandwidth silently — an
    earlier version of this fixture without it triggered a real ~2.5GB
    download in this repo's own development, caught partway through; see
    docs/journal/2026-09-02-eval-suite.md."""
    import os

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / ("models--" + model.quantized_model_id.replace("/", "--"))
    if not repo_dir.exists():
        return False
    return any(repo_dir.rglob(model.quantized_filename))


@pytest.fixture(scope="session")
def agent_runner_factory(request: pytest.FixtureRequest, inventory: tuple[Path, bytes]):
    """Model-gated tests only (`pytest.mark.model`). Returns a zero-arg
    factory building a fresh `AgentRunner` over one shared real
    `mistralrs.Runner` — a fresh wrapper per test (no leaked conversation
    state) over one loaded model (expensive to reload). Skips — never
    errors, never attempts a network download — when the GGUF isn't
    already in the local cache; see `_gguf_cached_locally`."""
    pytest.importorskip("mistralrs")
    from typing import cast

    from sumac import llm

    model_name = request.config.getoption("--eval-model") or llm.DEFAULT_MODEL_PRESET.name
    model = llm.model_preset(model_name)
    if not _gguf_cached_locally(model):
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


# --- assertion helpers ------------------------------------------------------
# Deliberately plain functions, not a generic scorer class — each test
# calls one or two of these directly and reads top-to-bottom.

UNIT_SYNONYMS: dict[str, str] = {
    "can": "can", "cans": "can",
    "jar": "jar", "jars": "jar",
    "carton": "carton", "cartons": "carton",
    "pack": "pack", "packs": "pack",
    "tub": "tub", "tubs": "tub",
    "box": "box", "boxes": "box",
    "bag": "bag", "bags": "bag",
    "bottle": "bottle", "bottles": "bottle",
    "jug": "jug", "jugs": "jug",
    "g": "g", "kg": "kg",
}  # fmt: skip


def _canon_unit(unit: str) -> str:
    return UNIT_SYNONYMS.get(unit.strip().lower(), unit.strip().lower())


def _canon_location(cfg: sumac_config.Config, value: str | None) -> str | None:
    if value is None:
        return None
    if value in cfg.known_locations:
        return value
    for loc_id in cfg.known_locations:
        if sumac_config.location_path(cfg.known_locations, loc_id) == value:
            return loc_id
    return value


def assert_no_writes(plan) -> None:  # noqa: ANN001
    assert plan.writes == (), f"expected no writes, got {plan.writes!r}"


def assert_write(
    plan,  # noqa: ANN001
    cfg: sumac_config.Config,
    *,
    kind,  # noqa: ANN001
    product_id: str,
    amount: str,
    unit: str,
    to_location: str | None = None,
    from_location: str | None = None,
) -> None:
    """Exactly one write in `plan`, matching every field given — unit and
    location are canonicalised (a plural unit and a display-path location
    are accepted), product and amount are compared exactly."""
    assert len(plan.writes) == 1, f"expected exactly one write, got {plan.writes!r}"
    w = plan.writes[0]
    assert w.kind == kind, f"expected kind={kind}, got {w.kind}"
    assert w.product_id.strip().lower() == product_id.strip().lower(), (
        f"expected product {product_id!r}, got {w.product_id!r}"
    )
    assert w.amount == Decimal(amount), f"expected amount {amount!r}, got {w.amount!r}"
    assert _canon_unit(w.unit) == _canon_unit(unit), f"expected unit {unit!r}, got {w.unit!r}"
    if to_location is not None:
        assert _canon_location(cfg, w.to_location) == to_location, (
            f"expected to_location {to_location!r}, got {w.to_location!r} "
            f"(canonicalised: {_canon_location(cfg, w.to_location)!r})"
        )
    if from_location is not None:
        assert _canon_location(cfg, w.from_location) == from_location, (
            f"expected from_location {from_location!r}, got {w.from_location!r} "
            f"(canonicalised: {_canon_location(cfg, w.from_location)!r})"
        )


def assert_classified(plan, kind) -> None:  # noqa: ANN001
    assert plan.trace, "expected a classify_request round in the trace, trace is empty"
    first = plan.trace[0]
    assert first.name == "classify_request", f"expected classify_request first, got {first.name}"
    actual = first.arguments.get("kind")
    assert actual == kind.value, f"expected classified as {kind.value!r}, got {actual!r}"


def assert_tool_called(plan, name: str, *, at_most: int | None = None) -> None:  # noqa: ANN001
    calls = [t.name for t in plan.trace if t.name == name]
    trace_names = [t.name for t in plan.trace]
    assert calls, f"expected {name!r} to be called at least once, trace: {trace_names}"
    if at_most is not None:
        assert len(calls) <= at_most, (
            f"{name!r} called {len(calls)} times, expected at most {at_most}"
        )


def is_ask_or_act(plan, *, max_reply_len: int = 400) -> str:  # noqa: ANN001
    """Returns `"act"` (wrote something), `"ask"` (empty writes, a
    question, no tool call beyond find, a short reply), or `"inaction"`
    (empty writes matching neither — the failure branch). A bare question
    mark alone isn't enough: it conflates a genuine clarifying question
    with a reply that rambles without acting."""
    if plan.writes:
        return "act"
    reply = plan.reply_text or ""
    domain_calls = [t.name for t in plan.trace if t.name != "classify_request"]
    only_find = all(name == "sumac_find_inventory" for name in domain_calls)
    if "?" in reply and only_find and len(reply) <= max_reply_len:
        return "ask"
    return "inaction"
