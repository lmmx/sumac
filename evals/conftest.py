"""Options, safety rails, and seeded fixtures for the eval suite — see
docs/journal/2026-09-02-eval-suite.md, "Safety rails" and "Fixture
families". The real household inventory lives outside this repository
(`chez/sumac_data`, per docs/journal/2026-09-02-eval-suite.md) and `sumac`
resolves its data directory from `SUMAC_DATA_DIR`, defaulting to `./data`
— every fixture here exists to make it structurally impossible for this
suite to read or write it, not merely unlikely."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from evals import seed as eval_seed
from evals.vocab import FAMILIES, FamilyVocab
from sumac import store as sumac_store

_EVAL_PASSPHRASE = eval_seed.EVAL_PASSPHRASE
_EVAL_OSUSER = eval_seed.EVAL_OSUSER
_REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--families",
        action="store",
        type=int,
        default=len(FAMILIES),
        help=f"Number of fixture families to build, from FAMILIES[:N] (default {len(FAMILIES)}).",
    )
    parser.addoption(
        "--eval-seed",
        action="store",
        type=int,
        default=None,
        help="Sampling seed for this session's model Runner (model-gated tests only).",
    )
    parser.addoption(
        "--eval-model",
        action="store",
        type=str,
        default=None,
        help="ModelPreset name (default: llm.DEFAULT_MODEL_PRESET).",
    )
    parser.addoption(
        "--eval-json",
        action="store",
        type=str,
        default=None,
        help="Path to write this session's per-case results as JSON.",
    )
    parser.addoption(
        "--eval-temperature",
        action="store",
        type=float,
        default=0.7,
        help="Sampling temperature for test_proposals.py (test_routing.py always uses 0.0).",
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
    too, so every family's `log:<osuser>` stream (`docs/LAYOUT.md`) is
    consistent regardless of which OS user actually runs the suite."""
    mp = pytest.MonkeyPatch()
    mp.setenv("SUMAC_PASSPHRASE", _EVAL_PASSPHRASE)
    mp.setenv("SUMAC_DATA_DIR", str(eval_root / "must-not-be-used"))
    mp.setattr("getpass.getuser", lambda: _EVAL_OSUSER)
    yield
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def _guard_store_append(eval_root: Path, _eval_environment: None):
    """Rail 3: `sumac.store.append` refuses to write outside `eval_root`,
    checked at the point of the write rather than the configuration that
    produced it — the rail that holds if rails 1 and 2 are ever edited
    wrongly. Legitimately exercised by every seeding call below, not just
    a defensive no-op: `test_seeding_writes_stay_inside_eval_root` in
    `test_scoring.py` confirms it doesn't false-positive on those."""
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
def eval_families(request: pytest.FixtureRequest) -> tuple[FamilyVocab, ...]:
    n = request.config.getoption("--families")
    return FAMILIES[:n]


@pytest.fixture(scope="session")
def family_fixtures(
    eval_families: tuple[FamilyVocab, ...], eval_root: Path, _guard_store_append: None
) -> dict[str, tuple[Path, bytes]]:
    """Rail 2: every family's data directory is asserted inside `eval_root`
    (itself under `tmp_path_factory`'s base) before it is handed to any
    test — a family whose seeding somehow escaped that boundary fails here,
    not silently later."""
    built: dict[str, tuple[Path, bytes]] = {}
    for family in eval_families:
        data_dir, key = eval_seed.build_family(eval_root, family, passphrase=_EVAL_PASSPHRASE)
        if not _is_within(data_dir, eval_root):
            raise pytest.UsageError(f"family {family.id} data_dir escaped eval_root: {data_dir}")
        built[family.id] = (data_dir, key)
    return built


def _gguf_cached_locally(model) -> bool:  # noqa: ANN001
    """Whether `model.quantized_filename` is already present in the local
    Hugging Face Hub cache. Checked *before* ever constructing a real
    `mistralrs.Runner` — a first run of `sumac ask` against an uncached
    preset downloads a multi-gigabyte GGUF file over the network
    (`llm._build_runner`'s own log line: "first run downloads it; may take
    a while"), and a try/except around that construction only catches an
    immediate error, not a slow download that never raises at all. This
    check is what makes `eval_runner` skip cleanly with no network attempt
    in an environment with nothing cached, rather than hang or spend
    bandwidth silently."""
    import os
    from pathlib import Path

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / ("models--" + model.quantized_model_id.replace("/", "--"))
    if not repo_dir.exists():
        return False
    return any(repo_dir.rglob(model.quantized_filename))


@pytest.fixture(scope="session")
def eval_runner(request: pytest.FixtureRequest):
    """Model-gated tests only (`pytest.mark.model`). Skips — never errors,
    and never attempts a network download — when the GGUF isn't already in
    the local cache. See `_gguf_cached_locally`: this fixture is the one
    place in the suite that may construct a real `mistralrs.Runner`, and it
    must never be reached in an environment with no GPU and no cached
    weights."""
    pytest.importorskip("mistralrs")
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
        runner = llm._build_runner(model, seed=seed_value)
    except Exception as e:  # noqa: BLE001 - last-resort guard; the cache check above is primary
        pytest.skip(f"could not load {model.quantized_model_id}: {e}")
    return runner, model


# --- results collection: one JSON file per pytest session (one epoch) -------
# `run.py` drives one pytest session per `--eval-seed`; `report.py` and
# `compare.py` read the resulting files. A hook, not only a fixture,
# because `pytest_sessionfinish` has no fixture access — `pytest_configure`
# stashes a plain list on `config` that both the fixture and the hook can
# reach (the documented pattern for hook/fixture-shared session state).


def pytest_configure(config: pytest.Config) -> None:
    config._eval_rows = []  # ty: ignore[unresolved-attribute]


@pytest.fixture(scope="session")
def eval_results_collector(request: pytest.FixtureRequest) -> list[dict]:
    return request.config._eval_rows  # ty: ignore[unresolved-attribute]


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _provenance(config: pytest.Config) -> dict:
    """Written into every `--eval-json` file's header.
    `compare.py` hard-errors on a model or family-set mismatch between two
    such files, and warns on a `prompt_constants_hash` mismatch — a prompt
    change is usually the thing being measured, a different family set is
    a different population. Does not hash the GGUF file on disk — locating
    it depends on mistral.rs's own cache layout, which this suite doesn't
    introspect; see docs/journal/2026-09-02-eval-suite.md, Missing."""
    from sumac import llm

    model_name = config.getoption("--eval-model") or llm.DEFAULT_MODEL_PRESET.name
    model = llm.model_preset(model_name)
    n_families = config.getoption("--families")
    prompt_text = "".join(
        [llm.CLASSIFIER_PROMPT, llm._FIND_PROMPT, llm._ADD_PROMPT, llm._REMOVE_PROMPT]
    )
    return {
        "git_sha": _git_sha(),
        "model_preset": model.name,
        "quantized_model_id": model.quantized_model_id,
        "quantized_filename": model.quantized_filename,
        "temperature_routing": 0.0,
        "temperature_proposals": config.getoption("--eval-temperature"),
        "top_p": llm.DEFAULT_TOP_P,
        "max_tokens": llm.DEFAULT_MAX_TOKENS,
        "eval_seed": config.getoption("--eval-seed"),
        "families": [f.id for f in FAMILIES[:n_families]],
        "prompt_constants_hash": hashlib.sha256(prompt_text.encode()).hexdigest()[:16],
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    json_path = config.getoption("--eval-json")
    if not json_path:
        return
    rows = getattr(config, "_eval_rows", [])
    payload = {"provenance": _provenance(config), "rows": rows}
    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
