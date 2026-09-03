"""Null-baseline `llm.SendsCompletions` stubs — see
docs/journal/2026-09-02-eval-suite.md, "Null baselines, always reported".
Each needs no model and no GGUF download; `test_scoring.py` runs the full
case table through all three and `report.py` prints each one's score as a
fixed row beside the model's in every summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

# No return-type annotation on `send_chat_completion_request`/`_respond`
# below, deliberately — matching `tests/test_llm.py`'s `FakeRunner`, which
# skips annotating that method's return type the same way. An explicit
# `-> SimpleNamespace` makes `ty` verify it against `llm.SendsCompletions`'s
# real `ChatCompletionResponse` return type and fail; these are fakes
# returning a duck-typed stand-in, the same way `FakeRunner` already does.


class _FixedResponseRunner:
    """Shared plumbing: returns whatever `_respond(request)` produces,
    shaped like a real `mistralrs.ChatCompletionResponse` — a plain-text
    message (no `tool_calls`) or a single scripted tool call."""

    def send_chat_completion_request(self, request: Any, model_id: str | None = None):
        message = self._respond(request)
        choice = SimpleNamespace(finish_reason="stop", index=0, message=message)
        return SimpleNamespace(choices=[choice])

    def _respond(self, request: Any):
        raise NotImplementedError

    @staticmethod
    def _tool_call(name: str, arguments: dict):
        function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
        tool_call = SimpleNamespace(function=function)
        return SimpleNamespace(content=None, role="assistant", tool_calls=[tool_call])

    @staticmethod
    def _text(content: str):
        return SimpleNamespace(content=content, role="assistant", tool_calls=None)


class DoNothingRunner(_FixedResponseRunner):
    """Every round, including the classify round, returns plain text with
    no tool call — which `AgentRunner._classify` treats as `reject`
    (`src/sumac/llm.py:807-824`), so every request short-circuits before
    the domain loop ever runs, regardless of its actual kind. Used to
    confirm the scorer isn't trivially satisfiable by inaction — a `Writes`
    case, and every non-`reject` `NoWrites`/`AskOrAct` case's expected
    `kind`, must fail against this, unconditionally."""

    def _respond(self, request: Any):
        return self._text("I'm not going to do anything with this request.")


class RejectEverythingRunner(_FixedResponseRunner):
    """Classifies every request `reject`, then (per `AgentRunner.propose`)
    the domain loop never runs at all. Passes every `reject` case and
    every `NoWrites`-on-`find` case; the point of the trace-based `find`
    assertions in `scoring.check_trace` is that this stub fails them, since
    it never calls `sumac_find_inventory`."""

    def _respond(self, request: Any):
        return self._tool_call("classify_request", {"kind": "reject"})


class AlwaysDiscoverRunner(_FixedResponseRunner):
    """Classifies every request `add`, then emits one fixed
    `sumac_discover_inventory` call regardless of what the request actually
    asked for, then a plain-text reply — round 1 classify, round 2 the
    discovery, round 3 a final reply (ending `_run_loop` with one write),
    round 4 another final reply with no tool call (the self-review round,
    which then keeps that one write rather than looping again — see
    `AgentRunner._maybe_self_review`). Fails every case whose gold write
    names a different product, amount, or location, which is every
    generated case except by coincidence."""

    _FIXED_CALL = {
        "product_id": "Baseline Product",
        "amount": "1",
        "unit": "unit",
        "to_location": "storage",
    }

    def __init__(self) -> None:
        self._round = 0

    def _respond(self, request: Any):
        self._round += 1
        if self._round == 1:
            return self._tool_call("classify_request", {"kind": "add"})
        if self._round == 2:
            return self._tool_call("sumac_discover_inventory", self._FIXED_CALL)
        return self._text("Added the baseline product.")


BASELINES: dict[str, type] = {
    "do-nothing": DoNothingRunner,
    "reject-everything": RejectEverythingRunner,
    "always-discover": AlwaysDiscoverRunner,
}


@dataclass(frozen=True, slots=True)
class BaselineResult:
    name: str
    passed: int
    total: int
    kind_correct: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_baseline(name: str, family_fixtures: dict, cases) -> BaselineResult:
    """Runs `cases` (an `evals.cases.EvalCase` sequence) through the
    baseline named `name` (a key of `BASELINES`) over `family_fixtures`
    (`{family_id: (data_dir, key)}`), scoring each with
    `scoring.score_case`. Shared by `test_scoring.py`'s null-floor check
    and `report.py`'s summary table, so both use the same definition of
    "passed" — see `scoring.score_case`'s docstring for why that includes
    classification, not only the write/trace outcome."""
    from evals import scoring
    from sumac import config as sumac_config
    from sumac import llm

    make_runner = BASELINES[name]
    passed = 0
    kind_correct = 0
    for case in cases:
        data_dir, key = family_fixtures[case.family_id]
        agent = llm.AgentRunner(data_dir, key, runner=make_runner())
        plan = agent.propose(case.prompt)
        cfg = sumac_config.build_config(data_dir, key)
        case_score = scoring.score_case(case, plan, cfg)
        if case_score.passed:
            passed += 1
        if case_score.kind_ok:
            kind_correct += 1
    return BaselineResult(name=name, passed=passed, total=len(cases), kind_correct=kind_correct)
