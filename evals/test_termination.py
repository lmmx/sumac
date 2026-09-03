"""Deterministic proof that `AgentRunner`'s round cap actually bounds a
pathologically-repeating model — no real model needed. This does not
reproduce the context-overflow failure itself (that needs a real small
context window filling up; see docs/journal/2026-09-02-eval-suite.md's
diagnosis), only confirms the mechanism meant to prevent it terminates
cleanly rather than looping forever or crashing.

The real fix for the diagnosed failure — `MAX_TOOL_ROUNDS` bounds round
*count*, not accumulated token count, and `_maybe_force_action`/
`_maybe_self_review` each add further rounds onto the same growing
`self._messages` list — is not made here. It's an `src/sumac/llm.py`
behaviour change of the same weight as the sampling-pinning commit, and
wants an explicit decision, not a silent edit alongside an eval suite
rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sumac import llm


class _RepeatedRejectionRunner:
    """Classifies every request `add`, then emits the exact same
    `sumac_discover_inventory` call forever, never producing a final
    plain-text reply. Run against a bare `tmp_path` with no seeded
    inventory (this test needs no real one — the fake never varies its
    call regardless of what it receives), the call is rejected as
    `unknown_location` every round rather than the real Basmati Rice
    request's `unit_unconvertible`; either way `decide` rejects it and
    `_propose_write` returns a `_rejected(...)` result, so the shape this
    test proves — the loop terminates under a persistent rejection — is
    the same one. A model stuck in this failure mode looks exactly like
    this from `AgentRunner`'s side: an endless stream of the same rejected
    call."""

    _BAD_CALL = {
        "product_id": "Basmati Rice",
        "amount": "1",
        "unit": "bag",
        "to_location": "pantry-white-unit-r1c1",
    }

    def __init__(self) -> None:
        self.rounds = 0

    def send_chat_completion_request(self, request, model_id=None):  # noqa: ANN001, ANN201
        self.rounds += 1
        if self.rounds == 1:
            function = SimpleNamespace(
                name="classify_request", arguments=json.dumps({"kind": "add"})
            )
        else:
            function = SimpleNamespace(
                name="sumac_discover_inventory", arguments=json.dumps(self._BAD_CALL)
            )
        message = SimpleNamespace(
            content=None, role="assistant", tool_calls=[SimpleNamespace(function=function)]
        )
        choice = SimpleNamespace(finish_reason="stop", index=0, message=message)
        return SimpleNamespace(choices=[choice])


def test_repeated_rejection_terminates_within_round_cap(tmp_path: Path) -> None:
    """Wires `_RepeatedRejectionRunner` straight into a real `AgentRunner`.
    Bounds checked: `propose()` returns rather than hanging, it never
    accumulates a write from a call `decide` rejected, and the fake
    runner's own round count stays within the documented cap: 1 classify
    round, up to `MAX_TOOL_ROUNDS` for the main loop, plus up to
    `MAX_TOOL_ROUNDS` more for `_maybe_force_action`'s extra `_run_loop`
    call (triggered here since an `add`-classified request produced no
    writes) — `_maybe_self_review` never adds a third batch in this
    scenario specifically, since it short-circuits on empty `plan.writes`
    before making any further round. Observed: exactly 41 rounds against
    that formula's 42-round ceiling, every one growing the same
    `self._messages` list with no token-count check anywhere — the
    mechanism the diagnosis names as bounding round count, not context
    size."""
    fake = _RepeatedRejectionRunner()
    agent = llm.AgentRunner(tmp_path, b"\x00" * 32, runner=fake)

    plan = agent.propose("Add 1 bag of Basmati Rice (1kg) next to the existing jug of Basmati Rice")

    max_possible_rounds = 1 + llm.MAX_TOOL_ROUNDS + llm.MAX_TOOL_ROUNDS
    assert fake.rounds <= max_possible_rounds, (
        f"the loop ran {fake.rounds} rounds against a documented cap of {max_possible_rounds} — "
        "the round-cap mechanism itself has regressed"
    )
    assert plan.writes == (), "a call decide.decide_change rejected was somehow accepted as a write"
