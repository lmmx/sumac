"""Aggregates repeated-epoch eval runs (`scripts/epoch-benchmark.sh`) into a
per-scenario, per-model comparison — plain counts and rates, deliberately
not p-values or effect-size machinery. An earlier draft of this suite built
exactly that (a paired McNemar test, observed intraclass correlation, a
two-way cluster bootstrap, a realised MDE) and none of it earned its
complexity at this suite's size — see
docs/journal/2026-09-02-eval-suite.md's original "Comparing two runs"
section, from before all of it was deleted. A per-scenario pass-count table
was the part that did: it's what actually answers "did this regression show
up consistently, or was that one run noise."

Usage: `uv run python -m evals.epoch_report runs/epochs/*/`

Each argument is a directory of `epoch-NN.json` files — one model (and,
independently, one prompt variant)'s worth, written by
`scripts/epoch-benchmark.sh` or a manual `pytest --eval-json` run. Grouping
is read from each file's own `"model"`/`"prompt_variant"` fields, not the
directory name, so a mixed or mis-organised directory still groups
correctly; a file with no `"prompt_variant"` field (written before that was
added) is treated as `"default"`.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_epochs(directory: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("epoch-*.json"))]


def _group_label(epoch: dict) -> str:
    """Groups by model + prompt_variant, same as before, plus `backend`
    (added for the Modal backend — a file with no `"backend"` field was
    written before that existed and is treated as `"local"`). A Modal and
    a local epoch for the same model/variant are never the same benchmark
    — see docs/journal/2026-09-04-modal-remote-inference-backend.md's
    "quantization parity" and "usage accounting" sections on why comparing
    them directly is misleading — so they must never silently share a row."""
    variant = epoch.get("prompt_variant", "default")
    label = epoch["model"] if variant == "default" else f"{epoch['model']} [{variant}]"
    backend = epoch.get("backend", "local")
    return label if backend == "local" else f"{label} ({backend})"


def _print_report(model_epochs: dict[str, list[dict]]) -> None:
    models = sorted(model_epochs)
    # Wide enough for the longest label ("model [prompt-variant]" can run past
    # any fixed width) — computed once, shared by both tables below so a long
    # label can't run its column into the next one.
    label_width = max(20, max((len(m) for m in models), default=0) + 2)

    print("EPOCH COMPARISON")
    print(
        f"  {'model':{label_width}s} {'epochs':>7s} {'attempts':>9s} "
        f"{'pass_rate':>10s} {'tok/s':>8s} {'time/scenario':>14s}"
    )
    for model in models:
        epochs = model_epochs[model]
        results = [r for e in epochs for r in e["results"]]
        passed = sum(1 for r in results if r["verdict"]["passed"])
        rates = [
            r["metrics"]["tokens_per_sec"]
            for r in results
            if r["metrics"].get("tokens_per_sec") is not None
        ]
        durations = [r["metrics"]["duration_s"] for r in results]
        mean_rate = sum(rates) / len(rates) if rates else float("nan")
        mean_duration = sum(durations) / len(durations) if durations else float("nan")
        pass_rate = passed / len(results) * 100 if results else float("nan")
        print(
            f"  {model:{label_width}s} {len(epochs):7d} {len(results):9d} "
            f"{pass_rate:9.1f}% {mean_rate:8.1f} {mean_duration:13.1f}s"
        )

    scenario_order: list[str] = []
    seen: set[str] = set()
    per_model_scenario: dict[str, dict[str, list[bool]]] = {}
    for model in models:
        by_scenario: dict[str, list[bool]] = defaultdict(list)
        for epoch in model_epochs[model]:
            for r in epoch["results"]:
                by_scenario[r["scenario"]].append(r["verdict"]["passed"])
                if r["scenario"] not in seen:
                    seen.add(r["scenario"])
                    scenario_order.append(r["scenario"])
        per_model_scenario[model] = by_scenario

    def counts(model: str, scenario: str) -> tuple[int, int]:
        outcomes = per_model_scenario[model].get(scenario, [])
        return sum(outcomes), len(outcomes)

    header = "  " + f"{'scenario':45s}" + "".join(m.rjust(label_width) for m in models)
    rows: list[str] = []
    disagreements: list[str] = []
    always_failing: list[str] = []
    for scenario in scenario_order:
        cell_counts = [counts(m, scenario) for m in models]
        cells = "".join(f"{p}/{t}".rjust(label_width) for p, t in cell_counts)
        row = "  " + f"{scenario:45s}" + cells
        rows.append(row)
        raw_passes = [p for p, _ in cell_counts]
        if max(raw_passes) - min(raw_passes) > 1:
            disagreements.append(row)
        if all(p == 0 for p, _ in cell_counts):
            always_failing.append(row)

    print("\nPER-SCENARIO PASS COUNTS")
    print(header)
    for row in rows:
        print(row)

    if disagreements:
        print("\nDISAGREEING SCENARIOS (pass count differs by more than 1 across models)")
        print(header)
        for row in disagreements:
            print(row)

    if always_failing:
        print("\nALWAYS-FAILING SCENARIOS (every model, every epoch)")
        print(header)
        for row in always_failing:
            print(row)


def main(argv: list[str]) -> None:
    if not argv:
        print("usage: uv run python -m evals.epoch_report DIR [DIR ...]", file=sys.stderr)
        raise SystemExit(2)

    model_epochs: dict[str, list[dict]] = defaultdict(list)
    for arg in argv:
        directory = Path(arg)
        epochs = _load_epochs(directory)
        if not epochs:
            print(f"warning: no epoch-*.json files in {directory}", file=sys.stderr)
            continue
        for epoch in epochs:
            model_epochs[_group_label(epoch)].append(epoch)

    if not model_epochs:
        print("no epoch data found", file=sys.stderr)
        raise SystemExit(1)

    _print_report(model_epochs)


if __name__ == "__main__":
    main(sys.argv[1:])
