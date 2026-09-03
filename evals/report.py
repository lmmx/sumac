"""Aggregates per-seed JSON files written by `evals.run` into a summary:
pass^k across epochs, the null-baseline floor rows, per-template pass
rates, a classification confusion matrix, and the blocked-case table
reported separately from the headline — see
docs/journal/2026-09-02-eval-suite.md, "Epochs and pass^k" and "Null
baselines, always reported".

Usage:
    uv run python -m evals.report runs/2026-09-02-qwen35/
    uv run python -m evals.report runs/2026-09-02-qwen35/ --no-baselines
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_runs(run_dir: Path) -> list[dict]:
    files = sorted(run_dir.glob("seed-*.json"))
    if not files:
        raise SystemExit(f"no seed-*.json files found in {run_dir}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def _check_provenance_consistent(runs: list[dict]) -> None:
    first = runs[0]["provenance"]
    for run in runs[1:]:
        prov = run["provenance"]
        if prov.get("model_preset") != first.get("model_preset"):
            raise SystemExit(
                "runs in this directory used different model presets — "
                "aggregate only same-population epochs together"
            )
        if prov.get("families") != first.get("families"):
            raise SystemExit(
                "runs in this directory used different family sets — "
                "aggregate only same-population epochs together"
            )
        if prov.get("prompt_constants_hash") != first.get("prompt_constants_hash"):
            print(
                "WARNING: prompt_constants_hash differs across runs in this directory — "
                "a prompt change may have landed mid-collection"
            )


def _aggregate(runs: list[dict], row_type: str) -> dict[str, dict]:
    by_case: dict[str, dict] = {}
    for run in runs:
        for row in run["rows"]:
            if row["type"] != row_type:
                continue
            entry = by_case.setdefault(
                row["case_id"],
                {
                    "n_epochs": 0,
                    "n_passed": 0,
                    "template": row.get("template"),
                    "tags": row.get("tags", []),
                    "blocked": row.get("blocked", False),
                },
            )
            entry["n_epochs"] += 1
            passed = row["correct"] if row_type == "routing" else row["passed"]
            entry["n_passed"] += int(bool(passed))
    return by_case


def _print_pass_k(by_case: dict[str, dict], label: str) -> None:
    if not by_case:
        print(f"\nno {label} rows found")
        return

    headline = {cid: e for cid, e in by_case.items() if not e["blocked"]}
    blocked = {cid: e for cid, e in by_case.items() if e["blocked"]}
    n_epochs = next(iter(by_case.values()))["n_epochs"]

    print(f"\n{label}: {len(headline)} headline cases, {n_epochs} epoch(s)")
    if headline:
        pass_k_count = sum(1 for e in headline.values() if e["n_passed"] == e["n_epochs"])
        pass_k_rate = pass_k_count / len(headline)
        print(f"  pass^{n_epochs}: {pass_k_count}/{len(headline)} ({pass_k_rate:.1%})")
    if blocked:
        blocked_k = sum(1 for e in blocked.values() if e["n_passed"] == e["n_epochs"])
        print(
            f"  blocked (excluded above — see location-reference taxonomy in the eval spec): "
            f"{blocked_k}/{len(blocked)} pass^{n_epochs}"
        )

    by_template: dict[str, list[float]] = defaultdict(list)
    for e in headline.values():
        by_template[e["template"] or "(hand-written)"].append(e["n_passed"] / e["n_epochs"])
    if by_template:
        print("  per-template mean pass rate:")
        for template, rates in sorted(by_template.items()):
            print(f"    {template:24s} {sum(rates) / len(rates):5.1%}  (n={len(rates)})")

    hard = {cid: e for cid, e in by_case.items() if "hard" in e["tags"]}
    if hard:
        failing = sorted(cid for cid, e in hard.items() if e["n_passed"] < e["n_epochs"])
        print(f"  hard cases with any epoch failing: {failing or 'none'}")


def _print_confusion_matrix(runs: list[dict]) -> None:
    confusion: Counter[tuple[str, str]] = Counter()
    for run in runs:
        for row in run["rows"]:
            if row["type"] != "routing":
                continue
            confusion[(row["expected_kind"], row["actual_kind"])] += 1
    if not confusion:
        return
    total = sum(confusion.values())
    correct = sum(n for (e, a), n in confusion.items() if e == a)
    print(f"\nclassification accuracy across all epochs: {correct}/{total} ({correct / total:.1%})")
    for (expected, actual), n in sorted(confusion.items()):
        marker = "" if expected == actual else "  <-- MISCLASSIFIED"
        print(f"  expected={expected:8s} actual={actual:8s}  n={n:4d}{marker}")


def _print_baselines(n_families: int) -> None:
    """Runs the three null baselines fresh (no model needed) over the same
    family count this run used, and prints their score beside the real
    numbers above — see docs/journal/2026-09-02-eval-suite.md, "Null
    baselines, always reported". Seeds its own families standalone
    (`seed.build_families_standalone`) since this script has no pytest
    session to seed through."""
    from evals.baselines import BASELINES, run_baseline
    from evals.cases import all_cases
    from evals.seed import build_families_standalone
    from evals.vocab import FAMILIES_BY_ID

    family_fixtures = build_families_standalone(n_families)
    families = tuple(FAMILIES_BY_ID[fid] for fid in family_fixtures)
    cases = all_cases(families=families)
    headline_cases = tuple(c for c in cases if "blocked" not in c.tags)

    print(f"\nnull baselines ({len(headline_cases)} headline cases, this run's family count):")
    for name in BASELINES:
        result = run_baseline(name, family_fixtures, headline_cases)
        print(
            f"  {name:20s} {result.passed:3d}/{result.total} ({result.pass_rate:5.1%})   "
            f"kind-correct {result.kind_correct:3d}/{result.total}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--no-baselines", action="store_true", help="Skip the (re-seeded) baseline rows."
    )
    args = parser.parse_args(argv)

    runs = _load_runs(args.run_dir)
    _check_provenance_consistent(runs)
    prov = runs[0]["provenance"]

    print(f"model: {prov.get('model_preset')} ({prov.get('quantized_model_id')})")
    temp_routing = prov.get("temperature_routing")
    temp_proposals = prov.get("temperature_proposals")
    print(f"temperature: routing={temp_routing} proposals={temp_proposals}")
    print(f"families: {prov.get('families')}")
    print(f"epochs: {len(runs)} (seeds: {[r['provenance'].get('eval_seed') for r in runs]})")

    _print_pass_k(_aggregate(runs, "routing"), "routing")
    _print_confusion_matrix(runs)
    _print_pass_k(_aggregate(runs, "proposal"), "proposals")

    if not args.no_baselines:
        _print_baselines(len(prov.get("families", [])) or 10)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
