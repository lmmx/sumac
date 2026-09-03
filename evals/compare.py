"""Paired comparison between two run directories — see
docs/journal/2026-09-02-eval-suite.md, "Comparing two runs" and "Post-hoc
power reporting". Reports the paired difference with an exact McNemar test,
the *observed* discordance and intraclass correlations (not the assumed
values the eval spec's sizing tables used to justify the default layout),
the realised minimum detectable effect for this comparison, per-template
pass rates, a two-way (family x template) pairs cluster bootstrap
confidence interval, and the named list of `hard`-tagged cases that
changed state.

Usage:
    uv run python -m evals.compare runs/a/ runs/b/
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path


def _load_runs(run_dir: Path) -> list[dict]:
    files = sorted(run_dir.glob("seed-*.json"))
    if not files:
        raise SystemExit(f"no seed-*.json files found in {run_dir}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def _check_provenance(run_a: list[dict], run_b: list[dict]) -> None:
    prov_a, prov_b = run_a[0]["provenance"], run_b[0]["provenance"]
    if prov_a.get("model_preset") != prov_b.get("model_preset"):
        raise SystemExit(
            f"model mismatch: {prov_a.get('model_preset')} vs {prov_b.get('model_preset')} — "
            "refusing to compare runs of different models"
        )
    if prov_a.get("families") != prov_b.get("families"):
        raise SystemExit(
            "family-set mismatch — refusing to compare runs over different populations"
        )
    if prov_a.get("prompt_constants_hash") != prov_b.get("prompt_constants_hash"):
        print(
            "WARNING: prompt_constants_hash differs between the two runs — "
            "a prompt change is usually the thing being measured here, so this may be expected"
        )


def _case_pass_rate(runs: list[dict], row_type: str) -> dict[str, dict]:
    """case_id -> {"rate": float, "template": str|None, "family_id": str,
    "tags": [...], "blocked": bool} — mean pass rate across every epoch in
    `runs` for that case."""
    by_case: dict[str, dict] = {}
    for run in runs:
        for row in run["rows"]:
            if row["type"] != row_type:
                continue
            entry = by_case.setdefault(
                row["case_id"],
                {
                    "n": 0,
                    "passed": 0,
                    "template": row.get("template"),
                    "family_id": row["family_id"],
                    "tags": row.get("tags", []),
                    "blocked": row.get("blocked", False),
                },
            )
            entry["n"] += 1
            passed = row["correct"] if row_type == "routing" else row["passed"]
            entry["passed"] += int(bool(passed))
    for entry in by_case.values():
        entry["rate"] = entry["passed"] / entry["n"] if entry["n"] else 0.0
    return by_case


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Exact two-tailed McNemar p-value via the binomial sign test: under
    the null, `b` (or `c`) ~ Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cumulative = sum(comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2 * cumulative / (2**n))


def _paired_binary(by_case_a: dict, by_case_b: dict, threshold: float = 0.5) -> dict:
    """Reduces each case's mean pass rate in both runs to a pass/fail via
    `threshold`, over cases present (and non-blocked) in both. Returns the
    McNemar table plus the discordant case ids, split by direction."""
    common = sorted(
        cid
        for cid in (set(by_case_a) & set(by_case_b))
        if not by_case_a[cid]["blocked"] and not by_case_b[cid]["blocked"]
    )
    b_to_worse: list[str] = []  # passed in A, failed in B
    c_to_better: list[str] = []  # failed in A, passed in B
    both_pass = both_fail = 0
    for cid in common:
        pass_a = by_case_a[cid]["rate"] >= threshold
        pass_b = by_case_b[cid]["rate"] >= threshold
        if pass_a and pass_b:
            both_pass += 1
        elif not pass_a and not pass_b:
            both_fail += 1
        elif pass_a and not pass_b:
            b_to_worse.append(cid)
        else:
            c_to_better.append(cid)
    return {
        "common": common,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "b_to_worse": b_to_worse,
        "c_to_better": c_to_better,
    }


def _realised_mde(q: float, n: int) -> float | None:
    """`delta = 2.8 sqrt(q/n)` — the paired-comparison MDE formula from
    docs/journal/2026-09-02-eval-suite.md's sizing tables, evaluated on the
    *observed* discordance `q` and case count `n` for this comparison,
    rather than an assumed value."""
    if n == 0:
        return None
    return 2.8 * (q / n) ** 0.5


def _icc_one_way(groups: dict[str, list[float]]) -> float | None:
    """ANOVA-based ICC(1) — ratio of between-group to total variance — for
    a one-way random-effects decomposition of `groups` (cluster id ->
    per-unit values, here each unit's 0/1 "outcome differs between the two
    runs" indicator). Not a full multilevel model; a defensible first
    estimate for whether family or template correlates the discordant
    outcomes more, which is what the two-way bootstrap below needs to
    outperform a naive single-axis one."""
    all_values = [v for vals in groups.values() for v in vals]
    n_total = len(all_values)
    k = len(groups)
    if n_total == 0 or k < 2:
        return None
    grand_mean = sum(all_values) / n_total
    ss_between = sum(
        len(vals) * (sum(vals) / len(vals) - grand_mean) ** 2 for vals in groups.values() if vals
    )
    ss_within = sum(
        (v - sum(vals) / len(vals)) ** 2 for vals in groups.values() if vals for v in vals
    )
    df_between = k - 1
    df_within = n_total - k
    if df_within <= 0 or df_between <= 0:
        return None
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    n_bar = n_total / k  # harmonic-mean-free approximation for unbalanced groups
    denom = ms_between + (n_bar - 1) * ms_within
    if denom == 0:
        return 0.0
    icc = (ms_between - ms_within) / denom
    return max(0.0, icc)  # negative ICC estimates are clamped to 0


def _two_way_cluster_bootstrap(
    by_case_a: dict, by_case_b: dict, *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float] | None:
    """Pairs cluster bootstrap (Cameron, Gelbach & Miller 2011) over both
    families and templates: each iteration resamples family ids and
    template ids independently, with replacement, and rebuilds the sample
    from every (resampled family, resampled template) cell. Restricted to
    generated cases (`template is not None`) over a full family x template
    grid — `hard` cases don't fit that grid and are reported separately as
    a named list, not folded into this interval. Returns
    `(observed_diff, ci_low, ci_high)` or `None` if the grid is empty."""
    common = set(by_case_a) & set(by_case_b)
    by_key: dict[tuple[str, str], tuple[float, float]] = {}
    for cid in common:
        a, b = by_case_a[cid], by_case_b[cid]
        if a["template"] is None or a["blocked"] or b["blocked"]:
            continue
        by_key[(a["family_id"], a["template"])] = (a["rate"], b["rate"])

    if not by_key:
        return None

    families = sorted({f for f, _t in by_key})
    templates = sorted({t for _f, t in by_key})

    def mean_diff(keys: list[tuple[str, str]]) -> float:
        diffs = [by_key[k][1] - by_key[k][0] for k in keys if k in by_key]
        return sum(diffs) / len(diffs) if diffs else 0.0

    observed = mean_diff(list(by_key.keys()))

    rng = random.Random(seed)
    boot_diffs: list[float] = []
    for _ in range(n_boot):
        resampled_families = [rng.choice(families) for _ in families]
        resampled_templates = [rng.choice(templates) for _ in templates]
        keys = [(f, t) for f in resampled_families for t in resampled_templates]
        boot_diffs.append(mean_diff(keys))

    boot_diffs.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = int(0.975 * n_boot) - 1
    return observed, boot_diffs[lo_idx], boot_diffs[max(lo_idx, hi_idx)]


def _print_comparison(label: str, by_case_a: dict, by_case_b: dict) -> None:
    if not by_case_a or not by_case_b:
        print(f"\nno {label} rows in one or both runs — skipping")
        return

    table = _paired_binary(by_case_a, by_case_b)
    n = len(table["common"])
    b = len(table["b_to_worse"])
    c = len(table["c_to_better"])
    if n == 0:
        print(f"\n{label}: no headline cases common to both runs")
        return

    p_value = _exact_mcnemar_p(b, c)
    q = (b + c) / n
    mde = _realised_mde(q, n)

    print(f"\n{label}: {n} common headline cases")
    print(f"  A: {table['both_pass'] + b}/{n} pass   B: {table['both_pass'] + c}/{n} pass")
    print(f"  discordant: {b} regressed (A pass, B fail), {c} improved (A fail, B pass)")
    print(f"  McNemar exact p-value: {p_value:.4f}")
    print(
        f"  observed discordance q = {q:.3f}   realised MDE at this n: {mde:.1%}"
        if mde is not None
        else "  n=0, no MDE"
    )

    # observed ICC over both clustering axes, on the discordant indicator
    by_family: dict[str, list[float]] = defaultdict(list)
    by_template: dict[str, list[float]] = defaultdict(list)
    for cid in table["common"]:
        discordant = 1.0 if cid in table["b_to_worse"] or cid in table["c_to_better"] else 0.0
        fam = by_case_a[cid]["family_id"]
        tmpl = by_case_a[cid]["template"] or "(hand-written)"
        by_family[fam].append(discordant)
        by_template[tmpl].append(discordant)

    icc_family = _icc_one_way(by_family)
    icc_template = _icc_one_way(by_template)
    if icc_family is not None:
        print(f"  observed ICC (family):   {icc_family:.3f}")
    if icc_template is not None:
        print(f"  observed ICC (template): {icc_template:.3f}")

    bootstrap = _two_way_cluster_bootstrap(by_case_a, by_case_b)
    if bootstrap is not None:
        observed, lo, hi = bootstrap
        print(
            f"  two-way (family x template) cluster bootstrap: mean diff {observed:+.1%}, "
            f"95% CI [{lo:+.1%}, {hi:+.1%}]"
        )

    # per-template rates, side by side
    templates = sorted({e["template"] for e in by_case_a.values() if e["template"]})
    if templates:
        print("  per-template pass rate (A -> B):")
        for template in templates:
            rates_a = [e["rate"] for e in by_case_a.values() if e["template"] == template]
            rates_b = [e["rate"] for e in by_case_b.values() if e["template"] == template]
            if rates_a and rates_b:
                mean_a = sum(rates_a) / len(rates_a)
                mean_b = sum(rates_b) / len(rates_b)
                arrow = "->" if abs(mean_b - mean_a) > 1e-9 else "=="
                print(f"    {template:24s} {mean_a:5.1%} {arrow} {mean_b:5.1%}")

    # hard cases, named
    hard_common = [cid for cid in table["common"] if "hard" in by_case_a[cid]["tags"]]
    if hard_common:
        flipped = [
            cid for cid in hard_common if cid in table["b_to_worse"] or cid in table["c_to_better"]
        ]
        print(f"  hard cases that changed state: {flipped or 'none'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_a", type=Path, help="Baseline run directory (e.g. before a prompt edit)."
    )
    parser.add_argument(
        "run_b", type=Path, help="Comparison run directory (e.g. after a prompt edit)."
    )
    args = parser.parse_args(argv)

    runs_a = _load_runs(args.run_a)
    runs_b = _load_runs(args.run_b)
    _check_provenance(runs_a, runs_b)

    print(f"A: {args.run_a} ({len(runs_a)} epoch(s))")
    print(f"B: {args.run_b} ({len(runs_b)} epoch(s))")

    _print_comparison(
        "routing", _case_pass_rate(runs_a, "routing"), _case_pass_rate(runs_b, "routing")
    )
    _print_comparison(
        "proposals", _case_pass_rate(runs_a, "proposal"), _case_pass_rate(runs_b, "proposal")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
