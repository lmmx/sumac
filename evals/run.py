"""Epoch orchestrator: one pytest session per `--eval-seed`, each in its
own process — see docs/journal/2026-09-02-eval-suite.md, "Epochs are
separate pytest sessions". `mistralrs.ChatCompletionRequest` has no `seed`
field; `mistralrs.Runner.__init__` does, and `AgentRunner` builds one
`Runner` per session, shared across every case in that session. Reusing
one session across epochs would make the RNG stream position at case N
depend on how many tokens every prior case generated in that same
session — so a `-k` filter would change what gets sampled, and a re-run of
one failing case wouldn't reproduce the sample that failed. Giving each
epoch its own process and its own `--eval-seed` avoids that: any single
epoch reproduces exactly from its seed alone.

Usage:
    uv run python -m evals.run --epochs 8 --out runs/2026-09-02-qwen35/
    uv run python -m evals.run --epochs 1 --out runs/dev/ --families 2 -k proposals

Does not abort on a non-zero pytest exit — a failing epoch still writes its
`--eval-json` file (`conftest.pytest_sessionfinish` runs regardless of
`exitstatus`), and losing epochs 2..N to a stopped run would destroy the
reliability signal those epochs exist to produce.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of independently-seeded sessions to run."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory for per-seed JSON files."
    )
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument(
        "--families", type=int, default=None, help="Passed through to pytest's --families."
    )
    parser.add_argument("--eval-model", type=str, default=None)
    parser.add_argument("--eval-temperature", type=float, default=None)
    parser.add_argument(
        "-k", "--filter", type=str, default=None,
        help="pytest -k expression, e.g. 'routing' or 'proposals' to run only one module.",
    )  # fmt: skip
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    failed_seeds: list[int] = []

    for i in range(args.epochs):
        seed = args.start_seed + i
        json_path = args.out / f"seed-{seed:02d}.json"
        cmd = [
            sys.executable, "-m", "pytest", "evals",
            "-m", "model",
            "--eval-seed", str(seed),
            "--eval-json", str(json_path),
        ]  # fmt: skip
        if args.families is not None:
            cmd += ["--families", str(args.families)]
        if args.eval_model is not None:
            cmd += ["--eval-model", args.eval_model]
        if args.eval_temperature is not None:
            cmd += ["--eval-temperature", str(args.eval_temperature)]
        if args.filter is not None:
            cmd += ["-k", args.filter]

        print(f"--- epoch {i + 1}/{args.epochs} (seed={seed}) ---")
        result = subprocess.run(cmd, cwd=_REPO_ROOT)
        if result.returncode != 0:
            failed_seeds.append(seed)
            note = " (still check for a JSON file — pytest_sessionfinish writes one regardless)"
            print(f"epoch seed={seed} exited {result.returncode}{note}")

    if failed_seeds:
        print(
            f"\n{len(failed_seeds)}/{args.epochs} epochs had a non-zero pytest exit: {failed_seeds}"
        )
    print(f"\nresults in {args.out} — aggregate with: uv run python -m evals.report {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
