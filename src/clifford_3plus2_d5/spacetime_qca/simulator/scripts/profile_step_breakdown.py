"""Profile sub-kernels inside the full-SM coupled spacetime-QCA step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from clifford_3plus2_d5.spacetime_qca.simulator.step_breakdown_profile import (
    default_step_breakdown_cases,
    run_spacetime_step_breakdown_profile,
)


def _parser() -> argparse.ArgumentParser:
    case_names = tuple(case.name for case in default_step_breakdown_cases())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=case_names,
        help="Profile one named step-breakdown case. May be repeated. Defaults to higgs_leapfrog_sm.",
    )
    parser.add_argument("--all-cases", action="store_true", help="Run every step-breakdown case explicitly.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Warmup calls before timed repeats.")
    parser.add_argument("--timed-runs", type=int, default=1, help="Timed repeated calls per case.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.all_cases and args.case:
        raise SystemExit("--all-cases cannot be combined with --case")
    selected_cases = tuple(args.case) if args.case else None
    if args.all_cases:
        selected_cases = tuple(case.name for case in default_step_breakdown_cases())
    payload = run_spacetime_step_breakdown_profile(
        case_names=selected_cases,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
