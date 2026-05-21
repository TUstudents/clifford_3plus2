"""Profile sub-kernels inside the full-SM coupled spacetime-QCA step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from clifford_3plus2_d5.spacetime_qca.simulator.step_breakdown_profile import (
    DEFAULT_FORCE_COMPARISON_CHUNK_SIZES,
    default_step_breakdown_cases,
    run_force_chunk_comparison,
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
    parser.add_argument(
        "--force-method",
        choices=("finite_difference", "finite_difference_batched", "analytic_staple"),
        default=None,
        help="Override force method for force-related cases only.",
    )
    parser.add_argument(
        "--force-chunk-size",
        type=int,
        default=None,
        help="Override batched force chunk size for force-related cases only.",
    )
    parser.add_argument(
        "--force-comparison",
        action="store_true",
        help="Run scalar-vs-batched SM left-force chunk comparison instead of regular cases.",
    )
    parser.add_argument(
        "--force-comparison-chunk-size",
        action="append",
        type=int,
        default=None,
        help="Chunk size for --force-comparison. May be repeated; defaults to 4,8,16,32.",
    )
    parser.add_argument("--warmup-runs", type=int, default=0, help="Warmup calls before timed repeats.")
    parser.add_argument("--timed-runs", type=int, default=1, help="Timed repeated calls per case.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.all_cases and args.case:
        raise SystemExit("--all-cases cannot be combined with --case")
    if args.force_comparison and (args.all_cases or args.case):
        raise SystemExit("--force-comparison cannot be combined with --case or --all-cases")
    if args.force_comparison:
        payload = run_force_chunk_comparison(
            chunk_sizes=tuple(args.force_comparison_chunk_size or DEFAULT_FORCE_COMPARISON_CHUNK_SIZES),
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    selected_cases = tuple(args.case) if args.case else None
    if args.all_cases:
        selected_cases = tuple(case.name for case in default_step_breakdown_cases())
    payload = run_spacetime_step_breakdown_profile(
        case_names=selected_cases,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        force_method=args.force_method,
        force_chunk_size=args.force_chunk_size,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
