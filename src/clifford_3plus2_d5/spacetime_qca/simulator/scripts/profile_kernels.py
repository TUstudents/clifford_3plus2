"""Warm-profile bounded spacetime-QCA simulator kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from clifford_3plus2_d5.spacetime_qca.simulator.kernel_profile import (
    default_kernel_profile_cases,
    run_spacetime_kernel_profile,
)


def _parser() -> argparse.ArgumentParser:
    case_names = tuple(case.name for case in default_kernel_profile_cases(include_current=True))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=case_names,
        help="Profile one named kernel case. May be repeated. Defaults to all non-current cases.",
    )
    parser.add_argument("--include-current", action="store_true", help="Include finite-difference matter-current probe.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup calls before timed repeats.")
    parser.add_argument("--timed-runs", type=int, default=3, help="Timed repeated calls per case.")
    parser.add_argument(
        "--force-method",
        choices=("autodiff", "finite_difference", "finite_difference_batched", "analytic_staple"),
        default=None,
        help="Override force method for step_no_matter cases only.",
    )
    parser.add_argument(
        "--force-chunk-size",
        type=int,
        default=None,
        help="Override finite-difference batched force chunk size for step_no_matter cases.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected_cases = tuple(args.case) if args.case else None
    payload = run_spacetime_kernel_profile(
        case_names=selected_cases,
        include_current=args.include_current or (selected_cases is not None and "matter_current_u1_y" in selected_cases),
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
