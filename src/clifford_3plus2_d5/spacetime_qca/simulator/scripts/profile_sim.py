"""Profile the scan-backed spacetime-QCA simulator on bounded tiny cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from clifford_3plus2_d5.spacetime_qca.simulator.profiling import (
    default_spacetime_profile_cases,
    run_spacetime_profile,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in default_spacetime_profile_cases()),
        help="Profile one named case. May be repeated. Defaults to all bounded cases.",
    )
    parser.add_argument("--include-jit", action="store_true", help="Also profile the simulator's internal JIT path.")
    parser.add_argument(
        "--force-method",
        choices=("autodiff", "finite_difference", "finite_difference_batched", "analytic_staple"),
        default=None,
        help="Override force method in selected simulator profiling cases.",
    )
    parser.add_argument(
        "--force-chunk-size",
        type=int,
        default=None,
        help="Override finite-difference batched force chunk size in selected profiling cases.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_spacetime_profile(
        case_names=tuple(args.case) if args.case else None,
        include_jit=args.include_jit,
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
