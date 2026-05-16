"""Run the complex-linear split lab scan and print JSON rows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from clifford_3plus2_d5.lepton.complex_scans import (
    run_complex_c3_split_scan,
    run_complex_c5_discovered_split_scan,
    run_complex_c5_split_scan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("c3", "c5", "c5-discovered"), default="c3")
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.profile == "c3":
        rows = run_complex_c3_split_scan(max_candidates=args.max_candidates)
    elif args.profile == "c5":
        rows = run_complex_c5_split_scan(max_candidates=args.max_candidates)
    else:
        rows = run_complex_c5_discovered_split_scan(max_candidates=args.max_candidates)
    print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
