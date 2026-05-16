"""Print the complex-linear split lab summary as JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from clifford_3plus2_d5.lepton.complex_scans import (
    complex_c3_summary,
    complex_c5_discovered_summary,
    complex_c5_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("c3", "c5", "c5-discovered"), default="c3")
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.profile == "c3":
        summary = complex_c3_summary(max_candidates=args.max_candidates)
    elif args.profile == "c5":
        summary = complex_c5_summary(max_candidates=args.max_candidates)
    else:
        summary = complex_c5_discovered_summary(max_candidates=args.max_candidates)
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
