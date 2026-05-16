"""Audit the first Lab B domain-wall candidate as JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from clifford_3plus2_d5.lepton.results import (
    audit_first_domain_wall_candidate,
    audit_first_physical_domain_wall_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--physical", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Timeout for the physical R12 exact-center audit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.physical:
        audit = audit_first_physical_domain_wall_candidate(
            max_pairs=args.max_pairs,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        audit = audit_first_domain_wall_candidate(max_pairs=args.max_pairs)
    print(json.dumps(asdict(audit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
