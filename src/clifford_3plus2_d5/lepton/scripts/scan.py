"""Run one bounded lepton-lab scan and print rows as JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from clifford_3plus2_d5.lepton.scans import (
    run_lab_a_bloch_scan,
    run_lab_a_onsite_scan,
    run_lab_a_wall_scan,
    run_lab_b_domain_wall_scan,
    run_lab_b_physical_domain_wall_scan,
    run_lab_b_strict_scan,
    run_lab_b_structural_wall_scan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan",
        choices=(
            "lab-a-onsite",
            "lab-a-bloch",
            "lab-a-wall",
            "lab-b-strict",
            "lab-b-structural-wall",
            "lab-b-domain-wall",
            "lab-b-physical-domain-wall",
        ),
    )
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--exact-domain-center", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.scan == "lab-a-onsite":
        rows = run_lab_a_onsite_scan(max_candidates=args.max_candidates)
    elif args.scan == "lab-a-bloch":
        rows = run_lab_a_bloch_scan(max_candidates=args.max_candidates)
    elif args.scan == "lab-a-wall":
        rows = run_lab_a_wall_scan(max_candidates=args.max_candidates, max_pairs=args.max_pairs)
    elif args.scan == "lab-b-strict":
        rows = run_lab_b_strict_scan(max_candidates=args.max_candidates)
    elif args.scan == "lab-b-structural-wall":
        rows = run_lab_b_structural_wall_scan(
            max_candidates=args.max_candidates,
            max_pairs=args.max_pairs,
        )
    elif args.scan == "lab-b-domain-wall":
        rows = run_lab_b_domain_wall_scan(
            max_candidates=args.max_candidates,
            max_pairs=args.max_pairs,
            verify_center_exact=args.exact_domain_center,
        )
    else:
        rows = run_lab_b_physical_domain_wall_scan(
            max_candidates=args.max_candidates,
            max_pairs=args.max_pairs,
        )
    print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
