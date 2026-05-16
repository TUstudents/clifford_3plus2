"""Print bounded lepton-lab scan summaries as JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from clifford_3plus2_d5.lepton.results import (
    lab_a_summary,
    lab_b_domain_wall_summary,
    lab_b_physical_domain_wall_summary,
    lab_b_strict_summary,
    lab_b_structural_wall_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-a-max", type=int, default=1)
    parser.add_argument("--lab-b-max", type=int, default=1)
    parser.add_argument("--lab-b-pairs", type=int, default=1)
    parser.add_argument(
        "--exact-domain-center",
        action="store_true",
        help="Use exact center verification for the domain-wall summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = {
        "lab_a": [
            asdict(summary) for summary in lab_a_summary(max_candidates_per_mechanism=args.lab_a_max)
        ],
        "lab_b_strict": asdict(lab_b_strict_summary(max_candidates=args.lab_b_max)),
        "lab_b_structural_wall": asdict(
            lab_b_structural_wall_summary(max_candidates=args.lab_b_max, max_pairs=args.lab_b_pairs)
        ),
        "lab_b_domain_wall": asdict(
            lab_b_domain_wall_summary(
                max_candidates=args.lab_b_max,
                max_pairs=args.lab_b_pairs,
                verify_center_exact=args.exact_domain_center,
            )
        ),
        "lab_b_physical_domain_wall": asdict(
            lab_b_physical_domain_wall_summary(
                max_candidates=args.lab_b_max,
                max_pairs=args.lab_b_pairs,
            )
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
