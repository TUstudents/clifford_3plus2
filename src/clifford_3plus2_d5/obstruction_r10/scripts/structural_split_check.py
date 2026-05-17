from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.obstruction_r10.search.addressability import (
    certificate_to_dict,
    rank_one_color_projector_controls,
    rank_one_weak_projector_controls,
    standard_block_projectors,
    structural_split_certificate,
)


VERDICT_CHOICES = ("structural_split", "candidate_only", "falsified")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the structural 3+2 split candidate.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the checker fails.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--include-rank-one-color",
        action="store_true",
        help="Include independently addressable color-axis projectors as a falsifier.",
    )
    parser.add_argument(
        "--include-rank-one-weak",
        action="store_true",
        help="Include independently addressable weak-axis projectors as a falsifier.",
    )
    parser.add_argument("--expect-verdict", choices=VERDICT_CHOICES)
    args = parser.parse_args()

    operators = standard_block_projectors()
    if args.include_rank_one_color:
        operators = operators + rank_one_color_projector_controls()
    if args.include_rank_one_weak:
        operators = operators + rank_one_weak_projector_controls()

    certificate = structural_split_certificate(operators=operators)
    payload = certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This verifies the exact structural split candidate only.")
        print("It does not prove QCA rule data force P_3/P_2.")
        print(
            "projector_identities_passed: "
            f"{str(payload['projector_identities_passed']).lower()}"
        )
        print(f"projectors_commute_with_J: {str(payload['projectors_commute_with_j']).lower()}")
        print(f"projector_3_rank: {payload['projector_3_rank']}")
        print(f"projector_2_rank: {payload['projector_2_rank']}")
        print(
            "rank_one_color_projectors_addressable: "
            f"{str(payload['rank_one_color_projectors_addressable']).lower()}"
        )
        print(
            "rank_one_weak_projectors_addressable: "
            f"{str(payload['rank_one_weak_projectors_addressable']).lower()}"
        )
        print(
            "addressability_algebra_safe: "
            f"{str(payload['addressability_algebra_safe']).lower()}"
        )
        print(
            "qca_supplies_structural_3plus2_split: "
            f"{str(payload['qca_supplies_structural_3plus2_split']).lower()}"
        )
        print(f"structural_split_verdict: {payload['structural_split_verdict']}")
        print(
            "structural_split_check_passed: "
            f"{str(payload['structural_split_check_passed']).lower()}"
        )
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if (
        args.expect_verdict is not None
        and certificate.structural_split_verdict != args.expect_verdict
    ):
        return 1
    if args.check and not payload["structural_split_check_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
