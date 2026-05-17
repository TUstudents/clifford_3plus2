from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.obstruction_r10.search.normalizer import (
    certificate_to_dict,
    full_u5_like_addressability_controls,
    normalizer_certificate,
)
from clifford_3plus2_d5.obstruction_r10.search.addressability import (
    off_block_mixer_control,
    rank_one_color_projector_controls,
    rank_one_weak_projector_controls,
    standard_block_projectors,
)


VERDICT_CHOICES = ("forced", "weak_orbit", "candidate_only", "falsified")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check forcedness and normalizer proxies for the 3+2 candidate."
    )
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
    parser.add_argument(
        "--include-off-block",
        action="store_true",
        help="Include a block-mixing control as a falsifier.",
    )
    parser.add_argument(
        "--include-full-u5-controls",
        action="store_true",
        help="Include a compact unsafe proxy for controls resolving all five modes.",
    )
    parser.add_argument("--expect-verdict", choices=VERDICT_CHOICES)
    args = parser.parse_args()

    operators = standard_block_projectors()
    if args.include_rank_one_color:
        operators = operators + rank_one_color_projector_controls()
    if args.include_rank_one_weak:
        operators = operators + rank_one_weak_projector_controls()
    if args.include_off_block:
        operators = operators + (off_block_mixer_control(),)
    if args.include_full_u5_controls:
        operators = operators + full_u5_like_addressability_controls()

    certificate = normalizer_certificate(addressable_operators=operators)
    payload = certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This checks forcedness and normalizer proxies for the candidate data.")
        print("It does not prove microscopic QCA rule data force J or P_3/P_2.")
        print(f"rule_data_source: {payload['rule_data_source']}")
        print(f"centralizer_dimension: {payload['centralizer_dimension']}")
        print(
            "orthogonal_normalizer_dimension: "
            f"{payload['orthogonal_normalizer_dimension']}"
        )
        print(
            "addressability_algebra_dimension: "
            f"{payload['addressability_algebra_dimension']}"
        )
        print(f"candidate_j_valid: {str(payload['candidate_j_valid']).lower()}")
        print(f"candidate_split_valid: {str(payload['candidate_split_valid']).lower()}")
        print(
            "candidate_j_preserved_by_normalizer: "
            f"{str(payload['candidate_j_preserved_by_normalizer']).lower()}"
        )
        print(
            "normalizer_preserves_declared_split: "
            f"{str(payload['normalizer_preserves_declared_split']).lower()}"
        )
        print(
            "continuous_j_alternatives_not_excluded: "
            f"{str(payload['continuous_j_alternatives_not_excluded']).lower()}"
        )
        print(
            "rank_three_projector_family_not_excluded: "
            f"{str(payload['rank_three_projector_family_not_excluded']).lower()}"
        )
        print(
            "rank_one_color_projectors_addressable: "
            f"{str(payload['rank_one_color_projectors_addressable']).lower()}"
        )
        print(
            "rank_one_weak_projectors_addressable: "
            f"{str(payload['rank_one_weak_projectors_addressable']).lower()}"
        )
        print(
            "off_block_controls_addressable: "
            f"{str(payload['off_block_controls_addressable']).lower()}"
        )
        print(
            "addressability_algebra_safe: "
            f"{str(payload['addressability_algebra_safe']).lower()}"
        )
        print(f"j_unique_or_forced: {str(payload['j_unique_or_forced']).lower()}")
        print(f"split_unique_or_forced: {str(payload['split_unique_or_forced']).lower()}")
        print(f"forcedness_verdict: {payload['forcedness_verdict']}")
        print(f"normalizer_check_passed: {str(payload['normalizer_check_passed']).lower()}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if args.expect_verdict is not None and certificate.forcedness_verdict != args.expect_verdict:
        return 1
    if args.check and not payload["normalizer_check_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
