from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.obstruction_r10.search.real_qca_branch import (
    certificate_to_dict,
    rank_one_pair_rotation_primitives,
    real_qca_branch_certificate,
    standard_real_qca_primitives,
)
from clifford_3plus2_d5.obstruction_r10.search.addressability import (
    off_block_mixer_control,
    rank_one_color_projector_controls,
    rank_one_weak_projector_controls,
    standard_block_projectors,
)


VERDICT_CHOICES = ("forced_candidate", "candidate_only", "falsified")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the Phase 8A stronger real-QCA-first branch."
    )
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the checker fails.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--max-depth", type=int, default=4)
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
        "--include-rank-one-pair",
        action="store_true",
        help="Include independently addressable rank-one pair rotations as a falsifier.",
    )
    parser.add_argument(
        "--include-off-block",
        action="store_true",
        help="Include a block-mixing control as a falsifier.",
    )
    parser.add_argument("--expect-verdict", choices=VERDICT_CHOICES)
    args = parser.parse_args()

    primitives = standard_real_qca_primitives()
    if args.include_rank_one_pair:
        primitives = primitives + rank_one_pair_rotation_primitives()

    addressable_operators = standard_block_projectors()
    if args.include_rank_one_color:
        addressable_operators = addressable_operators + rank_one_color_projector_controls()
    if args.include_rank_one_weak:
        addressable_operators = addressable_operators + rank_one_weak_projector_controls()
    if args.include_off_block:
        addressable_operators = addressable_operators + (off_block_mixer_control(),)

    certificate = real_qca_branch_certificate(
        primitives=primitives,
        addressable_operators=addressable_operators,
        max_depth=args.max_depth,
    )
    payload = certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This checks the stronger real-QCA-first branch candidate.")
        print("It does not prove microscopic QCA rule data force J or P_3/P_2.")
        print(f"branch_name: {payload['branch_name']}")
        print(f"candidate_word_found: {str(payload['candidate_word_found']).lower()}")
        print(f"candidate_word: {','.join(payload['candidate_word'])}")
        print(f"word_depth: {payload['word_depth']}")
        print(f"finite_depth: {str(payload['finite_depth']).lower()}")
        print(f"translation_invariant: {str(payload['translation_invariant']).lower()}")
        print(f"generates_j: {str(payload['generates_j']).lower()}")
        print(f"generates_split: {str(payload['generates_split']).lower()}")
        print(
            "j_forced_by_rule_space: "
            f"{str(payload['j_forced_by_rule_space']).lower()}"
        )
        print(
            "split_forced_by_rule_space: "
            f"{str(payload['split_forced_by_rule_space']).lower()}"
        )
        print(
            "forbidden_rank_one_controls_present: "
            f"{str(payload['forbidden_rank_one_controls_present']).lower()}"
        )
        print(
            "forbidden_off_block_controls_present: "
            f"{str(payload['forbidden_off_block_controls_present']).lower()}"
        )
        print(
            "addressability_algebra_safe: "
            f"{str(payload['addressability_algebra_safe']).lower()}"
        )
        print(f"normalizer_verdict: {payload['normalizer_verdict']}")
        print(f"real_qca_branch_verdict: {payload['real_qca_branch_verdict']}")
        print(
            "real_qca_branch_check_passed: "
            f"{str(payload['real_qca_branch_check_passed']).lower()}"
        )
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if (
        args.expect_verdict is not None
        and certificate.real_qca_branch_verdict != args.expect_verdict
    ):
        return 1
    if args.check and not payload["real_qca_branch_check_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
