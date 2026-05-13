from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.search.forced_j import certificate_to_dict, j_certificate
from clifford_3plus2_d5.search.gate_words import (
    rank_one_pair_rotations,
    standard_period_four_primitives,
)


VERDICT_CHOICES = ("forced_j", "candidate_only", "falsified")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a declared gate word produces J.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the checker fails.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--include-addressable-rank-one",
        action="store_true",
        help="Include independently addressable rank-one pair rotations as a falsifier.",
    )
    parser.add_argument("--expect-verdict", choices=VERDICT_CHOICES)
    args = parser.parse_args()

    primitives = standard_period_four_primitives()
    if args.include_addressable_rank_one:
        primitives = primitives + rank_one_pair_rotations()

    certificate = j_certificate(primitives=primitives)
    payload = certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This checks whether declared exact gate words can produce J.")
        print("It does not prove microscopic QCA rule data force J.")
        print(f"candidate_name: {payload['candidate_name']}")
        print(f"generated_by_gate_word: {str(payload['generated_by_gate_word']).lower()}")
        print(f"gate_word: {','.join(payload['gate_word'])}")
        print(f"J_squared_minus_identity: {str(payload['squares_to_minus_identity']).lower()}")
        print(f"J_orthogonal: {str(payload['is_real_orthogonal']).lower()}")
        print(
            "rank_one_pair_rotations_addressable: "
            f"{str(payload['rank_one_pair_rotations_addressable']).lower()}"
        )
        print(f"forced_j_check_passed: {str(payload['forced_j_check_passed']).lower()}")
        print(f"qca_forces_j: {str(payload['qca_forces_j']).lower()}")
        print(f"forced_j_verdict: {payload['forced_j_verdict']}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if args.expect_verdict is not None and certificate.verdict != args.expect_verdict:
        return 1
    if args.check and not payload["forced_j_check_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
