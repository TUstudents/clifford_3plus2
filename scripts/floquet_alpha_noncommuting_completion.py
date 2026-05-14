from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.floquet_alpha_noncommuting import (
    FloquetAlphaNoncommutingCompletionCertificate,
    floquet_alpha_noncommuting_candidates,
    floquet_alpha_noncommuting_completion_certificate,
)


def _certificate_to_dict(
    certificate: FloquetAlphaNoncommutingCompletionCertificate,
) -> dict[str, object]:
    return {
        "family": "floquet_alpha_noncommuting_completion",
        "candidate_name": certificate.candidate_name,
        "completed_j_index": certificate.completed_j_index,
        "completed_j_pair_orientation_signs": list(
            certificate.completed_j_pair_orientation_signs
        ),
        "w_in_previous_generated_algebra": certificate.w_in_previous_generated_algebra,
        "w_in_completed_generated_algebra": certificate.w_in_completed_generated_algebra,
        "w_in_completed_center": certificate.w_in_completed_center,
        "w_commutes_with_u1": certificate.w_commutes_with_u1,
        "w_commutes_with_u2": certificate.w_commutes_with_u2,
        "w_squares_to_minus_identity": certificate.w_squares_to_minus_identity,
        "w_orthogonal": certificate.w_orthogonal,
        "generated_algebra_dimension": certificate.generated_algebra_dimension,
        "center_dimension": certificate.center_dimension,
        "center_solved": certificate.center_solved,
        "central_idempotent_ranks": list(certificate.central_idempotent_ranks),
        "complementary_rank_6_4_pairs": certificate.complementary_rank_6_4_pairs,
        "lower_rank_central_idempotents": certificate.lower_rank_central_idempotents,
        "compatible_centralizer_dimension": certificate.compatible_centralizer_dimension,
        "compatible_j_solved": certificate.compatible_j_solved,
        "compatible_j_moduli_dimension": certificate.compatible_j_moduli_dimension,
        "compatible_complex_structure_count": (
            certificate.compatible_complex_structure_count
        ),
        "local_compatible_operator_dimension": (
            certificate.local_compatible_operator_dimension
        ),
        "local_compatible_j_solved": certificate.local_compatible_j_solved,
        "local_compatible_j_moduli_dimension": (
            certificate.local_compatible_j_moduli_dimension
        ),
        "local_compatible_complex_structure_count": (
            certificate.local_compatible_complex_structure_count
        ),
        "declared_w_is_local_compatible_j": (
            certificate.declared_w_is_local_compatible_j
        ),
        "strict_unique_j_found": certificate.strict_unique_j_found,
        "pass_completion_to_bridge": certificate.pass_completion_to_bridge,
        "completion_label": certificate.completion_label,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose adding one finite compatible J as a third alpha route layer."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the completion diagnostic regresses.",
    )
    parser.add_argument(
        "--pattern-index",
        type=int,
        default=0,
        help="Run one representative resonance pattern. Defaults to 0.",
    )
    parser.add_argument(
        "--j-index",
        type=int,
        default=0,
        help="Compatible J index to add as W. Defaults to 0.",
    )
    args = parser.parse_args()

    candidates = floquet_alpha_noncommuting_candidates(pattern_index=args.pattern_index)
    if not candidates:
        raise SystemExit(f"unknown pattern index: {args.pattern_index}")
    certificate = floquet_alpha_noncommuting_completion_certificate(
        candidates[0],
        j_index=args.j_index,
    )
    payload = _certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This declares one finite compatible J as a diagnostic third layer W.")
        print(f"candidate_name: {certificate.candidate_name}")
        print(f"completed_j_index: {certificate.completed_j_index}")
        print(
            "completed_j_pair_orientation_signs: "
            f"{certificate.completed_j_pair_orientation_signs}"
        )
        print(
            "w_in_previous_generated_algebra: "
            f"{str(certificate.w_in_previous_generated_algebra).lower()}"
        )
        print(
            "w_in_completed_generated_algebra: "
            f"{str(certificate.w_in_completed_generated_algebra).lower()}"
        )
        print(f"w_in_completed_center: {str(certificate.w_in_completed_center).lower()}")
        print(f"w_commutes_with_u1: {str(certificate.w_commutes_with_u1).lower()}")
        print(f"w_commutes_with_u2: {str(certificate.w_commutes_with_u2).lower()}")
        print(
            "w_squares_to_minus_identity: "
            f"{str(certificate.w_squares_to_minus_identity).lower()}"
        )
        print(f"w_orthogonal: {str(certificate.w_orthogonal).lower()}")
        print(f"generated_algebra_dimension: {certificate.generated_algebra_dimension}")
        print(f"center_dimension: {certificate.center_dimension}")
        print(f"center_solved: {str(certificate.center_solved).lower()}")
        print(f"central_idempotent_ranks: {list(certificate.central_idempotent_ranks)}")
        print(
            "complementary_rank_6_4_pairs: "
            f"{certificate.complementary_rank_6_4_pairs}"
        )
        print(
            "lower_rank_central_idempotents: "
            f"{certificate.lower_rank_central_idempotents}"
        )
        print(
            "compatible_centralizer_dimension: "
            f"{certificate.compatible_centralizer_dimension}"
        )
        print(f"compatible_j_moduli_dimension: {certificate.compatible_j_moduli_dimension}")
        print(
            "compatible_complex_structure_count: "
            f"{certificate.compatible_complex_structure_count}"
        )
        print(
            "local_compatible_operator_dimension: "
            f"{certificate.local_compatible_operator_dimension}"
        )
        print(
            "local_compatible_j_moduli_dimension: "
            f"{certificate.local_compatible_j_moduli_dimension}"
        )
        print(
            "local_compatible_complex_structure_count: "
            f"{certificate.local_compatible_complex_structure_count}"
        )
        print(
            "declared_w_is_local_compatible_j: "
            f"{str(certificate.declared_w_is_local_compatible_j).lower()}"
        )
        print(f"strict_unique_j_found: {str(certificate.strict_unique_j_found).lower()}")
        print(
            "pass_completion_to_bridge: "
            f"{str(certificate.pass_completion_to_bridge).lower()}"
        )
        print(f"completion_label: {certificate.completion_label}")
        print(f"load_bearing_qca_bridge: {str(certificate.load_bearing_qca_bridge).lower()}")

    if args.check:
        check_passed = (
            not certificate.w_in_previous_generated_algebra
            and certificate.w_in_completed_generated_algebra
            and certificate.w_in_completed_center
            and certificate.w_commutes_with_u1
            and certificate.w_commutes_with_u2
            and certificate.w_squares_to_minus_identity
            and certificate.w_orthogonal
            and certificate.center_solved
            and certificate.central_idempotent_ranks == (0, 4, 6, 10)
            and certificate.complementary_rank_6_4_pairs == 1
            and certificate.lower_rank_central_idempotents == 0
            and certificate.compatible_centralizer_dimension == 4
            and certificate.compatible_j_moduli_dimension == 0
            and certificate.compatible_complex_structure_count == 4
            and certificate.local_compatible_j_moduli_dimension == 0
            and certificate.local_compatible_complex_structure_count == 4
            and certificate.declared_w_is_local_compatible_j
            and not certificate.strict_unique_j_found
            and not certificate.pass_completion_to_bridge
            and certificate.completion_label
            == "completion_no_lower_rank_but_j_still_block_sign_ambiguous"
        )
        if not check_passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
