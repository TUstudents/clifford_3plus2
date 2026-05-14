from __future__ import annotations

import argparse
import json

import sympy as sp

from clifford_3plus2_d5.qca.floquet_alpha_noncommuting import (
    FloquetAlphaNoncommutingJDiagnostic,
    FloquetAlphaNoncommutingJGapCertificate,
    floquet_alpha_noncommuting_candidates,
    floquet_alpha_noncommuting_j_gap_certificate,
)


def _matrix_to_rows(matrix: sp.Matrix) -> list[list[str]]:
    return [[str(sp.simplify(matrix[row, column])) for column in range(matrix.cols)] for row in range(matrix.rows)]


def _diagnostic_to_dict(diagnostic: FloquetAlphaNoncommutingJDiagnostic) -> dict[str, object]:
    return {
        "index": diagnostic.index,
        "expression": [
            [name, str(value)] for name, value in diagnostic.expression
        ],
        "pair_orientation_signs": list(diagnostic.pair_orientation_signs),
        "in_generated_algebra": diagnostic.in_generated_algebra,
        "in_rule_local_center": diagnostic.in_rule_local_center,
        "equals_spectral_polarization_j": diagnostic.equals_spectral_polarization_j,
        "equals_negative_spectral_polarization_j": (
            diagnostic.equals_negative_spectral_polarization_j
        ),
        "commutes_with_u1": diagnostic.commutes_with_u1,
        "commutes_with_u2": diagnostic.commutes_with_u2,
        "squares_to_minus_identity": diagnostic.squares_to_minus_identity,
        "orthogonal": diagnostic.orthogonal,
        "matrix": _matrix_to_rows(diagnostic.matrix),
    }


def _certificate_to_dict(
    certificate: FloquetAlphaNoncommutingJGapCertificate,
) -> dict[str, object]:
    return {
        "family": "floquet_alpha_noncommuting_j_gap",
        "candidate_name": certificate.candidate_name,
        "compatible_j_count": certificate.compatible_j_count,
        "generated_algebra_dimension": certificate.generated_algebra_dimension,
        "center_dimension": certificate.center_dimension,
        "compatible_centralizer_dimension": certificate.compatible_centralizer_dimension,
        "compatible_j_solved": certificate.compatible_j_solved,
        "compatible_j_moduli_dimension": certificate.compatible_j_moduli_dimension,
        "generated_j_solved": certificate.generated_j_solved,
        "generated_complex_structure_count": (
            certificate.generated_complex_structure_count
        ),
        "local_compatible_j_solved": certificate.local_compatible_j_solved,
        "local_compatible_j_moduli_dimension": (
            certificate.local_compatible_j_moduli_dimension
        ),
        "local_compatible_complex_structure_count": (
            certificate.local_compatible_complex_structure_count
        ),
        "compatible_j_in_generated_algebra_count": (
            certificate.compatible_j_in_generated_algebra_count
        ),
        "compatible_j_in_rule_local_center_count": (
            certificate.compatible_j_in_rule_local_center_count
        ),
        "spectral_polarization_j_matched_count": (
            certificate.spectral_polarization_j_matched_count
        ),
        "forced_j_found": certificate.forced_j_found,
        "reason_for_forced_j_failure": certificate.reason_for_forced_j_failure,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
        "compatible_j_diagnostics": [
            _diagnostic_to_dict(diagnostic)
            for diagnostic in certificate.compatible_j_diagnostics
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain why the noncommuting Floquet-alpha route does not force J."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the exact J-gap diagnostics regress.",
    )
    parser.add_argument(
        "--pattern-index",
        type=int,
        default=0,
        help="Run one representative resonance pattern. Defaults to 0.",
    )
    args = parser.parse_args()

    candidates = floquet_alpha_noncommuting_candidates(pattern_index=args.pattern_index)
    if not candidates:
        raise SystemExit(f"unknown pattern index: {args.pattern_index}")
    certificate = floquet_alpha_noncommuting_j_gap_certificate(candidates[0])
    payload = _certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This extracts the finite compatible J gap for the noncommuting alpha route.")
        print(f"candidate_name: {certificate.candidate_name}")
        print(f"compatible_j_count: {certificate.compatible_j_count}")
        print(f"generated_algebra_dimension: {certificate.generated_algebra_dimension}")
        print(f"center_dimension: {certificate.center_dimension}")
        print(
            "compatible_centralizer_dimension: "
            f"{certificate.compatible_centralizer_dimension}"
        )
        print(f"compatible_j_solved: {str(certificate.compatible_j_solved).lower()}")
        print(f"compatible_j_moduli_dimension: {certificate.compatible_j_moduli_dimension}")
        print(f"generated_j_solved: {str(certificate.generated_j_solved).lower()}")
        print(
            "generated_complex_structure_count: "
            f"{certificate.generated_complex_structure_count}"
        )
        print(
            "local_compatible_complex_structure_count: "
            f"{certificate.local_compatible_complex_structure_count}"
        )
        print(
            "compatible_j_in_generated_algebra_count: "
            f"{certificate.compatible_j_in_generated_algebra_count}"
        )
        print(
            "compatible_j_in_rule_local_center_count: "
            f"{certificate.compatible_j_in_rule_local_center_count}"
        )
        print(
            "spectral_polarization_j_matched_count: "
            f"{certificate.spectral_polarization_j_matched_count}"
        )
        print(f"forced_j_found: {str(certificate.forced_j_found).lower()}")
        print(f"reason_for_forced_j_failure: {certificate.reason_for_forced_j_failure}")
        print(f"load_bearing_qca_bridge: {str(certificate.load_bearing_qca_bridge).lower()}")
        for diagnostic in certificate.compatible_j_diagnostics:
            print(
                "compatible_j: "
                f"index={diagnostic.index}, "
                f"pair_signs={diagnostic.pair_orientation_signs}, "
                f"in_generated={str(diagnostic.in_generated_algebra).lower()}, "
                f"in_local_center={str(diagnostic.in_rule_local_center).lower()}, "
                f"matches_spectral_j="
                f"{str(diagnostic.equals_spectral_polarization_j or diagnostic.equals_negative_spectral_polarization_j).lower()}"
            )

    if args.check:
        expected_pair_signs = {
            (1, 1, -1, 1, -1),
            (1, 1, -1, -1, 1),
            (-1, -1, 1, 1, -1),
            (-1, -1, 1, -1, 1),
        }
        check_passed = (
            certificate.compatible_j_count == 4
            and certificate.compatible_j_moduli_dimension == 0
            and certificate.compatible_j_in_generated_algebra_count == 0
            and certificate.compatible_j_in_rule_local_center_count == 0
            and certificate.spectral_polarization_j_matched_count == 0
            and not certificate.forced_j_found
            and certificate.reason_for_forced_j_failure
            == "compatible_j_finite_but_not_generated_or_rule_local"
            and {
                diagnostic.pair_orientation_signs
                for diagnostic in certificate.compatible_j_diagnostics
            }
            == expected_pair_signs
        )
        if not check_passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
