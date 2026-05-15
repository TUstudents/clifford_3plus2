from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.floquet_alpha import (
    FloquetAlphaTimeReversalCertificate,
    floquet_alpha_candidates,
    floquet_alpha_time_reversal_certificate,
)


def _certificate_to_dict(
    certificate: FloquetAlphaTimeReversalCertificate,
) -> dict[str, object]:
    return {
        "candidate_name": certificate.candidate_name,
        "time_reversal_name": certificate.time_reversal_name,
        "time_reversal_origin": certificate.time_reversal_origin,
        "k_real_orthogonal": certificate.k_real_orthogonal,
        "k_involution": certificate.k_involution,
        "k_conjugates_floquet_to_inverse": certificate.k_conjugates_floquet_to_inverse,
        "k_anticommutes_with_canonical_j": certificate.k_anticommutes_with_canonical_j,
        "k_in_generated_algebra": certificate.k_in_generated_algebra,
        "compatible_j_moduli_dimension_before_k": (
            certificate.compatible_j_moduli_dimension_before_k
        ),
        "k_fixed_compatible_j_moduli_dimension": (
            certificate.k_fixed_compatible_j_moduli_dimension
        ),
        "k_fixed_generated_complex_structure_count": (
            certificate.k_fixed_generated_complex_structure_count
        ),
        "k_fixed_local_compatible_complex_structure_count": (
            certificate.k_fixed_local_compatible_complex_structure_count
        ),
        "k_fixed_local_matches_canonical_orbit_count": (
            certificate.k_fixed_local_matches_canonical_orbit_count
        ),
        "k_reduces_full_moduli": certificate.k_reduces_full_moduli,
        "k_reduces_to_global_pm": certificate.k_reduces_to_global_pm,
        "strict_bridge_candidates": certificate.strict_bridge_candidates,
        "verdict": certificate.verdict,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Floquet-alpha with a declared time-reversal involution."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the time-reversal diagnostic regresses.",
    )
    parser.add_argument("--pattern-index", type=int, help="Run a single resonance pattern.")
    args = parser.parse_args()

    candidates = floquet_alpha_candidates()
    if args.pattern_index is not None:
        candidates = tuple(
            candidate for candidate in candidates if candidate.pattern_index == args.pattern_index
        )
        if not candidates:
            raise SystemExit(f"unknown pattern index: {args.pattern_index}")

    certificates = tuple(floquet_alpha_time_reversal_certificate(candidate) for candidate in candidates)
    payload = {
        "family": "floquet_alpha_time_reversal",
        "candidate_count": len(certificates),
        "k_real_orthogonal_candidates": sum(
            certificate.k_real_orthogonal for certificate in certificates
        ),
        "k_involution_candidates": sum(certificate.k_involution for certificate in certificates),
        "k_conjugates_floquet_to_inverse_candidates": sum(
            certificate.k_conjugates_floquet_to_inverse for certificate in certificates
        ),
        "k_anticommutes_with_canonical_j_candidates": sum(
            certificate.k_anticommutes_with_canonical_j for certificate in certificates
        ),
        "k_in_generated_algebra_candidates": sum(
            certificate.k_in_generated_algebra for certificate in certificates
        ),
        "compatible_j_moduli_dimension_before_k": (
            certificates[0].compatible_j_moduli_dimension_before_k
        ),
        "k_fixed_compatible_j_moduli_dimension": (
            certificates[0].k_fixed_compatible_j_moduli_dimension
        ),
        "k_fixed_local_compatible_complex_structure_count": (
            certificates[0].k_fixed_local_compatible_complex_structure_count
        ),
        "k_fixed_local_matches_canonical_orbit_count": (
            certificates[0].k_fixed_local_matches_canonical_orbit_count
        ),
        "k_reduces_full_moduli_candidates": sum(
            certificate.k_reduces_full_moduli for certificate in certificates
        ),
        "k_reduces_to_global_pm_candidates": sum(
            certificate.k_reduces_to_global_pm for certificate in certificates
        ),
        "strict_bridge_candidates": sum(
            certificate.strict_bridge_candidates for certificate in certificates
        ),
        "verdict_counts": {
            verdict: sum(certificate.verdict == verdict for certificate in certificates)
            for verdict in sorted({certificate.verdict for certificate in certificates})
        },
        "load_bearing_qca_bridge": False,
        "results": [_certificate_to_dict(certificate) for certificate in certificates],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This checks Floquet-alpha with a declared time-reversal involution.")
        print("K is diagnostic only unless generated by microscopic rule data.")
        print(f"candidate_count: {payload['candidate_count']}")
        print(f"k_real_orthogonal_candidates: {payload['k_real_orthogonal_candidates']}")
        print(f"k_involution_candidates: {payload['k_involution_candidates']}")
        print(
            "k_conjugates_floquet_to_inverse_candidates: "
            f"{payload['k_conjugates_floquet_to_inverse_candidates']}"
        )
        print(
            "k_anticommutes_with_canonical_j_candidates: "
            f"{payload['k_anticommutes_with_canonical_j_candidates']}"
        )
        print(f"k_in_generated_algebra_candidates: {payload['k_in_generated_algebra_candidates']}")
        print(
            "compatible_j_moduli_dimension_before_k: "
            f"{payload['compatible_j_moduli_dimension_before_k']}"
        )
        print(
            "k_fixed_compatible_j_moduli_dimension: "
            f"{payload['k_fixed_compatible_j_moduli_dimension']}"
        )
        print(
            "k_fixed_local_compatible_complex_structure_count: "
            f"{payload['k_fixed_local_compatible_complex_structure_count']}"
        )
        print(
            "k_fixed_local_matches_canonical_orbit_count: "
            f"{payload['k_fixed_local_matches_canonical_orbit_count']}"
        )
        print(
            "k_reduces_full_moduli_candidates: "
            f"{payload['k_reduces_full_moduli_candidates']}"
        )
        print(
            "k_reduces_to_global_pm_candidates: "
            f"{payload['k_reduces_to_global_pm_candidates']}"
        )
        print(f"strict_bridge_candidates: {payload['strict_bridge_candidates']}")
        print(f"verdict_counts: {payload['verdict_counts']}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")
        for certificate in certificates:
            print(
                "candidate: "
                f"{certificate.candidate_name}, "
                f"k_origin={certificate.time_reversal_origin}, "
                f"k_in_generated={str(certificate.k_in_generated_algebra).lower()}, "
                f"j_moduli_before={certificate.compatible_j_moduli_dimension_before_k}, "
                f"k_fixed_moduli={certificate.k_fixed_compatible_j_moduli_dimension}, "
                f"k_fixed_local_j_count={certificate.k_fixed_local_compatible_complex_structure_count}, "
                f"k_global_pm={str(certificate.k_reduces_to_global_pm).lower()}, "
                f"verdict={certificate.verdict}"
            )

    if args.check:
        check_passed = (
            payload["candidate_count"] == 10
            and payload["k_real_orthogonal_candidates"] == 10
            and payload["k_involution_candidates"] == 10
            and payload["k_conjugates_floquet_to_inverse_candidates"] == 10
            and payload["k_anticommutes_with_canonical_j_candidates"] == 10
            and payload["k_in_generated_algebra_candidates"] == 0
            and payload["compatible_j_moduli_dimension_before_k"] == 9
            and payload["k_fixed_compatible_j_moduli_dimension"] == 3
            and payload["k_fixed_local_compatible_complex_structure_count"] == 4
            and payload["k_fixed_local_matches_canonical_orbit_count"] == 2
            and payload["k_reduces_full_moduli_candidates"] == 10
            and payload["k_reduces_to_global_pm_candidates"] == 0
            and payload["strict_bridge_candidates"] == 0
            and payload["verdict_counts"]
            == {"declared_time_reversal_reduces_moduli_not_unique": 10}
            and not payload["load_bearing_qca_bridge"]
        )
        if not check_passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
