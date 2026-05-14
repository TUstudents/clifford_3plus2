from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.floquet_alpha import (
    FloquetAlphaPolarizationCertificate,
    floquet_alpha_candidates,
    floquet_alpha_polarization_certificate,
)


def _certificate_to_dict(certificate: FloquetAlphaPolarizationCertificate) -> dict[str, object]:
    return {
        "candidate_name": certificate.candidate_name,
        "exact_working_field": certificate.exact_working_field,
        "alpha_projector_rank": certificate.alpha_projector_rank,
        "eta_projector_rank": certificate.eta_projector_rank,
        "spectral_projectors_are_idempotent": certificate.spectral_projectors_are_idempotent,
        "spectral_projectors_are_complementary": (
            certificate.spectral_projectors_are_complementary
        ),
        "scaled_alpha_relation": certificate.scaled_alpha_relation,
        "scaled_alpha_square_relation": certificate.scaled_alpha_square_relation,
        "scaled_alpha_orthogonality_relation": (
            certificate.scaled_alpha_orthogonality_relation
        ),
        "scaled_alpha_commutes_with_projectors": (
            certificate.scaled_alpha_commutes_with_projectors
        ),
        "eta_j_square_relation": certificate.eta_j_square_relation,
        "eta_j_orthogonality_relation": certificate.eta_j_orthogonality_relation,
        "scaled_polarization_certified": certificate.scaled_polarization_certified,
        "normalized_j_requires_sqrt3": certificate.normalized_j_requires_sqrt3,
        "canonical_j_generated_by_floquet": certificate.canonical_j_generated_by_floquet,
        "canonical_j_squared_minus_identity": certificate.canonical_j_squared_minus_identity,
        "canonical_j_orthogonal": certificate.canonical_j_orthogonal,
        "canonical_j_commutes_with_projectors": certificate.canonical_j_commutes_with_projectors,
        "central_idempotent_ranks": list(certificate.central_idempotent_ranks),
        "complementary_rank_6_4_pairs": certificate.complementary_rank_6_4_pairs,
        "lower_rank_central_idempotents": certificate.lower_rank_central_idempotents,
        "strict_compatible_j_forced": certificate.strict_compatible_j_forced,
        "compatible_j_solved": certificate.compatible_j_solved,
        "alpha_plus_polarization_passed": certificate.alpha_plus_polarization_passed,
        "pass_strict_rule_to_bridge": certificate.pass_strict_rule_to_bridge,
        "verdict": certificate.verdict,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search Floquet-alpha with canonical polarization J extraction."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if alpha-plus fails.")
    parser.add_argument("--pattern-index", type=int, help="Run a single resonance pattern.")
    args = parser.parse_args()

    candidates = floquet_alpha_candidates()
    if args.pattern_index is not None:
        candidates = tuple(
            candidate for candidate in candidates if candidate.pattern_index == args.pattern_index
        )
        if not candidates:
            raise SystemExit(f"unknown pattern index: {args.pattern_index}")

    certificates = tuple(floquet_alpha_polarization_certificate(candidate) for candidate in candidates)
    payload = {
        "family": "floquet_alpha_plus",
        "candidate_count": len(certificates),
        "polarization_j_candidates": sum(
            certificate.alpha_plus_polarization_passed for certificate in certificates
        ),
        "scaled_polarization_certified_candidates": sum(
            certificate.scaled_polarization_certified for certificate in certificates
        ),
        "strict_bridge_candidates": sum(
            certificate.pass_strict_rule_to_bridge for certificate in certificates
        ),
        "strict_compatible_j_forced_candidates": sum(
            certificate.strict_compatible_j_forced for certificate in certificates
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
        print("This searches Floquet-alpha plus canonical polarization J extraction.")
        print("It reports the strict compatible-J obstruction separately.")
        print(f"candidate_count: {payload['candidate_count']}")
        print(f"polarization_j_candidates: {payload['polarization_j_candidates']}")
        print(
            "scaled_polarization_certified_candidates: "
            f"{payload['scaled_polarization_certified_candidates']}"
        )
        print(f"strict_compatible_j_forced_candidates: {payload['strict_compatible_j_forced_candidates']}")
        print(f"strict_bridge_candidates: {payload['strict_bridge_candidates']}")
        print(f"verdict_counts: {payload['verdict_counts']}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")
        for certificate in certificates:
            print(
                "candidate: "
                f"{certificate.candidate_name}, "
                f"alpha_eta_ranks=({certificate.alpha_projector_rank},"
                f"{certificate.eta_projector_rank}), "
                f"scaled_cert={str(certificate.scaled_polarization_certified).lower()}, "
                f"center_ranks={list(certificate.central_idempotent_ranks)}, "
                f"polarization_j={str(certificate.alpha_plus_polarization_passed).lower()}, "
                f"strict_forced_j={str(certificate.strict_compatible_j_forced).lower()}, "
                f"verdict={certificate.verdict}"
            )

    if args.check and payload["polarization_j_candidates"] != payload["candidate_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
