from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.defect_beta import (
    DefectBetaCertificate,
    defect_beta_candidates,
    defect_beta_certificate,
)


def _certificate_to_dict(certificate: DefectBetaCertificate) -> dict[str, object]:
    return {
        "candidate_name": certificate.candidate_name,
        "transition_count": certificate.transition_count,
        "monodromy_computed_from_transitions": certificate.monodromy_computed_from_transitions,
        "omega_projector_rank": certificate.omega_projector_rank,
        "i_projector_rank": certificate.i_projector_rank,
        "spectral_projectors_are_idempotent": certificate.spectral_projectors_are_idempotent,
        "spectral_projectors_are_complementary": (
            certificate.spectral_projectors_are_complementary
        ),
        "canonical_j_generated_by_monodromy": certificate.canonical_j_generated_by_monodromy,
        "canonical_j_squared_minus_identity": certificate.canonical_j_squared_minus_identity,
        "canonical_j_orthogonal": certificate.canonical_j_orthogonal,
        "canonical_j_commutes_with_projectors": certificate.canonical_j_commutes_with_projectors,
        "central_idempotent_ranks": list(certificate.central_idempotent_ranks),
        "complementary_rank_6_4_pairs": certificate.complementary_rank_6_4_pairs,
        "lower_rank_central_idempotents": certificate.lower_rank_central_idempotents,
        "strict_compatible_j_forced": certificate.strict_compatible_j_forced,
        "compatible_j_solved": certificate.compatible_j_solved,
        "beta_monodromy_passed": certificate.beta_monodromy_passed,
        "pass_strict_rule_to_bridge": certificate.pass_strict_rule_to_bridge,
        "verdict": certificate.verdict,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the defect-beta monodromy family.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if beta search fails.")
    parser.add_argument("--pattern-index", type=int, help="Run a single defect-charge pattern.")
    args = parser.parse_args()

    candidates = defect_beta_candidates()
    if args.pattern_index is not None:
        candidates = tuple(
            candidate for candidate in candidates if candidate.pattern_index == args.pattern_index
        )
        if not candidates:
            raise SystemExit(f"unknown pattern index: {args.pattern_index}")

    certificates = tuple(defect_beta_certificate(candidate) for candidate in candidates)
    payload = {
        "family": "defect_beta",
        "candidate_count": len(certificates),
        "monodromy_candidates": sum(certificate.beta_monodromy_passed for certificate in certificates),
        "strict_compatible_j_forced_candidates": sum(
            certificate.strict_compatible_j_forced for certificate in certificates
        ),
        "strict_bridge_candidates": sum(
            certificate.pass_strict_rule_to_bridge for certificate in certificates
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
        print("This searches the defect-beta monodromy primitive family.")
        print("It computes round-trip monodromy from wall transition functions.")
        print(f"candidate_count: {payload['candidate_count']}")
        print(f"monodromy_candidates: {payload['monodromy_candidates']}")
        print(f"strict_compatible_j_forced_candidates: {payload['strict_compatible_j_forced_candidates']}")
        print(f"strict_bridge_candidates: {payload['strict_bridge_candidates']}")
        print(f"verdict_counts: {payload['verdict_counts']}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")
        for certificate in certificates:
            print(
                "candidate: "
                f"{certificate.candidate_name}, "
                f"transitions={certificate.transition_count}, "
                f"omega_i_ranks=({certificate.omega_projector_rank},"
                f"{certificate.i_projector_rank}), "
                f"center_ranks={list(certificate.central_idempotent_ranks)}, "
                f"monodromy_j={str(certificate.beta_monodromy_passed).lower()}, "
                f"strict_forced_j={str(certificate.strict_compatible_j_forced).lower()}, "
                f"verdict={certificate.verdict}"
            )

    if args.check and payload["monodromy_candidates"] != payload["candidate_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
