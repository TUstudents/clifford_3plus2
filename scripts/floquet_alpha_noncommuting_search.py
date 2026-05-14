from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.floquet_alpha_noncommuting import (
    FloquetAlphaNoncommutingCertificate,
    floquet_alpha_noncommuting_candidates,
    floquet_alpha_noncommuting_certificate,
)


def _certificate_to_dict(
    certificate: FloquetAlphaNoncommutingCertificate,
) -> dict[str, object]:
    return {
        "candidate_name": certificate.candidate_name,
        "base_candidate_name": certificate.base_candidate_name,
        "twist_layer_name": certificate.twist_layer_name,
        "orientation_signs": list(certificate.orientation_signs),
        "mode_mapping": [list(item) for item in certificate.mode_mapping],
        "u1_u2_commute": certificate.u1_u2_commute,
        "u2_real_orthogonal": certificate.u2_real_orthogonal,
        "u2_preserves_alpha_projector": certificate.u2_preserves_alpha_projector,
        "u2_preserves_eta_projector": certificate.u2_preserves_eta_projector,
        "alpha_orientation_nonconstant": certificate.alpha_orientation_nonconstant,
        "eta_orientation_nonconstant": certificate.eta_orientation_nonconstant,
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
        "generated_j_solved": certificate.generated_j_solved,
        "generated_j_moduli_dimension": certificate.generated_j_moduli_dimension,
        "generated_complex_structure_count": certificate.generated_complex_structure_count,
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
        "forced_j_found": certificate.forced_j_found,
        "pass_strict_rule_to_bridge": certificate.pass_strict_rule_to_bridge,
        "rule_verdict": certificate.rule_verdict,
        "route_label": certificate.route_label,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def _payload(certificates: tuple[FloquetAlphaNoncommutingCertificate, ...]) -> dict[str, object]:
    compatible_dimensions = [
        certificate.compatible_centralizer_dimension for certificate in certificates
    ]
    return {
        "family": "floquet_alpha_noncommuting",
        "candidate_count": len(certificates),
        "noncommuting_candidates": sum(
            not certificate.u1_u2_commute for certificate in certificates
        ),
        "block_preserving_candidates": sum(
            certificate.u2_preserves_alpha_projector
            and certificate.u2_preserves_eta_projector
            for certificate in certificates
        ),
        "coarse_center_preserved_candidates": sum(
            certificate.complementary_rank_6_4_pairs > 0
            and certificate.lower_rank_central_idempotents == 0
            for certificate in certificates
        ),
        "compatible_j_zero_dimensional_candidates": sum(
            certificate.compatible_j_solved
            and certificate.compatible_j_moduli_dimension == 0
            for certificate in certificates
        ),
        "forced_j_candidates": sum(
            certificate.forced_j_found for certificate in certificates
        ),
        "strict_bridge_candidates": sum(
            certificate.pass_strict_rule_to_bridge for certificate in certificates
        ),
        "best_compatible_centralizer_dimension": (
            min(compatible_dimensions) if compatible_dimensions else None
        ),
        "route_label_counts": {
            label: sum(certificate.route_label == label for certificate in certificates)
            for label in sorted({certificate.route_label for certificate in certificates})
        },
        "load_bearing_qca_bridge": False,
        "results": [_certificate_to_dict(certificate) for certificate in certificates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search block-preserving noncommuting Floquet-alpha signed twists."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the representative route diagnostics regress.",
    )
    parser.add_argument(
        "--pattern-index",
        type=int,
        default=0,
        help="Run a single resonance pattern. Defaults to the representative pattern 0.",
    )
    parser.add_argument(
        "--all-patterns",
        action="store_true",
        help="Run every symmetric Floquet-alpha pattern.",
    )
    args = parser.parse_args()

    pattern_index = None if args.all_patterns else args.pattern_index
    candidates = floquet_alpha_noncommuting_candidates(pattern_index=pattern_index)
    if not candidates:
        raise SystemExit(f"unknown pattern index: {args.pattern_index}")

    certificates = tuple(
        floquet_alpha_noncommuting_certificate(candidate) for candidate in candidates
    )
    payload = _payload(certificates)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This searches block-preserving noncommuting Floquet-alpha signed twists.")
        print("The default is the representative pattern; use --all-patterns for symmetry checks.")
        print(f"candidate_count: {payload['candidate_count']}")
        print(f"noncommuting_candidates: {payload['noncommuting_candidates']}")
        print(f"block_preserving_candidates: {payload['block_preserving_candidates']}")
        print(
            "coarse_center_preserved_candidates: "
            f"{payload['coarse_center_preserved_candidates']}"
        )
        print(
            "compatible_j_zero_dimensional_candidates: "
            f"{payload['compatible_j_zero_dimensional_candidates']}"
        )
        print(f"forced_j_candidates: {payload['forced_j_candidates']}")
        print(f"strict_bridge_candidates: {payload['strict_bridge_candidates']}")
        print(
            "best_compatible_centralizer_dimension: "
            f"{payload['best_compatible_centralizer_dimension']}"
        )
        print(f"route_label_counts: {payload['route_label_counts']}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")
        for certificate in certificates:
            print(
                "candidate: "
                f"{certificate.candidate_name}, "
                f"commutes={str(certificate.u1_u2_commute).lower()}, "
                f"block_preserving="
                f"{str(certificate.u2_preserves_alpha_projector and certificate.u2_preserves_eta_projector).lower()}, "
                f"center_dim={certificate.center_dimension}, "
                f"centralizer_dim={certificate.compatible_centralizer_dimension}, "
                f"compatible_j_count={certificate.compatible_complex_structure_count}, "
                f"forced_j={str(certificate.forced_j_found).lower()}, "
                f"label={certificate.route_label}"
            )

    if args.check:
        representative_ok = all(
            not certificate.u1_u2_commute
            and certificate.u2_real_orthogonal
            and certificate.u2_preserves_alpha_projector
            and certificate.u2_preserves_eta_projector
            and certificate.complementary_rank_6_4_pairs == 1
            and certificate.lower_rank_central_idempotents == 0
            and certificate.compatible_j_solved
            and certificate.compatible_j_moduli_dimension == 0
            for certificate in certificates
        )
        if not representative_ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
