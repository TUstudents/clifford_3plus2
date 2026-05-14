from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.floquet_alpha import (
    FloquetAlphaSecondLayerCertificate,
    floquet_alpha_candidates,
    floquet_alpha_second_layer_certificate,
)


def _certificate_to_dict(certificate: FloquetAlphaSecondLayerCertificate) -> dict[str, object]:
    return {
        "candidate_name": certificate.candidate_name,
        "second_layer_name": certificate.second_layer_name,
        "u_v_commute": certificate.u_v_commute,
        "second_layer_real_orthogonal": certificate.second_layer_real_orthogonal,
        "alpha_cycle_order_certified": certificate.alpha_cycle_order_certified,
        "eta_swap_order_certified": certificate.eta_swap_order_certified,
        "generated_algebra_dimension": certificate.generated_algebra_dimension,
        "center_dimension": certificate.center_dimension,
        "compatible_centralizer_dimension": certificate.compatible_centralizer_dimension,
        "rule_center_solved": certificate.rule_center_solved,
        "rule_verdict": certificate.rule_verdict,
        "explicit_lower_rank_projector_ranks": list(
            certificate.explicit_lower_rank_projector_ranks
        ),
        "no_locking_guardrail_passed": certificate.no_locking_guardrail_passed,
        "compatible_centralizer_collapsed": certificate.compatible_centralizer_collapsed,
        "pass_strict_rule_to_bridge": certificate.pass_strict_rule_to_bridge,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the literal Floquet-alpha commuting cycle/swap second layer."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the cycle/swap layer passes the no-locking guardrail.",
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

    certificates = tuple(floquet_alpha_second_layer_certificate(candidate) for candidate in candidates)
    payload = {
        "family": "floquet_alpha_cycle_swap_second_layer",
        "candidate_count": len(certificates),
        "commuting_second_layer_candidates": sum(
            certificate.u_v_commute for certificate in certificates
        ),
        "order_certified_candidates": sum(
            certificate.alpha_cycle_order_certified and certificate.eta_swap_order_certified
            for certificate in certificates
        ),
        "compatible_centralizer_collapsed_candidates": sum(
            certificate.compatible_centralizer_collapsed for certificate in certificates
        ),
        "no_locking_guardrail_passed_candidates": sum(
            certificate.no_locking_guardrail_passed for certificate in certificates
        ),
        "strict_bridge_candidates": sum(
            certificate.pass_strict_rule_to_bridge for certificate in certificates
        ),
        "generated_algebra_dimension": certificates[0].generated_algebra_dimension,
        "center_dimension": certificates[0].center_dimension,
        "compatible_centralizer_dimension": certificates[0].compatible_centralizer_dimension,
        "explicit_lower_rank_projector_ranks": list(
            certificates[0].explicit_lower_rank_projector_ranks
        ),
        "load_bearing_qca_bridge": False,
        "results": [_certificate_to_dict(certificate) for certificate in certificates],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This checks the literal Floquet-alpha commuting cycle/swap second layer.")
        print(f"candidate_count: {payload['candidate_count']}")
        print(
            "commuting_second_layer_candidates: "
            f"{payload['commuting_second_layer_candidates']}"
        )
        print(f"order_certified_candidates: {payload['order_certified_candidates']}")
        print(
            "compatible_centralizer_collapsed_candidates: "
            f"{payload['compatible_centralizer_collapsed_candidates']}"
        )
        print(
            "no_locking_guardrail_passed_candidates: "
            f"{payload['no_locking_guardrail_passed_candidates']}"
        )
        print(f"strict_bridge_candidates: {payload['strict_bridge_candidates']}")
        print(f"generated_algebra_dimension: {payload['generated_algebra_dimension']}")
        print(f"center_dimension: {payload['center_dimension']}")
        print(
            "compatible_centralizer_dimension: "
            f"{payload['compatible_centralizer_dimension']}"
        )
        print(
            "explicit_lower_rank_projector_ranks: "
            f"{payload['explicit_lower_rank_projector_ranks']}"
        )
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")
        for certificate in certificates:
            print(
                "candidate: "
                f"{certificate.candidate_name}, "
                f"commutes={str(certificate.u_v_commute).lower()}, "
                f"orders={str(certificate.alpha_cycle_order_certified).lower()}/"
                f"{str(certificate.eta_swap_order_certified).lower()}, "
                f"center_dim={certificate.center_dimension}, "
                f"centralizer_dim={certificate.compatible_centralizer_dimension}, "
                f"lower_ranks={list(certificate.explicit_lower_rank_projector_ranks)}, "
                f"no_locking={str(certificate.no_locking_guardrail_passed).lower()}, "
                f"bridge={str(certificate.pass_strict_rule_to_bridge).lower()}"
            )

    if args.check and payload["no_locking_guardrail_passed_candidates"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
