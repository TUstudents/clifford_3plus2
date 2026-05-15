from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.spatial_1d import (
    Spatial1DUnseededCandidateCertificate,
    Spatial1DUnseededSearchSummary,
    spatial_1d_unseeded_search_summary,
)


def _candidate_to_dict(
    candidate: Spatial1DUnseededCandidateCertificate,
) -> dict[str, object]:
    return {
        "candidate_name": candidate.candidate_name,
        "layer_name": candidate.layer_name,
        "qca_shifts": list(candidate.qca_shifts),
        "qca_locality_radius": candidate.qca_locality_radius,
        "finite_radius": candidate.finite_radius,
        "coefficient_matrices_real": candidate.coefficient_matrices_real,
        "laurent_orthogonal": candidate.laurent_orthogonal,
        "seeded_coefficient_guardrail_passed": (
            candidate.seeded_coefficient_guardrail_passed
        ),
        "seeded_coefficient_witnesses": list(candidate.seeded_coefficient_witnesses),
        "coefficient_algebra_dimension": candidate.coefficient_algebra_dimension,
        "coefficient_center_dimension": candidate.coefficient_center_dimension,
        "central_idempotent_ranks": list(candidate.central_idempotent_ranks),
        "lower_rank_central_idempotents": candidate.lower_rank_central_idempotents,
        "coarse_6_4_center_pair": candidate.coarse_6_4_center_pair,
        "mode_windings": list(candidate.mode_windings),
        "computed_alpha_winding": candidate.computed_alpha_winding,
        "computed_eta_winding": candidate.computed_eta_winding,
        "computed_winding_gcd": candidate.computed_winding_gcd,
        "computed_winding_lcm": candidate.computed_winding_lcm,
        "orientation_choices_before_transport": (
            candidate.orientation_choices_before_transport
        ),
        "orientation_choices_after_transport": (
            candidate.orientation_choices_after_transport
        ),
        "sign_coupled_to_global_pm": candidate.sign_coupled_to_global_pm,
        "strict_bridge_candidate": candidate.strict_bridge_candidate,
        "route_label": candidate.route_label,
        "load_bearing_qca_bridge": candidate.load_bearing_qca_bridge,
    }


def _summary_to_dict(summary: Spatial1DUnseededSearchSummary) -> dict[str, object]:
    return {
        "family": "spatial_1d_unseeded",
        "candidate_count": summary.candidate_count,
        "unseeded_candidate_count": summary.unseeded_candidate_count,
        "seeded_guardrail_rejections": summary.seeded_guardrail_rejections,
        "laurent_orthogonal_candidates": summary.laurent_orthogonal_candidates,
        "unseeded_coarse_6_4_center_candidates": (
            summary.unseeded_coarse_6_4_center_candidates
        ),
        "unseeded_sign_coupled_candidates": summary.unseeded_sign_coupled_candidates,
        "unseeded_strict_bridge_candidates": summary.unseeded_strict_bridge_candidates,
        "lower_rank_center_rejections": summary.lower_rank_center_rejections,
        "route_label": summary.route_label,
        "load_bearing_qca_bridge": summary.load_bearing_qca_bridge,
        "candidates": [_candidate_to_dict(candidate) for candidate in summary.candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search conservative unseeded 1D spatial QCA candidates."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the unseeded spatial search diagnostics regress.",
    )
    args = parser.parse_args()

    summary = spatial_1d_unseeded_search_summary()
    payload = _summary_to_dict(summary)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This searches conservative unseeded 1D spatial QCA candidates.")
        print("It rejects coefficients that already contain the coarse projectors.")
        print(f"candidate_count: {summary.candidate_count}")
        print(f"unseeded_candidate_count: {summary.unseeded_candidate_count}")
        print(f"seeded_guardrail_rejections: {summary.seeded_guardrail_rejections}")
        print(f"laurent_orthogonal_candidates: {summary.laurent_orthogonal_candidates}")
        print(
            "unseeded_coarse_6_4_center_candidates: "
            f"{summary.unseeded_coarse_6_4_center_candidates}"
        )
        print(
            "unseeded_sign_coupled_candidates: "
            f"{summary.unseeded_sign_coupled_candidates}"
        )
        print(
            "unseeded_strict_bridge_candidates: "
            f"{summary.unseeded_strict_bridge_candidates}"
        )
        print(f"lower_rank_center_rejections: {summary.lower_rank_center_rejections}")
        print(f"route_label: {summary.route_label}")
        print(f"load_bearing_qca_bridge: {str(summary.load_bearing_qca_bridge).lower()}")
        for candidate in summary.candidates:
            witnesses = list(candidate.seeded_coefficient_witnesses)
            print(
                "candidate: "
                f"{candidate.candidate_name}, "
                f"seeded={str(not candidate.seeded_coefficient_guardrail_passed).lower()}, "
                f"center_ranks={list(candidate.central_idempotent_ranks)}, "
                f"lower_rank={candidate.lower_rank_central_idempotents}, "
                f"sign_coupled={str(candidate.sign_coupled_to_global_pm).lower()}, "
                f"route={candidate.route_label}, "
                f"witnesses={witnesses}"
            )

    if args.check:
        check_passed = (
            summary.candidate_count == 4
            and summary.unseeded_candidate_count == 3
            and summary.seeded_guardrail_rejections == 1
            and summary.laurent_orthogonal_candidates == 4
            and summary.unseeded_coarse_6_4_center_candidates == 0
            and summary.unseeded_sign_coupled_candidates == 0
            and summary.unseeded_strict_bridge_candidates == 0
            and summary.lower_rank_center_rejections == 1
            and summary.route_label == "unseeded_spatial_no_bridge_candidates"
            and not summary.load_bearing_qca_bridge
        )
        if not check_passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
