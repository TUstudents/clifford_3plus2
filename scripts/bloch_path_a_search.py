from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.bloch_rule import (
    BlochPathASearchSummary,
    BlochRuleVerdict,
    bloch_path_a_search_summary,
)


def _verdict_to_dict(verdict: BlochRuleVerdict) -> dict[str, object]:
    return {
        "rule_name": verdict.rule_name,
        "period": verdict.period,
        "layer_count": verdict.layer_count,
        "qca_locality_radius": verdict.qca_locality_radius,
        "all_layers_finite_radius": verdict.all_layers_finite_radius,
        "all_layers_laurent_orthogonal": verdict.all_layers_laurent_orthogonal,
        "all_symbols_unitary_on_samples": verdict.all_symbols_unitary_on_samples,
        "coefficient_matrices_real": verdict.coefficient_matrices_real,
        "coefficient_algebra_dimension": verdict.coefficient_algebra_dimension,
        "seed_guardrail_passed": verdict.seed_guardrail_passed,
        "raw_seed_witnesses": list(verdict.raw_seed_witnesses),
        "algebraic_seed_witnesses": list(verdict.algebraic_seed_witnesses),
        "sample_count": verdict.sample_count,
        "algebra_dimensions_by_sample": list(verdict.algebra_dimensions_by_sample),
        "center_dimensions_by_sample": list(verdict.center_dimensions_by_sample),
        "central_idempotent_rank_profiles_by_sample": [
            list(profile) for profile in verdict.central_idempotent_rank_profiles_by_sample
        ],
        "coarse_6_4_band_split_at_all_samples": (
            verdict.coarse_6_4_band_split_at_all_samples
        ),
        "lower_rank_center_at_any_sample": verdict.lower_rank_center_at_any_sample,
        "compatible_j_section_count": verdict.compatible_j_section_count,
        "transported_j_section_count": verdict.transported_j_section_count,
        "rule_generated_transported_j_section_count": (
            verdict.rule_generated_transported_j_section_count
        ),
        "alpha_winding": verdict.alpha_winding,
        "eta_winding": verdict.eta_winding,
        "winding_gcd": verdict.winding_gcd,
        "winding_lcm": verdict.winding_lcm,
        "winding_proxy_4_3": verdict.winding_proxy_4_3,
        "global_transport_pm_candidate": verdict.global_transport_pm_candidate,
        "strict_bridge_candidate": verdict.strict_bridge_candidate,
        "route_label": verdict.route_label,
        "load_bearing_qca_bridge": verdict.load_bearing_qca_bridge,
    }


def _summary_to_dict(summary: BlochPathASearchSummary) -> dict[str, object]:
    return {
        "family": "bloch_path_a",
        "candidate_count": summary.candidate_count,
        "seed_guardrail_rejections": summary.seed_guardrail_rejections,
        "unseeded_candidate_count": summary.unseeded_candidate_count,
        "stable_6_4_band_candidates": summary.stable_6_4_band_candidates,
        "topological_pm_candidates": summary.topological_pm_candidates,
        "rule_generated_j_section_candidates": (
            summary.rule_generated_j_section_candidates
        ),
        "strict_bridge_candidates": summary.strict_bridge_candidates,
        "route_label": summary.route_label,
        "load_bearing_qca_bridge": summary.load_bearing_qca_bridge,
        "candidates": [_verdict_to_dict(verdict) for verdict in summary.candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sampled Bloch Path-A spatial QCA diagnostics."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the Path-A diagnostics regress.",
    )
    args = parser.parse_args()

    summary = bloch_path_a_search_summary()
    payload = _summary_to_dict(summary)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This checks sampled Bloch Path-A spatial QCA candidates.")
        print("It rejects seeded P_alpha/P_eta coefficient algebra explicitly.")
        print(f"candidate_count: {summary.candidate_count}")
        print(f"seed_guardrail_rejections: {summary.seed_guardrail_rejections}")
        print(f"unseeded_candidate_count: {summary.unseeded_candidate_count}")
        print(f"stable_6_4_band_candidates: {summary.stable_6_4_band_candidates}")
        print(f"topological_pm_candidates: {summary.topological_pm_candidates}")
        print(
            "rule_generated_j_section_candidates: "
            f"{summary.rule_generated_j_section_candidates}"
        )
        print(f"strict_bridge_candidates: {summary.strict_bridge_candidates}")
        print(f"route_label: {summary.route_label}")
        print(f"load_bearing_qca_bridge: {str(summary.load_bearing_qca_bridge).lower()}")
        for verdict in summary.candidates:
            print(
                "candidate: "
                f"{verdict.rule_name}, "
                f"seed_passed={str(verdict.seed_guardrail_passed).lower()}, "
                f"stable_6_4={str(verdict.coarse_6_4_band_split_at_all_samples).lower()}, "
                f"pm={str(verdict.global_transport_pm_candidate).lower()}, "
                f"j_generated={verdict.rule_generated_transported_j_section_count}, "
                f"verdict={verdict.route_label}"
            )

    if args.check:
        check_passed = (
            summary.candidate_count == 6
            and summary.seed_guardrail_rejections >= 3
            and summary.unseeded_candidate_count >= 1
            and summary.topological_pm_candidates >= 1
            and summary.strict_bridge_candidates == 0
            and summary.route_label == "bloch_path_a_seeded_shape_only"
            and not summary.load_bearing_qca_bridge
        )
        if not check_passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
