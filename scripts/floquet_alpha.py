from __future__ import annotations

import argparse
import json

import sympy as sp

from clifford_3plus2_d5.qca.floquet_alpha import (
    FloquetAlphaPolarizationCertificate,
    FloquetAlphaSecondLayerCertificate,
    FloquetAlphaTimeReversalCertificate,
    floquet_alpha_candidates,
    floquet_alpha_polarization_certificate,
    floquet_alpha_rule_to_verdict,
    floquet_alpha_second_layer_certificate,
    floquet_alpha_time_reversal_certificate,
)
from clifford_3plus2_d5.qca.floquet_alpha_noncommuting import (
    FloquetAlphaNoncommutingCertificate,
    FloquetAlphaNoncommutingCompletionCertificate,
    FloquetAlphaNoncommutingJDiagnostic,
    FloquetAlphaNoncommutingJGapCertificate,
    floquet_alpha_noncommuting_candidates,
    floquet_alpha_noncommuting_certificate,
    floquet_alpha_noncommuting_completion_certificate,
    floquet_alpha_noncommuting_j_gap_certificate,
)
from clifford_3plus2_d5.qca.rule_verdict import result_to_dict


def _base_candidates(pattern_index: int | None):
    candidates = floquet_alpha_candidates()
    if pattern_index is None:
        return candidates
    selected = tuple(
        candidate for candidate in candidates if candidate.pattern_index == pattern_index
    )
    if not selected:
        raise SystemExit(f"unknown pattern index: {pattern_index}")
    return selected


def _matrix_to_rows(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [str(sp.simplify(matrix[row, column])) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def _polarization_to_dict(
    certificate: FloquetAlphaPolarizationCertificate,
) -> dict[str, object]:
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
        "generated_j_moduli_dimension": certificate.generated_j_moduli_dimension,
        "alpha_sector_centralizer_dimension": (
            certificate.alpha_sector_centralizer_dimension
        ),
        "eta_sector_centralizer_dimension": certificate.eta_sector_centralizer_dimension,
        "compatible_centralizer_dimension": certificate.compatible_centralizer_dimension,
        "compatible_j_moduli_dimension": certificate.compatible_j_moduli_dimension,
        "locality_radius_bound": certificate.locality_radius_bound,
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
        "strict_compatible_j_forced": certificate.strict_compatible_j_forced,
        "compatible_j_solved": certificate.compatible_j_solved,
        "alpha_plus_polarization_passed": certificate.alpha_plus_polarization_passed,
        "pass_strict_rule_to_bridge": certificate.pass_strict_rule_to_bridge,
        "verdict": certificate.verdict,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def _time_reversal_to_dict(
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


def _second_layer_to_dict(
    certificate: FloquetAlphaSecondLayerCertificate,
) -> dict[str, object]:
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


def _noncommuting_to_dict(
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


def _j_diagnostic_to_dict(
    diagnostic: FloquetAlphaNoncommutingJDiagnostic,
) -> dict[str, object]:
    return {
        "index": diagnostic.index,
        "expression": [[name, str(value)] for name, value in diagnostic.expression],
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


def _j_gap_to_dict(
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
            _j_diagnostic_to_dict(diagnostic)
            for diagnostic in certificate.compatible_j_diagnostics
        ],
    }


def _completion_to_dict(
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


def _run_base(args: argparse.Namespace) -> tuple[dict[str, object], list[str], bool]:
    results = tuple(
        floquet_alpha_rule_to_verdict(candidate)
        for candidate in _base_candidates(args.pattern_index)
    )
    payload = {
        "family": "floquet_alpha",
        "candidate_count": len(results),
        "bridge_candidates": sum(result.pass_rule_to_bridge for result in results),
        "rank_6_4_pair_candidates": sum(
            result.complementary_rank_6_4_pairs > 0 for result in results
        ),
        "rank_one_falsified_candidates": sum(
            bool(result.lower_rank_central_idempotents) for result in results
        ),
        "verdict_counts": {
            verdict: sum(result.verdict == verdict for result in results)
            for verdict in sorted({result.verdict for result in results})
        },
        "load_bearing_qca_bridge": False,
        "results": [result_to_dict(result) for result in results],
    }
    lines = [
        "This searches the Floquet-alpha physical primitive family.",
        "It exposes one mandatory quantized resonance layer per candidate.",
        f"candidate_count: {payload['candidate_count']}",
        f"rank_6_4_pair_candidates: {payload['rank_6_4_pair_candidates']}",
        f"rank_one_falsified_candidates: {payload['rank_one_falsified_candidates']}",
        f"bridge_candidates: {payload['bridge_candidates']}",
        f"verdict_counts: {payload['verdict_counts']}",
        f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}",
    ]
    lines.extend(
        "candidate: "
        f"{result.rule_name}, "
        f"center_ranks={[item.rank for item in result.central_idempotents]}, "
        f"pairs={result.complementary_rank_6_4_pairs}, "
        f"forced_j={str(result.forced_j_found).lower()}, "
        f"verdict={result.verdict}"
        for result in results
    )
    return payload, lines, payload["candidate_count"] != 0


def _run_plus(args: argparse.Namespace) -> tuple[dict[str, object], list[str], bool]:
    certificates = tuple(
        floquet_alpha_polarization_certificate(candidate)
        for candidate in _base_candidates(args.pattern_index)
    )
    payload = {
        "family": "floquet_alpha_plus",
        "candidate_count": len(certificates),
        "polarization_j_candidates": sum(
            certificate.alpha_plus_polarization_passed for certificate in certificates
        ),
        "scaled_polarization_certified_candidates": sum(
            certificate.scaled_polarization_certified for certificate in certificates
        ),
        "generated_j_moduli_dimension": certificates[0].generated_j_moduli_dimension,
        "compatible_centralizer_dimension": certificates[0].compatible_centralizer_dimension,
        "compatible_j_moduli_dimension": certificates[0].compatible_j_moduli_dimension,
        "locality_radius_bound": certificates[0].locality_radius_bound,
        "local_compatible_operator_dimension": (
            certificates[0].local_compatible_operator_dimension
        ),
        "local_compatible_j_moduli_dimension": (
            certificates[0].local_compatible_j_moduli_dimension
        ),
        "local_compatible_complex_structure_count": (
            certificates[0].local_compatible_complex_structure_count
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
        "results": [_polarization_to_dict(certificate) for certificate in certificates],
    }
    lines = [
        "This searches Floquet-alpha plus canonical polarization J extraction.",
        "It reports the strict compatible-J obstruction separately.",
        f"candidate_count: {payload['candidate_count']}",
        f"polarization_j_candidates: {payload['polarization_j_candidates']}",
        "scaled_polarization_certified_candidates: "
        f"{payload['scaled_polarization_certified_candidates']}",
        f"generated_j_moduli_dimension: {payload['generated_j_moduli_dimension']}",
        f"compatible_centralizer_dimension: {payload['compatible_centralizer_dimension']}",
        f"compatible_j_moduli_dimension: {payload['compatible_j_moduli_dimension']}",
        f"locality_radius_bound: {payload['locality_radius_bound']}",
        "local_compatible_operator_dimension: "
        f"{payload['local_compatible_operator_dimension']}",
        "local_compatible_j_moduli_dimension: "
        f"{payload['local_compatible_j_moduli_dimension']}",
        "local_compatible_complex_structure_count: "
        f"{payload['local_compatible_complex_structure_count']}",
        "strict_compatible_j_forced_candidates: "
        f"{payload['strict_compatible_j_forced_candidates']}",
        f"strict_bridge_candidates: {payload['strict_bridge_candidates']}",
        f"verdict_counts: {payload['verdict_counts']}",
        f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}",
    ]
    lines.extend(
        "candidate: "
        f"{certificate.candidate_name}, "
        f"alpha_eta_ranks=({certificate.alpha_projector_rank},"
        f"{certificate.eta_projector_rank}), "
        f"scaled_cert={str(certificate.scaled_polarization_certified).lower()}, "
        f"center_ranks={list(certificate.central_idempotent_ranks)}, "
        f"centralizer_dim={certificate.compatible_centralizer_dimension}, "
        f"j_moduli_dim={certificate.compatible_j_moduli_dimension}, "
        f"local_j_count={certificate.local_compatible_complex_structure_count}, "
        f"polarization_j={str(certificate.alpha_plus_polarization_passed).lower()}, "
        f"strict_forced_j={str(certificate.strict_compatible_j_forced).lower()}, "
        f"verdict={certificate.verdict}"
        for certificate in certificates
    )
    return payload, lines, payload["polarization_j_candidates"] == payload["candidate_count"]


def _run_time_reversal(args: argparse.Namespace) -> tuple[dict[str, object], list[str], bool]:
    certificates = tuple(
        floquet_alpha_time_reversal_certificate(candidate)
        for candidate in _base_candidates(args.pattern_index)
    )
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
        "results": [_time_reversal_to_dict(certificate) for certificate in certificates],
    }
    lines = [
        "This checks Floquet-alpha with a declared time-reversal involution.",
        "K is diagnostic only unless generated by microscopic rule data.",
        f"candidate_count: {payload['candidate_count']}",
        f"k_real_orthogonal_candidates: {payload['k_real_orthogonal_candidates']}",
        f"k_involution_candidates: {payload['k_involution_candidates']}",
        "k_conjugates_floquet_to_inverse_candidates: "
        f"{payload['k_conjugates_floquet_to_inverse_candidates']}",
        "k_anticommutes_with_canonical_j_candidates: "
        f"{payload['k_anticommutes_with_canonical_j_candidates']}",
        f"k_in_generated_algebra_candidates: {payload['k_in_generated_algebra_candidates']}",
        "compatible_j_moduli_dimension_before_k: "
        f"{payload['compatible_j_moduli_dimension_before_k']}",
        "k_fixed_compatible_j_moduli_dimension: "
        f"{payload['k_fixed_compatible_j_moduli_dimension']}",
        "k_fixed_local_compatible_complex_structure_count: "
        f"{payload['k_fixed_local_compatible_complex_structure_count']}",
        "k_fixed_local_matches_canonical_orbit_count: "
        f"{payload['k_fixed_local_matches_canonical_orbit_count']}",
        f"k_reduces_full_moduli_candidates: {payload['k_reduces_full_moduli_candidates']}",
        f"k_reduces_to_global_pm_candidates: {payload['k_reduces_to_global_pm_candidates']}",
        f"strict_bridge_candidates: {payload['strict_bridge_candidates']}",
        f"verdict_counts: {payload['verdict_counts']}",
        f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}",
    ]
    lines.extend(
        "candidate: "
        f"{certificate.candidate_name}, "
        f"k_origin={certificate.time_reversal_origin}, "
        f"k_in_generated={str(certificate.k_in_generated_algebra).lower()}, "
        f"j_moduli_before={certificate.compatible_j_moduli_dimension_before_k}, "
        f"k_fixed_moduli={certificate.k_fixed_compatible_j_moduli_dimension}, "
        f"k_fixed_local_j_count={certificate.k_fixed_local_compatible_complex_structure_count}, "
        f"k_global_pm={str(certificate.k_reduces_to_global_pm).lower()}, "
        f"verdict={certificate.verdict}"
        for certificate in certificates
    )
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
        and payload["verdict_counts"] == {"declared_time_reversal_reduces_moduli_not_unique": 10}
        and not payload["load_bearing_qca_bridge"]
    )
    return payload, lines, check_passed


def _run_second_layer(args: argparse.Namespace) -> tuple[dict[str, object], list[str], bool]:
    certificates = tuple(
        floquet_alpha_second_layer_certificate(candidate)
        for candidate in _base_candidates(args.pattern_index)
    )
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
        "results": [_second_layer_to_dict(certificate) for certificate in certificates],
    }
    lines = [
        "This checks the literal Floquet-alpha commuting cycle/swap second layer.",
        f"candidate_count: {payload['candidate_count']}",
        f"commuting_second_layer_candidates: {payload['commuting_second_layer_candidates']}",
        f"order_certified_candidates: {payload['order_certified_candidates']}",
        "compatible_centralizer_collapsed_candidates: "
        f"{payload['compatible_centralizer_collapsed_candidates']}",
        "no_locking_guardrail_passed_candidates: "
        f"{payload['no_locking_guardrail_passed_candidates']}",
        f"strict_bridge_candidates: {payload['strict_bridge_candidates']}",
        f"generated_algebra_dimension: {payload['generated_algebra_dimension']}",
        f"center_dimension: {payload['center_dimension']}",
        f"compatible_centralizer_dimension: {payload['compatible_centralizer_dimension']}",
        "explicit_lower_rank_projector_ranks: "
        f"{payload['explicit_lower_rank_projector_ranks']}",
        f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}",
    ]
    lines.extend(
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
        for certificate in certificates
    )
    return payload, lines, payload["no_locking_guardrail_passed_candidates"] == 0


def _noncommuting_candidates(args: argparse.Namespace):
    pattern_index = None if args.all_patterns else args.pattern_index
    candidates = floquet_alpha_noncommuting_candidates(pattern_index=pattern_index)
    if not candidates:
        raise SystemExit(f"unknown pattern index: {args.pattern_index}")
    return candidates


def _run_noncommuting(args: argparse.Namespace) -> tuple[dict[str, object], list[str], bool]:
    certificates = tuple(
        floquet_alpha_noncommuting_certificate(candidate)
        for candidate in _noncommuting_candidates(args)
    )
    compatible_dimensions = [
        certificate.compatible_centralizer_dimension for certificate in certificates
    ]
    payload = {
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
        "results": [_noncommuting_to_dict(certificate) for certificate in certificates],
    }
    lines = [
        "This searches block-preserving noncommuting Floquet-alpha signed twists.",
        "The default is the representative pattern; use --all-patterns for symmetry checks.",
        f"candidate_count: {payload['candidate_count']}",
        f"noncommuting_candidates: {payload['noncommuting_candidates']}",
        f"block_preserving_candidates: {payload['block_preserving_candidates']}",
        "coarse_center_preserved_candidates: "
        f"{payload['coarse_center_preserved_candidates']}",
        "compatible_j_zero_dimensional_candidates: "
        f"{payload['compatible_j_zero_dimensional_candidates']}",
        f"forced_j_candidates: {payload['forced_j_candidates']}",
        f"strict_bridge_candidates: {payload['strict_bridge_candidates']}",
        "best_compatible_centralizer_dimension: "
        f"{payload['best_compatible_centralizer_dimension']}",
        f"route_label_counts: {payload['route_label_counts']}",
        f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}",
    ]
    lines.extend(
        "candidate: "
        f"{certificate.candidate_name}, "
        f"commutes={str(certificate.u1_u2_commute).lower()}, "
        "block_preserving="
        f"{str(certificate.u2_preserves_alpha_projector and certificate.u2_preserves_eta_projector).lower()}, "
        f"center_dim={certificate.center_dimension}, "
        f"centralizer_dim={certificate.compatible_centralizer_dimension}, "
        f"compatible_j_count={certificate.compatible_complex_structure_count}, "
        f"forced_j={str(certificate.forced_j_found).lower()}, "
        f"label={certificate.route_label}"
        for certificate in certificates
    )
    check_passed = all(
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
    return payload, lines, check_passed


def _run_noncommuting_gap(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[str], bool]:
    candidates = floquet_alpha_noncommuting_candidates(pattern_index=args.pattern_index)
    if not candidates:
        raise SystemExit(f"unknown pattern index: {args.pattern_index}")
    certificate = floquet_alpha_noncommuting_j_gap_certificate(candidates[0])
    payload = _j_gap_to_dict(certificate)
    lines = [
        "This extracts the finite compatible J gap for the noncommuting alpha route.",
        f"candidate_name: {certificate.candidate_name}",
        f"compatible_j_count: {certificate.compatible_j_count}",
        f"generated_algebra_dimension: {certificate.generated_algebra_dimension}",
        f"center_dimension: {certificate.center_dimension}",
        f"compatible_centralizer_dimension: {certificate.compatible_centralizer_dimension}",
        f"compatible_j_solved: {str(certificate.compatible_j_solved).lower()}",
        f"compatible_j_moduli_dimension: {certificate.compatible_j_moduli_dimension}",
        f"generated_j_solved: {str(certificate.generated_j_solved).lower()}",
        f"generated_complex_structure_count: {certificate.generated_complex_structure_count}",
        "local_compatible_complex_structure_count: "
        f"{certificate.local_compatible_complex_structure_count}",
        "compatible_j_in_generated_algebra_count: "
        f"{certificate.compatible_j_in_generated_algebra_count}",
        "compatible_j_in_rule_local_center_count: "
        f"{certificate.compatible_j_in_rule_local_center_count}",
        "spectral_polarization_j_matched_count: "
        f"{certificate.spectral_polarization_j_matched_count}",
        f"forced_j_found: {str(certificate.forced_j_found).lower()}",
        f"reason_for_forced_j_failure: {certificate.reason_for_forced_j_failure}",
        f"load_bearing_qca_bridge: {str(certificate.load_bearing_qca_bridge).lower()}",
    ]
    lines.extend(
        "compatible_j: "
        f"index={diagnostic.index}, "
        f"pair_signs={diagnostic.pair_orientation_signs}, "
        f"in_generated={str(diagnostic.in_generated_algebra).lower()}, "
        f"in_local_center={str(diagnostic.in_rule_local_center).lower()}, "
        "matches_spectral_j="
        f"{str(diagnostic.equals_spectral_polarization_j or diagnostic.equals_negative_spectral_polarization_j).lower()}"
        for diagnostic in certificate.compatible_j_diagnostics
    )
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
    return payload, lines, check_passed


def _run_noncommuting_completion(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[str], bool]:
    candidates = floquet_alpha_noncommuting_candidates(pattern_index=args.pattern_index)
    if not candidates:
        raise SystemExit(f"unknown pattern index: {args.pattern_index}")
    certificate = floquet_alpha_noncommuting_completion_certificate(
        candidates[0],
        j_index=args.j_index,
    )
    payload = _completion_to_dict(certificate)
    lines = [
        "This declares one finite compatible J as a diagnostic third layer W.",
        f"candidate_name: {certificate.candidate_name}",
        f"completed_j_index: {certificate.completed_j_index}",
        "completed_j_pair_orientation_signs: "
        f"{certificate.completed_j_pair_orientation_signs}",
        "w_in_previous_generated_algebra: "
        f"{str(certificate.w_in_previous_generated_algebra).lower()}",
        "w_in_completed_generated_algebra: "
        f"{str(certificate.w_in_completed_generated_algebra).lower()}",
        f"w_in_completed_center: {str(certificate.w_in_completed_center).lower()}",
        f"w_commutes_with_u1: {str(certificate.w_commutes_with_u1).lower()}",
        f"w_commutes_with_u2: {str(certificate.w_commutes_with_u2).lower()}",
        f"w_squares_to_minus_identity: {str(certificate.w_squares_to_minus_identity).lower()}",
        f"w_orthogonal: {str(certificate.w_orthogonal).lower()}",
        f"generated_algebra_dimension: {certificate.generated_algebra_dimension}",
        f"center_dimension: {certificate.center_dimension}",
        f"center_solved: {str(certificate.center_solved).lower()}",
        f"central_idempotent_ranks: {list(certificate.central_idempotent_ranks)}",
        f"complementary_rank_6_4_pairs: {certificate.complementary_rank_6_4_pairs}",
        f"lower_rank_central_idempotents: {certificate.lower_rank_central_idempotents}",
        f"compatible_centralizer_dimension: {certificate.compatible_centralizer_dimension}",
        f"compatible_j_moduli_dimension: {certificate.compatible_j_moduli_dimension}",
        f"compatible_complex_structure_count: {certificate.compatible_complex_structure_count}",
        f"local_compatible_operator_dimension: {certificate.local_compatible_operator_dimension}",
        "local_compatible_j_moduli_dimension: "
        f"{certificate.local_compatible_j_moduli_dimension}",
        "local_compatible_complex_structure_count: "
        f"{certificate.local_compatible_complex_structure_count}",
        "declared_w_is_local_compatible_j: "
        f"{str(certificate.declared_w_is_local_compatible_j).lower()}",
        f"strict_unique_j_found: {str(certificate.strict_unique_j_found).lower()}",
        f"pass_completion_to_bridge: {str(certificate.pass_completion_to_bridge).lower()}",
        f"completion_label: {certificate.completion_label}",
        f"load_bearing_qca_bridge: {str(certificate.load_bearing_qca_bridge).lower()}",
    ]
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
    return payload, lines, check_passed


VARIANTS = {
    "base": _run_base,
    "plus": _run_plus,
    "second-layer": _run_second_layer,
    "noncommuting": _run_noncommuting,
    "noncommuting-gap": _run_noncommuting_gap,
    "noncommuting-completion": _run_noncommuting_completion,
    "time-reversal": _run_time_reversal,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Floquet-alpha route diagnostics from one canonical CLI."
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="base",
        help="Diagnostic variant to run. Defaults to the base Floquet-alpha search.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero on regression.")
    parser.add_argument("--pattern-index", type=int, default=None, help="Run one pattern.")
    parser.add_argument(
        "--all-patterns",
        action="store_true",
        help="Run every symmetric pattern for the noncommuting variant.",
    )
    parser.add_argument(
        "--j-index",
        type=int,
        default=0,
        help="Compatible J index for the noncommuting-completion variant.",
    )
    args = parser.parse_args()

    if args.variant in {"noncommuting", "noncommuting-gap", "noncommuting-completion"}:
        if args.pattern_index is None:
            args.pattern_index = 0
    elif args.all_patterns:
        raise SystemExit("--all-patterns is only valid with --variant noncommuting")

    payload, lines, check_passed = VARIANTS[args.variant](args)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"variant: {args.variant}")
        for line in lines:
            print(line)

    if args.check and not check_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
