from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import split_projectors_3_2
from clifford_3plus2_d5.qca.bloch_rule import (
    BlochRuleLayerInput,
    bloch_layer_laurent_orthogonal,
    bloch_path_a_polynomial_hop_layer,
    bloch_path_a_projector_free_combined_layer,
    bloch_path_a_projector_free_rule_to_verdict,
    bloch_path_a_search_summary,
    bloch_symbol_at_root,
)
from clifford_3plus2_d5.qca.spatial_1d import SpatialHoppingTerm, root_of_unity


ROOT = Path(__file__).resolve().parents[1]


def test_bloch_symbol_evaluates_laurent_terms_at_roots() -> None:
    layer = BlochRuleLayerInput(
        name="identity_shift",
        period=12,
        dimension=10,
        terms=(SpatialHoppingTerm(shift=4, matrix=identity(10)),),
    )

    assert bloch_symbol_at_root(layer, 1) == root_of_unity(12, 1) ** 4 * identity(10)


def test_bloch_path_a_summary_reports_seeded_shape_only() -> None:
    summary = bloch_path_a_search_summary()

    assert summary.candidate_count == 6
    assert summary.seed_guardrail_rejections == 4
    assert summary.unseeded_candidate_count == 2
    assert summary.stable_6_4_band_candidates == 0
    assert summary.topological_pm_candidates == 4
    assert summary.rule_generated_j_section_candidates == 0
    assert summary.strict_bridge_candidates == 0
    assert summary.route_label == "bloch_path_a_seeded_shape_only"
    assert not summary.load_bearing_qca_bridge

    by_name = {candidate.rule_name: candidate for candidate in summary.candidates}
    combined = by_name["path_a_combined_route1_route2"]
    assert not combined.seed_guardrail_passed
    assert combined.algebraic_seed_witnesses == (
        "coefficient_algebra:P_alpha",
        "coefficient_algebra:P_eta",
    )
    assert combined.global_transport_pm_candidate
    assert combined.rule_generated_transported_j_section_count == 0
    assert combined.route_label == "bloch_path_a_seeded_guardrail_rejected"

    identity_shift = by_name["path_a_unseeded_uniform_identity_shift"]
    assert identity_shift.seed_guardrail_passed
    assert not identity_shift.coarse_6_4_band_split_at_all_samples
    assert identity_shift.route_label == "bloch_path_a_no_stable_6_4_band_split"


def test_projector_free_combined_layer_has_no_raw_projector_coefficients() -> None:
    p_alpha, p_eta = split_projectors_3_2()
    layer = bloch_path_a_projector_free_combined_layer()

    assert layer.name == "path_a_projector_free_cycle_combined"
    assert tuple(term.shift for term in layer.bloch_terms) == (3, 4)
    assert layer.locality_radius == 4
    assert all(term.matrix not in (p_alpha, p_eta) for term in layer.bloch_terms)
    assert sorted(term.matrix.rank() for term in layer.bloch_terms) == [4, 6]


def test_polynomial_hop_layer_remains_laurent_orthogonal() -> None:
    layer = bloch_path_a_polynomial_hop_layer(
        terms_by_shift=(
            (3, ((3, 4), (4, 0))),
            (4, ((0, 1), (1, 2), (2, 3))),
        ),
        mixes_by_shift=((3, (0, 1)),),
        name_suffix="test",
    )
    bloch_layer = BlochRuleLayerInput(
        name=layer.name,
        period=12,
        dimension=10,
        terms=tuple(
            SpatialHoppingTerm(shift=term.shift, matrix=term.matrix) for term in layer.bloch_terms
        ),
    )

    assert layer.name == "path_a_polynomial_hop_p0_test"
    assert tuple(term.shift for term in layer.bloch_terms) == (3, 4)
    assert layer.locality_radius == 4
    assert sorted(term.matrix.rank() for term in layer.bloch_terms) == [4, 6]
    assert bloch_layer_laurent_orthogonal(bloch_layer)


def test_projector_free_rule_to_verdict_is_bounded_not_solved() -> None:
    verdict = bloch_path_a_projector_free_rule_to_verdict(
        max_generated_algebra_dimension=16,
        max_center_solve_dimension=4,
        max_j_solve_dimension=4,
    )

    assert verdict.bloch_period == 12
    assert verdict.bloch_sample_count == 12
    assert verdict.all_layers_real_orthogonal
    assert verdict.all_layers_local
    assert verdict.generated_algebra_dimension == 16
    assert not verdict.generated_algebra_closed
    assert not verdict.center_solved
    assert verdict.verdict == "not_solved"
    assert not verdict.pass_rule_to_bridge


def test_bloch_path_a_cli_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bloch_path_a_search.py",
            "--json",
            "--check",
            "--projector-free-max-algebra-dim",
            "16",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["family"] == "bloch_path_a"
    assert payload["candidate_count"] == 6
    assert payload["seed_guardrail_rejections"] == 4
    assert payload["unseeded_candidate_count"] == 2
    assert payload["topological_pm_candidates"] == 4
    assert payload["strict_bridge_candidates"] == 0
    assert payload["candidate_panel_route_label"] == "bloch_path_a_seeded_shape_only"
    assert payload["route_label"] == "bloch_path_a_projector_free_cap_boundary"
    assert payload["load_bearing_qca_bridge"] is False
    projector_free = payload["projector_free_rule_to_verdict"]
    assert projector_free["bloch_period"] == 12
    assert projector_free["bloch_sample_count"] == 12
    assert projector_free["generated_algebra_closed"] is False
    assert projector_free["verdict"] == "not_solved"


def test_bloch_path_a_stepwise_cli_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bloch_path_a_stepwise.py",
            "--json",
            "--check",
            "--max-candidates",
            "1",
            "--max-algebra-dim",
            "16",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["candidate_count"] == 1
    assert payload["closed_count"] == 0
    [candidate] = payload["results"]
    assert candidate["name"] == "path_a_projector_free_cycle_combined"
    assert candidate["generated_algebra_dimension"] == 16
    assert candidate["generated_algebra_closed"] is False
    assert candidate["route_label"] == "cap_exceeded_structured"
