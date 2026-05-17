from __future__ import annotations

from clifford_3plus2_d5.obstruction_r10.qca.rule_verdict import _bloch_layer_laurent_orthogonal
from clifford_3plus2_d5.obstruction_r10.qca.two_site_bloch import (
    two_site_bloch_certificate,
    two_site_bloch_forward_inverse_layer,
    two_site_split_step_search_summary,
)


def test_two_site_forward_inverse_layer_is_laurent_orthogonal() -> None:
    layer = two_site_bloch_forward_inverse_layer()

    assert layer.name == "two_site_bloch_forward_inverse_c12340_s44433"
    assert layer.matrix.shape == (20, 20)
    assert tuple(term.shift for term in layer.bloch_terms) == (3, 4, -3, -4)
    assert _bloch_layer_laurent_orthogonal(layer, dimension=20)


def test_two_site_uniform_certificate_closes_trivial_center() -> None:
    certificate = two_site_bloch_certificate(
        variant="uniform",
        max_generated_algebra_dimension=8,
    )

    assert certificate.variant == "uniform"
    assert certificate.coefficient_matrices_real
    assert certificate.laurent_orthogonal
    assert certificate.seed_guardrail_passed
    assert certificate.generated_algebra_closed
    assert certificate.generated_algebra_dimension == 4
    assert certificate.center_solved
    assert certificate.central_idempotent_ranks == (0, 20)
    assert certificate.effective_rank_6_4_pairs == 0
    assert certificate.route_label == "two_site_trivial_center_no_effective_split"
    assert not certificate.load_bearing_qca_bridge


def test_two_site_split_step_panel_reports_bounded_cap() -> None:
    summary = two_site_split_step_search_summary(
        max_candidates=1,
        max_generated_algebra_dimension=2,
        max_coefficient_algebra_dimension=2,
    )

    assert summary.candidate_count == 1
    assert summary.laurent_orthogonal_candidates == 1
    assert summary.closed_candidates == 0
    assert summary.strict_bridge_candidates == 0
    assert summary.route_label == "split_step_cap_boundary"
    assert summary.candidates[0].candidate_name == "uniform_sublattice_swap"
