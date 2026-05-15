from __future__ import annotations

from clifford_3plus2_d5.qca.rule_verdict import _bloch_layer_laurent_orthogonal
from clifford_3plus2_d5.qca.two_site_bloch import (
    two_site_bloch_certificate,
    two_site_bloch_forward_inverse_layer,
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
