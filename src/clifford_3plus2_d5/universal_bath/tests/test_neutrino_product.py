"""Tests for the Session 03 neutrino product bath."""

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import residual_projectors
from clifford_3plus2_d5.universal_bath.neutrino_product import (
    NEUTRINO_SOURCE_LABELS,
    alternate_tail_control_response,
    cross_moments,
    diagonal_moment_differences,
    finite_product_hamiltonian,
    frozen_neutrino_sources,
    neutrino_product_bath_payload,
    neutrino_response_diagonal,
    neutrino_tail_response,
    neutrino_target_response,
    product_return_matrix,
    rank_one_control_has_cross_return,
    wrong_source_control_response,
)
from clifford_3plus2_d5.universal_bath.source_dictionary import SourceStatus
from clifford_3plus2_d5.universal_bath.tail import (
    period_one_tail,
    silver_epsilon,
    silver_selected_z,
)


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def test_frozen_neutrino_sources_are_u_and_b_with_depths() -> None:
    collective, edge = frozen_neutrino_sources()

    assert (collective.label, edge.label) == NEUTRINO_SOURCE_LABELS
    assert collective.status == SourceStatus.FROZEN
    assert edge.status == SourceStatus.FROZEN
    assert collective.normal_depth == 1
    assert edge.normal_depth == 0
    assert collective.residual_components == {"a": 0, "u": 1, "b": 0}
    assert edge.residual_components == {"a": 0, "u": 0, "b": 1}


def test_product_hamiltonian_is_chain_tensor_family_identity() -> None:
    h_q = finite_product_hamiltonian(4)

    assert h_q.shape == (12, 12)
    assert _matrix_equal(h_q, h_q.T)


def test_product_bath_cross_return_moments_vanish() -> None:
    assert cross_moments(6, (0, 1, 2, 3, 4)) == (0, 0, 0, 0, 0)


def test_product_bath_diagonal_return_moments_are_equal() -> None:
    assert diagonal_moment_differences(6, (0, 1, 2, 3, 4)) == (0, 0, 0, 0, 0)


def test_product_return_matrix_is_scalar_on_neutrino_family_pair() -> None:
    for power in range(5):
        returns = product_return_matrix(6, power)
        assert returns[0, 1] == 0
        assert returns[1, 0] == 0
        assert sp.simplify(returns[0, 0] - returns[1, 1]) == 0


def test_neutrino_response_has_exact_tail_eigenstructure() -> None:
    z = sp.Symbol("z")
    tail = period_one_tail(z)
    diagonal = neutrino_response_diagonal(z)

    assert sp.simplify(diagonal[0, 0]) == 0
    assert sp.simplify(diagonal[1, 1] - tail**2) == 0
    assert sp.simplify(diagonal[2, 2] - 1) == 0
    assert all(diagonal[row, col] == 0 for row in range(3) for col in range(3) if row != col)


def test_neutrino_response_matches_target_at_silver_probe() -> None:
    z = silver_selected_z()
    projectors = residual_projectors()
    expected = sp.simplify(silver_epsilon() ** 2 * projectors["u"] + projectors["b"])

    assert _matrix_equal(neutrino_tail_response(z), expected)
    assert _matrix_equal(neutrino_target_response(), expected)


def test_negative_controls_are_rejected() -> None:
    target = neutrino_target_response()

    assert rank_one_control_has_cross_return()
    assert not _matrix_equal(wrong_source_control_response(), target)
    assert not _matrix_equal(alternate_tail_control_response(), target)


def test_neutrino_product_bath_payload_reports_core_pass() -> None:
    payload = neutrino_product_bath_payload()

    assert payload.final_verdict == "NEUTRINO_PRODUCT_BATH_CORE_PASS"
    assert payload.source_dictionary_pass
    assert payload.frozen_neutrino_sources == NEUTRINO_SOURCE_LABELS
    assert payload.checked_moment_powers == (0, 1, 2, 3, 4)
    assert payload.diagonal_moments_equal
    assert payload.cross_moments_zero
    assert payload.fixed_point_residual == 0
    assert payload.tail_value == silver_epsilon()
    assert payload.tail_value_matches_epsilon
    assert payload.response_matches_target
    assert sp.simplify(payload.mass_ratio - silver_epsilon() ** 2) == 0
    assert sp.simplify(payload.mass_squared_ratio - silver_epsilon() ** 4) == 0
    assert payload.rank_one_control_has_cross_return
    assert payload.wrong_source_control_rejected
    assert payload.alternate_tail_control_rejected
    assert payload.pmns_ckm_parked
