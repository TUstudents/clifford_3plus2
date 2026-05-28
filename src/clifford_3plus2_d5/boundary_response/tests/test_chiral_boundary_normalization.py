"""Tests for the V17 chiral boundary normalization no-go."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.chiral_boundary_normalization import (
    chiral_boundary_normalization_audit_payload,
    collective_negative_eigenvector,
    collective_positive_eigenvector,
    normalized_odd_collective_vector,
    orthogonal_chiral_swap,
    orthogonal_chiral_swap_forced_ratio,
    orthogonal_chiral_swap_phase_angle,
    primitive_ratio_from_collective_vector,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
    parity_preserving_generators,
    primitive_even_vector,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)
from clifford_3plus2_d5.boundary_response.quark_coin_rigidity import (
    isotropic_quark_phase_angle,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_normalized_odd_collective_has_unit_norm() -> None:
    odd_collective = normalized_odd_collective_vector()
    assert sp.simplify((odd_collective.T * odd_collective)[0, 0] - 1) == 0


def test_chiral_swap_is_orthogonal_involution() -> None:
    sigma = orthogonal_chiral_swap()
    identity = sp.eye(SHELL_DIMENSION)
    _assert_matrix_equal(sigma * sigma, identity)
    _assert_matrix_equal(sigma.T * sigma, identity)


def test_chiral_swap_exchanges_even_and_normalized_odd_collective() -> None:
    sigma = orthogonal_chiral_swap()
    even = primitive_even_vector()
    odd_collective = normalized_odd_collective_vector()
    _assert_matrix_equal(sigma * even, odd_collective)
    _assert_matrix_equal(sigma * odd_collective, even)


def test_chiral_swap_commutes_with_odd_s5_generators() -> None:
    sigma = orthogonal_chiral_swap()
    for generator in parity_preserving_generators():
        _assert_matrix_equal(generator * sigma, sigma * generator)


def test_chiral_swap_collective_eigenvectors_have_pm_one_ratios() -> None:
    sigma = orthogonal_chiral_swap()
    positive = collective_positive_eigenvector()
    negative = collective_negative_eigenvector()
    _assert_matrix_equal(sigma * positive, positive)
    _assert_matrix_equal(sigma * negative, -negative)
    assert sp.simplify(primitive_ratio_from_collective_vector(positive) - 1 / sp.sqrt(5)) == 0
    assert sp.simplify(primitive_ratio_from_collective_vector(negative) + 1 / sp.sqrt(5)) == 0


def test_orthogonal_chiral_swap_phase_is_pi_over_four_not_ckm_phase() -> None:
    ratio = orthogonal_chiral_swap_forced_ratio()
    assert sp.simplify(ratio - 1 / sp.sqrt(5)) == 0
    assert sp.simplify(orthogonal_chiral_swap_phase_angle() - sp.pi / 4) == 0
    assert sp.simplify(orthogonal_chiral_swap_phase_angle() - quark_boundary_phase_angle()) != 0
    assert sp.simplify(isotropic_quark_phase_angle(1) - quark_boundary_phase_angle()) == 0


def test_chiral_boundary_normalization_payload_reports_no_go_pass() -> None:
    payload = chiral_boundary_normalization_audit_payload()
    assert payload.final_verdict == "CHIRAL_BOUNDARY_NORMALIZATION_NO_GO_PASS"
    assert sp.simplify(payload.orthogonal_chiral_swap_forced_ratio - 1 / sp.sqrt(5)) == 0
    assert sp.simplify(payload.orthogonal_chiral_swap_phase - sp.pi / 4) == 0
    assert payload.ckm_flat_ratio == 1
    assert sp.simplify(payload.ckm_phase - quark_boundary_phase_angle()) == 0
    assert not payload.orthogonal_chiral_swap_derives_ckm_phase
