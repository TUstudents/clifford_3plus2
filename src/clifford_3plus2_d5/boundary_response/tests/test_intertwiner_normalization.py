"""Tests for the V18 algebraic intertwiner normalization gate."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.chiral_boundary_normalization import (
    even_projector,
    odd_collective_projector,
)
from clifford_3plus2_d5.boundary_response.intertwiner_normalization import (
    algebraic_intertwiner_audit_payload,
    even_odd_intertwiner,
    intertwiner_induced_phase,
    intertwiner_induced_ratio,
    intertwiner_is_s5_compatible,
    intertwiner_norm_square,
    s5_even_to_odd_intertwiner_basis,
    spectral_lift_has_full_casimir,
    spectral_lift_square,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
    primitive_odd_sum_vector,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_s5_even_to_odd_intertwiner_space_is_one_dimensional() -> None:
    basis = s5_even_to_odd_intertwiner_basis()
    assert len(basis) == 1
    _assert_matrix_equal(basis[0], primitive_odd_sum_vector())


def test_even_odd_intertwiner_maps_even_to_scaled_odd_sum() -> None:
    c = sp.Symbol("c", real=True)
    image = even_odd_intertwiner(c) * sp.Matrix([1, 0, 0, 0, 0, 0])
    _assert_matrix_equal(image, c * primitive_odd_sum_vector())


def test_even_odd_intertwiner_is_s5_compatible_for_free_scale() -> None:
    c = sp.Symbol("c", real=True)
    assert intertwiner_is_s5_compatible(c)
    assert intertwiner_is_s5_compatible(sp.Rational(1, 2))
    assert intertwiner_is_s5_compatible(2)


def test_intertwiner_norm_square_is_five_c_squared_p_even() -> None:
    c = sp.Symbol("c", real=True)
    _assert_matrix_equal(intertwiner_norm_square(c), 5 * c**2 * even_projector())


def test_spectral_lift_square_is_collective_not_full_casimir() -> None:
    c = sp.Symbol("c", real=True)
    expected = 5 * c**2 * (even_projector() + odd_collective_projector())
    _assert_matrix_equal(spectral_lift_square(c), expected)
    assert not spectral_lift_has_full_casimir(1)
    assert spectral_lift_square(1) != 5 * sp.eye(SHELL_DIMENSION)


def test_scale_controls_are_valid_but_change_phase() -> None:
    for ratio in (sp.Rational(1, 2), sp.Integer(2)):
        assert intertwiner_is_s5_compatible(ratio)
        assert sp.simplify(intertwiner_induced_phase(ratio) - quark_boundary_phase_angle()) != 0


def test_scale_one_recovers_ckm_phase_only_by_choice() -> None:
    assert intertwiner_induced_ratio(1) == 1
    assert sp.simplify(intertwiner_induced_phase(1) - quark_boundary_phase_angle()) == 0


def test_algebraic_intertwiner_payload_reports_free_norm_kill() -> None:
    payload = algebraic_intertwiner_audit_payload()
    assert payload.final_verdict == "ALGEBRAIC_INTERTWINER_FREE_NORM_KILL"
    assert payload.s5_intertwiner_dimension == 1
    _assert_matrix_equal(
        payload.norm_square,
        5 * sp.Symbol("c", real=True) ** 2 * even_projector(),
    )
    assert not payload.spectral_lift_full_casimir
    assert payload.scale_free_under_s5
    assert payload.ckm_scale == 1
    assert payload.ckm_phase_recovered_if_scale_one
    assert payload.unit_component_normalization_required
