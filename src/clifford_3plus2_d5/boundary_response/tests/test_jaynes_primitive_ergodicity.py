"""Tests for the V20 Jaynes primitive-ergodicity theorem."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.jaynes_primitive_ergodicity import (
    compressed_macro_entropy_alpha,
    compressed_macro_entropy_alpha_derivative,
    compressed_macro_entropy_alpha_max,
    density_commutes_with_odd_s5,
    jaynes_density_trace,
    jaynes_entropy_derivative,
    jaynes_entropy_max_alpha,
    jaynes_entropy_second_derivative,
    jaynes_primitive_density,
    jaynes_primitive_entropy,
    jaynes_primitive_ergodicity_audit_payload,
    phase_from_alpha,
    primitive_ratio_from_alpha,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_jaynes_density_is_trace_one_and_s5_invariant() -> None:
    alpha = sp.Symbol("alpha", positive=True)
    density = jaynes_primitive_density(alpha)
    assert sp.simplify(jaynes_density_trace(alpha) - 1) == 0
    assert density_commutes_with_odd_s5(alpha)
    assert density.shape == (SHELL_DIMENSION, SHELL_DIMENSION)


def test_jaynes_density_positive_for_probability_controls() -> None:
    for alpha in (sp.Rational(1, 6), sp.Rational(1, 3), sp.Rational(1, 2)):
        diagonal_entries = [jaynes_primitive_density(alpha)[index, index] for index in range(SHELL_DIMENSION)]
        assert all(entry > 0 for entry in diagonal_entries)
        assert sp.simplify(sum(diagonal_entries) - 1) == 0


def test_jaynes_entropy_derivatives_match_closed_forms() -> None:
    alpha = sp.Symbol("alpha", positive=True)
    expected_derivative = sp.log((1 - alpha) / (5 * alpha))
    expected_second = -1 / (alpha * (1 - alpha))
    assert sp.simplify(jaynes_entropy_derivative(alpha) - expected_derivative) == 0
    assert sp.simplify(jaynes_entropy_second_derivative(alpha) - expected_second) == 0
    assert sp.simplify(sp.diff(jaynes_primitive_entropy(alpha), alpha) - expected_derivative) == 0
    assert sp.simplify(sp.diff(expected_derivative, alpha) - expected_second) == 0


def test_unconstrained_jaynes_maximum_is_uniform_six_channel_density() -> None:
    alpha = jaynes_entropy_max_alpha()
    assert alpha == sp.Rational(1, 6)
    assert sp.simplify(jaynes_entropy_derivative(alpha)) == 0
    assert sp.simplify(jaynes_entropy_second_derivative(alpha) + sp.Rational(36, 5)) == 0
    _assert_matrix_equal(jaynes_primitive_density(alpha), sp.eye(SHELL_DIMENSION) / 6)
    assert sp.simplify(jaynes_primitive_entropy(alpha) - sp.log(6)) == 0


def test_jaynes_maximum_recovers_ckm_ratio_and_phase() -> None:
    alpha = jaynes_entropy_max_alpha()
    assert sp.simplify(primitive_ratio_from_alpha(alpha) - 1) == 0
    assert sp.simplify(phase_from_alpha(alpha) - quark_boundary_phase_angle()) == 0


def test_parity_bias_controls_remain_s5_invariant_but_not_ckm() -> None:
    for alpha in (sp.Rational(1, 2), sp.Rational(1, 3)):
        assert density_commutes_with_odd_s5(alpha)
        assert sp.simplify(phase_from_alpha(alpha) - quark_boundary_phase_angle()) != 0


def test_compressed_macrochannel_control_gives_v17_ratio_and_phase() -> None:
    alpha = sp.Symbol("alpha", positive=True)
    max_alpha = compressed_macro_entropy_alpha_max()
    assert max_alpha == sp.Rational(1, 2)
    assert sp.simplify(compressed_macro_entropy_alpha_derivative(max_alpha)) == 0
    assert sp.simplify(
        sp.diff(compressed_macro_entropy_alpha(alpha), alpha)
        - compressed_macro_entropy_alpha_derivative(alpha)
    ) == 0
    assert sp.simplify(primitive_ratio_from_alpha(max_alpha) - 1 / sp.sqrt(5)) == 0
    assert sp.simplify(phase_from_alpha(max_alpha) - sp.pi / 4) == 0
    assert sp.simplify(phase_from_alpha(max_alpha) - quark_boundary_phase_angle()) != 0


def test_jaynes_primitive_ergodicity_payload_reports_theorem_pass() -> None:
    payload = jaynes_primitive_ergodicity_audit_payload()
    assert payload.final_verdict == "JAYNES_PRIMITIVE_ERGODICITY_THEOREM_PASS"
    assert payload.maximizing_alpha == sp.Rational(1, 6)
    assert sp.simplify(payload.induced_ratio - 1) == 0
    assert sp.simplify(payload.induced_phase - quark_boundary_phase_angle()) == 0
    assert payload.density_uniform
    assert payload.s5_invariant
    assert payload.parity_bias_controls_leave_ratio_free
    assert payload.compressed_max_alpha == sp.Rational(1, 2)
    assert sp.simplify(payload.compressed_max_ratio - 1 / sp.sqrt(5)) == 0
    assert sp.simplify(payload.compressed_max_phase - sp.pi / 4) == 0
    assert payload.ckm_phase_recovered
    assert payload.compressed_control_rejected
    assert "no parity-bias" in payload.remaining_physical_input
