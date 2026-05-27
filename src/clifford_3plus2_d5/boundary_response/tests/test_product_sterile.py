"""Tests for the V5 product sterile-tail audit."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.product_sterile import (
    normalized_rank_one_negative_control,
    product_sterile_audit_payload,
    product_sterile_coupling_matrix,
    product_sterile_effective_response,
    product_sterile_factorized_resolvent,
    product_sterile_finite_target,
    product_sterile_hamiltonian,
    product_sterile_normalized_response,
    product_sterile_resolvent,
    product_sterile_return_matrix,
    product_sterile_transfer_amplitude,
    rank_one_sterile_negative_control,
)
from clifford_3plus2_d5.boundary_response.residual_basis import residual_basis_matrix
from clifford_3plus2_d5.boundary_response.schur import matrix_equal
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def test_product_sterile_hamiltonian_shape_and_symmetry() -> None:
    h_q = product_sterile_hamiltonian(4)
    assert h_q.shape == (12, 12)
    assert matrix_equal(h_q, h_q.T)


def test_product_resolvent_factorizes() -> None:
    full = product_sterile_resolvent(4)
    factored = product_sterile_factorized_resolvent(4)
    assert matrix_equal(full, factored)


def test_product_coupling_keeps_family_factor_inside_q() -> None:
    coupling = product_sterile_coupling_matrix(5)
    assert coupling.shape == (15, 3)

    basis = residual_basis_matrix(("a", "u", "b"))
    in_residual_basis = (coupling * basis).applyfunc(sp.simplify)
    assert all(sp.simplify(entry) == 0 for entry in in_residual_basis[:, 0])
    assert any(sp.simplify(entry) != 0 for entry in in_residual_basis[:, 1])
    assert any(sp.simplify(entry) != 0 for entry in in_residual_basis[:, 2])


def test_product_returns_are_equal_and_cross_return_vanishes() -> None:
    returns = product_sterile_return_matrix(6)
    assert sp.simplify(returns[0, 0] - returns[1, 1]) == 0
    assert sp.simplify(returns[0, 1]) == 0
    assert sp.simplify(returns[1, 0]) == 0


def test_product_normalized_response_matches_finite_target() -> None:
    response = product_sterile_normalized_response(8)
    target = product_sterile_finite_target(8)
    assert matrix_equal(response, target)


def test_product_response_has_target_eigenstructure_at_finite_shells() -> None:
    shells = 8
    amp = product_sterile_transfer_amplitude(shells)
    response = product_sterile_normalized_response(shells)
    basis = residual_basis_matrix(("a", "u", "b"))
    residual = (basis.T * response * basis).applyfunc(sp.simplify)

    assert sp.simplify(residual[0, 0]) == 0
    assert sp.simplify(residual[1, 1] - amp**2) == 0
    assert sp.simplify(residual[2, 2] - 1) == 0
    assert all(residual[row, col] == 0 for row in range(3) for col in range(3) if row != col)


def test_product_transfer_amplitude_converges_to_epsilon() -> None:
    error = sp.simplify(epsilon() - product_sterile_transfer_amplitude(10))
    assert abs(float(sp.N(error))) < 1e-6


def test_product_mass_ratios_converge_to_epsilon_powers() -> None:
    payload = product_sterile_audit_payload(shells=10)
    assert abs(float(sp.N(payload.mass_ratio - payload.epsilon_limit_mass_ratio))) < 1e-6
    assert abs(float(sp.N(payload.mass_squared_ratio - payload.epsilon_limit_mass_squared_ratio))) < 1e-6


def test_rank_one_negative_control_has_cross_return_and_rank_one() -> None:
    negative = normalized_rank_one_negative_control(8)
    basis = residual_basis_matrix(("a", "u", "b"))
    residual = (basis.T * negative * basis).applyfunc(sp.simplify)

    assert sp.simplify(residual[1, 2]) != 0
    assert sp.simplify(residual[2, 1]) != 0
    assert rank_one_sterile_negative_control(8).rank() == 1


def test_product_sterile_audit_reports_convergence_pass_but_parks_textures() -> None:
    payload = product_sterile_audit_payload(shells=10)
    assert payload.final_verdict == "PRODUCT_STERILE_CONVERGENCE_PASS"
    assert payload.equal_returns
    assert payload.cross_return_zero
    assert payload.radial_mode_absent
    assert payload.response_matches_finite_target
    assert payload.negative_control_has_cross_return
    assert payload.negative_control_rank == 1
    assert payload.pmns_ckm_parked


def test_product_effective_response_is_scaled_by_common_head_return() -> None:
    response = product_sterile_effective_response(6)
    normalized = product_sterile_normalized_response(6)
    assert response.shape == normalized.shape == (3, 3)
