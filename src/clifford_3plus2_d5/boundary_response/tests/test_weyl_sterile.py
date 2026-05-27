"""Tests for the V6 exact Weyl-function sterile theorem."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import transfer_probe
from clifford_3plus2_d5.boundary_response.product_sterile import (
    normalized_rank_one_negative_control,
    product_sterile_normalized_response,
    product_sterile_transfer_amplitude,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    k_nu_operator,
    residual_basis_matrix,
)
from clifford_3plus2_d5.boundary_response.schur import matrix_equal
from clifford_3plus2_d5.boundary_response.transfer import epsilon
from clifford_3plus2_d5.boundary_response.weyl_sterile import (
    semi_infinite_weyl_function,
    weyl_branch_limit,
    weyl_fixed_point_residual,
    weyl_product_sterile_normalized_response,
    weyl_product_sterile_response,
    weyl_quadratic_residual,
    weyl_sterile_audit_payload,
    weyl_transfer_amplitude,
)


def test_weyl_function_satisfies_quadratic_identity() -> None:
    z = sp.symbols("z", positive=True)
    assert weyl_quadratic_residual(z) == 0


def test_weyl_function_satisfies_fixed_point_identity() -> None:
    z = sp.symbols("z", positive=True)
    assert weyl_fixed_point_residual(z) == 0


def test_weyl_branch_is_decaying_at_infinity() -> None:
    z = sp.symbols("z", positive=True)
    assert weyl_branch_limit(z) == 1


def test_weyl_value_at_transfer_probe_is_epsilon() -> None:
    z = transfer_probe()
    assert sp.simplify(semi_infinite_weyl_function(z) - epsilon()) == 0
    assert sp.simplify(weyl_transfer_amplitude(z) - epsilon()) == 0


def test_weyl_product_normalized_response_has_exact_eigenstructure() -> None:
    z = sp.symbols("z", positive=True)
    m_z = semi_infinite_weyl_function(z)
    response = weyl_product_sterile_normalized_response(z)
    basis = residual_basis_matrix(("a", "u", "b"))
    residual = (basis.T * response * basis).applyfunc(sp.simplify)

    assert sp.simplify(residual[0, 0]) == 0
    assert sp.simplify(residual[1, 1] - m_z**2) == 0
    assert sp.simplify(residual[2, 2] - 1) == 0
    assert all(residual[row, col] == 0 for row in range(3) for col in range(3) if row != col)


def test_weyl_product_response_is_scaled_by_weyl_head_return() -> None:
    z = sp.symbols("z", positive=True)
    m_z = semi_infinite_weyl_function(z)
    response = weyl_product_sterile_response(z)
    normalized = weyl_product_sterile_normalized_response(z)
    assert matrix_equal(response, (m_z * normalized).applyfunc(sp.simplify))


def test_weyl_response_at_transfer_probe_matches_k_nu_exactly() -> None:
    assert matrix_equal(weyl_product_sterile_normalized_response(), k_nu_operator())


def test_finite_v5_transfer_converges_to_weyl_value() -> None:
    error = sp.simplify(product_sterile_transfer_amplitude(10) - semi_infinite_weyl_function(transfer_probe()))
    assert abs(float(sp.N(error))) < 1e-6


def test_finite_v5_response_converges_to_weyl_response() -> None:
    finite = product_sterile_normalized_response(10)
    exact = weyl_product_sterile_normalized_response()
    error = finite - exact
    error_norm = sp.simplify(sum(entry**2 for entry in error))
    assert abs(float(sp.N(error_norm))) < 1e-10


def test_rank_one_negative_control_remains_cross_term_contaminated() -> None:
    negative = normalized_rank_one_negative_control(8)
    basis = residual_basis_matrix(("a", "u", "b"))
    residual = (basis.T * negative * basis).applyfunc(sp.simplify)
    assert sp.simplify(residual[1, 2]) != 0
    assert sp.simplify(residual[2, 1]) != 0


def test_weyl_audit_reports_limit_pass_and_parks_textures() -> None:
    payload = weyl_sterile_audit_payload()
    assert payload.final_verdict == "PRODUCT_STERILE_LIMIT_PASS"
    assert payload.weyl_value == epsilon()
    assert payload.fixed_point_residual == 0
    assert payload.quadratic_residual == 0
    assert payload.branch_limit == 1
    assert payload.response_matches_target
    assert payload.negative_control_has_cross_return
    assert payload.pmns_ckm_parked
