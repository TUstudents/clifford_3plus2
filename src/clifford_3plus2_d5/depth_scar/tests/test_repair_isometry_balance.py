"""V10 tests for projected unitarity balance."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_isometry import (
    active_domain_basis,
    active_identity,
    active_projector,
    active_repair_block,
    active_repair_domain_matrix,
    active_repair_norm_matrix,
    leakage_domain_matrix,
    leakage_norm_matrix,
    repaired_range_projector,
    unitarity_balance_formula_pass,
    unitarity_balance_residual,
)


def test_active_and_repaired_projectors_are_expected_subspaces() -> None:
    assert active_projector() == sp.diag(0, 1, 1)
    assert repaired_range_projector() == sp.diag(1, 1, 0)
    assert active_domain_basis() == sp.Matrix([[0, 0], [1, 0], [0, 1]])
    assert active_identity() == sp.eye(2)


def test_active_repair_and_leakage_norms_are_diagonal() -> None:
    alpha, beta, ell_a, ell_b = sp.symbols("alpha beta ell_a ell_b", real=True)

    assert active_repair_block(alpha, beta) == sp.Matrix(
        [
            [0, alpha, 0],
            [0, 0, beta],
            [0, 0, 0],
        ]
    )
    assert active_repair_domain_matrix(alpha, beta) == sp.Matrix([[alpha, 0], [0, beta], [0, 0]])
    assert leakage_domain_matrix(ell_a, ell_b) == sp.Matrix([[ell_a, 0], [0, ell_b]])
    assert active_repair_norm_matrix(alpha, beta) == sp.diag(alpha**2, beta**2)
    assert leakage_norm_matrix(ell_a, ell_b) == sp.diag(ell_a**2, ell_b**2)


def test_unitarity_balance_identity_has_expected_form() -> None:
    alpha, beta, ell_a, ell_b = sp.symbols("alpha beta ell_a ell_b", real=True)
    residual = unitarity_balance_residual(alpha, beta, ell_a, ell_b)

    assert residual == sp.Matrix(
        [
            [alpha**2 + ell_a**2 - 1, 0],
            [0, beta**2 + ell_b**2 - 1],
        ]
    )
    assert unitarity_balance_formula_pass()
