"""Tests for residual-family projectors and symmetry."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import (
    commutator,
    is_s3_invariant,
    is_selected_s2_invariant,
    k_nu_operator,
    residual_basis_matrix,
    residual_projectors,
    residual_vectors,
    s3_centralizer_template,
    s3_permutation_matrices,
)
from clifford_3plus2_d5.boundary_response.schur import matrix_equal
from clifford_3plus2_d5.boundary_response.transfer import epsilon_squared


def test_residual_basis_is_orthonormal() -> None:
    basis = residual_basis_matrix(("a", "u", "b"))
    assert matrix_equal(basis.T * basis, sp.eye(3))


def test_projectors_are_orthogonal_idempotents_and_complete() -> None:
    projectors = residual_projectors()
    total = sp.zeros(3, 3)
    for name, proj in projectors.items():
        assert matrix_equal(proj * proj, proj), name
        total += proj

    names = tuple(projectors)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            assert matrix_equal(projectors[left_name] * projectors[right_name], sp.zeros(3, 3))

    assert matrix_equal(total, sp.eye(3))


def test_k_nu_has_expected_projector_eigen_actions() -> None:
    vectors = residual_vectors()
    k_nu = k_nu_operator()
    assert matrix_equal(k_nu * vectors["a"], sp.zeros(3, 1))
    assert matrix_equal(k_nu * vectors["u"], epsilon_squared() * vectors["u"])
    assert matrix_equal(k_nu * vectors["b"], vectors["b"])


def test_k_nu_preserves_selected_s2_but_not_full_s3() -> None:
    k_nu = k_nu_operator()
    assert is_selected_s2_invariant(k_nu)
    assert not is_s3_invariant(k_nu)


def test_s3_centralizer_template_commutes_with_all_s3() -> None:
    alpha, beta = sp.symbols("alpha beta")
    template = s3_centralizer_template(alpha, beta)
    for op in s3_permutation_matrices():
        assert matrix_equal(commutator(template, op), sp.zeros(3, 3))


def test_s3_centralizer_cannot_split_a_and_b() -> None:
    alpha, beta = sp.symbols("alpha beta")
    vectors = residual_vectors()
    template = s3_centralizer_template(alpha, beta)
    assert matrix_equal(template * vectors["a"], beta * vectors["a"])
    assert matrix_equal(template * vectors["b"], beta * vectors["b"])
