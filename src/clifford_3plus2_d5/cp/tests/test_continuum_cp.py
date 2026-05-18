"""Tests for ``continuum_cp.py`` — O(ε²) BCC continuum decomposition."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.cp.continuum_cp import (
    bloch_order_two,
    continuum_cp_audit_payload,
    cp_action_on_operator,
    cp_even_part,
    cp_irrep_decomposition,
    cp_irrep_norm_table,
    cp_odd_part,
    effective_hamiltonian_first_correction,
    h1_is_hermitian,
    h1_total_norm_squared,
    polynomial_matrix_norm_squared,
    symbolic_momentum,
)
from clifford_3plus2_d5.cp.cubic_harmonics import IRREP_NAMES


def test_bloch_order_two_is_4x4() -> None:
    b2 = bloch_order_two()
    assert b2.shape == (4, 4)


def test_h1_is_hermitian() -> None:
    assert h1_is_hermitian()


def test_h1_is_nonzero_and_block_diagonal_in_chiral_structure() -> None:
    h1 = effective_hamiltonian_first_correction()
    assert h1 != sp.zeros(4, 4)
    # H^(1) is block-diagonal: right and left Weyl blocks decouple.
    # Off-diagonal 2x2 blocks should be zero.
    for r in range(2):
        for c in range(2, 4):
            assert sp.simplify(h1[r, c]) == 0
            assert sp.simplify(h1[c, r]) == 0


def test_cp_even_part_of_zero_matrix_is_zero() -> None:
    zero = sp.zeros(4, 4)
    assert cp_even_part(zero) == sp.zeros(4, 4)
    assert cp_odd_part(zero) == sp.zeros(4, 4)


def test_cp_action_squared_is_identity() -> None:
    # CP² applied to an operator returns the original (CP is order 2 up to sign
    # under the antiunitary convention used here).
    h1 = effective_hamiltonian_first_correction()
    twice = cp_action_on_operator(cp_action_on_operator(h1))
    assert (twice - h1).applyfunc(sp.simplify) == sp.zeros(4, 4)


def test_cp_decomposition_reconstructs() -> None:
    h1 = effective_hamiltonian_first_correction()
    reconstruction = (cp_even_part(h1) + cp_odd_part(h1)).applyfunc(sp.simplify)
    assert (reconstruction - h1).applyfunc(sp.simplify) == sp.zeros(4, 4)


def test_cp_even_part_is_zero_for_h1() -> None:
    # The load-bearing finding: H^(1) is purely CP-odd.
    h1 = effective_hamiltonian_first_correction()
    even = cp_even_part(h1).applyfunc(sp.simplify)
    assert even == sp.zeros(4, 4)


def test_cp_odd_part_equals_h1() -> None:
    h1 = effective_hamiltonian_first_correction()
    odd = cp_odd_part(h1).applyfunc(sp.simplify)
    assert (odd - h1).applyfunc(sp.simplify) == sp.zeros(4, 4)


def test_h1_total_norm_squared_is_twelve() -> None:
    assert h1_total_norm_squared() == 12


def test_cp_irrep_table_localizes_in_T2g() -> None:
    table = cp_irrep_norm_table()
    # CP-odd part lives entirely in T2g
    assert table[("CP-odd", "T2g")] == 12
    assert table[("CP-odd", "A1g")] == 0
    assert table[("CP-odd", "Eg")] == 0
    # CP-even part is zero
    for irrep in IRREP_NAMES:
        assert table[("CP-even", irrep)] == 0


def test_cp_irrep_decomposition_reconstructs_h1_by_summation() -> None:
    h1 = effective_hamiltonian_first_correction()
    decomp = cp_irrep_decomposition()
    total = sp.zeros(4, 4)
    for matrix in decomp.values():
        total = total + matrix
    assert (total - h1).applyfunc(sp.simplify) == sp.zeros(4, 4)


def test_continuum_cp_audit_payload_passes() -> None:
    payload = continuum_cp_audit_payload()
    assert payload.h1_is_hermitian
    assert payload.h1_total_norm_squared == 12
    assert payload.cp_even_total_norm_squared == 0
    assert payload.cp_odd_total_norm_squared == 12
    assert payload.cp_odd_carrying_irreps == ("T2g",)
    assert payload.cp_violating_fraction_at_order_epsilon == 1
    assert "PASS" in payload.verdict
    assert "T2g" in payload.interpretation


def test_polynomial_norm_handles_complex_coefficients() -> None:
    _, kx, ky, kz = symbolic_momentum()
    k = (kx, ky, kz)
    matrix = sp.Matrix([[sp.I * kx * ky]])  # |I|² = 1
    assert polynomial_matrix_norm_squared(matrix, k) == 1


def test_polynomial_norm_sums_complex_squared_moduli() -> None:
    _, kx, ky, kz = symbolic_momentum()
    k = (kx, ky, kz)
    # Entry has two monomials with coefficients I and 1
    matrix = sp.Matrix([[sp.I * ky * kz + kx * ky]])
    # ||·||² = |I|² + |1|² = 1 + 1 = 2
    assert polynomial_matrix_norm_squared(matrix, k) == 2
