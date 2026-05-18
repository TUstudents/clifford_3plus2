"""Tests for ``discrete_symmetries.py`` — alpha-1 P/T/C operator construction."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.cp.discrete_symmetries import (
    all_seven_operators,
    charge_conjugation_operator,
    charge_conjugation_spinor,
    conjugation_pattern,
    cp_operator,
    cpt_operator,
    ct_operator,
    parity_operator,
    parity_spinor,
    pt_operator,
    time_reversal_operator,
    time_reversal_spinor,
)


def test_parity_spinor_is_gamma_zero() -> None:
    from clifford_3plus2_d5.cp.reuse import gamma0

    assert parity_spinor() == gamma0()


def test_parity_is_unitary_self_inverse() -> None:
    P = parity_spinor()
    assert (P * P - sp.eye(4)).applyfunc(sp.simplify) == sp.zeros(4)


def test_time_reversal_spinor_squares_to_identity() -> None:
    T = time_reversal_spinor()
    assert (T * T - sp.eye(4)).applyfunc(sp.simplify) == sp.zeros(4)


def test_charge_conjugation_spinor_equals_time_reversal() -> None:
    # In our chiral basis, the algebraic conditions on C and T coincide.
    assert (charge_conjugation_spinor() - time_reversal_spinor()).applyfunc(
        sp.simplify,
    ) == sp.zeros(4)


def test_seven_named_operators() -> None:
    names = tuple(op.name for op in all_seven_operators())
    assert names == ("P", "T", "C", "PT", "CP", "CT", "CPT")


def test_conjugation_pattern_parity() -> None:
    assert conjugation_pattern(parity_operator()) == (1, -1, -1, -1)


def test_conjugation_pattern_time_reversal() -> None:
    assert conjugation_pattern(time_reversal_operator()) == (-1, 1, 1, 1)


def test_conjugation_pattern_charge_conjugation() -> None:
    assert conjugation_pattern(charge_conjugation_operator()) == (-1, 1, 1, 1)


def test_conjugation_pattern_pt() -> None:
    assert conjugation_pattern(pt_operator()) == (-1, -1, -1, -1)


def test_conjugation_pattern_cp() -> None:
    assert conjugation_pattern(cp_operator()) == (-1, -1, -1, -1)


def test_conjugation_pattern_ct() -> None:
    assert conjugation_pattern(ct_operator()) == (1, 1, 1, 1)


def test_conjugation_pattern_cpt() -> None:
    assert conjugation_pattern(cpt_operator()) == (1, -1, -1, -1)


def test_composite_flags() -> None:
    # PT: both antiunitary contributions cancel? No — exactly one antiunitary.
    assert pt_operator().antiunitary
    # CT: two antiunitaries → unitary.
    assert not ct_operator().antiunitary
    # CPT: three antiunitary-or-not.  C antiunitary + P unitary + T antiunitary
    # = two K's = unitary.
    assert not cpt_operator().antiunitary
