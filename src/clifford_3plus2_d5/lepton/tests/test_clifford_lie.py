"""Guardrails for the Session 15 Clifford Lie data layer."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_lie import (
    all_in_span,
    annihilates_vector,
    basis_span_dimension,
    chiral_basis_consistency_check,
    chiral_so8_bivector_basis,
    clifford_g2_basis,
    clifford_su3_basis,
    complement_to_g2_in_so8,
    complement_to_su3_in_g2,
    imaginary_octonion_transverse_to_e7_basis,
    is_skew_symmetric,
    unit_octonion_vector,
)


def test_chiral_bivectors_are_so8_basis() -> None:
    bivectors = chiral_so8_bivector_basis()
    assert chiral_basis_consistency_check()
    assert len(bivectors) == 28
    assert basis_span_dimension(bivectors) == 28
    assert all(is_skew_symmetric(generator) for generator in bivectors)


def test_g2_lives_in_so7() -> None:
    g2 = clifford_g2_basis()
    e0 = unit_octonion_vector(0)
    assert len(g2) == 14
    assert basis_span_dimension(g2) == 14
    assert all(is_skew_symmetric(generator) for generator in g2)
    assert all(annihilates_vector(generator, e0) for generator in g2)


def test_su3_lives_in_g2_and_fixes_e7() -> None:
    su3 = clifford_su3_basis()
    g2 = clifford_g2_basis()
    e7 = unit_octonion_vector(7)
    assert len(su3) == 8
    assert basis_span_dimension(su3) == 8
    assert all(is_skew_symmetric(generator) for generator in su3)
    assert all(annihilates_vector(generator, e7) for generator in su3)
    assert all_in_span(su3, g2)


def test_lie_bases_are_in_bivector_span() -> None:
    bivectors = chiral_so8_bivector_basis()
    assert all_in_span(clifford_g2_basis(), bivectors)
    assert all_in_span(clifford_su3_basis(), bivectors)


def test_complements_have_expected_dimensions() -> None:
    g2 = clifford_g2_basis()
    su3 = clifford_su3_basis()
    g2_complement = complement_to_g2_in_so8()
    su3_complement = complement_to_su3_in_g2()

    assert len(g2_complement) == 14
    assert basis_span_dimension(g2_complement) == 14
    assert basis_span_dimension((*g2, *g2_complement)) == 28

    assert len(su3_complement) == 6
    assert basis_span_dimension(su3_complement) == 6
    assert basis_span_dimension((*su3, *su3_complement)) == 14

    transverse = imaginary_octonion_transverse_to_e7_basis()
    assert len(transverse) == 6
    assert basis_span_dimension(transverse) == 6
    assert all(vector.shape == (8, 1) for vector in transverse)
    assert all(vector[0] == 0 and vector[7] == 0 for vector in transverse)
    assert sum(transverse, sp.zeros(8, 1)).shape == (8, 1)
