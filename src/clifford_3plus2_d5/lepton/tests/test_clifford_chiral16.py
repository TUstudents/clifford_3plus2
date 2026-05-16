"""Session 17 guardrails for the Cl(0,10) chiral-16 carrier."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_chiral16 import (
    all_commute_with_chosen_j,
    chiral16_block_matrix,
    chiral16_basis_matrix,
    chiral16_su3_generator_is_valid,
    chosen_chiral16_complex_structure,
    cl010_chirality_operator,
    cl010_chirality_projectors,
    cl010_gamma_matrices,
    cl010_relations_pass,
    cl010_volume_element,
    cl02_electroweak_audit,
    cl02_factor_complex_structures_on_chiral16,
    lifted_su3_generators_chiral16,
    lifted_su3_span_dimension,
)
from clifford_3plus2_d5.lepton.continuum import matrix_zero


def _same(left: sp.Matrix, right: sp.Matrix) -> bool:
    return matrix_zero(left - right)


def test_cl010_gamma_relations_and_volume() -> None:
    gammas = cl010_gamma_matrices()
    assert len(gammas) == 10
    assert all(gamma.shape == (64, 64) for gamma in gammas)
    assert cl010_relations_pass(gammas)

    volume = cl010_volume_element(gammas)
    assert _same(volume * volume, -sp.eye(64))


def test_cl010_chirality_projectors_have_chiral16_rank() -> None:
    chirality = cl010_chirality_operator()
    assert _same(chirality * chirality, sp.eye(64))

    plus, minus = cl010_chirality_projectors()
    assert _same(plus * plus, plus)
    assert _same(minus * minus, minus)
    assert _same(plus * minus, sp.zeros(64))
    assert _same(plus + minus, sp.eye(64))
    assert (plus.rank(), minus.rank()) == (32, 32)
    assert chiral16_basis_matrix("+").shape == (64, 32)


def test_odd_gammas_swap_chirality_and_even_words_preserve_it() -> None:
    gammas = cl010_gamma_matrices()
    assert chiral16_block_matrix(gammas[0]) is None

    bivector = chiral16_block_matrix(gammas[0] * gammas[1])
    assert bivector is not None
    assert bivector.shape == (32, 32)


def test_cl02_supplies_chirality_preserving_complex_structures() -> None:
    candidates = cl02_factor_complex_structures_on_chiral16("+")
    assert len(candidates) == 2
    assert all(_same(candidate * candidate, -sp.eye(32)) for candidate in candidates)

    chosen = chosen_chiral16_complex_structure()
    assert _same(chosen * chosen, -sp.eye(32))
    assert _same(chosen.T * chosen, sp.eye(32))
    assert any(_same(chosen, candidate) for candidate in candidates)


def test_lifted_su3_generators_span_su3_and_commute_with_chosen_j() -> None:
    generators = lifted_su3_generators_chiral16()
    assert len(generators) == 8
    assert lifted_su3_span_dimension() == 8
    assert all(chiral16_su3_generator_is_valid(generator) for generator in generators)
    assert all_commute_with_chosen_j(generators)


def test_cl02_electroweak_audit_is_honest_about_su2() -> None:
    audit = cl02_electroweak_audit()
    assert audit["cl02_j_candidate_count"] == 2
    assert audit["chosen_j_commutes_with_su3"] is True
    assert audit["spin_02_dimension"] == 1
    assert audit["spin_02_can_supply_su2"] is False
    assert "Spin(0,2) is U(1), not SU(2)" in str(audit["note"])
