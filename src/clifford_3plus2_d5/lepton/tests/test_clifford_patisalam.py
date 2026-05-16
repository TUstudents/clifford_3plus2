"""Session 18 tests for the Pati-Salam Cl(0,6) tensor Cl(0,4) audit."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_patisalam import (
    cl04_chosen_commutant_j,
    cl04_commutant_complex_structures,
    cl04_gamma_matrices,
    cl04_relations_pass,
    cl04_volume_element,
    cl06_gamma_matrices,
    cl06_relations_pass,
    patisalam_all_commute_with_chosen_j,
    patisalam_audit_payload,
    patisalam_chiral16_basis_matrix,
    patisalam_chiral16_block_matrix,
    patisalam_chirality_operator,
    patisalam_chirality_projectors,
    patisalam_chosen_complex_structure,
    patisalam_cl010_gamma_matrices,
    patisalam_cl010_relations_pass,
    patisalam_cl010_volume_element,
    patisalam_j_compatibility_audit,
    patisalam_spin_span_dimensions,
    patisalam_su2_factors_commute,
    spin04_generators_chiral16,
    spin04_simple_bivector_j,
    spin04_simple_bivector_j_audit,
    spin06_generators_chiral16,
    su2_l_generators_from_spin04,
    su2_r_generators_from_spin04,
    su4_generators_from_spin06,
)
from clifford_3plus2_d5.lepton.continuum import matrix_zero


def _same(left: sp.Matrix, right: sp.Matrix) -> bool:
    return matrix_zero(left - right)


def test_cl06_and_cl04_relations() -> None:
    cl06 = cl06_gamma_matrices()
    cl04 = cl04_gamma_matrices()
    assert len(cl06) == 6
    assert len(cl04) == 4
    assert all(gamma.shape == (8, 8) for gamma in (*cl06, *cl04))
    assert cl06_relations_pass(cl06)
    assert cl04_relations_pass(cl04)

    omega4 = cl04_volume_element(cl04)
    assert _same(omega4 * omega4, sp.eye(8))
    assert all(_same(omega4 * gamma, -gamma * omega4) for gamma in cl04)


def test_patisalam_cl010_relations_and_chirality() -> None:
    gammas = patisalam_cl010_gamma_matrices()
    assert len(gammas) == 10
    assert all(gamma.shape == (64, 64) for gamma in gammas)
    assert patisalam_cl010_relations_pass(gammas)

    volume = patisalam_cl010_volume_element(gammas)
    assert _same(volume * volume, -sp.eye(64))

    chirality = patisalam_chirality_operator(gammas)
    assert _same(chirality * chirality, sp.eye(64))
    plus, minus = patisalam_chirality_projectors(gammas)
    assert _same(plus * plus, plus)
    assert _same(minus * minus, minus)
    assert _same(plus + minus, sp.eye(64))
    assert (plus.rank(), minus.rank()) == (32, 32)
    assert patisalam_chiral16_basis_matrix("+").shape == (64, 32)


def test_odd_gammas_swap_chirality_and_even_words_preserve_it() -> None:
    gammas = patisalam_cl010_gamma_matrices()
    assert patisalam_chiral16_block_matrix(gammas[0]) is None

    bivector = patisalam_chiral16_block_matrix(gammas[0] * gammas[1])
    assert bivector is not None
    assert bivector.shape == (32, 32)


def test_patisalam_lie_dimensions_and_su2_commutation() -> None:
    dimensions = patisalam_spin_span_dimensions()
    assert dimensions == {
        "spin06_su4": 15,
        "spin04": 6,
        "su2_l": 3,
        "su2_r": 3,
    }
    assert len(spin06_generators_chiral16()) == 15
    assert len(spin04_generators_chiral16()) == 6
    assert len(su4_generators_from_spin06()) == 15
    assert len(su2_l_generators_from_spin04()) == 3
    assert len(su2_r_generators_from_spin04()) == 3
    assert patisalam_su2_factors_commute()


def test_commutant_j_is_complex_and_preserves_patisalam_algebra() -> None:
    assert len(cl04_commutant_complex_structures()) == 3
    j4 = cl04_chosen_commutant_j()
    assert _same(j4 * j4, -sp.eye(8))

    chosen_j = patisalam_chosen_complex_structure()
    assert _same(chosen_j * chosen_j, -sp.eye(32))
    assert _same(chosen_j.T * chosen_j, sp.eye(32))
    assert patisalam_all_commute_with_chosen_j(su4_generators_from_spin06())
    assert patisalam_all_commute_with_chosen_j(su2_l_generators_from_spin04())
    assert patisalam_all_commute_with_chosen_j(su2_r_generators_from_spin04())


def test_simple_spin04_bivector_j_breaks_both_su2_factors_to_cartans() -> None:
    simple_j = spin04_simple_bivector_j()
    assert _same(simple_j * simple_j, -sp.eye(32))

    audit = spin04_simple_bivector_j_audit()
    assert audit["squares_to_minus_identity"] is True
    assert audit["commuting_su2_l_generators"] == 1
    assert audit["commuting_su2_r_generators"] == 1


def test_patisalam_audit_payload_is_stable() -> None:
    audit = patisalam_audit_payload()
    assert audit["factorization"] == "Cl(0,10) = Cl(0,6) tensor Cl(0,4)"
    assert audit["cl010_relations_pass"] is True
    assert audit["chirality_ranks"] == (32, 32)
    assert audit["spin06_su4_dimension"] == 15
    assert audit["spin04_dimension"] == 6
    assert audit["su2_l_dimension"] == 3
    assert audit["su2_r_dimension"] == 3
    assert audit["su2_l_su2_r_commute"] is True

    j_audit = patisalam_j_compatibility_audit()
    assert j_audit["chosen_j_commutes_with_su4"] is True
    assert j_audit["chosen_j_commutes_with_su2_l"] is True
    assert j_audit["chosen_j_commutes_with_su2_r"] is True
