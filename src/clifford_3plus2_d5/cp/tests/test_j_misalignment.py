"""Tests for ``j_misalignment.py`` — beta audit.

Slow: J-decomposition reconstruction and multi-element β-multi payload
are exact symbolic SymPy computations over the chiral-16 carrier with
~32×32 matrices; each test takes ~2 minutes.
"""

from __future__ import annotations

import pytest
import sympy as sp

pytestmark = pytest.mark.slow

from clifford_3plus2_d5.cp.j_misalignment import (
    beta_audit_payload,
    cp_violating_fraction,
    j_decomposition,
    lift_cl04_j_to_chiral16,
    viable_j_candidates,
)
from clifford_3plus2_d5.cp.reuse import (
    cl04_commutant_complex_structures,
    higgs_like_charge_shift_candidate,
    patisalam_chosen_complex_structure,
)


def test_cl04_has_three_complex_structure_candidates() -> None:
    assert len(cl04_commutant_complex_structures()) == 3


def test_viable_j_candidates_returns_at_least_one() -> None:
    candidates = viable_j_candidates()
    assert len(candidates) >= 1
    for j in candidates:
        assert j.shape == (32, 32)


def test_j_decomposition_reconstructs_matrix() -> None:
    higgs = higgs_like_charge_shift_candidate()
    j = patisalam_chosen_complex_structure()
    m_c, m_a = j_decomposition(higgs, j)
    assert (m_c + m_a - higgs).applyfunc(sp.simplify) == sp.zeros(higgs.rows, higgs.cols)


def test_j_decomposition_components_have_correct_commutation() -> None:
    higgs = higgs_like_charge_shift_candidate()
    j = patisalam_chosen_complex_structure()
    m_c, m_a = j_decomposition(higgs, j)
    zero = sp.zeros(higgs.rows, higgs.cols)
    assert (m_c * j - j * m_c).applyfunc(sp.simplify) == zero
    assert (m_a * j + j * m_a).applyfunc(sp.simplify) == zero


def test_higgs_does_not_commute_with_chosen_j() -> None:
    higgs = higgs_like_charge_shift_candidate()
    j = patisalam_chosen_complex_structure()
    commutator = (higgs * j - j * higgs).applyfunc(sp.simplify)
    assert commutator != sp.zeros(higgs.rows, higgs.cols)


def test_cp_violating_fraction_is_exactly_one_half() -> None:
    higgs = higgs_like_charge_shift_candidate()
    j = patisalam_chosen_complex_structure()
    fraction = cp_violating_fraction(higgs, j)
    assert fraction == sp.Rational(1, 2)


def test_beta_audit_payload_passes_with_maximal_mixing() -> None:
    payload = beta_audit_payload()
    assert payload.passes
    assert "BETA PASS" in payload.verdict
    assert payload.j_candidate_count_in_cl04 == 3
    assert payload.chosen_j_cp_violating_fraction == sp.Rational(1, 2)
    assert payload.chosen_j_cp_violating_fraction_float == 0.5
    assert payload.higgs_frobenius_norm_squared == 256
    assert payload.chosen_j_commuting_norm_squared == 128
    assert payload.chosen_j_anticommuting_norm_squared == 128
    assert "maximal mixing" in payload.interpretation


# ---- Multi-element beta audit tests ----


def test_multi_element_beta_payload_passes_robustly() -> None:
    from clifford_3plus2_d5.cp.j_misalignment import multi_element_beta_audit_payload

    payload = multi_element_beta_audit_payload()
    assert payload.basis_dimension == 4
    assert payload.passes
    assert payload.all_equal_one_half
    assert not payload.any_zero
    assert "ROBUST PASS" in payload.verdict


def test_multi_element_per_basis_all_one_half() -> None:
    from clifford_3plus2_d5.cp.j_misalignment import multi_element_beta_audit_payload

    payload = multi_element_beta_audit_payload()
    half = sp.Rational(1, 2)
    assert len(payload.per_basis_fractions) == 4
    for fraction in payload.per_basis_fractions:
        assert fraction == half


def test_multi_element_per_transpose_all_one_half() -> None:
    from clifford_3plus2_d5.cp.j_misalignment import multi_element_beta_audit_payload

    payload = multi_element_beta_audit_payload()
    half = sp.Rational(1, 2)
    assert len(payload.per_transpose_fractions) == 4
    for fraction in payload.per_transpose_fractions:
        assert fraction == half


def test_multi_element_interpretation_mentions_universal() -> None:
    from clifford_3plus2_d5.cp.j_misalignment import multi_element_beta_audit_payload

    payload = multi_element_beta_audit_payload()
    assert "universal" in payload.interpretation
