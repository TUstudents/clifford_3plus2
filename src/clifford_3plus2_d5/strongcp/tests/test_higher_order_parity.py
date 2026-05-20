"""Tests for ``higher_order_parity.py``.

Slow: exact symbolic BCH extraction at O(ε²) (computing H^(2) from
the ε³ coefficient of log U on the BCC Dirac Bloch operator) is
SymPy-heavy and takes 10-20s per test.
"""

from __future__ import annotations

import pytest
import sympy as sp

pytestmark = pytest.mark.slow

from clifford_3plus2_d5.strongcp.higher_order_parity import (
    effective_hamiltonian_second_correction,
    h1_from_bch_matches_cp_implementation,
    h2_a2u_component_is_zero,
    h2_decomposition,
    h2_is_chirality_parity_odd,
    h2_is_hermitian,
    h2_lives_entirely_in_t1u,
    higher_order_parity_payload,
)


def test_h1_bch_matches_cp_implementation() -> None:
    assert h1_from_bch_matches_cp_implementation()


def test_h2_is_hermitian() -> None:
    assert h2_is_hermitian()


def test_h2_is_chirality_parity_odd() -> None:
    assert h2_is_chirality_parity_odd()


def test_h2_a2u_component_is_zero() -> None:
    # The load-bearing test: H^(2) cannot contribute to θ_QCD.
    assert h2_a2u_component_is_zero()


def test_h2_lives_entirely_in_T1u() -> None:
    assert h2_lives_entirely_in_t1u()


def test_h2_decomposition_t1u_nontrivial() -> None:
    decomp = h2_decomposition()
    t1u = decomp["T1u"]
    # Sanity: H^(2) itself is non-zero, so the T_{1u} component must be non-zero too.
    assert t1u != sp.zeros(4, 4)


def test_h2_has_degree_3_in_k() -> None:
    h2 = effective_hamiltonian_second_correction()
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    for r in range(h2.rows):
        for c in range(h2.cols):
            entry = sp.expand(h2[r, c])
            if entry == 0:
                continue
            poly = sp.Poly(entry, kx, ky, kz)
            for monomial in poly.monoms():
                assert sum(monomial) == 3


def test_higher_order_parity_payload_selection_rule_applies() -> None:
    payload = higher_order_parity_payload()
    assert payload.h1_cross_check_passes
    assert payload.h2_is_hermitian
    assert payload.h2_chirality_parity_odd
    assert payload.h2_a2u_component_zero
    assert payload.h2_entirely_in_t1u
    assert payload.selection_rule_applies
    assert "PARITY SELECTION RULE" in payload.verdict


def test_payload_irrep_summary_only_t1u_nontrivial() -> None:
    payload = higher_order_parity_payload()
    assert payload.h2_irrep_summary == {"A2u": False, "T2u": False, "T1u": True}
