"""Tests for ``discrete_symmetries.py`` — alpha-1 P/T/C operator construction.

Updated 2026-05-20: charge_conjugation_spinor now returns i·γ² (standard
physical chiral-basis C-matrix), distinct from T's γ²γ⁰.  Conjugation
patterns for C and its composites change accordingly.
"""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.cp.discrete_symmetries import (
    all_seven_operators,
    bloch_particle_hole_spinor,
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


def test_charge_conjugation_spinor_is_i_gamma_two() -> None:
    """Standard physical C in chiral basis: M = i·γ²."""
    from clifford_3plus2_d5.cp.reuse import gamma_spatial_matrices

    _, gamma2, _ = gamma_spatial_matrices()
    expected = (sp.I * gamma2).applyfunc(sp.simplify)
    assert (charge_conjugation_spinor() - expected).applyfunc(sp.simplify) == sp.zeros(4)


def test_charge_conjugation_spinor_squares_to_identity() -> None:
    """(iγ²)² = -I·(γ²)² = -I·(-I) = I."""
    C = charge_conjugation_spinor()
    assert (C * C - sp.eye(4)).applyfunc(sp.simplify) == sp.zeros(4)


def test_bloch_particle_hole_matches_legacy_C_matrix() -> None:
    """Pre-2026-05-20 ``charge_conjugation_spinor`` returned γ²γ⁰ — same
    matrix as T.  Kept under bloch_particle_hole_spinor for traceability."""
    assert (
        bloch_particle_hole_spinor() - time_reversal_spinor()
    ).applyfunc(sp.simplify) == sp.zeros(4)


def test_seven_named_operators() -> None:
    names = tuple(op.name for op in all_seven_operators())
    assert names == ("P", "T", "C", "PT", "CP", "CT", "CPT")


def test_conjugation_pattern_parity() -> None:
    assert conjugation_pattern(parity_operator()) == (1, -1, -1, -1)


def test_conjugation_pattern_time_reversal() -> None:
    assert conjugation_pattern(time_reversal_operator()) == (-1, 1, 1, 1)


def test_conjugation_pattern_charge_conjugation() -> None:
    """Standard physical C with M = iγ²: anticommutes with all γ^μ."""
    assert conjugation_pattern(charge_conjugation_operator()) == (-1, -1, -1, -1)


def test_conjugation_pattern_pt() -> None:
    assert conjugation_pattern(pt_operator()) == (-1, -1, -1, -1)


def test_conjugation_pattern_cp() -> None:
    """CP under standard C: γ⁰ anticommutes, γ^i commute."""
    assert conjugation_pattern(cp_operator()) == (-1, 1, 1, 1)


def test_conjugation_pattern_ct() -> None:
    """CT under standard C: γ⁰ commutes, γ^i anticommute."""
    assert conjugation_pattern(ct_operator()) == (1, -1, -1, -1)


def test_conjugation_pattern_cpt() -> None:
    """CPT under standard C: commutes with all γ^μ (identity-like)."""
    assert conjugation_pattern(cpt_operator()) == (1, 1, 1, 1)


def test_composite_flags() -> None:
    # PT: both antiunitary contributions cancel? No — exactly one antiunitary.
    assert pt_operator().antiunitary
    # CT: two antiunitaries → unitary.
    assert not ct_operator().antiunitary
    # CPT: three antiunitary-or-not.  C antiunitary + P unitary + T antiunitary
    # = two K's = unitary.
    assert not cpt_operator().antiunitary


def test_hamiltonian_signs_all_plus_one_for_massless_walk() -> None:
    """For BCC Dirac (massless), every named operator has A H A⁻¹ = +H(k_image)."""
    for op in all_seven_operators():
        assert op.hamiltonian_sign == +1, (
            f"{op.name} has hamiltonian_sign={op.hamiltonian_sign}, expected +1"
        )


def test_hamiltonian_sign_metadata_matches_derived_action() -> None:
    """Derive-check: for each operator A, compute A H A^{-1} for massless
    H = α·k and confirm it equals ``op.hamiltonian_sign · H(k_image)``
    where k_image = -k if op.momentum_flip else +k.

    This is the load-bearing regression test for the
    walk_respects_symmetry XNOR(antiunitary, hamiltonian_sign==+1)
    criterion: the metadata must match the derived action.
    """
    from clifford_3plus2_d5.cp.reuse import dirac_hamiltonian, same_matrix

    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    H_k = dirac_hamiltonian(kx, ky, kz)
    H_minus_k = dirac_hamiltonian(-kx, -ky, -kz)

    for op in all_seven_operators():
        M = op.spinor_matrix
        Minv = M.inv().applyfunc(sp.simplify)
        inner = H_k.applyfunc(sp.conjugate) if op.antiunitary else H_k
        derived = (M * inner * Minv).applyfunc(sp.simplify)
        h_image = H_minus_k if op.momentum_flip else H_k
        expected = (op.hamiltonian_sign * h_image).applyfunc(sp.simplify)
        assert same_matrix(derived, expected), (
            f"{op.name}: A·H·A^-1 ≠ {op.hamiltonian_sign:+d} · "
            f"H({'(-k)' if op.momentum_flip else 'k'}).  "
            f"Derived: {derived}.  Expected: {expected}."
        )


def test_kind_labels_are_present() -> None:
    """Every operator has a semantic kind label."""
    kinds = {op.name: op.kind for op in all_seven_operators()}
    assert kinds["P"] == "physical_P"
    assert kinds["T"] == "physical_T"
    assert kinds["C"] == "physical_C"
    assert kinds["CP"].startswith("composite_")
    assert kinds["CT"].startswith("composite_")
    assert kinds["PT"].startswith("composite_")
    assert kinds["CPT"].startswith("composite_")


def test_compose_multiplies_hamiltonian_signs() -> None:
    """Composition: (B ∘ A) H (B ∘ A)⁻¹ = s_A · s_B · H(k_image).

    For all our operators with s = +1, the composite sign stays +1.
    """
    assert cp_operator().hamiltonian_sign == +1
    assert ct_operator().hamiltonian_sign == +1
    assert pt_operator().hamiltonian_sign == +1
    assert cpt_operator().hamiltonian_sign == +1
