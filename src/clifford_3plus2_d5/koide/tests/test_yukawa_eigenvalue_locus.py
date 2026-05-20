"""Tests for ``yukawa_eigenvalue_locus.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.koide.yukawa_eigenvalue_locus import (
    bcc_z3_orbit,
    bcc_z3_yukawa_matrix,
    cone_predicted_mass_vector,
    example_v_stars,
    is_circulant,
    koide_special_ratio_numerical,
    koide_special_ratio_symbolic,
    pdg_mass_ratio_matches_special,
    special_mass_ratio_at_cone,
    yukawa_eigenvalue_triple,
    yukawa_koide_K,
    yukawa_locus_payload,
    z3_yukawa_has_degenerate_eigenvalues,
)


def test_bcc_z3_orbit_has_three_vectors() -> None:
    v = sp.Matrix([1, 0, 0])
    orbit = bcc_z3_orbit(v)
    assert len(orbit) == 3
    for o in orbit:
        assert o.shape == (3, 1)


def test_bcc_z3_orbit_returns_to_start_after_three_iterations() -> None:
    v = sp.Matrix([2, 1, 0])
    o0, o1, o2 = bcc_z3_orbit(v)
    # After three more rotations (R^3 = I) should be back to v.
    from clifford_3plus2_d5.koide.reuse import body_diagonal_rotation_matrix

    R = body_diagonal_rotation_matrix()
    assert (R * o2 - o0).applyfunc(sp.simplify) == sp.zeros(3, 1)


def test_yukawa_matrix_is_circulant_for_generic_v_star() -> None:
    v = sp.Matrix([2, 1, 0])
    Y = bcc_z3_yukawa_matrix(v)
    assert is_circulant(Y)


def test_yukawa_matrix_is_circulant_for_diagonal_v_star() -> None:
    v = sp.Matrix([1, 1, 1])
    Y = bcc_z3_yukawa_matrix(v)
    assert is_circulant(Y)


def test_yukawa_eigenvalues_always_have_two_fold_degeneracy() -> None:
    for v in example_v_stars().values():
        assert z3_yukawa_has_degenerate_eigenvalues(v)


def test_eigenvalue_triple_matches_analytical_form() -> None:
    v = sp.Matrix([2, 1, 0])
    lam_1, lam_2, lam_3 = yukawa_eigenvalue_triple(v)
    # Manually: |v_t|² = ((2+1+0)/3)² · 3 = 3, |v_o|² = 5 - 3 = 2.
    # λ_1 = 3·3 = 9, λ_2 = λ_3 = (3/2)·2 = 3.
    assert sp.simplify(lam_1 - 9) == 0
    assert sp.simplify(lam_2 - 3) == 0
    assert sp.simplify(lam_3 - 3) == 0


def test_koide_special_ratio_is_three_plus_two_root_two() -> None:
    r = koide_special_ratio_symbolic()
    assert sp.simplify(r - (3 + 2 * sp.sqrt(2))) == 0
    assert abs(koide_special_ratio_numerical() - 5.828427) < 1e-4


def test_special_mass_ratio_is_two_r_squared() -> None:
    r = koide_special_ratio_symbolic()
    expected = sp.simplify(2 * r**2)
    actual = special_mass_ratio_at_cone()
    assert sp.simplify(actual - expected) == 0
    # Numerical: 2 * (3 + 2√2)² = 2 * (17 + 12√2) ≈ 67.97
    assert abs(float(actual) - 67.9411) < 1e-3


def test_koide_K_at_special_ratio_is_exactly_two_thirds() -> None:
    # Construct v_* with |v_t|/|v_o| = r* explicitly.
    r = koide_special_ratio_symbolic()
    # v_o = (1, -1, 0)/√2 has |v_o| = 1. v_t = r·(1,1,1)/√3 has |v_t| = r.
    v_o_dir = sp.Matrix([1, -1, 0]) / sp.sqrt(2)
    v_t_dir = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    v_star = (r * v_t_dir + v_o_dir).applyfunc(sp.simplify)
    K = yukawa_koide_K(v_star)
    assert sp.simplify(K - sp.Rational(2, 3)) == 0


def test_mass_vector_at_cone_is_degenerate_pair() -> None:
    v_o_mag = sp.Integer(1)
    v_mass = cone_predicted_mass_vector(v_o_mag)
    # Should be (√(3·r²), √(3/2), √(3/2))
    assert sp.simplify(v_mass[1, 0] - v_mass[2, 0]) == 0
    assert sp.simplify(v_mass[0, 0] - v_mass[1, 0]) != 0


def test_pdg_not_in_z3_equivariant_locus() -> None:
    # PDG has all-distinct masses; Z₃ locus has 2-fold degenerate masses.
    assert not pdg_mass_ratio_matches_special()


def test_yukawa_locus_payload_consistent() -> None:
    p = yukawa_locus_payload()
    assert p.z3_yukawa_is_circulant
    assert p.eigenvalues_always_degenerate
    assert not p.pdg_compatible_with_z3_locus
    assert p.locus_is_one_parameter_family_on_cone
    assert "CONSISTENT WITH KOIDE" in p.interpretation
    assert "does NOT predict" in p.interpretation
