"""Tests for ``lattice_topological_charge.py`` (Phase SC-4)."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.plaquette import (
    canonical_bcc_plaquette_shapes,
)
from clifford_3plus2_d5.strongcp.lattice_topological_charge import (
    a2u_projection_of_pair_tensor,
    canonical_plaquette_shape_under_inversion,
    confirm_gauge_independence_symbolic,
    f_is_antihermitian,
    f_munu_from_plaquette_holonomy,
    gauge_group_algebra_dimension,
    gauge_group_dimension,
    identity_plus_anti_hermitian_test_holonomy,
    lattice_topological_charge_payload,
    num_indices_required_for_4d_q,
    num_spatial_directions,
    plaquette_inversion_permutation,
    plaquette_pair_tensor_is_real_symmetric,
    plaquette_pair_tensor_symbolic,
    plaquette_rep_is_parity_even,
    spatial_only_q_is_dimensionally_trivial,
)


# =============================================================================
# SC-4a: F_{ij} extraction
# =============================================================================


def test_f_munu_from_identity_is_zero() -> None:
    H = sp.eye(2)
    F = f_munu_from_plaquette_holonomy(H)
    assert F == sp.zeros(2, 2)


def test_f_munu_from_test_holonomy_is_hermitian() -> None:
    H = identity_plus_anti_hermitian_test_holonomy(n_dim=2)
    F = f_munu_from_plaquette_holonomy(H)
    assert f_is_antihermitian(F)


def test_f_munu_from_su3_test_holonomy_is_hermitian() -> None:
    H = identity_plus_anti_hermitian_test_holonomy(n_dim=3)
    F = f_munu_from_plaquette_holonomy(H)
    assert f_is_antihermitian(F)


def test_f_munu_extracts_epsilon_squared_coefficient() -> None:
    # For H = I + i ε² A with A Hermitian, F should equal ε² A.
    H = identity_plus_anti_hermitian_test_holonomy(n_dim=2)
    F = f_munu_from_plaquette_holonomy(H)
    epsilon = sp.symbols("epsilon", positive=True)
    a1, a2, a3 = sp.symbols("a1 a2 a3", real=True)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    expected = (epsilon**2 * (a1 * sx + a2 * sy + a3 * sz)).applyfunc(sp.simplify)
    diff = (F - expected).applyfunc(sp.simplify)
    assert diff == sp.zeros(2, 2)


# =============================================================================
# SC-4b: spatial-only Q dimensional triviality
# =============================================================================


def test_num_spatial_directions_is_three() -> None:
    assert num_spatial_directions() == 3


def test_num_indices_for_4d_q_is_four() -> None:
    assert num_indices_required_for_4d_q() == 4


def test_spatial_only_q_is_dimensionally_trivial() -> None:
    # 3 spatial < 4 required indices → spatial Q ≡ 0 by antisymmetry of ε.
    assert spatial_only_q_is_dimensionally_trivial()


# =============================================================================
# SC-4c: cubic-irrep decomposition
# =============================================================================


def test_canonical_plaquette_inversion_returns_same_shape() -> None:
    shapes = canonical_bcc_plaquette_shapes()
    for shape in shapes:
        inverted = canonical_plaquette_shape_under_inversion(shape)
        # Inversion + canonicalization returns one of the 6 canonical shapes.
        assert inverted in shapes


def test_plaquette_inversion_permutation_is_identity() -> None:
    perm = plaquette_inversion_permutation()
    assert perm == tuple(range(6))


def test_plaquette_rep_is_parity_even() -> None:
    assert plaquette_rep_is_parity_even()


def test_pair_tensor_for_identity_holonomies_is_zero() -> None:
    # F = 0 for all 6 plaquettes (e.g., flat gauge config).
    F_zero = tuple(sp.zeros(2, 2) for _ in range(6))
    T = plaquette_pair_tensor_symbolic(F_zero)
    assert T == sp.zeros(6, 6)


def test_pair_tensor_validates_six_plaquettes() -> None:
    import pytest

    F_too_few = tuple(sp.zeros(2, 2) for _ in range(5))
    with pytest.raises(ValueError):
        plaquette_pair_tensor_symbolic(F_too_few)


def test_pair_tensor_real_symmetric_for_concrete_hermitian_Fs() -> None:
    # Use a set of 6 distinct Hermitian 2×2 matrices.
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    F_set = (sx, sy, sz, sx + sy, sy + sz, sz + sx)
    T = plaquette_pair_tensor_symbolic(F_set)
    assert plaquette_pair_tensor_is_real_symmetric(T)


def test_a2u_projection_of_pair_tensor_is_zero() -> None:
    # Structural: projection is the zero map because plaquette rep is parity-even.
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    F_set = (sx, sy, sz, sx + sy, sy + sz, sz + sx)
    T = plaquette_pair_tensor_symbolic(F_set)
    projected = a2u_projection_of_pair_tensor(T)
    assert projected == sp.zeros(6, 6)


# =============================================================================
# SC-4d: gauge-group independence
# =============================================================================


def test_gauge_group_dimensions() -> None:
    assert gauge_group_dimension("SU2_L") == 2
    assert gauge_group_dimension("SU2_R") == 2
    assert gauge_group_dimension("SU3") == 3
    assert gauge_group_dimension("SU4_PS") == 4


def test_gauge_group_algebra_dimensions() -> None:
    # SU(N) has N² − 1 generators.
    assert gauge_group_algebra_dimension("SU2_L") == 3
    assert gauge_group_algebra_dimension("SU2_R") == 3
    assert gauge_group_algebra_dimension("SU3") == 8
    assert gauge_group_algebra_dimension("SU4_PS") == 15


def test_gauge_group_dimension_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        gauge_group_dimension("U1_em")  # type: ignore[arg-type]


def test_gauge_independence_for_su2_L() -> None:
    assert confirm_gauge_independence_symbolic("SU2_L")


def test_gauge_independence_for_su2_R() -> None:
    assert confirm_gauge_independence_symbolic("SU2_R")


def test_gauge_independence_for_su4_ps() -> None:
    assert confirm_gauge_independence_symbolic("SU4_PS")


# =============================================================================
# SC-4e: combined audit
# =============================================================================


def test_payload_all_consistent() -> None:
    p = lattice_topological_charge_payload()
    assert p.f_munu_extraction_implemented
    assert p.spatial_only_q_dimensionally_trivial
    assert p.plaquette_inversion_permutation_is_identity
    assert p.plaquette_rep_is_parity_even
    assert p.a2u_projection_of_pair_tensor_is_zero
    assert p.gauge_independence_su2_l
    assert p.gauge_independence_su2_r
    assert p.gauge_independence_su4_ps


def test_payload_verdict_is_confirms() -> None:
    p = lattice_topological_charge_payload()
    assert "SC-4 CONFIRMS" in p.final_verdict


def test_payload_interpretation_mentions_key_findings() -> None:
    p = lattice_topological_charge_payload()
    text = p.interpretation
    assert "dimensionally trivial" in text
    assert "parity-even" in text
    assert "Sym²" in text or "Sym2" in text or "g-irrep" in text
    assert "SU(2)_L" in text and "SU(4)_PS" in text
