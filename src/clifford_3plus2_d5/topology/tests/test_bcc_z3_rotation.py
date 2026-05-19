"""Tests for ``bcc_z3_rotation.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.topology.bcc_z3_rotation import (
    apply_rotation_to_direction,
    bcc_direction_permutation,
    body_diagonal_rotation_matrix,
    cycle_lengths,
    dirac_spinor_lift,
    dirac_spinor_lift_4d,
    dirac_spinor_lift_cube,
    dirac_spinor_lift_cubes_to_minus_identity,
    dirac_spinor_lift_is_su2,
    permutation_cycles,
)


def test_rotation_matrix_is_order_three() -> None:
    R = body_diagonal_rotation_matrix()
    assert (R**3 - sp.eye(3)).applyfunc(sp.simplify) == sp.zeros(3)


def test_rotation_matrix_is_orthogonal_and_determinant_one() -> None:
    R = body_diagonal_rotation_matrix()
    assert (R.T * R - sp.eye(3)).applyfunc(sp.simplify) == sp.zeros(3)
    assert R.det() == 1


def test_rotation_fixes_diagonal_corners() -> None:
    plus = apply_rotation_to_direction((1, 1, 1))
    minus = apply_rotation_to_direction((-1, -1, -1))
    assert plus == (1, 1, 1)
    assert minus == (-1, -1, -1)


def test_permutation_has_cycle_structure_one_one_three_three() -> None:
    assert cycle_lengths() == (3, 3, 1, 1)


def test_permutation_total_length_is_eight() -> None:
    perm = bcc_direction_permutation()
    assert len(perm) == 8
    # Verify it IS a permutation (image hits each index exactly once)
    assert sorted(perm) == list(range(8))


def test_cycles_partition_eight_indices() -> None:
    cycles = permutation_cycles()
    union: set[int] = set()
    for cycle in cycles:
        union.update(cycle)
    assert union == set(range(8))


def test_dirac_spinor_lift_is_su2() -> None:
    assert dirac_spinor_lift_is_su2()


def test_dirac_spinor_lift_cubes_to_minus_identity() -> None:
    # SO(3) order-3 rotation lifts to SU(2) element with U^3 = -I.
    assert dirac_spinor_lift_cubes_to_minus_identity()


def test_dirac_spinor_lift_cube_explicitly() -> None:
    cube = dirac_spinor_lift_cube()
    assert (cube + sp.eye(2)).applyfunc(sp.simplify) == sp.zeros(2)


def test_dirac_4d_lift_is_block_diagonal() -> None:
    u4 = dirac_spinor_lift_4d()
    u2 = dirac_spinor_lift()
    assert u4.shape == (4, 4)
    # Upper-left 2x2 block should equal u2.
    upper = u4[:2, :2]
    lower = u4[2:, 2:]
    assert (upper - u2).applyfunc(sp.simplify) == sp.zeros(2)
    assert (lower - u2).applyfunc(sp.simplify) == sp.zeros(2)
    # Off-diagonal blocks zero.
    assert u4[:2, 2:].applyfunc(sp.simplify) == sp.zeros(2)
    assert u4[2:, :2].applyfunc(sp.simplify) == sp.zeros(2)
