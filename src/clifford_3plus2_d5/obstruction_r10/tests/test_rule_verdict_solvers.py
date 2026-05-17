import sympy as sp

from clifford_3plus2_d5.obstruction_r10.qca.rule_verdict import (
    center_basis_of_generated_algebra,
    solve_central_idempotent_rank_profile,
    solve_central_idempotents,
)


def test_coordinate_idempotent_solver_handles_mixed_center_basis() -> None:
    p2a = sp.diag(1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    p2b = sp.diag(0, 0, 1, 1, 0, 0, 0, 0, 0, 0)
    p2c = sp.diag(0, 0, 0, 0, 1, 1, 0, 0, 0, 0)
    p4 = sp.diag(0, 0, 0, 0, 0, 0, 1, 1, 1, 1)
    center_basis = (
        sp.eye(10),
        p2a + 2 * p2b,
        p2b - p2c,
        p2a + p4,
    )

    solved, idempotents = solve_central_idempotents(center_basis, dimension=10)

    assert solved
    assert sorted(item.rank for item in idempotents) == [
        0,
        2,
        2,
        2,
        4,
        4,
        4,
        4,
        6,
        6,
        6,
        6,
        8,
        8,
        8,
        10,
    ]


def test_generated_center_uses_generators_inside_known_algebra_basis() -> None:
    left = sp.diag(1, -1)
    swap = sp.Matrix(
        [
            [0, 1],
            [1, 0],
        ]
    )
    algebra_basis = (sp.eye(2), left, swap, left * swap)

    center = center_basis_of_generated_algebra(algebra_basis, (left, swap), dimension=2)

    assert len(center) == 1
    assert center[0] == sp.eye(2)


def test_rank_profile_solver_fast_rejects_rank_2_8_involution() -> None:
    central_involution = sp.diag(1, 1, 1, 1, 1, 1, 1, 1, -1, -1)

    solved, ranks = solve_central_idempotent_rank_profile(
        (sp.eye(10), central_involution),
        dimension=10,
    )

    assert solved
    assert ranks == (0, 2, 8, 10)
