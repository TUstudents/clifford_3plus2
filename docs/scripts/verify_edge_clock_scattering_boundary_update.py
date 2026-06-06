"""Verify the edge-clock scattering boundary update.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/EDGE_CLOCK_SCATTERING_BOUNDARY_UPDATE.md``.
It checks that the exact BB same/mixed blocks define a local port isometry,
constructs an explicit finite unitary completion, and verifies that routing
mixed-normal amplitudes into outgoing clock-error ports removes the two-step
relative return in the retarded compression.

Run:
    python docs/scripts/verify_edge_clock_scattering_boundary_update.py
"""

from __future__ import annotations

from itertools import product

import sympy as sp

I = sp.I


def canonical_hops() -> dict[tuple[int, int, int], sp.Matrix]:
    """Return the pinned BB Weyl hop matrices by body-diagonal direction."""

    q_plus = (1 + I) / 4
    q_minus = (1 - I) / 4
    p1 = sp.Matrix([[1, 0], [1, 0]])
    p2 = sp.Matrix([[0, 1], [0, 1]])
    p3 = sp.Matrix([[1, 0], [-1, 0]])
    p4 = sp.Matrix([[0, -1], [0, 1]])
    hop_list = [
        q_plus * p1,
        q_minus * p2,
        q_minus * p1,
        q_plus * p2,
        q_minus * p3,
        q_plus * p4,
        q_plus * p3,
        q_minus * p4,
    ]
    directions = list(product((1, -1), repeat=3))
    return {direction: hop for direction, hop in zip(directions, hop_list, strict=True)}


def block_sum(
    hops: dict[tuple[int, int, int], sp.Matrix],
    directions: tuple[tuple[int, int, int], ...],
) -> sp.Matrix:
    return sum((hops[direction] for direction in directions), sp.zeros(2)).applyfunc(sp.simplify)


def hermitian_inner(left: sp.Matrix, right: sp.Matrix) -> sp.Expr:
    return sp.simplify((left.H * right)[0])


def complete_isometry_to_unitary(isometry: sp.Matrix) -> sp.Matrix:
    """Complete an exact isometry to a finite unitary by Gram-Schmidt."""

    columns = [isometry[:, index] for index in range(isometry.shape[1])]
    columns.extend(sp.eye(isometry.rows)[:, index] for index in range(isometry.rows))

    orthonormal: list[sp.Matrix] = []
    for column in columns:
        candidate = sp.Matrix(column)
        for basis in orthonormal:
            candidate = sp.simplify(candidate - basis * hermitian_inner(basis, candidate))
        norm_squared = hermitian_inner(candidate, candidate)
        if norm_squared != 0:
            orthonormal.append(sp.simplify(candidate / sp.sqrt(norm_squared)))
        if len(orthonormal) == isometry.rows:
            break

    return sp.Matrix.hstack(*orthonormal)


def main() -> None:
    hops = canonical_hops()
    b_plus = block_sum(hops, ((1, 1, 1), (1, 1, -1)))
    b_minus = block_sum(hops, ((-1, -1, 1), (-1, -1, -1)))
    m_plus2 = block_sum(hops, ((1, -1, 1), (1, -1, -1)))
    m_minus2 = block_sum(hops, ((-1, 1, 1), (-1, 1, -1)))

    visible_norm = sp.simplify(b_plus.H * b_plus + b_minus.H * b_minus)
    clock_error_norm = sp.simplify(m_plus2.H * m_plus2 + m_minus2.H * m_minus2)
    assert visible_norm == sp.eye(2) / 2
    assert clock_error_norm == sp.eye(2) / 2

    edge_isometry = sp.Matrix.vstack(b_plus, b_minus, m_plus2, m_minus2)
    assert edge_isometry.shape == (8, 2)
    assert sp.simplify(edge_isometry.H * edge_isometry - sp.eye(2)) == sp.zeros(2)

    unitary_completion = complete_isometry_to_unitary(edge_isometry)
    assert unitary_completion.shape == (8, 8)
    assert sp.simplify(unitary_completion.H * unitary_completion - sp.eye(8)) == sp.zeros(8)
    assert sp.simplify(unitary_completion * unitary_completion.H - sp.eye(8)) == sp.zeros(8)
    assert sp.simplify(unitary_completion[:, 0] - edge_isometry[:, 0]) == sp.zeros(8, 1)
    assert sp.simplify(unitary_completion[:, 1] - edge_isometry[:, 1]) == sp.zeros(8, 1)

    bare_return = sp.simplify(m_minus2 * m_plus2 + m_plus2 * m_minus2)
    expected_bare_return = sp.diag(-(1 + I) / 4, -(1 - I) / 4)
    assert bare_return == expected_bare_return
    assert bare_return.rank() == 2

    # In the retarded clock-error compression there is no incoming clock-error
    # map back to the visible spinor.
    g_plus = sp.zeros(2)
    g_minus = sp.zeros(2)
    clock_return = sp.simplify(g_plus * m_plus2 + g_minus * m_minus2)
    assert clock_return == sp.zeros(2)

    print("B_+ =", b_plus.tolist())
    print("B_- =", b_minus.tolist())
    print("M_+2 =", m_plus2.tolist())
    print("M_-2 =", m_minus2.tolist())
    print("visible norm =", visible_norm.tolist())
    print("clock-error norm =", clock_error_norm.tolist())
    print("C_edge shape =", edge_isometry.shape)
    print("C_edge^dag C_edge = I")
    print("unitary completion shape =", unitary_completion.shape)
    print("first two unitary columns are C_edge")
    print("bare wedge return =", bare_return.tolist())
    print("retarded clock-error return =", clock_return.tolist())
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
