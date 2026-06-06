"""Verify the microscopic BB-QCA edge update specification.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/MICROSCOPIC_BB_QCA_EDGE_UPDATE.md``.
It checks the finite algebra of the complete local scattering rule:

* exact BB same-normal and mixed-normal edge blocks;
* norm split ``1/2 I + 1/2 I``;
* local edge scattering isometry and finite unitary completion;
* retarded block-triangular compression has visible powers ``A**t``;
* recurrent wedge control restores the nonzero two-step return;
* visible survival transfer has silver eigenvalues at ``zeta=1/sqrt(2)``.

Run:
    python docs/scripts/verify_microscopic_bb_qca_edge_update.py
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


def radial_transfer(zeta: sp.Expr) -> sp.Matrix:
    return sp.Matrix(
        [
            [1 / (2 * zeta), I / (2 * zeta)],
            [-I / (2 * zeta), 2 * zeta + 1 / (2 * zeta)],
        ]
    )


def top_left(matrix: sp.Matrix, visible_dim: int = 2) -> sp.Matrix:
    return matrix[:visible_dim, :visible_dim]


def main() -> None:
    hops = canonical_hops()

    b_plus = block_sum(hops, ((1, 1, 1), (1, 1, -1)))
    b_minus = block_sum(hops, ((-1, -1, 1), (-1, -1, -1)))
    m_plus2 = block_sum(hops, ((1, -1, 1), (1, -1, -1)))
    m_minus2 = block_sum(hops, ((-1, 1, 1), (-1, 1, -1)))

    visible_norm = sp.simplify(b_plus.H * b_plus + b_minus.H * b_minus)
    mixed_norm = sp.simplify(m_plus2.H * m_plus2 + m_minus2.H * m_minus2)
    assert visible_norm == sp.eye(2) / 2
    assert mixed_norm == sp.eye(2) / 2
    assert sp.simplify(visible_norm + mixed_norm - sp.eye(2)) == sp.zeros(2)

    edge_isometry = sp.Matrix.vstack(b_plus, b_minus, m_plus2, m_minus2)
    assert edge_isometry.shape == (8, 2)
    assert sp.simplify(edge_isometry.H * edge_isometry - sp.eye(2)) == sp.zeros(2)

    local_unitary = complete_isometry_to_unitary(edge_isometry)
    assert local_unitary.shape == (8, 8)
    assert sp.simplify(local_unitary.H * local_unitary - sp.eye(8)) == sp.zeros(8)
    assert sp.simplify(local_unitary[:, :2] - edge_isometry) == sp.zeros(8, 2)

    a_visible = sp.simplify(b_plus + b_minus)
    hidden_dim = 8
    e_emit = sp.Matrix.vstack(m_plus2, m_minus2, sp.zeros(4, 2))
    outgoing_shift = sp.zeros(hidden_dim)
    outgoing_shift[4:8, 0:4] = sp.eye(4)
    retarded_update = sp.Matrix.vstack(
        sp.Matrix.hstack(a_visible, sp.zeros(2, hidden_dim)),
        sp.Matrix.hstack(e_emit, outgoing_shift),
    )

    for power in range(1, 6):
        assert sp.simplify(top_left(retarded_update**power) - a_visible**power) == sp.zeros(2)

    g_return = sp.Matrix.hstack(m_minus2, m_plus2, sp.zeros(2, 4))
    closed_update = sp.Matrix.vstack(
        sp.Matrix.hstack(a_visible, g_return),
        sp.Matrix.hstack(e_emit, outgoing_shift),
    )
    bare_return = sp.simplify(m_minus2 * m_plus2 + m_plus2 * m_minus2)
    expected_bare_return = sp.diag(-(1 + I) / 4, -(1 - I) / 4)
    assert bare_return == expected_bare_return
    assert bare_return.rank() == 2
    assert sp.simplify(top_left(closed_update**2) - a_visible**2) == bare_return

    zeta = sp.symbols("zeta", positive=True)
    transfer = radial_transfer(zeta)
    trace = sp.simplify(sp.trace(transfer))
    assert sp.simplify(trace - (2 * zeta + 1 / zeta)) == 0
    zeta_star = sp.sqrt(2) / 2
    eigs = {sp.simplify(eigenvalue) for eigenvalue in transfer.subs(zeta, zeta_star).eigenvals()}
    assert eigs == {sp.sqrt(2) - 1, sp.sqrt(2) + 1}

    print("B_+ =", b_plus.tolist())
    print("B_- =", b_minus.tolist())
    print("M_+2 =", m_plus2.tolist())
    print("M_-2 =", m_minus2.tolist())
    print("visible norm =", visible_norm.tolist())
    print("mixed norm =", mixed_norm.tolist())
    print("C_edge^dag C_edge = I")
    print("local unitary completion shape =", local_unitary.shape)
    print("retarded visible powers: P T_R^t iota = A^t for t=1..5")
    print("bare recurrent correction =", bare_return.tolist())
    print("transfer trace =", trace)
    print("silver eigenvalues =", sorted(eigs, key=sp.N))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
