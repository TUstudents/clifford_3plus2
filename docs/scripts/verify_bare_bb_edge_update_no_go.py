"""Verify the bare BB edge-update no-go.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BARE_BB_EDGE_UPDATE_AUDIT.md``.
It checks that the pinned Bialynicki-Birula BCC Weyl coin, restricted to the
codimension-two edge at zero tangential momentum, sends a diagonal ``q=0``
state into same-normal and mixed-normal branches with exact norm split
``1/2 + 1/2`` and that no nonzero Weyl spinor kills both mixed-normal leakage
channels.

Run:
    python docs/scripts/verify_bare_bb_edge_update_no_go.py
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


def main() -> None:
    hops = canonical_hops()

    b_plus = block_sum(hops, ((1, 1, 1), (1, 1, -1)))
    b_minus = block_sum(hops, ((-1, -1, 1), (-1, -1, -1)))
    m_plus2 = block_sum(hops, ((1, -1, 1), (1, -1, -1)))
    m_minus2 = block_sum(hops, ((-1, 1, 1), (-1, 1, -1)))

    assert b_plus.rank() == b_minus.rank() == 1
    assert m_plus2.rank() == m_minus2.rank() == 1

    same_norm = sp.simplify(b_plus.H * b_plus + b_minus.H * b_minus)
    mixed_norm = sp.simplify(m_plus2.H * m_plus2 + m_minus2.H * m_minus2)
    assert same_norm == sp.eye(2) / 2
    assert mixed_norm == sp.eye(2) / 2
    assert sp.simplify(same_norm + mixed_norm - sp.eye(2)) == sp.zeros(2)

    kernel_plus = m_plus2.nullspace()
    kernel_minus = m_minus2.nullspace()
    assert len(kernel_plus) == len(kernel_minus) == 1

    v_plus = kernel_plus[0]
    v_minus = kernel_minus[0]
    assert sp.simplify(m_plus2 * v_plus) == sp.zeros(2, 1)
    assert sp.simplify(m_minus2 * v_minus) == sp.zeros(2, 1)

    # Distinct null lines imply zero intersection for two one-dimensional
    # subspaces in a two-dimensional spinor space.
    kernel_span_rank = sp.Matrix.hstack(v_plus, v_minus).rank()
    assert kernel_span_rank == 2

    # Solving both leakage equations directly confirms that the simultaneous
    # no-leakage spinor is trivial.
    psi0, psi1 = sp.symbols("psi0 psi1")
    psi = sp.Matrix([psi0, psi1])
    equations = list(m_plus2 * psi) + list(m_minus2 * psi)
    solution = sp.solve(equations, (psi0, psi1), dict=True)
    assert solution == [{psi0: 0, psi1: 0}]

    print("B_+ =", b_plus.tolist())
    print("B_- =", b_minus.tolist())
    print("M_+2 =", m_plus2.tolist())
    print("M_-2 =", m_minus2.tolist())
    print("same-normal norm =", same_norm.tolist())
    print("mixed-normal norm =", mixed_norm.tolist())
    print("total norm =", (same_norm + mixed_norm).tolist())
    print("ker M_+2 =", [v.tolist() for v in kernel_plus])
    print("ker M_-2 =", [v.tolist() for v in kernel_minus])
    print("kernel span rank =", kernel_span_rank)
    print("simultaneous no-leakage solution =", solution)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
