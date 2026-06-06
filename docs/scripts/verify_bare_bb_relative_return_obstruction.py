"""Verify the bare BB relative-return obstruction.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md``.
It checks that the mixed-normal BB blocks have orthogonal one-tick leakage
norms but nonzero two-tick return to the diagonal scar:

    R_rel^(2) = M_-2 M_+2 + M_+2 M_-2

and that this return matrix is full rank.

Run:
    python docs/scripts/verify_bare_bb_relative_return_obstruction.py
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
    m_plus2 = block_sum(hops, ((1, -1, 1), (1, -1, -1)))
    m_minus2 = block_sum(hops, ((-1, 1, 1), (-1, 1, -1)))

    one_tick_norm = sp.simplify(m_plus2.H * m_plus2 + m_minus2.H * m_minus2)
    assert one_tick_norm == sp.eye(2) / 2
    cross_plus_minus = sp.simplify(m_plus2.H * m_minus2)
    cross_minus_plus = sp.simplify(m_minus2.H * m_plus2)
    assert cross_plus_minus == sp.zeros(2)
    assert cross_minus_plus == sp.zeros(2)

    return_plus_then_minus = sp.simplify(m_minus2 * m_plus2)
    return_minus_then_plus = sp.simplify(m_plus2 * m_minus2)
    relative_return = sp.simplify(return_plus_then_minus + return_minus_then_plus)
    expected = sp.diag(-(1 + I) / 4, -(1 - I) / 4)
    assert relative_return == expected
    assert return_plus_then_minus.rank() == 1
    assert return_minus_then_plus.rank() == 1
    assert relative_return.rank() == 2
    assert sp.simplify(relative_return.det() - sp.Rational(1, 8)) == 0

    # The return acts on both basis spinors, so it cannot be removed by a
    # nonzero spinor subspace.
    e0 = sp.Matrix([1, 0])
    e1 = sp.Matrix([0, 1])
    assert relative_return * e0 != sp.zeros(2, 1)
    assert relative_return * e1 != sp.zeros(2, 1)

    print("M_+2 =", m_plus2.tolist())
    print("M_-2 =", m_minus2.tolist())
    print("M_+2^dag M_-2 =", cross_plus_minus.tolist())
    print("M_-2^dag M_+2 =", cross_minus_plus.tolist())
    print("one-tick leakage norm =", one_tick_norm.tolist())
    print("M_-2 M_+2 =", return_plus_then_minus.tolist())
    print("M_+2 M_-2 =", return_minus_then_plus.tolist())
    print("R_rel^(2) =", relative_return.tolist())
    print("rank R_rel^(2) =", relative_return.rank())
    print("det R_rel^(2) =", relative_return.det())
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
