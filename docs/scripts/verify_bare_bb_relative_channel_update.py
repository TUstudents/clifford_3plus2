"""Verify the unprojected bare BB relative-channel update.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BARE_BB_RELATIVE_CHANNEL_UPDATE.md``.
It checks that after changing from two normal depths ``(r1,r2)`` to
trace/relative coordinates ``(r,q)``, the pinned BB edge update has the symbol

    U_rel(x,y) = x B_+ + x^-1 B_- + y M_+2 + y^-1 M_-2

and that this full unprojected symbol is unitary for independent unit-modulus
Bloch phases x and y. The projected visible branch and mixed relative branch
each carry exactly half the norm.

Run:
    python docs/scripts/verify_bare_bb_relative_channel_update.py
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
    transitions = {
        (1, 1): (1, 0, "B_+"),
        (-1, -1): (-1, 0, "B_-"),
        (1, -1): (0, 2, "M_+2"),
        (-1, 1): (0, -2, "M_-2"),
    }
    for (s1, s2), (delta_r, delta_q, _label) in transitions.items():
        assert delta_r == sp.Rational(s1 + s2, 2)
        assert delta_q == s1 - s2

    hops = canonical_hops()
    b_plus = block_sum(hops, ((1, 1, 1), (1, 1, -1)))
    b_minus = block_sum(hops, ((-1, -1, 1), (-1, -1, -1)))
    m_plus2 = block_sum(hops, ((1, -1, 1), (1, -1, -1)))
    m_minus2 = block_sum(hops, ((-1, 1, 1), (-1, 1, -1)))

    x, y = sp.symbols("x y", nonzero=True)
    u_rel = x * b_plus + x**-1 * b_minus + y * m_plus2 + y**-1 * m_minus2
    u_rel_dagger = (
        x**-1 * b_plus.H
        + x * b_minus.H
        + y**-1 * m_plus2.H
        + y * m_minus2.H
    )

    assert sp.simplify(u_rel_dagger * u_rel - sp.eye(2)) == sp.zeros(2)
    assert sp.simplify(u_rel * u_rel_dagger - sp.eye(2)) == sp.zeros(2)

    visible_symbol = x * b_plus + x**-1 * b_minus
    visible_dagger = x**-1 * b_plus.H + x * b_minus.H
    mixed_symbol = y * m_plus2 + y**-1 * m_minus2
    mixed_dagger = y**-1 * m_plus2.H + y * m_minus2.H

    visible_norm = sp.simplify(visible_dagger * visible_symbol)
    mixed_norm = sp.simplify(mixed_dagger * mixed_symbol)
    assert visible_norm == sp.eye(2) / 2
    assert mixed_norm == sp.eye(2) / 2
    assert sp.simplify(visible_norm + mixed_norm - sp.eye(2)) == sp.zeros(2)

    # The mixed relative channel has both shift directions with nonzero blocks,
    # so y is a genuine closed Bloch phase, not a retarded branch choice.
    assert m_plus2 != sp.zeros(2)
    assert m_minus2 != sp.zeros(2)
    assert m_plus2.rank() == m_minus2.rank() == 1

    print("normal-sign transitions =", transitions)
    print("B_+ =", b_plus.tolist())
    print("B_- =", b_minus.tolist())
    print("M_+2 =", m_plus2.tolist())
    print("M_-2 =", m_minus2.tolist())
    print("U_rel(x,y) =", u_rel.tolist())
    print("U_rel^dag U_rel = I")
    print("U_rel U_rel^dag = I")
    print("visible branch norm =", visible_norm.tolist())
    print("mixed relative branch norm =", mixed_norm.tolist())
    print("mixed shift ranks =", (m_plus2.rank(), m_minus2.rank()))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
