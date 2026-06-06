"""Verify minimality of the edge-clock scattering completion.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/EDGE_CLOCK_SCATTERING_MINIMALITY.md``.
It checks:

* the visible same-normal BB output has norm ``1/2 I``;
* any hidden output completing it must have rank two;
* the BB mixed-normal output supplies exactly the missing norm;
* the orientation-resolved clock-error sector is the BB mixed sector stacked as
  ``(M_+2, M_-2)``;
* retarded no-incoming hidden maps remove the first return.

Run:
    python docs/scripts/verify_edge_clock_scattering_minimality.py
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

    visible = sp.Matrix.vstack(b_plus, b_minus)
    mixed = sp.Matrix.vstack(m_plus2, m_minus2)
    edge = sp.Matrix.vstack(visible, mixed)

    visible_norm = sp.simplify(visible.H * visible)
    mixed_norm = sp.simplify(mixed.H * mixed)
    assert visible_norm == sp.eye(2) / 2
    assert mixed_norm == sp.eye(2) / 2
    assert sp.simplify(edge.H * edge - sp.eye(2)) == sp.zeros(2)

    missing_norm = sp.simplify(sp.eye(2) - visible_norm)
    assert missing_norm == sp.eye(2) / 2
    assert missing_norm.rank() == 2

    # A hidden output H must satisfy H^dag H = missing_norm. Since the right
    # side has rank two, H must have at least two rows/channels.
    h0, h1 = sp.symbols("h0 h1")
    one_channel = sp.Matrix([[h0, h1]])
    one_channel_norm = one_channel.H * one_channel
    assert one_channel_norm.rank() <= 1
    assert one_channel_norm.rank() < missing_norm.rank()

    assert mixed.rank() == 2
    assert m_plus2.rank() == 1
    assert m_minus2.rank() == 1
    assert sp.simplify(m_plus2.H * m_minus2) == sp.zeros(2)
    assert sp.simplify(m_minus2.H * m_plus2) == sp.zeros(2)

    bare_return = sp.simplify(m_minus2 * m_plus2 + m_plus2 * m_minus2)
    assert bare_return == sp.diag(-(1 + I) / 4, -(1 - I) / 4)

    g_plus = sp.zeros(2)
    g_minus = sp.zeros(2)
    retarded_return = sp.simplify(g_plus * m_plus2 + g_minus * m_minus2)
    assert retarded_return == sp.zeros(2)

    print("visible norm V^dag V =", visible_norm.tolist())
    print("missing norm I - V^dag V =", missing_norm.tolist())
    print("rank missing norm =", missing_norm.rank())
    print("one hidden scalar channel rank <=", one_channel_norm.rank())
    print("mixed norm M^dag M =", mixed_norm.tolist())
    print("rank mixed output =", mixed.rank())
    print("rank M_+2, M_-2 =", (m_plus2.rank(), m_minus2.rank()))
    print("one-tick clock-error cross terms vanish")
    print("bare return =", bare_return.tolist())
    print("retarded return =", retarded_return.tolist())
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
