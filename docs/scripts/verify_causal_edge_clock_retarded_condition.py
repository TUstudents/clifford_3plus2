"""Verify the causal edge-clock retarded condition.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md``.
It checks the exact block algebra behind the note:

* a retarded lower-triangular update ``[[A, 0], [E, S]]`` has visible powers
  exactly ``A**t``;
* adding a hidden-to-visible return block ``G`` produces the two-step term
  ``G E``;
* with the BB mixed-normal blocks, ``G E`` is exactly the bare relative-return
  obstruction.

Run:
    python docs/scripts/verify_causal_edge_clock_retarded_condition.py
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


def top_left(matrix: sp.Matrix, visible_dim: int = 2) -> sp.Matrix:
    return matrix[:visible_dim, :visible_dim]


def top_right(matrix: sp.Matrix, visible_dim: int = 2) -> sp.Matrix:
    return matrix[:visible_dim, visible_dim:]


def main() -> None:
    hops = canonical_hops()
    b_plus = block_sum(hops, ((1, 1, 1), (1, 1, -1)))
    b_minus = block_sum(hops, ((-1, -1, 1), (-1, -1, -1)))
    m_plus2 = block_sum(hops, ((1, -1, 1), (1, -1, -1)))
    m_minus2 = block_sum(hops, ((-1, 1, 1), (-1, 1, -1)))

    # A finite local representative of the visible branch. The triangular
    # theorem is independent of this choice; using the BB same-normal sum keeps
    # the certificate tied to the boundary blocks.
    a_visible = sp.simplify(b_plus + b_minus)

    # Hidden space is two orientation-resolved clock-error spinor ports plus a
    # one-step downstream buffer. E emits into the first four hidden components.
    hidden_dim = 8
    e_emit = sp.Matrix.vstack(m_plus2, m_minus2, sp.zeros(4, 2))
    outgoing_shift = sp.zeros(hidden_dim)
    outgoing_shift[4:8, 0:4] = sp.eye(4)

    zero_vh = sp.zeros(2, hidden_dim)
    retarded_update = sp.Matrix.vstack(
        sp.Matrix.hstack(a_visible, zero_vh),
        sp.Matrix.hstack(e_emit, outgoing_shift),
    )

    for power in range(1, 6):
        retarded_power = sp.simplify(retarded_update**power)
        assert sp.simplify(top_left(retarded_power) - a_visible**power) == sp.zeros(2)
        assert top_right(retarded_power) == sp.zeros(2, hidden_dim)

    # The closed recurrent control maps chi_+ back with M_-2 and chi_- back
    # with M_+2, reproducing the bare q=+-2 -> q=0 return.
    g_return = sp.Matrix.hstack(m_minus2, m_plus2, sp.zeros(2, 4))
    closed_update = sp.Matrix.vstack(
        sp.Matrix.hstack(a_visible, g_return),
        sp.Matrix.hstack(e_emit, outgoing_shift),
    )

    bare_return = sp.simplify(m_minus2 * m_plus2 + m_plus2 * m_minus2)
    expected_bare_return = sp.diag(-(1 + I) / 4, -(1 - I) / 4)
    assert bare_return == expected_bare_return
    assert bare_return.rank() == 2

    closed_two_step = sp.simplify(closed_update**2)
    recurrent_correction = sp.simplify(top_left(closed_two_step) - a_visible**2)
    assert recurrent_correction == bare_return
    assert recurrent_correction != sp.zeros(2)

    retarded_two_step = sp.simplify(retarded_update**2)
    retarded_correction = sp.simplify(top_left(retarded_two_step) - a_visible**2)
    assert retarded_correction == sp.zeros(2)

    print("A visible =", a_visible.tolist())
    print("E emit shape =", e_emit.shape)
    print("outgoing shift nilpotent S^2 =", (outgoing_shift**2).tolist())
    print("retarded visible powers: P T_R^t iota = A^t for t=1..5")
    print("retarded two-step correction =", retarded_correction.tolist())
    print("bare wedge return G E =", bare_return.tolist())
    print("closed recurrent two-step correction =", recurrent_correction.tolist())
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
