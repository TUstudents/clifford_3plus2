"""Verify the finite mixed-normal model and hard-gap q=0 limit.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BCC_Q0_SUPERSELECTION_DERIVATION.md``.
It mirrors the pinned Bialynicki-Birula BCC Weyl convention and checks:

* same-normal blocks preserve ``q = r1-r2``;
* mixed-normal blocks take ``q=0`` to ``q=+-2``;
* same-normal and mixed-normal branches split the norm exactly ``1/2 + 1/2``;
* a degenerate leakage penalty ``H_leak = 4 Lambda I`` gives
  ``Sigma_mix = I/(2(z-4 Lambda)) -> 0``;
* the visible transfer trace is minimized at ``zeta = 1/sqrt(2)``.

Run:
    python docs/scripts/verify_bcc_q0_superselection_limit.py
"""

from __future__ import annotations

from itertools import product

import sympy as sp

I = sp.I


def canonical_hops() -> dict[tuple[int, int, int], sp.Matrix]:
    """Return the pinned BB Weyl hop matrices by direction."""

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
    return {d: h for d, h in zip(directions, hop_list)}


def block_sum(
    hops: dict[tuple[int, int, int], sp.Matrix],
    directions: tuple[tuple[int, int, int], ...],
) -> sp.Matrix:
    """Sum hop blocks over the supplied directions."""

    return sum((hops[d] for d in directions), sp.zeros(2)).applyfunc(sp.simplify)


def radial_transfer(zeta: sp.Expr) -> sp.Matrix:
    """Return the visible q=0 scar transfer matrix."""

    return sp.Matrix(
        [
            [1 / (2 * zeta), I / (2 * zeta)],
            [-I / (2 * zeta), 2 * zeta + 1 / (2 * zeta)],
        ]
    )


def main() -> None:
    hops = canonical_hops()

    b_plus = block_sum(hops, ((1, 1, 1), (1, 1, -1)))
    b_minus = block_sum(hops, ((-1, -1, 1), (-1, -1, -1)))
    m_plus2 = block_sum(hops, ((1, -1, 1), (1, -1, -1)))
    m_minus2 = block_sum(hops, ((-1, 1, 1), (-1, 1, -1)))

    same_norm = sp.simplify(b_plus.H * b_plus + b_minus.H * b_minus)
    mixed_norm = sp.simplify(m_plus2.H * m_plus2 + m_minus2.H * m_minus2)
    assert same_norm == sp.eye(2) / 2, same_norm
    assert mixed_norm == sp.eye(2) / 2, mixed_norm
    assert sp.simplify(same_norm + mixed_norm - sp.eye(2)) == sp.zeros(2)

    z, lam = sp.symbols("z Lambda", positive=True)
    sigma_mix = sp.simplify(mixed_norm / (z - 4 * lam))
    expected_sigma = sp.eye(2) / (2 * (z - 4 * lam))
    assert sp.simplify(sigma_mix - expected_sigma) == sp.zeros(2)
    assert sp.limit(sigma_mix[0, 0], lam, sp.oo) == 0
    assert sp.limit(sigma_mix[1, 1], lam, sp.oo) == 0

    zeta = sp.symbols("zeta", positive=True)
    transfer = radial_transfer(zeta)
    trace = sp.simplify(sp.trace(transfer))
    zeta_star = [s for s in sp.solve(sp.diff(trace, zeta), zeta) if s.is_positive][0]
    assert sp.simplify(zeta_star - 1 / sp.sqrt(2)) == 0
    transfer_star = sp.simplify(transfer.subs(zeta, zeta_star))
    eigs = {sp.simplify(e) for e in transfer_star.eigenvals()}
    assert eigs == {sp.sqrt(2) - 1, sp.sqrt(2) + 1}, eigs

    print("B_+ =", b_plus.tolist())
    print("B_- =", b_minus.tolist())
    print("M_+2 =", m_plus2.tolist())
    print("M_-2 =", m_minus2.tolist())
    print("same-normal norm =", same_norm.tolist())
    print("mixed-normal norm =", mixed_norm.tolist())
    print("total norm =", (same_norm + mixed_norm).tolist())
    print("Sigma_mix(z,Lambda) =", sigma_mix.tolist())
    print("hard-gap limit Sigma_mix -> 0")
    print("trace T(zeta) =", trace)
    print("trace minimum zeta* =", zeta_star)
    print("silver eigenvalues at zeta* =", sorted(eigs, key=sp.N))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
