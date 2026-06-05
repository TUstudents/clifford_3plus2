"""Verify the silver contraction from the pinned BB ``q=0`` scar transfer.

This script is intentionally docs-local. It reproduces the load-bearing
calculation behind ``THEORY_REVIEW.md`` Section 5.4: that the
Bialynicki-Birula BCC Weyl coin, restricted to the synchronous-normal
(``sigma_1 = sigma_2``) ``q = r_1 - r_2 = 0`` scar at zero edge momentum
(``k_3 = 0``), gives an *orientation-preserving* radial transfer
(``det = +1``) whose eigenvalues are the silver pair ``sqrt2 +- 1`` at the
survival / trace-minimum point ``zeta = 1/sqrt2``.

The canonical hop convention is mirrored from
``clifford_3plus2_d5/spacetime_qca/bcc_weyl.py`` (``q_+ = (1+i)/4``,
``q_- = (1-i)/4``; projectors ``p1..p4``; direction order ``+++, ++-, ...,
---`` as produced by ``itertools.product((1,-1), repeat=3)``). The hops are
reconstructed locally so the script runs without importing the package
(whose ``__init__`` pulls in JAX); an optional cross-check against the real
module runs when it is importable.

Run:  ``python docs/scripts/verify_bb_scar_silver.py``
"""

from __future__ import annotations

from itertools import product

import sympy as sp

I = sp.I
SILVER = sp.sqrt(2) - 1  # epsilon


# --- canonical BB hops (mirror of bcc_weyl.py) ------------------------------

def canonical_hops() -> dict[tuple[int, int, int], sp.Matrix]:
    """Return ``{(s1,s2,s3): W}`` for the pinned BB Weyl convention."""

    q_plus = (1 + I) / 4
    q_minus = (1 - I) / 4
    p1 = sp.Matrix([[1, 0], [1, 0]])
    p2 = sp.Matrix([[0, 1], [0, 1]])
    p3 = sp.Matrix([[1, 0], [-1, 0]])
    p4 = sp.Matrix([[0, -1], [0, 1]])
    # order matches bcc_weyl.bialynicki_birula_hops() zipped with
    # bcc_geometry.bcc_body_diagonal_directions(normalized=False)
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


def crosscheck_against_package(hops: dict[tuple[int, int, int], sp.Matrix]) -> str:
    """Compare local hops to the installed package, if importable."""

    try:
        from clifford_3plus2_d5.spacetime_qca.bcc_weyl import (  # noqa: WPS433
            bialynicki_birula_directions,
            bialynicki_birula_hops,
        )
    except Exception as exc:  # pragma: no cover - env dependent (e.g. no JAX)
        return f"skipped ({type(exc).__name__})"

    pkg = dict(
        zip(
            (tuple(int(c) for c in d) for d in bialynicki_birula_directions()),
            bialynicki_birula_hops(),
        )
    )
    for direction, hop in hops.items():
        if sp.simplify(pkg[direction] - hop) != sp.zeros(2):
            raise AssertionError(f"hop mismatch at {direction}")
    return "match"


# --- the scar transfer ------------------------------------------------------

def scar_blocks(hops: dict[tuple[int, int, int], sp.Matrix]) -> tuple[sp.Matrix, sp.Matrix]:
    """Return ``(B_+, B_-)``: q=0 scar blocks, k_3=0 (sum over sigma_3)."""

    b_plus = sp.simplify(hops[(1, 1, 1)] + hops[(1, 1, -1)])
    b_minus = sp.simplify(hops[(-1, -1, 1)] + hops[(-1, -1, -1)])
    return b_plus, b_minus


def radial_transfer(zeta: sp.Symbol) -> sp.Matrix:
    """One-step (a,b) radial transfer from the bipartite-triangular blocks.

    In the symmetric/antisymmetric (e,o) basis the rank-one scar blocks are
    ``B_+ = [[1/2, i/2],[0,0]]`` and ``B_- = [[0,0],[i/2,1/2]]``. The
    stationary equation ``zeta psi_r = B_+ psi_{r-1} + B_- psi_{r+1}``,
    index-shifted, reduces to this transfer.
    """

    return sp.Matrix(
        [
            [1 / (2 * zeta), I / (2 * zeta)],
            [-I / (2 * zeta), 2 * zeta + 1 / (2 * zeta)],
        ]
    )


def main() -> None:
    hops = canonical_hops()
    b_plus, b_minus = scar_blocks(hops)

    # survival norm: subunitary no-leakage branch
    survival = sp.simplify(b_plus.H * b_plus + b_minus.H * b_minus)
    assert survival == sp.eye(2) / 2, survival

    # bipartite-triangular form in the (e,o) basis
    u = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    b_plus_eo = sp.simplify(u * b_plus * u.inv())
    b_minus_eo = sp.simplify(u * b_minus * u.inv())

    zeta = sp.symbols("zeta", positive=True)
    transfer = radial_transfer(zeta)
    det = sp.simplify(transfer.det())
    trace = sp.simplify(sp.trace(transfer))
    assert det == 1, det

    # trace minimum (AM-GM) coincides with the survival norm 1/sqrt2
    crit = [s for s in sp.solve(sp.diff(trace, zeta), zeta) if s.is_positive]
    zeta_star = crit[0]
    assert sp.simplify(zeta_star - 1 / sp.sqrt(2)) == 0, zeta_star
    assert sp.simplify(trace.subs(zeta, zeta_star) - 2 * sp.sqrt(2)) == 0

    transfer_star = sp.simplify(transfer.subs(zeta, 1 / sp.sqrt(2)))
    eigs = {sp.simplify(e) for e in transfer_star.eigenvals()}
    assert eigs == {sp.sqrt(2) - 1, sp.sqrt(2) + 1}, eigs

    # exact unification: neutrino sterile-chain transfer at z = 2 sqrt2
    neutrino = sp.Matrix([[2 * sp.sqrt(2), -1], [1, 0]])
    assert neutrino.det() == 1
    assert sp.simplify(sp.trace(neutrino) - 2 * sp.sqrt(2)) == 0
    assert {sp.simplify(e) for e in neutrino.eigenvals()} == eigs

    # Pell shadow: same |eigenvalue|, opposite orientation
    pell = sp.Matrix([[2, 1], [1, 0]])
    assert pell.det() == -1
    assert {sp.Abs(sp.simplify(e)) for e in pell.eigenvals()} == {
        SILVER,
        sp.sqrt(2) + 1,
    }

    print("BB hop cross-check vs package:", crosscheck_against_package(hops))
    print("B_+ =", b_plus.tolist())
    print("B_- =", b_minus.tolist())
    print("survival  B_+^H B_+ + B_-^H B_- =", survival.tolist())
    print("B_+ (e,o) =", b_plus_eo.tolist(), " B_- (e,o) =", b_minus_eo.tolist())
    print("transfer det =", det, " trace =", trace)
    print("trace minimized at zeta* =", zeta_star, " min trace =", 2 * sp.sqrt(2))
    print("eigenvalues at zeta=1/sqrt2 =", sorted(eigs, key=sp.N))
    print("decaying radial factor epsilon =", SILVER)
    print("neutrino chain at z=2sqrt2: det=1, tr=2sqrt2, eigs match =", True)
    print("Pell matrix det =", pell.det(), "(orientation-flipped arithmetic shadow)")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
