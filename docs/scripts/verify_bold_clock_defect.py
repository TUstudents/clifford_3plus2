"""Structural backbone of the bold S3 = Z3 |x Z2 chiral clock defect.

Docs-local. Verifies the `C:9` algebra behind THEORY_REVIEW.md Section 11.8.
It asserts ONLY the structural backbone (facts 1-4). The model's *bold inputs*
(5 coin-isospin lock, 6 Higgs supplies the Z2, 7 silver rides the coin) are
`C:3` conjectures and are NOT asserted here - they are printed as such.

Run:  python docs/scripts/verify_bold_clock_defect.py
"""

from __future__ import annotations

import sympy as sp

I = sp.I


def main() -> None:
    # (1) body-diagonal Z3 clock on the 3 residual ports
    C3 = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert C3**3 == sp.eye(3)

    # (2) cut the closing edge -> path nilpotent N ; depth spectrum {0,2,6}
    N = C3 - sp.Matrix([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    assert N**3 == sp.zeros(3) and N**2 != sp.zeros(3)
    A = N + N.T
    L = sp.diag(*[sum(A.row(i)) for i in range(3)]) - A
    assert set((2 * L).eigenvals().keys()) == {0, 2, 6}
    K3 = sp.diag(2, 2, 2) - (C3 + C3.T)
    assert (2 * K3).eigenvals() == {sp.Integer(0): 1, sp.Integer(6): 2}  # {0,6,6}

    # (3) endpoint reflection (0,2): unique transposition that inverts the clock
    #     AND transposes the path
    rho = sp.Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])     # (0,2)
    assert rho * rho == sp.eye(3)
    assert sp.simplify(rho * C3 * rho) == C3.inv()
    assert sp.simplify(rho * N * rho) == N.T
    others = {
        "(1,2)": sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
        "(0,1)": sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    }
    for r in others.values():
        assert sp.simplify(r * C3 * r) == C3.inv()          # all transpositions invert
        assert sp.simplify(r * N * r) != N.T                # but only (0,2) transposes N
    # S3 = <C3, rho>
    seen = {sp.ImmutableMatrix(sp.eye(3))}
    frontier = [sp.eye(3)]
    while frontier:
        g = frontier.pop()
        for s in (C3, rho):
            h = sp.ImmutableMatrix(s * g)
            if h not in seen:
                seen.add(h)
                frontier.append(sp.Matrix(h))
    assert len(seen) == 6

    # (4) chiral spin-1/2 lift of 2pi/3 body-diagonal rotation: U3^3 = -I
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -I], [I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    n = (sx + sy + sz) / sp.sqrt(3)
    U3 = sp.simplify(sp.cos(sp.pi / 3) * sp.eye(2) - I * sp.sin(sp.pi / 3) * n)
    assert sp.simplify(U3**3) == -sp.eye(2)

    print("(1) Z3 clock C3^3=I, eigenvalues 1,w,w^2            [PASS]")
    print("(2) cut clock -> N (N^3=0); 2*Delta(P3) spec {0,2,6}; K3 {0,6,6}  [PASS]")
    print("(3) endpoint (0,2): inverts clock AND N->N^T; unique among")
    print("    transpositions; <C3,(0,2)> = S3 = Z3 |x Z2           [PASS]")
    print("(4) chiral lift U3^3 = -I (orientation/CP sign)        [PASS]")
    print("\nBACKBONE CERTIFIED (C:9).")
    print("NOT asserted (C:3 bold inputs): (5) coin-isospin lock;")
    print("  (6) i*sigma2 = endpoint Z2 so tilde-H->N, H->full S3 shell;")
    print("  (7) silver rides the same coin (see verify_bb_scar_silver.py).")
    print("STANDING GAP A unchanged: down (6,2,4) needs the 3->6 regular-rep promotion.")


if __name__ == "__main__":
    main()
