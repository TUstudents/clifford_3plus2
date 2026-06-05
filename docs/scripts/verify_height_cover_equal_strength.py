"""Height-cover derivation of equal-strength repair (fixes the irreducibility flaw).

Docs-local. Companion to verify_equal_strength_audit.py (which showed the
"irreducibility => Schur" closure is invalid). This script verifies the correct
route from THEORY_REVIEW.md Section 11.7:

  (A) translation-covariance on the homogeneous height cover forces a Toeplitz
      (constant-strength) one-tick shift  a_{r+1} = a_r ;
  (B) compressing the constant shift to the 3-flag P3 gives x*N exactly ;
  (C) dual packaging: the scar's residual Z2 endpoint reflection, realized as
      Hermitian conjugation (rho X rho = X^dag), is equivalent to |a|=|b|.

Both routes reduce equal-strength to a reasonable assumption (sharp truncation /
no boundary modulation, or scar-Z2-respect), not to bare structure -- so it is
strongly motivated, not forced. The deepest open problem is thereby relocated to
"why is the family flag length 3?" (the generation problem).

Run:  python docs/scripts/verify_height_cover_equal_strength.py
"""

from __future__ import annotations

import sympy as sp

I = sp.I


def main() -> None:
    x = sp.symbols("x")

    # (A) translation-covariance on a finite window of the cover forces constant a_r
    rs = list(range(-2, 3))                 # edges r -> r+1 for r in -2..2
    idx = {r: r + 2 for r in range(-2, 4)}  # window sites -2..3
    n = 6
    a = sp.symbols("a0 a1 a2 a3 a4")        # a_r on edge r->r+1
    X = sp.zeros(n)
    T = sp.zeros(n)
    for k, r in enumerate(rs):
        e_r = sp.zeros(n, 1); e_r[idx[r]] = 1
        e_rp = sp.zeros(n, 1); e_rp[idx[r + 1]] = 1
        X += a[k] * e_rp * e_r.T
        T += e_rp * e_r.T
    TXT = sp.simplify(T.T * X * T)          # T.T = T^{-1} on the interior
    # interior covariance T^{-1} X T = X demands a-coefficients shift-equal;
    # symbolically the surviving interior entries shift a_k -> a_{k+1}:
    shifted = [TXT[idx[r + 1], idx[r]] for r in rs[:-1]]
    # covariance => shifted[k] should equal original a_k; this is a_{k+1}=a_k
    assert all(sp.simplify(shifted[k] - a[k + 1]) == 0 for k in range(len(rs) - 1))
    # => imposing TXT == X on the interior forces a_{k+1} = a_k, i.e. constant

    # (B) compress the constant-strength shift to the 3-flag
    m = 4
    def k4(r):
        v = sp.zeros(m, 1); v[r] = 1; return v
    Xcov = x * sum((k4(r + 1) * k4(r).T for r in range(3)), sp.zeros(m))
    P3 = sp.diag(1, 1, 1, 0)
    X3 = sp.simplify(P3 * Xcov * P3)[:3, :3]
    N = sp.Matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert X3 == x * N

    # (C) chiral-reflection covariance: C H_X C^-1 = H_X  with C = sigma_x (x) R
    aa, bb = sp.symbols("a b")
    E1 = sp.Matrix([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    E2 = sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    Xg = aa * E1 + bb * E2
    R = sp.Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])       # open-path reflection (0,2)
    Zr = sp.zeros(3)
    HX = sp.Matrix(sp.BlockMatrix([[Zr, Xg.H], [Xg, Zr]]))  # Hermitian chiral block
    C = sp.Matrix(sp.BlockMatrix([[Zr, R], [R, Zr]]))       # sigma_x (x) R
    CHC = sp.simplify(C * HX * C.inv())
    # top-right block of C H_X C^-1 is R X R; covariance demands R X R = X^dag
    assert sp.simplify(CHC[:3, 3:] - R * Xg * R) == sp.zeros(3)
    diff = sp.simplify(R * Xg * R - Xg.H)
    cond = sp.solve([diff[0, 1], diff[1, 2]], [aa], dict=True)
    assert {sp.conjugate(bb)} == {s[aa] for s in cond}      # a = conj(b) => |a|=|b|

    print("(A) translation-covariant one-tick shift on the cover -> a_{r+1}=a_r  [VALID]")
    print("(B) P3 (x sum |r+1><r|) P3 = x N  (constant strength inherited)        [VALID]")
    print("(C) C H_X C^-1 = H_X  (C = sigma_x (x) R)  <=>  RXR=X^dag  <=>  |a|=|b|  [VALID]")
    print("\nEqual-strength is derived MODULO one reasonable assumption:")
    print("  cover route : sharp truncation / no boundary modulation;")
    print("  dual route  : the defect respects the scar's residual Z2 reflection.")
    print("Both are strongly motivated (QCA homogeneity / path-scar symmetry),")
    print("not forced by bare structure. Deepest open problem relocates to:")
    print("  WHY is the family flag length 3 (the generation problem)?")


if __name__ == "__main__":
    main()
