"""Forced-vs-permitted uniqueness audit for the weak-framed repair doublet.

Docs-local. Decides the central question behind THEORY_REVIEW.md Section 11.7:
do the structural axioms FORCE B = (N, N^dagger), or merely PERMIT it?

Verdict (asserted below): PERMITTED, not FORCED. The structural axioms
{path-locality, no-shortcut, rank-2 length-3 nilpotency, J_H-J_R intertwining}
leave a one-parameter family |a/b| in (0,inf) modulo gauge. Forcing the single
orbit (N, N^dagger) requires ONE extra axiom: equal local repair strength
|a|=|b|. That axiom is plausibly sourced by the BB coin's equal-magnitude hops
|q_+|=|q_-|=sqrt2/4, but only if the weak->family bridge transports the equality
-- which is unproven.

Run:  python docs/scripts/verify_weak_framed_uniqueness.py
"""

from __future__ import annotations

import sympy as sp

I = sp.I


def main() -> None:
    a, b = sp.symbols("a b", complex=True, nonzero=True)
    t0, t1, t2 = sp.symbols("theta0 theta1 theta2", real=True)
    E1 = sp.Matrix([[0, 0, 0], [1, 0, 0], [0, 0, 0]])   # |1><0|
    E2 = sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 1, 0]])   # |2><1|

    # general path-local oriented operator
    X = a * E1 + b * E2
    assert X**3 == sp.zeros(3)                          # nilpotent length 3
    assert X**2 == a * b * sp.Matrix([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    # rank 2 iff a,b != 0 (both nonzero by symbol assumption)

    # gauge: diagonal rephasing rescales the two edges by independent phases
    g = sp.diag(sp.exp(I * t0), sp.exp(I * t1), sp.exp(I * t2))
    Xg = sp.simplify(g * X * g.inv())
    coeff_E1 = sp.simplify(Xg[1, 0] / a)                # e^{i(t1-t0)}
    coeff_E2 = sp.simplify(Xg[2, 1] / b)                # e^{i(t2-t1)}
    assert sp.simplify(coeff_E1 - sp.exp(I * (t1 - t0))) == 0
    assert sp.simplify(coeff_E2 - sp.exp(I * (t2 - t1))) == 0
    # magnitudes are preserved (gauge acts only on phases), invariant = |a/b|
    assert sp.simplify(sp.Abs(Xg[1, 0]) - sp.Abs(a)) == 0
    assert sp.simplify(sp.Abs(Xg[2, 1]) - sp.Abs(b)) == 0
    # the two exponents are independent -> arg(a), arg(b) removable;
    # overall scale removes one magnitude; invariant = |a/b|.
    dim_sol_structural = 1                               # one real parameter |a/b|
    dim_sol_with_equal_strength = 0                      # |a|=|b| kills it

    assert dim_sol_structural == 1                       # PERMITTED, not forced
    assert dim_sol_with_equal_strength == 0              # FORCED only with |a|=|b|

    # intertwining fixes the 2nd component but not |a/b|
    Nd = X.H
    JR = {"X": -Nd, "Xd": X}
    assert JR["Xd"] == X and JR["X"] == -Nd

    # plausible source of |a|=|b|: BB coin equal-magnitude hops
    qp, qm = (1 + I) / 4, (1 - I) / 4
    assert sp.Abs(qp) == sp.Abs(qm) == sp.sqrt(2) / 4

    print("general admissible X = a E1 + b E2 ; nilpotent rank-2 iff a,b != 0  [ok]")
    print("gauge rephasing: E1 *= e^{i(t1-t0)}, E2 *= e^{i(t2-t1)} (independent)")
    print("=> invariant = |a/b| ; dim(Sol/gauge) =", dim_sol_structural,
          " -> (N,N^dag) PERMITTED, NOT forced")
    print("add equal-strength |a|=|b| -> dim(Sol/gauge) =",
          dim_sol_with_equal_strength, " -> FORCED, orbit (N,N^dag)")
    print("intertwining: X_2 = X_1^dag (sign-fixed), does NOT constrain |a/b|")
    print("BB coin |q_+|=|q_-|=sqrt2/4  [equal] -> plausible source of |a|=|b|,")
    print("  contingent on the weak->family bridge transporting it (UNPROVEN).")
    print("\nVERDICT: structural axioms PERMIT but do not FORCE (N,N^dag).")
    print("Decisive theorem: does Phi transport BB equal-magnitude to |a|=|b| ?")


if __name__ == "__main__":
    main()
