"""Verify the relative-depth gap classification.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BCC_RELATIVE_DEPTH_ORDER_PARAMETER.md``.
It checks that reflection symmetry forces an even local potential in
``q = r1-r2``, that the lowest nontrivial term gives the adjacent leakage gap
``V(+-2)-V(0)=4 Lambda`` at quadratic order, and that this reproduces the
mixed-normal Schur feedback used in the q=0 superselection note.

Run:
    python docs/scripts/verify_relative_depth_gap_classification.py
"""

from __future__ import annotations

import sympy as sp


def main() -> None:
    q, z, lam, kappa = sp.symbols("q z Lambda kappa", positive=True)
    odd = sp.symbols("odd")

    generic = odd * q + lam * q**2 + kappa * q**4
    reflection_difference = sp.expand(generic.subs(q, -q) - generic)
    odd_solution = sp.solve(sp.Eq(reflection_difference, 0), odd)[0]
    assert odd_solution == 0

    potential = lam * q**2 + kappa * q**4
    v0 = sp.simplify(potential.subs(q, 0))
    vp = sp.simplify(potential.subs(q, 2))
    vm = sp.simplify(potential.subs(q, -2))
    assert v0 == 0
    assert vp == vm
    assert sp.simplify(vp - (4 * lam + 16 * kappa)) == 0

    quadratic_gap = sp.simplify((lam * q**2).subs(q, 2) - (lam * q**2).subs(q, 0))
    assert quadratic_gap == 4 * lam

    mixed_norm = sp.eye(2) / 2
    sigma_mix = sp.simplify(mixed_norm / (z - quadratic_gap))
    expected = sp.eye(2) / (2 * (z - 4 * lam))
    assert sp.simplify(sigma_mix - expected) == sp.zeros(2)
    assert sp.limit(sigma_mix[0, 0], lam, sp.oo) == 0
    assert sp.limit(sigma_mix[1, 1], lam, sp.oo) == 0

    print("reflection evenness forces odd coefficient =", odd_solution)
    print("V(0) =", v0)
    print("V(+2) = V(-2) =", vp)
    print("quadratic adjacent gap =", quadratic_gap)
    print("Sigma_mix(z,Lambda) =", sigma_mix.tolist())
    print("hard-gap limit Sigma_mix -> 0")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
