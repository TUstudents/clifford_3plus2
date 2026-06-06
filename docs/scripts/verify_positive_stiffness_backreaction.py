"""Verify the positive-stiffness backreaction certificate.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BCC_POSITIVE_STIFFNESS_BACKREACTION.md``.
It checks that a local mismatch constraint ``K_rel=q`` gives the positive
backreaction ``g K^dagger K = g q^2``, that the adjacent leakage sectors have
gap ``4g``, and that the induced mixed-normal Schur feedback has the same form
as the previous q=0 superselection certificate.

Run:
    python docs/scripts/verify_positive_stiffness_backreaction.py
"""

from __future__ import annotations

import sympy as sp


def main() -> None:
    q, z, g = sp.symbols("q z g", positive=True)

    k_rel = q
    h_lock = sp.simplify(g * k_rel * k_rel)
    assert h_lock == g * q**2
    assert sp.simplify(h_lock.subs(q, -q) - h_lock) == 0

    h0 = sp.simplify(h_lock.subs(q, 0))
    h_plus = sp.simplify(h_lock.subs(q, 2))
    h_minus = sp.simplify(h_lock.subs(q, -2))
    assert h0 == 0
    assert h_plus == h_minus == 4 * g

    sampled_values = [sp.simplify(h_lock.subs(q, value)) for value in (-3, -2, -1, 0, 1, 2, 3)]
    sampled_at_g1 = [value.subs(g, 1) for value in sampled_values]
    assert all(value >= 0 for value in sampled_at_g1)
    assert min(sampled_at_g1) == 0

    mixed_norm = sp.eye(2) / 2
    sigma_mix = sp.simplify(mixed_norm / (z - h_plus))
    expected = sp.eye(2) / (2 * (z - 4 * g))
    assert sp.simplify(sigma_mix - expected) == sp.zeros(2)
    assert sp.limit(sigma_mix[0, 0], g, sp.oo) == 0
    assert sp.limit(sigma_mix[1, 1], g, sp.oo) == 0

    print("K_rel =", k_rel)
    print("H_lock = g K^dagger K =", h_lock)
    print("reflection invariant H_lock(-q)-H_lock(q) = 0")
    print("H_lock(0) =", h0)
    print("H_lock(+2) = H_lock(-2) =", h_plus)
    print("sampled symbolic values =", sampled_values)
    print("sampled nonnegative values at g=1 =", sampled_at_g1)
    print("Sigma_mix(z,g) =", sigma_mix.tolist())
    print("hard-stiffness limit Sigma_mix -> 0")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
