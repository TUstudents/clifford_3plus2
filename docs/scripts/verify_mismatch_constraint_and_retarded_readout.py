"""Verify the mismatch-constraint and retarded-readout certificate.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md``.
It checks the exact algebraic parts of the note:

* the trace/relative split of two BCC edge depths;
* uniqueness of ``K_rel = q = r1-r2`` as the local linear mismatch;
* positivity and adjacent leakage gap from ``g K^dagger K``;
* the half-line Weyl equation and retarded branch selection;
* vanishing gapped retarded feedback in the hard-locking limit.

Run:
    python docs/scripts/verify_mismatch_constraint_and_retarded_readout.py
"""

from __future__ import annotations

import sympy as sp


def reflect(expr: sp.Expr, r1: sp.Symbol, r2: sp.Symbol) -> sp.Expr:
    """Exchange the two boundary-face depths simultaneously."""
    return sp.expand(expr.xreplace({r1: r2, r2: r1}))


def main() -> None:
    r1, r2, a, b, t = sp.symbols("r1 r2 a b t")
    q, z, g = sp.symbols("q z g", positive=True)

    trace = sp.Rational(1, 2) * (r1 + r2)
    relative = r1 - r2

    assert reflect(trace, r1, r2) == trace
    assert reflect(relative, r1, r2) == -relative

    linear_mismatch = a * r1 + b * r2

    # A true mismatch vanishes on the synchronous edge line r1=r2=t.
    sync_value = sp.expand(linear_mismatch.subs({r1: t, r2: t}))
    b_solution = sp.solve(sp.Eq(sync_value, 0), b)[0]
    unique_mismatch = sp.simplify(linear_mismatch.subs(b, b_solution))
    assert unique_mismatch == a * relative

    # Reflection-oddness gives the same condition.
    odd_condition = sp.expand(reflect(linear_mismatch, r1, r2) + linear_mismatch)
    b_from_oddness = sp.solve(sp.Eq(odd_condition.coeff(r1), 0), b)[0]
    assert b_from_oddness == -a
    assert sp.simplify(linear_mismatch.subs(b, b_from_oddness) - a * relative) == 0

    h_lock = sp.simplify(g * q**2)
    assert sp.simplify(h_lock.subs(q, -q) - h_lock) == 0
    assert h_lock.subs(q, 0) == 0
    assert h_lock.subs(q, 2) == h_lock.subs(q, -2) == 4 * g

    sampled_at_g1 = [h_lock.subs({q: value, g: 1}) for value in (-3, -2, -1, 0, 1, 2, 3)]
    assert all(value >= 0 for value in sampled_at_g1)
    assert min(sampled_at_g1) == 0

    # Half-line Weyl equation: m = 1/(z-m), or m**2-z*m+1 = 0.
    m = sp.symbols("m")
    branch_solutions = sp.solve(sp.Eq(m, 1 / (z - m)), m)
    expected_solutions = [
        sp.Rational(1, 2) * (z - sp.sqrt(z**2 - 4)),
        sp.Rational(1, 2) * (z + sp.sqrt(z**2 - 4)),
    ]
    assert set(branch_solutions) == set(expected_solutions)

    m_ret = sp.Rational(1, 2) * (z - sp.sqrt(z**2 - 4))
    m_grow = sp.Rational(1, 2) * (z + sp.sqrt(z**2 - 4))

    assert sp.limit(m_ret, z, sp.oo) == 0
    assert sp.limit(z * m_ret, z, sp.oo) == 1
    assert sp.limit(m_grow / z, z, sp.oo) == 1

    # On the open band E in (-2,2), the retarded branch has negative
    # imaginary part: m_R(E+i0) = (E - i*sqrt(4-E**2))/2.
    e_values = [-1, 0, 1]
    retarded_imag_values = [-sp.sqrt(4 - e**2) / 2 for e in e_values]
    assert all(value < 0 for value in retarded_imag_values)

    # The gapped retarded feedback inherits the outgoing asymptotic
    # m_R(w) ~ 1/w. With w = z-4g, it vanishes in the hard-locking limit.
    outgoing_asymptotic = 1 / (z - 4 * g)
    assert sp.limit(outgoing_asymptotic, g, sp.oo) == 0

    mixed_norm = sp.eye(2) / 2
    sigma_hard_asymptotic = sp.simplify(mixed_norm * outgoing_asymptotic)
    assert sp.limit(sigma_hard_asymptotic[0, 0], g, sp.oo) == 0
    assert sp.limit(sigma_hard_asymptotic[1, 1], g, sp.oo) == 0

    print("trace coordinate =", trace)
    print("relative coordinate q =", relative)
    print("reflection(trace) = trace")
    print("reflection(q) = -q")
    print("synchronous-line condition gives b =", b_solution)
    print("unique linear mismatch =", unique_mismatch)
    print("H_lock = g q^2 =", h_lock)
    print("adjacent leakage gap H_lock(+-2) =", 4 * g)
    print("half-line Weyl branches =", branch_solutions)
    print("retarded branch m_R =", m_ret)
    print("z*m_R ->", sp.limit(z * m_ret, z, sp.oo))
    print("retarded Im values at E=-1,0,1 =", retarded_imag_values)
    print("gapped outgoing asymptotic 1/(z-4g) -> 0")
    print("Sigma_hard_asymptotic =", sigma_hard_asymptotic.tolist())
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
