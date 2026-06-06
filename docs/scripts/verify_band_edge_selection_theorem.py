"""Verify the BB band-edge / marginal-stability selection theorem.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BAND_EDGE_SELECTION_THEOREM.md``.
It checks the exact algebraic backbone of the sidecar:

* pinned BB same-normal scar blocks have survival norm ``1/2 I``;
* the visible transfer has ``det = 1`` and ``tr = 2*zeta + 1/zeta``;
* the trace minimum occurs at the survival amplitude ``zeta = 1/sqrt(2)``;
* at that point the transfer eigenvalues are the silver pair ``sqrt(2) +- 1``;
* the corresponding scalar half-line Weyl attenuation at ``z = 2*sqrt(2)`` is
  ``sqrt(2) - 1``;
* the retarded attenuation has a nondegenerate maximum there, with
  ``-d2 log(lambda)/d zeta2 = 2*sqrt(2)``, the Gaussian curvature used by the
  regular-retarded Laplace selection theorem.

Run:
    python docs/scripts/verify_band_edge_selection_theorem.py
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


def scar_blocks(hops: dict[tuple[int, int, int], sp.Matrix]) -> tuple[sp.Matrix, sp.Matrix]:
    """Return the q=0 same-normal BB blocks at zero tangential momentum."""

    b_plus = sp.simplify(hops[(1, 1, 1)] + hops[(1, 1, -1)])
    b_minus = sp.simplify(hops[(-1, -1, 1)] + hops[(-1, -1, -1)])
    return b_plus, b_minus


def radial_transfer(zeta: sp.Expr) -> sp.Matrix:
    """Return the visible BB scar transfer matrix in the (e,o) basis."""

    return sp.Matrix(
        [
            [1 / (2 * zeta), I / (2 * zeta)],
            [-I / (2 * zeta), 2 * zeta + 1 / (2 * zeta)],
        ]
    )


def retarded_weyl(z: sp.Expr) -> sp.Expr:
    """Return the decaying branch of m = 1/(z-m)."""

    return sp.Rational(1, 2) * (z - sp.sqrt(z**2 - 4))


def main() -> None:
    hops = canonical_hops()
    b_plus, b_minus = scar_blocks(hops)

    survival = sp.simplify(b_plus.H * b_plus + b_minus.H * b_minus)
    assert survival == sp.eye(2) / 2
    c = sp.sqrt(survival[0, 0])
    assert sp.simplify(c - 1 / sp.sqrt(2)) == 0

    zeta = sp.symbols("zeta", positive=True)
    transfer = radial_transfer(zeta)
    determinant = sp.simplify(transfer.det())
    trace = sp.simplify(sp.trace(transfer))
    assert determinant == 1
    assert sp.simplify(trace - (2 * zeta + 1 / zeta)) == 0

    critical_points = [point for point in sp.solve(sp.diff(trace, zeta), zeta) if point.is_positive]
    assert len(critical_points) == 1
    zeta_star = critical_points[0]
    assert sp.simplify(zeta_star - c) == 0
    assert sp.simplify(sp.diff(trace, zeta, 2).subs(zeta, zeta_star)) > 0

    trace_star = sp.simplify(trace.subs(zeta, zeta_star))
    assert trace_star == 2 * sp.sqrt(2)
    transfer_star = sp.simplify(transfer.subs(zeta, zeta_star))
    eigenvalues = {sp.simplify(eigenvalue) for eigenvalue in transfer_star.eigenvals()}
    assert eigenvalues == {sp.sqrt(2) - 1, sp.sqrt(2) + 1}

    attenuation = sp.simplify(retarded_weyl(trace_star))
    assert attenuation == sp.sqrt(2) - 1
    assert sp.simplify(1 / attenuation - (sp.sqrt(2) + 1)) == 0

    # In the evanescent region, lowering the transfer trace increases the
    # decaying branch. Hence the trace minimum is the marginal/least-attenuated
    # point inside this admissible BB family.
    tau = sp.symbols("tau", positive=True)
    attenuation_tau = retarded_weyl(tau)
    derivative = sp.simplify(sp.diff(attenuation_tau, tau))
    assert derivative.subs(tau, 3) < 0

    attenuation_zeta = sp.simplify(retarded_weyl(trace))
    attenuation_prime_star = sp.simplify(sp.diff(attenuation_zeta, zeta).subs(zeta, zeta_star))
    attenuation_second_star = sp.simplify(
        sp.diff(attenuation_zeta, zeta, 2).subs(zeta, zeta_star)
    )
    assert attenuation_prime_star == 0
    assert sp.simplify(attenuation_second_star - (-4 + 2 * sp.sqrt(2))) == 0
    assert attenuation_second_star < 0

    log_curvature = sp.simplify(
        -sp.diff(sp.log(attenuation_zeta), zeta, 2).subs(zeta, zeta_star)
    )
    assert log_curvature == 2 * sp.sqrt(2)

    print("survival norm =", survival.tolist())
    print("survival amplitude c =", c)
    print("transfer determinant =", determinant)
    print("transfer trace =", trace)
    print("trace minimum zeta* =", zeta_star)
    print("trace(zeta*) =", trace_star)
    print("silver eigenvalues =", sorted(eigenvalues, key=sp.N))
    print("retarded Weyl attenuation m_R(2sqrt2) =", attenuation)
    print("d m_R / d tau at tau=3 =", derivative.subs(tau, 3))
    print("d lambda / d zeta at zeta* =", attenuation_prime_star)
    print("d2 lambda / d zeta2 at zeta* =", attenuation_second_star)
    print("-d2 log(lambda) / d zeta2 at zeta* =", log_curvature)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
