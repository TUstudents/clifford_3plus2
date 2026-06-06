"""Reproduce the up-sector nilpotent Taylor Clebsch study.

This script is intentionally docs-local. It does not add package APIs; it only
verifies the exact coefficient arithmetic used by
``docs/up_sector_clebsch_taylor_study.md``.
"""

from __future__ import annotations

import sympy as sp


def taylor_profile(x: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the scalar Taylor grade profile ``(x^2/2,x,1)``."""

    return (sp.simplify(x**2 / 2), sp.simplify(x), sp.Integer(1))


def is_positive_top_normalized(vector: tuple[sp.Expr, ...]) -> bool:
    """Return true if all entries are positive and top-normalized by 1."""

    return (
        sp.simplify(vector[-1] - 1) == 0
        and all(0 <= sp.N(value) <= 1 for value in vector)
    )


def main() -> None:
    x = 1 / sp.sqrt(2)
    target = (sp.Rational(1, 4), 1 / sp.sqrt(2), sp.Integer(1))
    rational_control = (sp.Rational(1, 4), sp.Rational(3, 4), sp.Integer(1))
    old = (sp.Rational(1, 2), sp.sqrt(2), sp.Integer(1))

    profile = taylor_profile(x)
    inferred_x_from_light = sp.sqrt(2 * target[0])
    charm_ratio_to_rational = sp.simplify(target[1] / rational_control[1])

    print("Taylor x:", x)
    print("Taylor profile:", profile)
    print("target:", target)
    print("Taylor profile equals target:", profile == target)
    print("x inferred from C_u(light)=1/4:", inferred_x_from_light)
    print("nearby rational control:", rational_control)
    print("Taylor charm / rational charm:", charm_ratio_to_rational)
    print("old vector:", old)
    print("old vector is positive top-normalized profile:", is_positive_top_normalized(old))

    assert profile == target
    assert inferred_x_from_light == x
    assert rational_control != target
    assert 0 < sp.N(charm_ratio_to_rational) < 1
    assert not is_positive_top_normalized(old)


if __name__ == "__main__":
    main()
