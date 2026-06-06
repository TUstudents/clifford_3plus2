"""Nilpotent Taylor scalar response for the up-sector Clebsches."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def nilpotent_flag() -> sp.Matrix:
    """Return the length-3 nilpotent repair flag in filtration order."""

    return sp.Matrix(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )


def nilpotent_order_is_three(flag: sp.Matrix | None = None) -> bool:
    """Return true when ``N^3=0`` and ``N^2 != 0``."""

    selected = nilpotent_flag() if flag is None else flag
    return selected**3 == sp.zeros(3, 3) and selected**2 != sp.zeros(3, 3)


def taylor_repair_amplitude() -> sp.Expr:
    """Return the preferred normalized one-step scalar repair amplitude."""

    return 1 / sp.sqrt(2)


def taylor_kernel_matrix(x: sp.Expr | None = None) -> sp.Matrix:
    """Return the truncated nilpotent Taylor kernel ``exp(xN)``."""

    amplitude = taylor_repair_amplitude() if x is None else sp.sympify(x)
    flag = nilpotent_flag()
    return (sp.eye(3) + amplitude * flag + (amplitude**2 / 2) * flag**2).applyfunc(
        sp.simplify
    )


def taylor_shell_profile(x: sp.Expr | None = None) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return grade readout ``(x^2/2, x, 1)`` from the Taylor kernel."""

    amplitude = taylor_repair_amplitude() if x is None else sp.sympify(x)
    return (
        sp.simplify(amplitude**2 / 2),
        sp.simplify(amplitude),
        sp.Integer(1),
    )


def up_clebsch_vector() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the preferred up-sector scalar Clebsch vector."""

    return (sp.Rational(1, 4), 1 / sp.sqrt(2), sp.Integer(1))


def empirical_rational_up_control() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the nearby data-suggested rational vector."""

    return (sp.Rational(1, 4), sp.Rational(3, 4), sp.Integer(1))


def old_up_clebsch_vector() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the older up-sector vector rejected by the scalar profile gate."""

    return (sp.Rational(1, 2), sp.sqrt(2), sp.Integer(1))


def bernstein_cumulative_alternative() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the previously studied cumulative Bernstein alternative."""

    return empirical_rational_up_control()


def is_positive_scalar_grade_profile(vector: tuple[sp.Expr, ...]) -> bool:
    """Return true when vector can be a top-normalized positive grade profile."""

    if sp.simplify(vector[-1] - 1) != 0:
        return False
    return all(0 <= sp.N(value) <= 1 for value in vector)


def one_step_amplitude_from_charm(vector: tuple[sp.Expr, ...]) -> sp.Expr:
    """Return the one-step amplitude inferred from the charm coefficient."""

    return sp.simplify(vector[1])


@dataclass(frozen=True)
class TaylorUpAuditPayload:
    """Audit payload for the up-sector nilpotent Taylor response."""

    final_verdict: str
    repair_amplitude: sp.Expr
    taylor_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    empirical_rational_control: tuple[sp.Expr, sp.Expr, sp.Expr]
    old_sqrt2_control: tuple[sp.Expr, sp.Expr, sp.Expr]
    nilpotent_order_three: bool
    taylor_matches_target: bool
    empirical_rational_control_close: bool
    old_sqrt2_vector_rejected: bool
    interpretation: str


def taylor_up_audit_payload() -> TaylorUpAuditPayload:
    """Return the up-sector Taylor story verdict."""

    target = up_clebsch_vector()
    rational = empirical_rational_up_control()
    old = old_up_clebsch_vector()
    profile = taylor_shell_profile()
    rational_close = sp.N(abs(profile[1] / rational[1] - 1)) < sp.N(sp.Rational(1, 10))
    old_rejected = not is_positive_scalar_grade_profile(old)
    checks_pass = (
        nilpotent_order_is_three()
        and profile == target
        and rational_close
        and old_rejected
        and taylor_kernel_matrix()[0, 2] == sp.Rational(1, 4)
    )

    if checks_pass:
        final_verdict = "NILPOTENT_TAYLOR_UP_CLEBSCH_PASS"
        interpretation = (
            "The length-3 nilpotent Taylor kernel exp(xN) gives grade profile "
            "(x^2/2,x,1). With normalized scalar repair amplitude x=1/sqrt(2), "
            "this is exactly (1/4,1/sqrt(2),1). The nearby rational "
            "(1/4,3/4,1) remains a data-oriented alternative, while the old "
            "sqrt(2) charm vector is rejected as a scalar positive profile."
        )
    else:
        final_verdict = "NILPOTENT_TAYLOR_UP_CLEBSCH_KILL"
        interpretation = (
            "The nilpotent order, Taylor profile, normalized-amplitude target, "
            "nearby-rational control, or old-vector rejection failed."
        )

    return TaylorUpAuditPayload(
        final_verdict=final_verdict,
        repair_amplitude=taylor_repair_amplitude(),
        taylor_profile=profile,
        empirical_rational_control=rational,
        old_sqrt2_control=old,
        nilpotent_order_three=nilpotent_order_is_three(),
        taylor_matches_target=profile == target,
        empirical_rational_control_close=bool(rational_close),
        old_sqrt2_vector_rejected=old_rejected,
        interpretation=interpretation,
    )
