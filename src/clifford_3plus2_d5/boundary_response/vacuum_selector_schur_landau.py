"""V34 Schur-shell origin of the tetrahedral cubic.

V33 proves that the first selector-capable tetrahedral Landau anisotropy is the
cubic invariant

    C(h) = sum_i (h . v_i)^3 = 8*x*y*z/sqrt(3).

V34 audits a minimal four-channel tetrahedral boundary shell that naturally
produces the same invariant in a Schur/log-determinant expansion.  The shell
couplings are

    d_i(h) = h . v_i,

so the diagonal shell response is ``D(h) = diag(d_i(h))``.  Its power sums are
exact:

    Tr D     = 0,
    Tr D^2  = 4 |h|^2 / 3,
    Tr D^3  = C(h).

Therefore the cubic term is not an arbitrary nonlinearity; it is the first odd
nonzero moment of the tetrahedral shell.  The sign remains physical: reversing
the oriented coupling reverses the cubic branch while leaving the quadratic
term unchanged.  Proper tetrahedral symmetry produces the invariant but does
not choose the positive selector branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector_condensation import (
    norm_squared,
    order_parameter_symbols,
    tetrahedral_cubic_polynomial,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_landau import (
    lowest_selector_anisotropy_degree,
    tetrahedral_landau_audit_payload,
    tetrahedral_rotation_group,
    transform_polynomial,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_cubic_invariant,
    tetrahedral_selector_candidates,
)
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import Vector3, vector_dot

REMAINING_DECLARED_INPUTS_AFTER_SCHUR_SHELL_LANDAU = (
    "oriented_boundary_shell_selects_positive_cubic_branch",
)


def _same_expr(left: sp.Expr, right: sp.Expr) -> bool:
    return sp.simplify(left - right) == 0


def schur_shell_symbols() -> tuple[sp.Symbol, sp.Symbol]:
    """Return the oriented shell coupling ``eta`` and Schur sign ``s``."""

    eta = sp.symbols("eta", real=True)
    schur_sign = sp.symbols("s", real=True)
    return eta, schur_sign


def tetrahedral_shell_couplings(order_parameter: Vector3) -> tuple[sp.Expr, ...]:
    """Return the four exit couplings ``d_i(h) = h.v_i``."""

    return tuple(
        sp.simplify(vector_dot(order_parameter, selector))
        for selector in tetrahedral_selector_candidates()
    )


def tetrahedral_shell_power_sum(
    degree: int,
    order_parameter: Vector3 | None = None,
) -> sp.Expr:
    """Return ``p_degree(h) = sum_i (h.v_i)^degree``."""

    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if order_parameter is None:
        order_parameter = order_parameter_symbols()
    return sp.simplify(
        sum(coupling**degree for coupling in tetrahedral_shell_couplings(order_parameter))
    )


def tetrahedral_power_sum_identities_hold() -> bool:
    """Return true when the first three shell moments match V34 targets."""

    h = order_parameter_symbols()
    return (
        tetrahedral_shell_power_sum(1, h) == 0
        and _same_expr(tetrahedral_shell_power_sum(2, h), sp.Rational(4, 3) * norm_squared(h))
        and _same_expr(tetrahedral_shell_power_sum(3, h), tetrahedral_cubic_polynomial())
    )


def schur_shell_landau_series(
    order_parameter: Vector3 | None = None,
    *,
    eta: sp.Expr | None = None,
    schur_sign: sp.Expr | None = None,
    expansion_order: int = 3,
) -> sp.Expr:
    """Return the finite Schur/log-det series through ``expansion_order``."""

    if expansion_order < 1:
        raise ValueError("expansion_order must be positive")
    if order_parameter is None:
        order_parameter = order_parameter_symbols()
    default_eta, default_sign = schur_shell_symbols()
    if eta is None:
        eta = default_eta
    if schur_sign is None:
        schur_sign = default_sign
    return sp.expand(
        sp.sympify(schur_sign)
        * sum(
            sp.sympify(eta) ** degree
            * tetrahedral_shell_power_sum(degree, order_parameter)
            / degree
            for degree in range(1, expansion_order + 1)
        )
    )


def schur_shell_quadratic_coefficient(
    *,
    eta: sp.Expr | None = None,
    schur_sign: sp.Expr | None = None,
) -> sp.Expr:
    """Return the coefficient multiplying ``|h|^2`` through cubic order."""

    default_eta, default_sign = schur_shell_symbols()
    if eta is None:
        eta = default_eta
    if schur_sign is None:
        schur_sign = default_sign
    return sp.simplify(sp.sympify(schur_sign) * sp.sympify(eta) ** 2 * sp.Rational(2, 3))


def schur_shell_cubic_coefficient(
    *,
    eta: sp.Expr | None = None,
    schur_sign: sp.Expr | None = None,
) -> sp.Expr:
    """Return the coefficient multiplying ``C(h)`` through cubic order."""

    default_eta, default_sign = schur_shell_symbols()
    if eta is None:
        eta = default_eta
    if schur_sign is None:
        schur_sign = default_sign
    return sp.simplify(sp.sympify(schur_sign) * sp.sympify(eta) ** 3 / 3)


def schur_series_matches_radial_plus_cubic() -> bool:
    """Return true when the cubic-order series contains only radial plus cubic."""

    eta, schur_sign = schur_shell_symbols()
    h = order_parameter_symbols()
    expected = (
        schur_shell_quadratic_coefficient(eta=eta, schur_sign=schur_sign) * norm_squared(h)
        + schur_shell_cubic_coefficient(eta=eta, schur_sign=schur_sign)
        * tetrahedral_cubic_polynomial()
    )
    return _same_expr(schur_shell_landau_series(h, eta=eta, schur_sign=schur_sign), expected)


def eta_sign_flips_only_cubic_branch() -> bool:
    """Return true when ``eta -> -eta`` leaves quadratic and flips cubic."""

    eta, schur_sign = schur_shell_symbols()
    return (
        _same_expr(
            schur_shell_quadratic_coefficient(eta=-eta, schur_sign=schur_sign),
            schur_shell_quadratic_coefficient(eta=eta, schur_sign=schur_sign),
        )
        and _same_expr(
            schur_shell_cubic_coefficient(eta=-eta, schur_sign=schur_sign),
            -schur_shell_cubic_coefficient(eta=eta, schur_sign=schur_sign),
        )
    )


def schur_sign_flips_all_shell_terms() -> bool:
    """Return true when ``s -> -s`` flips the whole shell contribution."""

    eta, schur_sign = schur_shell_symbols()
    h = order_parameter_symbols()
    return _same_expr(
        schur_shell_landau_series(h, eta=eta, schur_sign=-schur_sign),
        -schur_shell_landau_series(h, eta=eta, schur_sign=schur_sign),
    )


def paired_unoriented_shell_series(order_parameter: Vector3 | None = None) -> sp.Expr:
    """Return the paired ``eta`` and ``-eta`` shell control series."""

    eta, schur_sign = schur_shell_symbols()
    if order_parameter is None:
        order_parameter = order_parameter_symbols()
    return sp.expand(
        schur_shell_landau_series(order_parameter, eta=eta, schur_sign=schur_sign)
        + schur_shell_landau_series(order_parameter, eta=-eta, schur_sign=schur_sign)
    )


def paired_unoriented_shell_cancels_cubic() -> bool:
    """Return true when a paired unoriented shell cancels the cubic term."""

    eta, schur_sign = schur_shell_symbols()
    h = order_parameter_symbols()
    expected = 2 * schur_shell_quadratic_coefficient(
        eta=eta,
        schur_sign=schur_sign,
    ) * norm_squared(h)
    return _same_expr(paired_unoriented_shell_series(h), expected)


def proper_tetrahedral_rotations_preserve_shell_series() -> bool:
    """Return true when proper tetrahedral rotations preserve the shell series."""

    eta, schur_sign = schur_shell_symbols()
    h = order_parameter_symbols()
    series = schur_shell_landau_series(h, eta=eta, schur_sign=schur_sign)
    return all(
        _same_expr(transform_polynomial(series, rotation), series)
        for rotation in tetrahedral_rotation_group()
    )


def inversion_flips_cubic_and_is_not_proper_tetrahedral() -> bool:
    """Return true when inversion flips the cubic and is outside the proper group."""

    x, y, z = order_parameter_symbols()
    cubic = tetrahedral_cubic_polynomial()
    inverted_cubic = sp.expand(cubic.subs({x: -x, y: -y, z: -z}, simultaneous=True))
    inversion = -sp.eye(3)
    return (
        _same_expr(inverted_cubic, -cubic)
        and inversion.det() == -1
        and all(inversion != rotation for rotation in tetrahedral_rotation_group())
    )


def cubic_origin_matches_v33() -> bool:
    """Return true when the shell cubic is the V33 first selector anisotropy."""

    return (
        lowest_selector_anisotropy_degree() == 3
        and _same_expr(tetrahedral_shell_power_sum(3), tetrahedral_cubic_invariant(order_parameter_symbols()))
        and tetrahedral_landau_audit_payload().final_verdict
        == "TETRAHEDRAL_INVARIANT_LANDAU_MINIMALITY_PASS"
    )


@dataclass(frozen=True)
class SchurShellLandauAuditPayload:
    """Verdict payload for the V34 Schur-shell Landau-origin gate."""

    final_verdict: str
    p1_zero: bool
    p2_radial: bool
    p3_matches_cubic: bool
    series_radial_plus_cubic: bool
    cubic_origin_matches_v33: bool
    eta_sign_flips_only_cubic_branch: bool
    schur_sign_flips_all_terms: bool
    paired_unoriented_shell_cancels_cubic: bool
    proper_rotations_preserve_series: bool
    inversion_flips_cubic_and_is_not_proper: bool
    v33_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def schur_shell_landau_audit_payload() -> SchurShellLandauAuditPayload:
    """Return the V34 Schur-shell origin verdict."""

    h = order_parameter_symbols()
    p1_zero = tetrahedral_shell_power_sum(1, h) == 0
    p2_radial = _same_expr(tetrahedral_shell_power_sum(2, h), sp.Rational(4, 3) * norm_squared(h))
    p3_matches = _same_expr(tetrahedral_shell_power_sum(3, h), tetrahedral_cubic_polynomial())
    series_matches = schur_series_matches_radial_plus_cubic()
    cubic_origin = cubic_origin_matches_v33()
    eta_flip = eta_sign_flips_only_cubic_branch()
    schur_flip = schur_sign_flips_all_shell_terms()
    paired_cancels = paired_unoriented_shell_cancels_cubic()
    rotations_preserve = proper_tetrahedral_rotations_preserve_shell_series()
    inversion_control = inversion_flips_cubic_and_is_not_proper_tetrahedral()
    v33 = (
        tetrahedral_landau_audit_payload().final_verdict
        == "TETRAHEDRAL_INVARIANT_LANDAU_MINIMALITY_PASS"
    )

    checks_pass = (
        p1_zero
        and p2_radial
        and p3_matches
        and tetrahedral_power_sum_identities_hold()
        and series_matches
        and cubic_origin
        and eta_flip
        and schur_flip
        and paired_cancels
        and rotations_preserve
        and inversion_control
        and v33
    )

    if checks_pass:
        final_verdict = "SCHUR_SHELL_TETRAHEDRAL_CUBIC_ORIGIN_PASS_SIGN_FREE"
        interpretation = (
            "The four-channel tetrahedral boundary shell has p1=0, radial "
            "p2=4|h|^2/3, and p3 equal to the V32/V33 tetrahedral cubic. "
            "Its cubic-order Schur/log-det expansion therefore produces the "
            "first selector-capable Landau anisotropy from shell moments. "
            "The oriented coupling sign remains free: eta reversal flips the "
            "cubic branch, a paired unoriented shell cancels it, and proper "
            "tetrahedral rotations do not choose between selector and "
            "antipodal branches."
        )
    else:
        final_verdict = "SCHUR_SHELL_TETRAHEDRAL_CUBIC_ORIGIN_KILL"
        interpretation = (
            "The shell power sums, Schur series, sign controls, tetrahedral "
            "invariance, inversion control, or V33 regression failed."
        )

    return SchurShellLandauAuditPayload(
        final_verdict=final_verdict,
        p1_zero=p1_zero,
        p2_radial=p2_radial,
        p3_matches_cubic=p3_matches,
        series_radial_plus_cubic=series_matches,
        cubic_origin_matches_v33=cubic_origin,
        eta_sign_flips_only_cubic_branch=eta_flip,
        schur_sign_flips_all_terms=schur_flip,
        paired_unoriented_shell_cancels_cubic=paired_cancels,
        proper_rotations_preserve_series=rotations_preserve,
        inversion_flips_cubic_and_is_not_proper=inversion_control,
        v33_recovered=v33,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_SCHUR_SHELL_LANDAU,
        interpretation=interpretation,
    )
