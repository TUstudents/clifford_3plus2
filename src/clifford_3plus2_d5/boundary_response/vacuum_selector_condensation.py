"""V32 continuous tetrahedral vacuum-selector condensation gate.

V31 audited the tetrahedral selector anisotropy only on a finite list of
candidate order parameters.  V32 promotes that gate to a continuous
order-parameter statement on ``R^3``.

For the four normalized tetrahedral BCC exits ``v_i``, define

    C(h) = sum_i (h . v_i)^3.

For ``h = (x,y,z)``, the cubic invariant is exactly

    C(h) = 8*x*y*z/sqrt(3).

On the unit sphere this has four maxima at the BCC selector directions
``C = 8/9`` and four antipodal minima ``C = -8/9``.  The nonnegative locking
potential

    V_lock(h) = (|h|^2 - 1)^2 + (C(h) - 8/9)^2

therefore has zeroes exactly at the accepted selector branch among the audited
stationary candidates and controls.

This still does not derive microscopic QCA condensation.  It proves the
narrower gate: if a local vacuum field realizes this continuous tetrahedral
locking potential, the resulting order parameter selects one of the four BCC
tetrahedral exits rather than an arbitrary direction or the antipodal branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector import (
    generic_order_parameter,
    zero_order_parameter,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_antipodal_controls,
    tetrahedral_cubic_invariant,
    tetrahedral_midpoint_controls,
    tetrahedral_selector_candidates,
    vacuum_selector_potential_audit_payload,
)
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import Vector3

REMAINING_DECLARED_INPUTS_AFTER_SELECTOR_CONDENSATION = (
    "local_vacuum_realizes_tetrahedral_selector_locking_potential",
)


def _same_expr(left: sp.Expr, right: sp.Expr) -> bool:
    return sp.simplify(left - right) == 0


def order_parameter_symbols() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    """Return the continuous order-parameter coordinates ``(x,y,z)``."""

    x, y, z = sp.symbols("x y z", real=True)
    return x, y, z


def symbolic_order_parameter() -> Vector3:
    """Return the symbolic continuous order parameter ``h = (x,y,z)``."""

    return order_parameter_symbols()


def norm_squared(order_parameter: Vector3) -> sp.Expr:
    """Return ``|h|^2`` for an exact symbolic three-vector."""

    return sp.simplify(sum(component**2 for component in order_parameter))


def tetrahedral_cubic_polynomial() -> sp.Expr:
    """Return the closed-form tetrahedral cubic ``8*x*y*z/sqrt(3)``."""

    x, y, z = order_parameter_symbols()
    return sp.simplify(8 * x * y * z / sp.sqrt(3))


def cubic_polynomial_matches_invariant() -> bool:
    """Return true when ``sum_i (h.v_i)^3`` equals ``8*x*y*z/sqrt(3)``."""

    return _same_expr(
        tetrahedral_cubic_invariant(symbolic_order_parameter()),
        tetrahedral_cubic_polynomial(),
    )


def selector_locking_potential(order_parameter: Vector3) -> sp.Expr:
    """Return the continuous selector-locking potential ``V_lock(h)``."""

    return sp.simplify(
        (norm_squared(order_parameter) - 1) ** 2
        + (tetrahedral_cubic_invariant(order_parameter) - sp.Rational(8, 9)) ** 2
    )


def wrong_sign_locking_potential(order_parameter: Vector3) -> sp.Expr:
    """Return the locking potential selecting the antipodal branch."""

    return sp.simplify(
        (norm_squared(order_parameter) - 1) ** 2
        + (tetrahedral_cubic_invariant(order_parameter) + sp.Rational(8, 9)) ** 2
    )


def radial_locking_potential(order_parameter: Vector3) -> sp.Expr:
    """Return the radial-only control potential."""

    return sp.simplify((norm_squared(order_parameter) - 1) ** 2)


def coordinate_axis_controls() -> tuple[Vector3, ...]:
    """Return the six unit coordinate-axis stationary controls."""

    one = sp.Integer(1)
    zero = sp.Integer(0)
    return (
        (one, zero, zero),
        (-one, zero, zero),
        (zero, one, zero),
        (zero, -one, zero),
        (zero, zero, one),
        (zero, zero, -one),
    )


def unit_sphere_stationary_candidates() -> tuple[Vector3, ...]:
    """Return the complete Lagrange-stationary candidate list for ``C|S^2``.

    Case analysis for ``C = 8*x*y*z/sqrt(3)``:

    - if all coordinates are nonzero, the Lagrange equations force
      ``x^2 = y^2 = z^2 = 1/3``;
    - if any coordinate is zero, stationarity forces at least two coordinates
      to vanish, giving the six coordinate axes.
    """

    return (
        tetrahedral_selector_candidates()
        + tetrahedral_antipodal_controls()
        + coordinate_axis_controls()
    )


def unit_sphere_lagrange_residual(candidate: Vector3) -> tuple[sp.Expr, ...]:
    """Return Lagrange residuals for stationary points of ``C`` on ``|h|=1``."""

    x, y, z = order_parameter_symbols()
    cubic = tetrahedral_cubic_polynomial()
    gradient = tuple(sp.diff(cubic, variable) for variable in (x, y, z))
    substituted_gradient = tuple(
        sp.simplify(component.subs({x: candidate[0], y: candidate[1], z: candidate[2]}))
        for component in gradient
    )

    lagrange_multiplier = sp.Integer(0)
    for grad_component, h_component in zip(
        substituted_gradient,
        candidate,
        strict=True,
    ):
        if sp.simplify(h_component) != 0:
            lagrange_multiplier = sp.simplify(grad_component / (2 * h_component))
            break

    return tuple(
        sp.simplify(grad_component - 2 * lagrange_multiplier * h_component)
        for grad_component, h_component in zip(
            substituted_gradient,
            candidate,
            strict=True,
        )
    )


def stationary_candidates_satisfy_lagrange_equations() -> bool:
    """Return true when every audited candidate satisfies the sphere equations."""

    return all(
        residual == (0, 0, 0)
        for residual in (
            unit_sphere_lagrange_residual(candidate)
            for candidate in unit_sphere_stationary_candidates()
        )
    )


def unit_sphere_max_value() -> sp.Expr:
    """Return the exact maximum of ``C`` on the audited unit-sphere candidates."""

    return max(
        (tetrahedral_cubic_invariant(candidate) for candidate in unit_sphere_stationary_candidates()),
        key=lambda value: float(sp.N(value)),
    )


def selector_minimizers() -> tuple[int, ...]:
    """Return selector indices with zero locking potential."""

    potentials = tuple(
        selector_locking_potential(candidate)
        for candidate in tetrahedral_selector_candidates()
    )
    return tuple(index for index, value in enumerate(potentials) if sp.simplify(value) == 0)


def locking_potential_zero_only_at_selectors() -> bool:
    """Return true when audited candidates/controls have zero only at selectors."""

    selectors = tetrahedral_selector_candidates()
    non_selectors = (
        tetrahedral_antipodal_controls()
        + coordinate_axis_controls()
        + tetrahedral_midpoint_controls()
        + (zero_order_parameter(), generic_order_parameter())
    )
    return all(selector_locking_potential(candidate) == 0 for candidate in selectors) and all(
        sp.simplify(selector_locking_potential(control)) > 0
        for control in non_selectors
    )


def antipodal_controls_rejected() -> bool:
    """Return true when antipodes have positive accepted locking potential."""

    return all(
        sp.simplify(selector_locking_potential(control)) > 0
        for control in tetrahedral_antipodal_controls()
    )


def axis_controls_rejected() -> bool:
    """Return true when unit coordinate axes have positive locking potential."""

    return all(
        sp.simplify(selector_locking_potential(control)) > 0
        for control in coordinate_axis_controls()
    )


def midpoint_controls_rejected() -> bool:
    """Return true when midpoint controls have positive locking potential."""

    return all(
        sp.simplify(selector_locking_potential(control)) > 0
        for control in tetrahedral_midpoint_controls()
    )


def radial_only_control_rejected() -> bool:
    """Return true when radial locking alone leaves all unit directions degenerate."""

    selector = tetrahedral_selector_candidates()[0]
    antipode = tetrahedral_antipodal_controls()[0]
    axis = coordinate_axis_controls()[0]
    return (
        radial_locking_potential(selector) == 0
        and radial_locking_potential(antipode) == 0
        and radial_locking_potential(axis) == 0
    )


def cubic_only_control_rejected() -> bool:
    """Return true when the cubic-only energy is unbounded along a selector ray."""

    t = sp.symbols("t", positive=True)
    selector = tetrahedral_selector_candidates()[0]
    ray = tuple(sp.simplify(t * component) for component in selector)
    cubic_energy = -tetrahedral_cubic_invariant(ray)
    return sp.limit(cubic_energy, t, sp.oo) == -sp.oo


def wrong_sign_control_rejected() -> bool:
    """Return true when the wrong-sign lock selects the antipodal branch."""

    selector = tetrahedral_selector_candidates()[0]
    antipode = tetrahedral_antipodal_controls()[0]
    return (
        wrong_sign_locking_potential(antipode) == 0
        and sp.simplify(wrong_sign_locking_potential(selector)) > 0
    )


def v31_recovered() -> bool:
    """Return true when the finite-candidate V31 selector-potential gate passes."""

    return (
        vacuum_selector_potential_audit_payload().final_verdict
        == "TETRAHEDRAL_SELECTOR_POTENTIAL_PASS"
    )


@dataclass(frozen=True)
class VacuumSelectorCondensationAuditPayload:
    """Verdict payload for the V32 continuous selector-condensation gate."""

    final_verdict: str
    cubic_polynomial_matches: bool
    unit_sphere_max_value: sp.Expr
    selector_minimizers: tuple[int, ...]
    antipodal_controls_rejected: bool
    axis_controls_rejected: bool
    locking_potential_zero_only_at_selectors: bool
    radial_only_control_rejected: bool
    cubic_only_control_rejected: bool
    wrong_sign_control_rejected: bool
    v31_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def vacuum_selector_condensation_audit_payload() -> VacuumSelectorCondensationAuditPayload:
    """Return the V32 continuous tetrahedral selector-condensation verdict."""

    polynomial_matches = cubic_polynomial_matches_invariant()
    max_value = unit_sphere_max_value()
    minimizers = selector_minimizers()
    antipodal_rejected = antipodal_controls_rejected()
    axes_rejected = axis_controls_rejected()
    zero_only_selectors = locking_potential_zero_only_at_selectors()
    radial_rejected = radial_only_control_rejected()
    cubic_rejected = cubic_only_control_rejected()
    wrong_sign_rejected = wrong_sign_control_rejected()
    v31 = v31_recovered()

    checks_pass = (
        polynomial_matches
        and stationary_candidates_satisfy_lagrange_equations()
        and max_value == sp.Rational(8, 9)
        and minimizers == (0, 1, 2, 3)
        and antipodal_rejected
        and axes_rejected
        and midpoint_controls_rejected()
        and zero_only_selectors
        and radial_rejected
        and cubic_rejected
        and wrong_sign_rejected
        and v31
    )

    if checks_pass:
        final_verdict = "CONTINUOUS_TETRAHEDRAL_SELECTOR_CONDENSATION_PASS"
        interpretation = (
            "The tetrahedral cubic invariant reduces exactly to "
            "8*x*y*z/sqrt(3). On the unit sphere its audited stationary "
            "maxima are the four BCC selector directions with C=8/9. The "
            "nonnegative locking potential has zeroes only on that accepted "
            "selector branch among selectors, antipodes, axes, midpoints, "
            "zero, and generic controls. Radial-only, cubic-only, and "
            "wrong-sign controls fail. The remaining physical input is that "
            "the local vacuum realizes this continuous locking potential."
        )
    else:
        final_verdict = "CONTINUOUS_TETRAHEDRAL_SELECTOR_CONDENSATION_KILL"
        interpretation = (
            "The cubic polynomial identity, stationary-point audit, selector "
            "zero set, controls, or V31 regression failed."
        )

    return VacuumSelectorCondensationAuditPayload(
        final_verdict=final_verdict,
        cubic_polynomial_matches=polynomial_matches,
        unit_sphere_max_value=max_value,
        selector_minimizers=minimizers,
        antipodal_controls_rejected=antipodal_rejected,
        axis_controls_rejected=axes_rejected,
        locking_potential_zero_only_at_selectors=zero_only_selectors,
        radial_only_control_rejected=radial_rejected,
        cubic_only_control_rejected=cubic_rejected,
        wrong_sign_control_rejected=wrong_sign_rejected,
        v31_recovered=v31,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_SELECTOR_CONDENSATION,
        interpretation=interpretation,
    )
