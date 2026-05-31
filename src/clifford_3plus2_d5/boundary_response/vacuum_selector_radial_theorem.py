"""V42 analytic radial theorem for BB-induced selector breaking.

V41 was a grid audit.  V42 records the continuum-leading theorem behind that
grid result.  The already-audited Bialynicki-Birula Weyl Hamiltonian has
occupied leading energy ``-|k|``.  Along a unit selector ray this gives the
one-dimensional radial model

    E(r) = -c r + m^2 r^2 + lambda r^4,       c > 0.

The origin is unstable because the right derivative is ``-c``.  A positive
quartic makes the energy tend to ``+infinity`` at large radius.  Continuity
therefore forces a finite nonzero minimum.  For the default ``m^2 = 0`` case
the minimum is explicit: ``r = (c/(4 lambda))**(1/3)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import isclose

from clifford_3plus2_d5.boundary_response.vacuum_selector_bb_induced_breaking import (
    DEFAULT_BACKREACTION_MASS_SQUARED,
    DEFAULT_BACKREACTION_QUARTIC,
    REMAINING_DECLARED_INPUTS_AFTER_BB_INDUCED_BREAKING,
    bb_induced_radial_breaking_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_TEST_EPSILON,
    FILLED_BAND_ZERO_TOLERANCE,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_filled_band_potential import (
    filled_band_selector_branch_gap,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_higgs_stabilizer import (
    RADIAL_STABILIZER_TOLERANCE,
)

DEFAULT_SELECTOR_SLOPE = 1.0


def continuum_occupied_weyl_energy(momentum_norm: float, *, slope: float = DEFAULT_SELECTOR_SLOPE) -> float:
    """Return the leading occupied Weyl energy ``-slope*|k|``."""

    if slope < 0.0:
        raise ValueError("selector slope must be nonnegative")
    return -slope * abs(momentum_norm)


def analytic_radial_energy(
    radius: float,
    *,
    slope: float = DEFAULT_SELECTOR_SLOPE,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> float:
    """Return ``-c r + m^2 r^2 + lambda r^4`` for ``r >= 0``."""

    if radius < 0.0:
        raise ValueError("radius must be nonnegative")
    return -slope * radius + mass_squared * radius * radius + quartic * radius**4


def analytic_radial_derivative(
    radius: float,
    *,
    slope: float = DEFAULT_SELECTOR_SLOPE,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> float:
    """Return the derivative of the analytic radial model for ``r >= 0``."""

    if radius < 0.0:
        raise ValueError("radius must be nonnegative")
    return -slope + 2.0 * mass_squared * radius + 4.0 * quartic * radius**3


def analytic_radial_second_derivative(
    radius: float,
    *,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> float:
    """Return the second derivative of the analytic radial model."""

    if radius < 0.0:
        raise ValueError("radius must be nonnegative")
    return 2.0 * mass_squared + 12.0 * quartic * radius * radius


def analytic_origin_is_destabilized(*, slope: float = DEFAULT_SELECTOR_SLOPE) -> bool:
    """Return true when the right derivative at the origin is negative."""

    return slope > 0.0 and analytic_radial_derivative(0.0, slope=slope) < 0.0


def analytic_quartic_bounds_radius(*, quartic: float = DEFAULT_BACKREACTION_QUARTIC) -> bool:
    """Return true when the quartic term dominates positively at infinity."""

    return quartic > 0.0


def analytic_finite_nonzero_minimum_exists(
    *,
    slope: float = DEFAULT_SELECTOR_SLOPE,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> bool:
    """Return the compact theorem condition for finite nonzero radial closure."""

    return analytic_origin_is_destabilized(slope=slope) and analytic_quartic_bounds_radius(quartic=quartic)


def massless_quartic_stationary_radius(
    *,
    slope: float = DEFAULT_SELECTOR_SLOPE,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> float:
    """Return the positive stationary radius for ``m^2 = 0``."""

    if slope <= 0.0:
        raise ValueError("selector slope must be positive")
    if quartic <= 0.0:
        raise ValueError("quartic coefficient must be positive")
    return (slope / (4.0 * quartic)) ** (1.0 / 3.0)


def massless_quartic_stationary_condition_passes(
    *,
    slope: float = DEFAULT_SELECTOR_SLOPE,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
    tolerance: float = RADIAL_STABILIZER_TOLERANCE,
) -> bool:
    """Return true when the explicit stationary radius solves ``E'(r)=0``."""

    radius = massless_quartic_stationary_radius(slope=slope, quartic=quartic)
    return abs(
        analytic_radial_derivative(
            radius,
            slope=slope,
            mass_squared=0.0,
            quartic=quartic,
        )
    ) <= tolerance


def massless_quartic_stationary_point_is_minimum(
    *,
    slope: float = DEFAULT_SELECTOR_SLOPE,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> bool:
    """Return true when the explicit stationary point has positive curvature."""

    radius = massless_quartic_stationary_radius(slope=slope, quartic=quartic)
    return analytic_radial_second_derivative(radius, mass_squared=0.0, quartic=quartic) > 0.0


def analytic_minimum_lowers_origin(
    *,
    slope: float = DEFAULT_SELECTOR_SLOPE,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> bool:
    """Return true when the analytic minimum has energy below the origin."""

    radius = massless_quartic_stationary_radius(slope=slope, quartic=quartic)
    return analytic_radial_energy(radius, slope=slope, mass_squared=0.0, quartic=quartic) < 0.0


def analytic_branch_selection_survives_at_minimum(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when V36 branch signs survive at the analytic radius."""

    radius = massless_quartic_stationary_radius()
    right_gap = filled_band_selector_branch_gap(
        "right",
        epsilon_value=epsilon_value,
        radial_scale=radius,
    )
    left_gap = filled_band_selector_branch_gap(
        "left",
        epsilon_value=epsilon_value,
        radial_scale=radius,
    )
    dirac_gap = filled_band_selector_branch_gap(
        "dirac",
        epsilon_value=epsilon_value,
        radial_scale=radius,
    )
    return right_gap > tolerance and left_gap < -tolerance and abs(dirac_gap) <= tolerance


def v41_recovered() -> bool:
    """Return true when the V41 grid-audit gate still passes."""

    return (
        bb_induced_radial_breaking_audit_payload().final_verdict
        == "BB_INDUCED_RADIAL_BREAKING_PASS"
    )


@dataclass(frozen=True)
class AnalyticRadialBreakingTheoremAuditPayload:
    """Verdict payload for the V42 analytic radial theorem."""

    final_verdict: str
    selector_slope: float
    mass_squared: float
    quartic: float
    stationary_radius: float
    energy_at_origin: float
    energy_at_stationary_radius: float
    origin_destabilized: bool
    quartic_bounded: bool
    finite_nonzero_minimum_exists: bool
    stationary_condition: bool
    stationary_point_minimum: bool
    minimum_lowers_origin: bool
    branch_selection_survives: bool
    zero_slope_control_rejected: bool
    zero_quartic_control_rejected: bool
    negative_quartic_control_rejected: bool
    v41_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


@cache
def analytic_radial_breaking_theorem_audit_payload() -> AnalyticRadialBreakingTheoremAuditPayload:
    """Return the V42 analytic radial theorem verdict."""

    radius = massless_quartic_stationary_radius()
    energy_at_origin = analytic_radial_energy(0.0)
    energy_at_radius = analytic_radial_energy(radius)
    origin_destabilized = analytic_origin_is_destabilized()
    quartic_bounded = analytic_quartic_bounds_radius()
    finite_minimum = analytic_finite_nonzero_minimum_exists()
    stationary = massless_quartic_stationary_condition_passes()
    local_minimum = massless_quartic_stationary_point_is_minimum()
    lowers_origin = analytic_minimum_lowers_origin()
    branch_survives = analytic_branch_selection_survives_at_minimum()
    zero_slope_rejected = not analytic_finite_nonzero_minimum_exists(slope=0.0)
    zero_quartic_rejected = not analytic_finite_nonzero_minimum_exists(quartic=0.0)
    negative_quartic_rejected = not analytic_quartic_bounds_radius(quartic=-1.0)
    v41 = v41_recovered()

    prerequisites_pass = (
        origin_destabilized
        and quartic_bounded
        and finite_minimum
        and stationary
        and local_minimum
        and lowers_origin
        and branch_survives
        and zero_slope_rejected
        and zero_quartic_rejected
        and negative_quartic_rejected
        and v41
        and isclose(radius, (1.0 / 4.0) ** (1.0 / 3.0), rel_tol=0.0, abs_tol=1.0e-12)
    )

    if prerequisites_pass:
        final_verdict = "ANALYTIC_RADIAL_BREAKING_THEOREM_PASS"
        interpretation = (
            "The continuum-leading occupied Weyl band gives -c r along the "
            "selector ray.  For c>0 and lambda>0, the origin is unstable and "
            "the quartic bounds the profile, forcing a finite nonzero radial "
            "minimum.  The V36 branch sign survives at the analytic radius."
        )
    else:
        final_verdict = "ANALYTIC_RADIAL_BREAKING_THEOREM_KILL"
        interpretation = (
            "The analytic radial model failed the origin-instability, quartic "
            "boundedness, explicit stationary-point, branch-survival, or V41 "
            "regression diagnostics."
        )

    return AnalyticRadialBreakingTheoremAuditPayload(
        final_verdict=final_verdict,
        selector_slope=DEFAULT_SELECTOR_SLOPE,
        mass_squared=DEFAULT_BACKREACTION_MASS_SQUARED,
        quartic=DEFAULT_BACKREACTION_QUARTIC,
        stationary_radius=radius,
        energy_at_origin=energy_at_origin,
        energy_at_stationary_radius=energy_at_radius,
        origin_destabilized=origin_destabilized,
        quartic_bounded=quartic_bounded,
        finite_nonzero_minimum_exists=finite_minimum,
        stationary_condition=stationary,
        stationary_point_minimum=local_minimum,
        minimum_lowers_origin=lowers_origin,
        branch_selection_survives=branch_survives,
        zero_slope_control_rejected=zero_slope_rejected,
        zero_quartic_control_rejected=zero_quartic_rejected,
        negative_quartic_control_rejected=negative_quartic_rejected,
        v41_recovered=v41,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_BB_INDUCED_BREAKING,
        interpretation=interpretation,
    )
