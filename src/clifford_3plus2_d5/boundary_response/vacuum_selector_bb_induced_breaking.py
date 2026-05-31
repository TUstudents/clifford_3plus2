"""V41 BB-induced radial breaking with positive quartic backreaction.

V40 narrows the radial input to a bounded broken quartic Higgs/backreaction
phase.  V41 checks whether the broken sign itself has to be inserted.  The
free BB filled-band even energy from V38 already destabilizes the origin along
the selector ray; a positive quartic backreaction can therefore bound the
profile and create a finite nonzero selector radius without adding a negative
Higgs mass by hand.

This remains a sufficiency theorem: it assumes a positive quartic
backreaction.  It does not derive that quartic coefficient from microscopic
gauge/Higgs dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_TEST_EPSILON,
    FILLED_BAND_ZERO_TOLERANCE,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_filled_band_potential import (
    filled_band_selector_branch_gap,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_higgs_origin import (
    higgs_radial_landau_origin_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_higgs_stabilizer import (
    RADIAL_STABILIZER_TOLERANCE,
    higgs_backreaction_radial_stabilizer_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_radial_no_go import (
    free_bb_has_no_finite_radial_minimum,
    radial_energy_finite_differences,
    radial_selector_energy_samples,
)

Helicity = Literal["right", "left", "dirac"]

BB_INDUCED_RADIAL_AUDIT_RADII = (
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.5,
    1.75,
    2.0,
)
DEFAULT_BACKREACTION_MASS_SQUARED = 0.0
DEFAULT_BACKREACTION_QUARTIC = 1.0
POSITIVE_MASS_CONTROL = 1.0

REMAINING_DECLARED_INPUTS_AFTER_BB_INDUCED_BREAKING = (
    "positive_quartic_backreaction_bounds_selector_radius",
)


def quartic_backreaction_potential(
    radius: float,
    *,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> float:
    """Return ``m^2 r^2 + lambda r^4`` for the selector radius."""

    return mass_squared * radius * radius + quartic * radius**4


def quartic_backreaction_is_bounded(*, quartic: float = DEFAULT_BACKREACTION_QUARTIC) -> bool:
    """Return true when the quartic backreaction bounds the radial profile."""

    return quartic > 0.0


def quartic_backreaction_alone_is_unbroken(
    *,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
    radii: tuple[float, ...] = BB_INDUCED_RADIAL_AUDIT_RADII,
) -> bool:
    """Return true when the positive quartic sector alone minimizes at ``r=0``."""

    if mass_squared < 0.0 or not quartic_backreaction_is_bounded(quartic=quartic):
        return False
    energies = tuple(
        quartic_backreaction_potential(
            radius,
            mass_squared=mass_squared,
            quartic=quartic,
        )
        for radius in radii
    )
    return energies[0] == min(energies)


def free_bb_destabilizes_origin(
    *,
    radii: tuple[float, ...] = BB_INDUCED_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> bool:
    """Return true when V38's free even energy lowers immediately away from zero."""

    energies = radial_selector_energy_samples(radii=radii, epsilon_value=epsilon_value)
    differences = radial_energy_finite_differences(radii=radii, epsilon_value=epsilon_value)
    return (
        abs(energies[0]) <= RADIAL_STABILIZER_TOLERANCE
        and energies[1] < energies[0] - RADIAL_STABILIZER_TOLERANCE
        and all(difference < -RADIAL_STABILIZER_TOLERANCE for difference in differences[:4])
    )


def bb_induced_radial_energy_samples(
    *,
    radii: tuple[float, ...] = BB_INDUCED_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> tuple[float, ...]:
    """Return ``E_free_even(r) + m^2 r^2 + lambda r^4`` on the audit grid."""

    free_energies = radial_selector_energy_samples(radii=radii, epsilon_value=epsilon_value)
    return tuple(
        free_energy
        + quartic_backreaction_potential(
            radius,
            mass_squared=mass_squared,
            quartic=quartic,
        )
        for radius, free_energy in zip(radii, free_energies, strict=True)
    )


def _global_minimum_radii(
    energies: tuple[float, ...],
    radii: tuple[float, ...],
    *,
    tolerance: float = RADIAL_STABILIZER_TOLERANCE,
) -> tuple[float, ...]:
    minimum = min(energies)
    return tuple(
        radius
        for radius, energy in zip(radii, energies, strict=True)
        if abs(energy - minimum) <= tolerance
    )


def bb_induced_radial_minimum_candidates(
    *,
    radii: tuple[float, ...] = BB_INDUCED_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> tuple[float, ...]:
    """Return grid radii minimizing the BB-induced radial profile."""

    return _global_minimum_radii(
        bb_induced_radial_energy_samples(
            radii=radii,
            epsilon_value=epsilon_value,
            mass_squared=mass_squared,
            quartic=quartic,
        ),
        radii,
    )


def bb_induced_has_finite_nonzero_minimum(
    *,
    radii: tuple[float, ...] = BB_INDUCED_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> bool:
    """Return true when the BB-induced profile minimizes away from endpoints."""

    minima = bb_induced_radial_minimum_candidates(
        radii=radii,
        epsilon_value=epsilon_value,
        mass_squared=mass_squared,
        quartic=quartic,
    )
    return bool(minima) and all(radii[0] < radius < radii[-1] for radius in minima)


def bb_induced_branch_gaps_at_minima(
    helicity: Helicity,
    *,
    radii: tuple[float, ...] = BB_INDUCED_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
) -> tuple[float, ...]:
    """Return V36 branch gaps at the BB-induced radial minima."""

    return tuple(
        filled_band_selector_branch_gap(
            helicity,
            epsilon_value=epsilon_value,
            radial_scale=radius,
        )
        for radius in bb_induced_radial_minimum_candidates(
            radii=radii,
            epsilon_value=epsilon_value,
            mass_squared=mass_squared,
            quartic=quartic,
        )
    )


def bb_induced_branch_selection_survives_at_minimum(
    *,
    radii: tuple[float, ...] = BB_INDUCED_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    mass_squared: float = DEFAULT_BACKREACTION_MASS_SQUARED,
    quartic: float = DEFAULT_BACKREACTION_QUARTIC,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when right/left/Dirac branch diagnostics survive."""

    right_gaps = bb_induced_branch_gaps_at_minima(
        "right",
        radii=radii,
        epsilon_value=epsilon_value,
        mass_squared=mass_squared,
        quartic=quartic,
    )
    left_gaps = bb_induced_branch_gaps_at_minima(
        "left",
        radii=radii,
        epsilon_value=epsilon_value,
        mass_squared=mass_squared,
        quartic=quartic,
    )
    dirac_gaps = bb_induced_branch_gaps_at_minima(
        "dirac",
        radii=radii,
        epsilon_value=epsilon_value,
        mass_squared=mass_squared,
        quartic=quartic,
    )
    return (
        bool(right_gaps)
        and all(gap > tolerance for gap in right_gaps)
        and all(gap < -tolerance for gap in left_gaps)
        and all(abs(gap) <= tolerance for gap in dirac_gaps)
    )


def zero_quartic_recovers_free_bb_no_go(
    *,
    radii: tuple[float, ...] = BB_INDUCED_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> bool:
    """Return true when removing the quartic recovers the V38 no-go shape."""

    return (
        not bb_induced_has_finite_nonzero_minimum(
            radii=radii,
            epsilon_value=epsilon_value,
            quartic=0.0,
        )
        and free_bb_has_no_finite_radial_minimum(
            radii=radii,
            epsilon_value=epsilon_value,
        )
    )


def positive_mass_control_keeps_broken_minimum() -> bool:
    """Return true when a nonnegative mass term still allows BB-induced breaking."""

    return bb_induced_has_finite_nonzero_minimum(
        mass_squared=POSITIVE_MASS_CONTROL,
        quartic=DEFAULT_BACKREACTION_QUARTIC,
    )


def v39_recovered() -> bool:
    """Return true when the V39 radial-stabilizer gate still passes."""

    return (
        higgs_backreaction_radial_stabilizer_audit_payload().final_verdict
        == "HIGGS_BACKREACTION_RADIAL_STABILIZER_PASS"
    )


def v40_recovered() -> bool:
    """Return true when the V40 Higgs radial-origin gate still passes."""

    return (
        higgs_radial_landau_origin_audit_payload().final_verdict
        == "HIGGS_RADIAL_LANDAU_UNIQUENESS_PASS"
    )


@dataclass(frozen=True)
class BBInducedRadialBreakingAuditPayload:
    """Verdict payload for the V41 BB-induced radial-breaking theorem gate."""

    final_verdict: str
    audited_radii: tuple[float, ...]
    free_radial_even_energies: tuple[float, ...]
    backreaction_energies: tuple[float, ...]
    total_radial_energies: tuple[float, ...]
    total_minimum_candidates: tuple[float, ...]
    positive_mass_control_minima: tuple[float, ...]
    right_branch_gaps_at_minima: tuple[float, ...]
    left_branch_gaps_at_minima: tuple[float, ...]
    dirac_branch_gaps_at_minima: tuple[float, ...]
    free_origin_destabilized: bool
    quartic_bounded: bool
    backreaction_alone_unbroken: bool
    finite_nonzero_minimum: bool
    branch_selection_survives: bool
    zero_quartic_recovers_no_go: bool
    negative_quartic_control_unbounded: bool
    positive_mass_control_broken: bool
    v39_recovered: bool
    v40_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


@cache
def bb_induced_radial_breaking_audit_payload() -> BBInducedRadialBreakingAuditPayload:
    """Return the V41 BB-induced radial-breaking verdict."""

    free_energies = radial_selector_energy_samples(radii=BB_INDUCED_RADIAL_AUDIT_RADII)
    backreaction_energies = tuple(
        quartic_backreaction_potential(radius)
        for radius in BB_INDUCED_RADIAL_AUDIT_RADII
    )
    total_energies = bb_induced_radial_energy_samples()
    minima = bb_induced_radial_minimum_candidates()
    positive_mass_minima = bb_induced_radial_minimum_candidates(
        mass_squared=POSITIVE_MASS_CONTROL,
    )
    right_gaps = bb_induced_branch_gaps_at_minima("right")
    left_gaps = bb_induced_branch_gaps_at_minima("left")
    dirac_gaps = bb_induced_branch_gaps_at_minima("dirac")
    origin_destabilized = free_bb_destabilizes_origin()
    bounded = quartic_backreaction_is_bounded()
    backreaction_unbroken = quartic_backreaction_alone_is_unbroken()
    finite_minimum = bb_induced_has_finite_nonzero_minimum()
    branch_survives = bb_induced_branch_selection_survives_at_minimum()
    quartic_no_go = zero_quartic_recovers_free_bb_no_go()
    negative_quartic_unbounded = not quartic_backreaction_is_bounded(quartic=-1.0)
    positive_mass_broken = positive_mass_control_keeps_broken_minimum()
    v39 = v39_recovered()
    v40 = v40_recovered()

    prerequisites_pass = (
        origin_destabilized
        and bounded
        and backreaction_unbroken
        and finite_minimum
        and branch_survives
        and quartic_no_go
        and negative_quartic_unbounded
        and positive_mass_broken
        and v39
        and v40
    )

    if prerequisites_pass:
        final_verdict = "BB_INDUCED_RADIAL_BREAKING_PASS"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_BB_INDUCED_BREAKING
        interpretation = (
            "The free BB filled-band even energy destabilizes the selector "
            "origin.  A positive quartic backreaction that is unbroken by "
            "itself then bounds the profile and induces a finite nonzero "
            "selector radius while preserving the chiral branch gap."
        )
    else:
        final_verdict = "BB_INDUCED_RADIAL_BREAKING_KILL"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_BB_INDUCED_BREAKING
        interpretation = (
            "The audited free-BB destabilization plus positive quartic "
            "backreaction did not produce a finite selector minimum with the "
            "required branch diagnostics."
        )

    return BBInducedRadialBreakingAuditPayload(
        final_verdict=final_verdict,
        audited_radii=BB_INDUCED_RADIAL_AUDIT_RADII,
        free_radial_even_energies=free_energies,
        backreaction_energies=backreaction_energies,
        total_radial_energies=total_energies,
        total_minimum_candidates=minima,
        positive_mass_control_minima=positive_mass_minima,
        right_branch_gaps_at_minima=right_gaps,
        left_branch_gaps_at_minima=left_gaps,
        dirac_branch_gaps_at_minima=dirac_gaps,
        free_origin_destabilized=origin_destabilized,
        quartic_bounded=bounded,
        backreaction_alone_unbroken=backreaction_unbroken,
        finite_nonzero_minimum=finite_minimum,
        branch_selection_survives=branch_survives,
        zero_quartic_recovers_no_go=quartic_no_go,
        negative_quartic_control_unbounded=negative_quartic_unbounded,
        positive_mass_control_broken=positive_mass_broken,
        v39_recovered=v39,
        v40_recovered=v40,
        remaining_declared_inputs=remaining_inputs,
        interpretation=interpretation,
    )
