"""V39 Higgs/backreaction radial-stabilization sufficiency gate.

V38 proves that the free Bialynicki-Birula filled-band functional supplies a
helicity-locked selector branch but no finite radial minimum.  V39 couples that
same selector amplitude to the minimal radial stabilizer already used by the
spacetime-QCA Higgs sector,

    V_H(r) = lambda (r^2 - v^2)^2.

This is a sufficiency theorem, not a derivation of the Higgs potential from
the free BB walk.  It checks that a positive Mexican-hat/backreaction radial
sector closes the V38 no-go while leaving the V35-V37 chiral branch-selection
structure intact.
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
from clifford_3plus2_d5.boundary_response.vacuum_selector_microscopic_potential import (
    filled_band_even_radial_energy,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_selector_candidates,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_radial_no_go import (
    free_bb_has_no_finite_radial_minimum,
    free_bb_radial_stabilization_audit_payload,
    radial_selector_energy_samples,
)

Helicity = Literal["right", "left", "dirac"]

RADIAL_STABILIZER_AUDIT_RADII = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
RADIAL_STABILIZER_TOLERANCE = 1.0e-9
DEFAULT_SELECTOR_VEV_SQUARED = 1.0
DEFAULT_SELECTOR_QUARTIC = 1.0

REMAINING_DECLARED_INPUTS_AFTER_HIGGS_BACKREACTION_STABILIZER = (
    "higgs_or_backreaction_sector_supplies_mexican_hat_radial_term",
)


def mexican_hat_radial_potential(
    radius: float,
    *,
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
) -> float:
    """Return ``lambda (r^2 - v^2)^2`` for the selector amplitude radius."""

    return quartic * (radius * radius - vev_squared) ** 2


def mexican_hat_radial_potential_samples(
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
) -> tuple[float, ...]:
    """Return radial stabilizer samples on the audited grid."""

    return tuple(
        mexican_hat_radial_potential(
            radius,
            vev_squared=vev_squared,
            quartic=quartic,
        )
        for radius in radii
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


def mexican_hat_minimum_candidates(
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
) -> tuple[float, ...]:
    """Return grid radii that minimize the radial stabilizer alone."""

    return _global_minimum_radii(
        mexican_hat_radial_potential_samples(
            radii=radii,
            vev_squared=vev_squared,
            quartic=quartic,
        ),
        radii,
    )


def stabilized_radial_even_energy(
    radius: float,
    *,
    helicity: Helicity = "right",
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
) -> float:
    """Return free BB even radial energy plus the Higgs/backreaction term."""

    selector = tetrahedral_selector_candidates()[0]
    return filled_band_even_radial_energy(
        selector,
        helicity,
        epsilon_value=epsilon_value,
        radial_scale=radius,
    ) + mexican_hat_radial_potential(
        radius,
        vev_squared=vev_squared,
        quartic=quartic,
    )


def stabilized_radial_energy_samples(
    *,
    helicity: Helicity = "right",
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
) -> tuple[float, ...]:
    """Return stabilized radial energy samples on the audited grid."""

    return tuple(
        stabilized_radial_even_energy(
            radius,
            helicity=helicity,
            epsilon_value=epsilon_value,
            vev_squared=vev_squared,
            quartic=quartic,
        )
        for radius in radii
    )


def stabilized_radial_minimum_candidates(
    *,
    helicity: Helicity = "right",
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
    tolerance: float = RADIAL_STABILIZER_TOLERANCE,
) -> tuple[float, ...]:
    """Return grid radii that minimize the stabilized radial energy."""

    return _global_minimum_radii(
        stabilized_radial_energy_samples(
            helicity=helicity,
            radii=radii,
            epsilon_value=epsilon_value,
            vev_squared=vev_squared,
            quartic=quartic,
        ),
        radii,
        tolerance=tolerance,
    )


def stabilized_has_finite_interior_minimum(
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
) -> bool:
    """Return true when the stabilized grid minimum is away from endpoints."""

    minima = stabilized_radial_minimum_candidates(
        radii=radii,
        epsilon_value=epsilon_value,
        vev_squared=vev_squared,
        quartic=quartic,
    )
    return bool(minima) and all(radii[0] < radius < radii[-1] for radius in minima)


def stabilized_branch_gaps_at_minima(
    helicity: Helicity,
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
) -> tuple[float, ...]:
    """Return V36 branch gaps at the stabilized radial minima."""

    return tuple(
        filled_band_selector_branch_gap(
            helicity,
            epsilon_value=epsilon_value,
            radial_scale=radius,
        )
        for radius in stabilized_radial_minimum_candidates(
            radii=radii,
            epsilon_value=epsilon_value,
            vev_squared=vev_squared,
            quartic=quartic,
        )
    )


def stabilized_branch_selection_survives_at_minimum(
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when right/left/Dirac branch diagnostics survive."""

    right_gaps = stabilized_branch_gaps_at_minima(
        "right",
        radii=radii,
        epsilon_value=epsilon_value,
        vev_squared=vev_squared,
        quartic=quartic,
    )
    left_gaps = stabilized_branch_gaps_at_minima(
        "left",
        radii=radii,
        epsilon_value=epsilon_value,
        vev_squared=vev_squared,
        quartic=quartic,
    )
    dirac_gaps = stabilized_branch_gaps_at_minima(
        "dirac",
        radii=radii,
        epsilon_value=epsilon_value,
        vev_squared=vev_squared,
        quartic=quartic,
    )
    return (
        bool(right_gaps)
        and all(gap > tolerance for gap in right_gaps)
        and all(gap < -tolerance for gap in left_gaps)
        and all(abs(gap) <= tolerance for gap in dirac_gaps)
    )


def quartic_zero_recovers_free_bb_no_go(
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> bool:
    """Return true when removing the stabilizer recovers the V38 no-go."""

    stabilized_without_quartic = stabilized_radial_energy_samples(
        radii=radii,
        epsilon_value=epsilon_value,
        quartic=0.0,
    )
    free_samples = radial_selector_energy_samples(
        radii=radii,
        epsilon_value=epsilon_value,
    )
    return (
        all(
            abs(left - right) <= RADIAL_STABILIZER_TOLERANCE
            for left, right in zip(stabilized_without_quartic, free_samples, strict=True)
        )
        and not stabilized_has_finite_interior_minimum(
            radii=radii,
            epsilon_value=epsilon_value,
            quartic=0.0,
        )
        and free_bb_has_no_finite_radial_minimum(
            radii=radii,
            epsilon_value=epsilon_value,
        )
    )


def zero_vev_pure_radial_sector_is_unbroken(
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
) -> bool:
    """Return true when a pure ``v^2=0`` radial sector minimizes at the origin."""

    return mexican_hat_minimum_candidates(
        radii=radii,
        vev_squared=0.0,
        quartic=quartic,
    ) == (0.0,)


def v38_recovered() -> bool:
    """Return true when the V38 free-radial no-go gate still passes."""

    return (
        free_bb_radial_stabilization_audit_payload().final_verdict
        == "FREE_BB_RADIAL_STABILIZATION_NO_GO_PASS"
    )


@dataclass(frozen=True)
class HiggsBackreactionRadialStabilizerAuditPayload:
    """Verdict payload for the V39 radial-stabilizer sufficiency gate."""

    final_verdict: str
    audited_radii: tuple[float, ...]
    free_radial_even_energies: tuple[float, ...]
    higgs_radial_energies: tuple[float, ...]
    stabilized_radial_energies: tuple[float, ...]
    stabilized_minimum_candidates: tuple[float, ...]
    right_branch_gaps_at_minima: tuple[float, ...]
    left_branch_gaps_at_minima: tuple[float, ...]
    dirac_branch_gaps_at_minima: tuple[float, ...]
    finite_interior_minimum: bool
    branch_selection_survives: bool
    quartic_zero_recovers_no_go: bool
    zero_vev_pure_sector_unbroken: bool
    v38_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


@cache
def higgs_backreaction_radial_stabilizer_audit_payload() -> HiggsBackreactionRadialStabilizerAuditPayload:
    """Return the V39 Higgs/backreaction radial-stabilizer verdict."""

    free_energies = radial_selector_energy_samples(radii=RADIAL_STABILIZER_AUDIT_RADII)
    higgs_energies = mexican_hat_radial_potential_samples()
    stabilized_energies = stabilized_radial_energy_samples()
    minima = stabilized_radial_minimum_candidates()
    right_gaps = stabilized_branch_gaps_at_minima("right")
    left_gaps = stabilized_branch_gaps_at_minima("left")
    dirac_gaps = stabilized_branch_gaps_at_minima("dirac")
    finite_minimum = stabilized_has_finite_interior_minimum()
    branch_survives = stabilized_branch_selection_survives_at_minimum()
    quartic_no_go = quartic_zero_recovers_free_bb_no_go()
    zero_vev_unbroken = zero_vev_pure_radial_sector_is_unbroken()
    v38 = v38_recovered()

    prerequisites_pass = (
        finite_minimum
        and branch_survives
        and quartic_no_go
        and zero_vev_unbroken
        and v38
    )

    if prerequisites_pass:
        final_verdict = "HIGGS_BACKREACTION_RADIAL_STABILIZER_PASS"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_HIGGS_BACKREACTION_STABILIZER
        interpretation = (
            "The free BB selector branch becomes a finite-radius broken "
            "selector once coupled to a positive Mexican-hat/backreaction "
            "radial sector.  The chiral branch sign survives at the minimum, "
            "while removing the quartic stabilizer recovers the V38 no-go."
        )
    else:
        final_verdict = "HIGGS_BACKREACTION_RADIAL_STABILIZER_KILL"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_HIGGS_BACKREACTION_STABILIZER
        interpretation = (
            "The proposed Higgs/backreaction radial term did not close the "
            "V38 radial no-go without breaking the selector diagnostics."
        )

    return HiggsBackreactionRadialStabilizerAuditPayload(
        final_verdict=final_verdict,
        audited_radii=RADIAL_STABILIZER_AUDIT_RADII,
        free_radial_even_energies=free_energies,
        higgs_radial_energies=higgs_energies,
        stabilized_radial_energies=stabilized_energies,
        stabilized_minimum_candidates=minima,
        right_branch_gaps_at_minima=right_gaps,
        left_branch_gaps_at_minima=left_gaps,
        dirac_branch_gaps_at_minima=dirac_gaps,
        finite_interior_minimum=finite_minimum,
        branch_selection_survives=branch_survives,
        quartic_zero_recovers_no_go=quartic_no_go,
        zero_vev_pure_sector_unbroken=zero_vev_unbroken,
        v38_recovered=v38,
        remaining_declared_inputs=remaining_inputs,
        interpretation=interpretation,
    )
