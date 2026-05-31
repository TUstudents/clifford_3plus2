"""V38 free BB radial-stabilization no-go gate.

V37 shows that the free Bialynicki-Birula filled-band eigenphase functional
has the right even/odd microscopic selector structure.  V38 asks whether that
same free functional also stabilizes a finite selector amplitude.

It does not on the audited small-momentum selector ray: the parity-even
filled-band energy decreases monotonically with radius, while the chiral odd
branch gap remains positive away from the origin.  The free walk supplies the
branch direction, not the radial Higgs/Landau stabilization.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_TEST_EPSILON,
    FILLED_BAND_ZERO_TOLERANCE,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_filled_band_potential import (
    filled_band_selector_branch_gap,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_microscopic_potential import (
    filled_band_even_radial_energy,
    dirac_odd_selector_cancels_on_samples,
    microscopic_selector_potential_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_selector_candidates,
)

FREE_BB_RADIAL_AUDIT_RADII = (0.0, 0.125, 0.25, 0.5, 1.0)
FREE_BB_RADIAL_TOLERANCE = 1.0e-9

REMAINING_DECLARED_INPUTS_AFTER_FREE_BB_RADIAL_NO_GO = (
    "radial_stabilization_requires_interaction_or_backreaction",
)


def radial_selector_energy_samples(
    *,
    helicity: str = "right",
    radii: tuple[float, ...] = FREE_BB_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> tuple[float, ...]:
    """Return free filled-band even energies along one selector ray."""

    selector = tetrahedral_selector_candidates()[0]
    return tuple(
        filled_band_even_radial_energy(
            selector,
            helicity,  # type: ignore[arg-type]
            epsilon_value=epsilon_value,
            radial_scale=radius,
        )
        for radius in radii
    )


def radial_energy_finite_differences(
    *,
    helicity: str = "right",
    radii: tuple[float, ...] = FREE_BB_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> tuple[float, ...]:
    """Return forward finite differences of the radial energy samples."""

    energies = radial_selector_energy_samples(
        helicity=helicity,
        radii=radii,
        epsilon_value=epsilon_value,
    )
    return tuple(right - left for left, right in zip(energies, energies[1:], strict=False))


def radial_energy_is_monotone_decreasing(
    *,
    helicity: str = "right",
    radii: tuple[float, ...] = FREE_BB_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = FREE_BB_RADIAL_TOLERANCE,
) -> bool:
    """Return true when the free radial energy strictly decreases on the grid."""

    return all(
        difference < -tolerance
        for difference in radial_energy_finite_differences(
            helicity=helicity,
            radii=radii,
            epsilon_value=epsilon_value,
        )
    )


def free_bb_radial_minimum_candidates(
    *,
    helicity: str = "right",
    radii: tuple[float, ...] = FREE_BB_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = FREE_BB_RADIAL_TOLERANCE,
) -> tuple[float, ...]:
    """Return interior grid radii that look like finite local minima."""

    energies = radial_selector_energy_samples(
        helicity=helicity,
        radii=radii,
        epsilon_value=epsilon_value,
    )
    return tuple(
        radii[index]
        for index in range(1, len(radii) - 1)
        if energies[index] < energies[index - 1] - tolerance
        and energies[index] < energies[index + 1] - tolerance
    )


def free_bb_has_no_finite_radial_minimum(
    *,
    helicity: str = "right",
    radii: tuple[float, ...] = FREE_BB_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> bool:
    """Return true when no interior finite-radius minimum is found."""

    return not free_bb_radial_minimum_candidates(
        helicity=helicity,
        radii=radii,
        epsilon_value=epsilon_value,
    )


def branch_gap_samples(
    *,
    radii: tuple[float, ...] = FREE_BB_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> tuple[float, ...]:
    """Return V36 right-Weyl branch gaps along the same radius grid."""

    return tuple(
        filled_band_selector_branch_gap(
            "right",
            epsilon_value=epsilon_value,
            radial_scale=radius,
        )
        for radius in radii
    )


def branch_gap_is_stable_on_nonzero_radii(
    *,
    radii: tuple[float, ...] = FREE_BB_RADIAL_AUDIT_RADII,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when the branch gap is zero at origin and positive afterward."""

    gaps = branch_gap_samples(radii=radii, epsilon_value=epsilon_value)
    return abs(gaps[0]) <= tolerance and all(gap > tolerance for gap in gaps[1:])


def v37_recovered() -> bool:
    """Return true when the V37 microscopic filled-band potential gate passes."""

    return (
        microscopic_selector_potential_audit_payload().final_verdict
        == "MICROSCOPIC_FILLED_BAND_SELECTOR_POTENTIAL_PASS"
    )


@dataclass(frozen=True)
class FreeBBRadialNoGoAuditPayload:
    """Verdict payload for the V38 free BB radial-stabilization no-go."""

    final_verdict: str
    audited_radii: tuple[float, ...]
    radial_even_energies: tuple[float, ...]
    radial_finite_differences: tuple[float, ...]
    finite_minimum_candidates: tuple[float, ...]
    branch_gaps: tuple[float, ...]
    radial_energy_monotone: bool
    no_finite_radial_minimum: bool
    branch_gap_stable: bool
    dirac_odd_selector_cancelled: bool
    v37_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def free_bb_radial_stabilization_audit_payload() -> FreeBBRadialNoGoAuditPayload:
    """Return the V38 free BB radial-stabilization no-go verdict."""

    energies = radial_selector_energy_samples()
    differences = radial_energy_finite_differences()
    minima = free_bb_radial_minimum_candidates()
    gaps = branch_gap_samples()
    monotone = radial_energy_is_monotone_decreasing()
    no_minimum = free_bb_has_no_finite_radial_minimum()
    branch_stable = branch_gap_is_stable_on_nonzero_radii()
    dirac_cancelled = dirac_odd_selector_cancels_on_samples()
    v37 = v37_recovered()

    if monotone and no_minimum and branch_stable and dirac_cancelled and v37:
        final_verdict = "FREE_BB_RADIAL_STABILIZATION_NO_GO_PASS"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_FREE_BB_RADIAL_NO_GO
        interpretation = (
            "The free BB filled-band functional supplies the helicity-locked "
            "selector branch but not finite-radius stabilization.  A radial "
            "Higgs/Landau lock requires interaction, constraint, or "
            "backreaction input beyond the free walk."
        )
    elif minima:
        final_verdict = "FREE_BB_RADIAL_STABILIZATION_FOUND"
        remaining_inputs: tuple[str, ...] = ()
        interpretation = (
            "The audited free BB radial energy contains an interior finite "
            "local minimum.  This would upgrade radial stabilization from an "
            "external input to a free-walk effect on the audited grid."
        )
    else:
        final_verdict = "FREE_BB_RADIAL_AUDIT_KILL"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_FREE_BB_RADIAL_NO_GO
        interpretation = (
            "The free BB radial audit failed monotonicity, branch-stability, "
            "Dirac-cancellation, or V37-regression diagnostics."
        )

    return FreeBBRadialNoGoAuditPayload(
        final_verdict=final_verdict,
        audited_radii=FREE_BB_RADIAL_AUDIT_RADII,
        radial_even_energies=energies,
        radial_finite_differences=differences,
        finite_minimum_candidates=minima,
        branch_gaps=gaps,
        radial_energy_monotone=monotone,
        no_finite_radial_minimum=no_minimum,
        branch_gap_stable=branch_stable,
        dirac_odd_selector_cancelled=dirac_cancelled,
        v37_recovered=v37,
        remaining_declared_inputs=remaining_inputs,
        interpretation=interpretation,
    )
