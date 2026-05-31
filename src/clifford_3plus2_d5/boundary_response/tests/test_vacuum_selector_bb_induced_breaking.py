"""Tests for the V41 BB-induced radial-breaking gate."""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.vacuum_selector_bb_induced_breaking import (
    BB_INDUCED_RADIAL_AUDIT_RADII,
    POSITIVE_MASS_CONTROL,
    REMAINING_DECLARED_INPUTS_AFTER_BB_INDUCED_BREAKING,
    bb_induced_branch_gaps_at_minima,
    bb_induced_branch_selection_survives_at_minimum,
    bb_induced_has_finite_nonzero_minimum,
    bb_induced_radial_breaking_audit_payload,
    bb_induced_radial_energy_samples,
    bb_induced_radial_minimum_candidates,
    free_bb_destabilizes_origin,
    positive_mass_control_keeps_broken_minimum,
    quartic_backreaction_alone_is_unbroken,
    quartic_backreaction_is_bounded,
    quartic_backreaction_potential,
    v39_recovered,
    v40_recovered,
    zero_quartic_recovers_free_bb_no_go,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_ZERO_TOLERANCE,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_radial_no_go import (
    radial_energy_finite_differences,
    radial_selector_energy_samples,
)


def test_free_bb_even_energy_destabilizes_origin() -> None:
    energies = radial_selector_energy_samples(radii=BB_INDUCED_RADIAL_AUDIT_RADII)
    differences = radial_energy_finite_differences(radii=BB_INDUCED_RADIAL_AUDIT_RADII)

    assert energies[0] == 0.0
    assert energies[1] < energies[0]
    assert all(difference < 0.0 for difference in differences[:4])
    assert free_bb_destabilizes_origin()


def test_positive_quartic_alone_is_bounded_and_unbroken() -> None:
    assert quartic_backreaction_is_bounded(quartic=1.0)
    assert not quartic_backreaction_is_bounded(quartic=-1.0)
    assert quartic_backreaction_alone_is_unbroken()
    assert quartic_backreaction_alone_is_unbroken(mass_squared=POSITIVE_MASS_CONTROL)
    assert quartic_backreaction_potential(0.0) == 0.0
    assert quartic_backreaction_potential(1.0) > 0.0


def test_positive_quartic_closes_free_bb_radial_no_go() -> None:
    energies = bb_induced_radial_energy_samples()
    minima = bb_induced_radial_minimum_candidates()

    assert minima == (0.625,)
    assert BB_INDUCED_RADIAL_AUDIT_RADII[0] < minima[0] < BB_INDUCED_RADIAL_AUDIT_RADII[-1]
    assert min(energies) == energies[BB_INDUCED_RADIAL_AUDIT_RADII.index(minima[0])]
    assert bb_induced_has_finite_nonzero_minimum()


def test_positive_mass_control_still_allows_bb_induced_breaking() -> None:
    minima = bb_induced_radial_minimum_candidates(mass_squared=POSITIVE_MASS_CONTROL)

    assert minima == (0.375,)
    assert positive_mass_control_keeps_broken_minimum()


def test_negative_controls_recover_no_go_or_unboundedness() -> None:
    assert not bb_induced_has_finite_nonzero_minimum(quartic=0.0)
    assert zero_quartic_recovers_free_bb_no_go()
    assert not quartic_backreaction_is_bounded(quartic=-1.0)


def test_branch_selection_survives_at_bb_induced_minimum() -> None:
    right_gaps = bb_induced_branch_gaps_at_minima("right")
    left_gaps = bb_induced_branch_gaps_at_minima("left")
    dirac_gaps = bb_induced_branch_gaps_at_minima("dirac")

    assert all(gap > FILLED_BAND_ZERO_TOLERANCE for gap in right_gaps)
    assert all(gap < -FILLED_BAND_ZERO_TOLERANCE for gap in left_gaps)
    assert all(abs(gap) <= FILLED_BAND_ZERO_TOLERANCE for gap in dirac_gaps)
    assert bb_induced_branch_selection_survives_at_minimum()


def test_bb_induced_radial_breaking_payload_reports_pass() -> None:
    payload = bb_induced_radial_breaking_audit_payload()

    assert payload.final_verdict == "BB_INDUCED_RADIAL_BREAKING_PASS"
    assert payload.total_minimum_candidates == (0.625,)
    assert payload.positive_mass_control_minima == (0.375,)
    assert payload.free_origin_destabilized
    assert payload.quartic_bounded
    assert payload.backreaction_alone_unbroken
    assert payload.finite_nonzero_minimum
    assert payload.branch_selection_survives
    assert payload.zero_quartic_recovers_no_go
    assert payload.negative_quartic_control_unbounded
    assert payload.positive_mass_control_broken
    assert payload.v39_recovered
    assert payload.v40_recovered
    assert v39_recovered()
    assert v40_recovered()
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_BB_INDUCED_BREAKING
    assert payload.remaining_declared_inputs == (
        "positive_quartic_backreaction_bounds_selector_radius",
    )
