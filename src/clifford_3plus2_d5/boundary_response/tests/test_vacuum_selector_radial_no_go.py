"""Tests for the V38 free BB radial-stabilization no-go gate."""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_ZERO_TOLERANCE,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_radial_no_go import (
    REMAINING_DECLARED_INPUTS_AFTER_FREE_BB_RADIAL_NO_GO,
    branch_gap_is_stable_on_nonzero_radii,
    branch_gap_samples,
    free_bb_has_no_finite_radial_minimum,
    free_bb_radial_minimum_candidates,
    free_bb_radial_stabilization_audit_payload,
    radial_energy_finite_differences,
    radial_energy_is_monotone_decreasing,
    radial_selector_energy_samples,
    v37_recovered,
)


def test_free_bb_radial_energy_is_monotone_decreasing() -> None:
    energies = radial_selector_energy_samples()
    differences = radial_energy_finite_differences()

    assert energies[0] == 0.0
    assert all(energy < 0 for energy in energies[1:])
    assert all(difference < 0 for difference in differences)
    assert radial_energy_is_monotone_decreasing()


def test_free_bb_has_no_interior_radial_minimum() -> None:
    assert free_bb_radial_minimum_candidates() == ()
    assert free_bb_has_no_finite_radial_minimum()


def test_branch_gap_stays_positive_after_origin() -> None:
    gaps = branch_gap_samples()

    assert abs(gaps[0]) <= FILLED_BAND_ZERO_TOLERANCE
    assert all(gap > FILLED_BAND_ZERO_TOLERANCE for gap in gaps[1:])
    assert branch_gap_is_stable_on_nonzero_radii()
    assert v37_recovered()


def test_free_bb_radial_no_go_payload_reports_pass() -> None:
    payload = free_bb_radial_stabilization_audit_payload()

    assert payload.final_verdict == "FREE_BB_RADIAL_STABILIZATION_NO_GO_PASS"
    assert payload.radial_energy_monotone
    assert payload.no_finite_radial_minimum
    assert payload.finite_minimum_candidates == ()
    assert payload.branch_gap_stable
    assert payload.dirac_odd_selector_cancelled
    assert payload.v37_recovered
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_FREE_BB_RADIAL_NO_GO
    assert payload.remaining_declared_inputs == (
        "radial_stabilization_requires_interaction_or_backreaction",
    )
