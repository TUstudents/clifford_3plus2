"""Tests for the V39 Higgs/backreaction radial-stabilizer gate."""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_ZERO_TOLERANCE,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_higgs_stabilizer import (
    RADIAL_STABILIZER_AUDIT_RADII,
    REMAINING_DECLARED_INPUTS_AFTER_HIGGS_BACKREACTION_STABILIZER,
    higgs_backreaction_radial_stabilizer_audit_payload,
    mexican_hat_minimum_candidates,
    mexican_hat_radial_potential_samples,
    quartic_zero_recovers_free_bb_no_go,
    stabilized_branch_gaps_at_minima,
    stabilized_branch_selection_survives_at_minimum,
    stabilized_has_finite_interior_minimum,
    stabilized_radial_energy_samples,
    stabilized_radial_minimum_candidates,
    zero_vev_pure_radial_sector_is_unbroken,
)


def test_mexican_hat_has_expected_grid_minimum() -> None:
    energies = mexican_hat_radial_potential_samples()

    assert mexican_hat_minimum_candidates() == (1.0,)
    assert energies[RADIAL_STABILIZER_AUDIT_RADII.index(1.0)] == 0.0
    assert zero_vev_pure_radial_sector_is_unbroken()


def test_stabilized_radial_energy_has_finite_interior_minimum() -> None:
    energies = stabilized_radial_energy_samples()
    minima = stabilized_radial_minimum_candidates()

    assert minima == (1.0,)
    assert stabilized_has_finite_interior_minimum()
    assert min(energies) == energies[RADIAL_STABILIZER_AUDIT_RADII.index(1.0)]


def test_branch_selection_survives_at_stabilized_minimum() -> None:
    right_gaps = stabilized_branch_gaps_at_minima("right")
    left_gaps = stabilized_branch_gaps_at_minima("left")
    dirac_gaps = stabilized_branch_gaps_at_minima("dirac")

    assert all(gap > FILLED_BAND_ZERO_TOLERANCE for gap in right_gaps)
    assert all(gap < -FILLED_BAND_ZERO_TOLERANCE for gap in left_gaps)
    assert all(abs(gap) <= FILLED_BAND_ZERO_TOLERANCE for gap in dirac_gaps)
    assert stabilized_branch_selection_survives_at_minimum()


def test_zero_quartic_recovers_free_bb_radial_no_go() -> None:
    assert not stabilized_has_finite_interior_minimum(quartic=0.0)
    assert quartic_zero_recovers_free_bb_no_go()


def test_higgs_backreaction_radial_stabilizer_payload_reports_pass() -> None:
    payload = higgs_backreaction_radial_stabilizer_audit_payload()

    assert payload.final_verdict == "HIGGS_BACKREACTION_RADIAL_STABILIZER_PASS"
    assert payload.stabilized_minimum_candidates == (1.0,)
    assert payload.finite_interior_minimum
    assert payload.branch_selection_survives
    assert payload.quartic_zero_recovers_no_go
    assert payload.zero_vev_pure_sector_unbroken
    assert payload.v38_recovered
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_HIGGS_BACKREACTION_STABILIZER
    assert payload.remaining_declared_inputs == (
        "higgs_or_backreaction_sector_supplies_mexican_hat_radial_term",
    )
