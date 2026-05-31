"""Tests for the V37 microscopic filled-band selector potential gate."""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_ZERO_TOLERANCE,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_microscopic_potential import (
    MICROSCOPIC_FILLED_BAND_ENERGY_TOLERANCE,
    REMAINING_DECLARED_INPUTS_AFTER_MICROSCOPIC_FILLED_BAND_POTENTIAL,
    dirac_odd_selector_cancels_on_samples,
    even_energy_is_tetrahedral_on_candidates,
    even_energy_permutation_symmetric,
    even_energy_radial_monotone_on_selector,
    filled_band_effective_energy,
    filled_band_energy_decomposes_into_even_plus_odd,
    filled_band_even_radial_energy,
    filled_band_odd_selector_energy,
    microscopic_selector_potential_audit_payload,
    odd_selector_branch_sign_stable_over_radii,
    selector_candidate_even_energies,
    trace_polynomial_probe_remains_blind,
    v36_recovered,
)


def test_filled_band_energy_splits_into_even_and_odd_parts() -> None:
    sample = (1.0, 2.0, 3.0)
    reverse = (-1.0, -2.0, -3.0)
    forward = filled_band_effective_energy(sample, "right")
    backward = filled_band_effective_energy(reverse, "right")
    even = filled_band_even_radial_energy(sample, "right")
    odd = filled_band_odd_selector_energy(sample, "right")

    assert abs(forward - (even + odd)) <= MICROSCOPIC_FILLED_BAND_ENERGY_TOLERANCE
    assert abs(backward - (even - odd)) <= MICROSCOPIC_FILLED_BAND_ENERGY_TOLERANCE
    assert odd < 0
    assert filled_band_energy_decomposes_into_even_plus_odd()


def test_even_energy_is_tetrahedral_and_radial_on_audited_samples() -> None:
    selectors = selector_candidate_even_energies("right")
    antipodes = selector_candidate_even_energies("right", branch="antipode")

    assert all(
        abs(value - selectors[0]) <= MICROSCOPIC_FILLED_BAND_ENERGY_TOLERANCE
        for value in selectors[1:] + antipodes
    )
    assert selectors[0] < 0
    assert even_energy_is_tetrahedral_on_candidates()
    assert even_energy_permutation_symmetric()
    assert even_energy_radial_monotone_on_selector()


def test_odd_selector_is_stable_and_vector_sector_cancels() -> None:
    assert odd_selector_branch_sign_stable_over_radii()
    assert dirac_odd_selector_cancels_on_samples()
    assert trace_polynomial_probe_remains_blind()
    assert v36_recovered()


def test_microscopic_selector_potential_payload_reports_pass() -> None:
    payload = microscopic_selector_potential_audit_payload()

    assert payload.final_verdict == "MICROSCOPIC_FILLED_BAND_SELECTOR_POTENTIAL_PASS"
    assert payload.decomposition_identity
    assert payload.even_tetrahedral_on_candidates
    assert payload.even_permutation_symmetric
    assert payload.even_radial_monotone
    assert payload.odd_branch_sign_stable
    assert payload.dirac_odd_selector_cancelled
    assert payload.trace_probe_blind
    assert payload.v36_recovered
    assert all(gap > FILLED_BAND_ZERO_TOLERANCE for gap in payload.right_branch_gaps_by_radius)
    assert (
        payload.remaining_declared_inputs
        == REMAINING_DECLARED_INPUTS_AFTER_MICROSCOPIC_FILLED_BAND_POTENTIAL
    )
    assert payload.remaining_declared_inputs == ()
