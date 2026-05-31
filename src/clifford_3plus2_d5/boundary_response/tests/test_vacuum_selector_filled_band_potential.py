"""Tests for the V36 chiral BB branch-selection potential gate."""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_RATIO_TOLERANCE,
    FILLED_BAND_ZERO_TOLERANCE,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_filled_band_potential import (
    REMAINING_DECLARED_INPUTS_AFTER_FILLED_BAND_BRANCH_SELECTION,
    dirac_pair_has_no_tetrahedral_branch_gap,
    filled_band_selector_branch_energies,
    filled_band_selector_branch_gap,
    filled_band_selector_branch_ratios,
    filled_band_selector_potential_audit_payload,
    left_weyl_reverses_tetrahedral_branch,
    right_weyl_selects_tetrahedral_branch,
    selector_ratios_match_v35_a2u,
    v35_recovered,
    zero_axis_controls_have_zero_energy,
)


def test_right_weyl_lowers_selectors_over_antipodes() -> None:
    selectors = filled_band_selector_branch_energies("right")
    antipodes = filled_band_selector_branch_energies("right", branch="antipode")
    gap = filled_band_selector_branch_gap("right")

    assert all(value < 0 for value in selectors)
    assert all(value > 0 for value in antipodes)
    assert all(abs(value - selectors[0]) <= FILLED_BAND_ZERO_TOLERANCE for value in selectors[1:])
    assert all(abs(value - antipodes[0]) <= FILLED_BAND_ZERO_TOLERANCE for value in antipodes[1:])
    assert all(abs(left + right) <= FILLED_BAND_ZERO_TOLERANCE for left, right in zip(selectors, antipodes, strict=True))
    assert gap > 0
    assert right_weyl_selects_tetrahedral_branch()


def test_left_weyl_reverses_selector_branch() -> None:
    right_selectors = filled_band_selector_branch_energies("right")
    left_selectors = filled_band_selector_branch_energies("left")

    assert all(
        abs(right + left) <= FILLED_BAND_ZERO_TOLERANCE
        for right, left in zip(right_selectors, left_selectors, strict=True)
    )
    assert filled_band_selector_branch_gap("left") < 0
    assert left_weyl_reverses_tetrahedral_branch()


def test_dirac_pair_cancels_branch_gap_and_axes_are_controls() -> None:
    selectors = filled_band_selector_branch_energies("dirac")
    antipodes = filled_band_selector_branch_energies("dirac", branch="antipode")

    assert all(abs(value) <= FILLED_BAND_ZERO_TOLERANCE for value in selectors + antipodes)
    assert abs(filled_band_selector_branch_gap("dirac")) <= FILLED_BAND_ZERO_TOLERANCE
    assert dirac_pair_has_no_tetrahedral_branch_gap()
    assert zero_axis_controls_have_zero_energy()


def test_selector_ratios_match_v35_a2u_ratio() -> None:
    selector_ratios = filled_band_selector_branch_ratios("right")
    antipode_ratios = filled_band_selector_branch_ratios("right", branch="antipode")
    reference = selector_ratios[0]

    assert reference < 0
    assert all(
        abs(ratio - reference) <= FILLED_BAND_RATIO_TOLERANCE
        for ratio in selector_ratios[1:] + antipode_ratios
    )
    assert selector_ratios_match_v35_a2u()
    assert v35_recovered()


def test_filled_band_selector_potential_payload_reports_pass() -> None:
    payload = filled_band_selector_potential_audit_payload()

    assert payload.final_verdict == "CHIRAL_BB_BRANCH_SELECTION_PASS"
    assert payload.right_branch_gap > 0
    assert payload.left_branch_gap < 0
    assert abs(payload.dirac_branch_gap) <= FILLED_BAND_ZERO_TOLERANCE
    assert payload.right_branch_selected
    assert payload.left_branch_reversed
    assert payload.dirac_branch_cancelled
    assert payload.zero_axis_controls_zero
    assert payload.selector_ratios_match_a2u
    assert payload.v35_recovered
    assert (
        payload.remaining_declared_inputs
        == REMAINING_DECLARED_INPUTS_AFTER_FILLED_BAND_BRANCH_SELECTION
    )
    assert payload.remaining_declared_inputs == ()
