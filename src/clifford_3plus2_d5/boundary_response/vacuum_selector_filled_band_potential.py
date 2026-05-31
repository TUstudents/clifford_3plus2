"""V36 chiral BB branch-selection potential gate.

V35 proves that the single-Weyl Bialynicki-Birula walk has a real
helicity-locked ``A2u`` filled-band selector term.  V36 evaluates that
selector on the actual V27/V32 tetrahedral vacuum candidates.

The four accepted tetrahedral selector representatives all have positive
``xyz`` product, while their antipodes have negative product.  The right-Weyl
filled-band selector therefore lowers the accepted branch and raises the
antipodal branch.  The left-Weyl block reverses the sign, and the Dirac/vector
pair cancels it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector import (
    zero_order_parameter,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_RATIO_TOLERANCE,
    FILLED_BAND_TEST_EPSILON,
    FILLED_BAND_ZERO_TOLERANCE,
    MomentumSample,
    Sector,
    chiral_bb_selector_sign_audit_payload,
    filled_band_parity_odd_energy,
    filled_band_selector_ratio,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_antipodal_controls,
    tetrahedral_selector_candidates,
)
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import Vector3

Helicity = Literal["right", "left", "dirac"]

REMAINING_DECLARED_INPUTS_AFTER_FILLED_BAND_BRANCH_SELECTION: tuple[str, ...] = ()


def _to_momentum_sample(
    vector: Vector3,
    *,
    radial_scale: float = 1.0,
) -> MomentumSample:
    """Convert an exact symbolic order-parameter vector to a numeric sample."""

    return tuple(float(sp.N(radial_scale * component)) for component in vector)  # type: ignore[return-value]


def _all_close(values: tuple[float, ...], *, tolerance: float) -> bool:
    reference = values[0]
    return all(abs(value - reference) <= tolerance for value in values[1:])


def _all_opposite(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    tolerance: float,
) -> bool:
    return all(abs(left_value + right_value) <= tolerance for left_value, right_value in zip(left, right, strict=True))


def filled_band_selector_candidate_energy(
    candidate: Vector3,
    helicity: Helicity,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
) -> float:
    """Return the filled-band selector energy on one candidate direction."""

    return filled_band_parity_odd_energy(
        helicity,
        _to_momentum_sample(candidate, radial_scale=radial_scale),
        epsilon_value=epsilon_value,
    )


def filled_band_selector_branch_energies(
    helicity: Helicity,
    *,
    branch: Literal["selector", "antipode"] = "selector",
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
) -> tuple[float, ...]:
    """Return filled-band energies for the selector branch or its antipodes."""

    candidates = (
        tetrahedral_selector_candidates()
        if branch == "selector"
        else tetrahedral_antipodal_controls()
    )
    return tuple(
        filled_band_selector_candidate_energy(
            candidate,
            helicity,
            epsilon_value=epsilon_value,
            radial_scale=radial_scale,
        )
        for candidate in candidates
    )


def filled_band_selector_branch_gap(
    helicity: Helicity,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
) -> float:
    """Return ``min(E_antipode)-max(E_selector)`` for one helicity."""

    selectors = filled_band_selector_branch_energies(
        helicity,
        branch="selector",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    antipodes = filled_band_selector_branch_energies(
        helicity,
        branch="antipode",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    return min(antipodes) - max(selectors)


def filled_band_selector_branch_ratios(
    helicity: Sector,
    *,
    branch: Literal["selector", "antipode"] = "selector",
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
) -> tuple[float, ...]:
    """Return normalized selector ratios for a branch of tetrahedral candidates."""

    candidates = (
        tetrahedral_selector_candidates()
        if branch == "selector"
        else tetrahedral_antipodal_controls()
    )
    return tuple(
        filled_band_selector_ratio(
            helicity,
            _to_momentum_sample(candidate, radial_scale=radial_scale),
            epsilon_value=epsilon_value,
        )
        for candidate in candidates
    )


def zero_axis_controls_have_zero_energy(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when the origin and coordinate axes have zero selector energy."""

    controls = (
        zero_order_parameter(),
        (sp.Integer(1), sp.Integer(0), sp.Integer(0)),
        (sp.Integer(0), sp.Integer(1), sp.Integer(0)),
        (sp.Integer(0), sp.Integer(0), sp.Integer(1)),
    )
    return all(
        abs(
            filled_band_selector_candidate_energy(
                control,
                "right",
                epsilon_value=epsilon_value,
                radial_scale=radial_scale,
            )
        )
        <= tolerance
        for control in controls
    )


def right_weyl_selects_tetrahedral_branch(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when right-Weyl energies lower selectors over antipodes."""

    selectors = filled_band_selector_branch_energies(
        "right",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    antipodes = filled_band_selector_branch_energies(
        "right",
        branch="antipode",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    return (
        _all_close(selectors, tolerance=tolerance)
        and _all_close(antipodes, tolerance=tolerance)
        and _all_opposite(selectors, antipodes, tolerance=tolerance)
        and max(selectors) < min(antipodes)
    )


def left_weyl_reverses_tetrahedral_branch(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when left-Weyl energies reverse the right-Weyl branch."""

    right_selectors = filled_band_selector_branch_energies(
        "right",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    left_selectors = filled_band_selector_branch_energies(
        "left",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    left_gap = filled_band_selector_branch_gap(
        "left",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    return _all_opposite(right_selectors, left_selectors, tolerance=tolerance) and left_gap < 0


def dirac_pair_has_no_tetrahedral_branch_gap(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when the vector/Dirac pair cancels the branch gap."""

    selectors = filled_band_selector_branch_energies(
        "dirac",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    antipodes = filled_band_selector_branch_energies(
        "dirac",
        branch="antipode",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    return all(abs(value) <= tolerance for value in selectors + antipodes)


def selector_ratios_match_v35_a2u(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
    tolerance: float = FILLED_BAND_RATIO_TOLERANCE,
) -> bool:
    """Return true when selector candidates carry one normalized ``A2u`` ratio."""

    selector_ratios = filled_band_selector_branch_ratios(
        "right",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    antipode_ratios = filled_band_selector_branch_ratios(
        "right",
        branch="antipode",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    return (
        _all_close(selector_ratios, tolerance=tolerance)
        and _all_close(antipode_ratios, tolerance=tolerance)
        and _all_close(selector_ratios + antipode_ratios, tolerance=tolerance)
    )


def v35_recovered() -> bool:
    """Return true when the V35 filled-band selector-sign gate passes."""

    return (
        chiral_bb_selector_sign_audit_payload().final_verdict
        == "CHIRAL_BB_FILLED_BAND_SELECTOR_SIGN_PASS"
    )


@dataclass(frozen=True)
class FilledBandSelectorPotentialAuditPayload:
    """Verdict payload for the V36 chiral BB branch-selection gate."""

    final_verdict: str
    right_selector_energies: tuple[float, ...]
    right_antipode_energies: tuple[float, ...]
    left_selector_energies: tuple[float, ...]
    left_antipode_energies: tuple[float, ...]
    dirac_selector_energies: tuple[float, ...]
    dirac_antipode_energies: tuple[float, ...]
    right_branch_gap: float
    left_branch_gap: float
    dirac_branch_gap: float
    right_branch_selected: bool
    left_branch_reversed: bool
    dirac_branch_cancelled: bool
    zero_axis_controls_zero: bool
    selector_ratios_match_a2u: bool
    v35_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def filled_band_selector_potential_audit_payload() -> FilledBandSelectorPotentialAuditPayload:
    """Return the V36 chiral BB branch-selection verdict."""

    right_selectors = filled_band_selector_branch_energies("right")
    right_antipodes = filled_band_selector_branch_energies("right", branch="antipode")
    left_selectors = filled_band_selector_branch_energies("left")
    left_antipodes = filled_band_selector_branch_energies("left", branch="antipode")
    dirac_selectors = filled_band_selector_branch_energies("dirac")
    dirac_antipodes = filled_band_selector_branch_energies("dirac", branch="antipode")
    right_gap = filled_band_selector_branch_gap("right")
    left_gap = filled_band_selector_branch_gap("left")
    dirac_gap = filled_band_selector_branch_gap("dirac")
    right_selected = right_weyl_selects_tetrahedral_branch()
    left_reversed = left_weyl_reverses_tetrahedral_branch()
    dirac_cancelled = dirac_pair_has_no_tetrahedral_branch_gap()
    zero_controls = zero_axis_controls_have_zero_energy()
    ratios_match = selector_ratios_match_v35_a2u()
    v35 = v35_recovered()

    prerequisites_pass = (
        right_selected
        and left_reversed
        and dirac_cancelled
        and zero_controls
        and ratios_match
        and v35
    )

    if prerequisites_pass:
        final_verdict = "CHIRAL_BB_BRANCH_SELECTION_PASS"
        remaining_inputs: tuple[str, ...] = ()
        interpretation = (
            "The V35 single-Weyl filled-band A2u term lowers the accepted "
            "tetrahedral selector branch over its antipodes.  The sign is "
            "locked to helicity, and the Dirac/vector pair cancels the branch "
            "gap."
        )
    else:
        final_verdict = "CHIRAL_BB_BRANCH_SELECTION_KILL"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_FILLED_BAND_BRANCH_SELECTION
        interpretation = (
            "The filled-band A2u term did not produce the required branch "
            "selection pattern on the tetrahedral selector candidates."
        )

    return FilledBandSelectorPotentialAuditPayload(
        final_verdict=final_verdict,
        right_selector_energies=right_selectors,
        right_antipode_energies=right_antipodes,
        left_selector_energies=left_selectors,
        left_antipode_energies=left_antipodes,
        dirac_selector_energies=dirac_selectors,
        dirac_antipode_energies=dirac_antipodes,
        right_branch_gap=right_gap,
        left_branch_gap=left_gap,
        dirac_branch_gap=dirac_gap,
        right_branch_selected=right_selected,
        left_branch_reversed=left_reversed,
        dirac_branch_cancelled=dirac_cancelled,
        zero_axis_controls_zero=zero_controls,
        selector_ratios_match_a2u=ratios_match,
        v35_recovered=v35,
        remaining_declared_inputs=remaining_inputs,
        interpretation=interpretation,
    )
