"""V37 microscopic filled-band selector potential gate.

V36 evaluates the chiral BB filled-band selector only on the tetrahedral
candidate branches.  V37 keeps the same BB eigenphase functional but splits it
into parity-even and parity-odd pieces:

    E_even(h) = [E_occ(h) + E_occ(-h)] / 2
    E_odd(h)  = [E_occ(h) - E_occ(-h)] / 2

The odd part is the V35/V36 helicity-locked selector.  The even part is a
radial filled-band contribution on the audited samples.  This gate therefore
checks the microscopic Bloch potential behind the earlier Landau-style
selector audits without claiming a quartic radial stabilization theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_TEST_EPSILON,
    FILLED_BAND_ZERO_TOLERANCE,
    MomentumSample,
    bb_scalar_h2_xyz_coefficient,
    filled_band_parity_odd_energy,
    occupied_filled_band_energy,
    scalar_filled_band_selector_sign_passes,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_filled_band_potential import (
    filled_band_selector_branch_gap,
    filled_band_selector_potential_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_antipodal_controls,
    tetrahedral_selector_candidates,
)
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import Vector3

Helicity = Literal["right", "left", "dirac"]

MICROSCOPIC_FILLED_BAND_ENERGY_TOLERANCE = 1.0e-9
MICROSCOPIC_RADIAL_SCALES = (0.25, 0.5, 1.0)
MICROSCOPIC_PERMUTATION_SAMPLES: tuple[MomentumSample, ...] = (
    (1.0, 2.0, 3.0),
    (2.0, 1.0, 3.0),
    (3.0, 2.0, 1.0),
)

REMAINING_DECLARED_INPUTS_AFTER_MICROSCOPIC_FILLED_BAND_POTENTIAL: tuple[str, ...] = ()


def _to_momentum_sample(
    vector: Vector3,
    *,
    radial_scale: float = 1.0,
) -> MomentumSample:
    """Convert an exact symbolic order-parameter vector to a numeric sample."""

    return tuple(float(sp.N(radial_scale * component)) for component in vector)  # type: ignore[return-value]


def _negate_sample(sample: MomentumSample) -> MomentumSample:
    return -sample[0], -sample[1], -sample[2]


def _all_close(values: tuple[float, ...], *, tolerance: float) -> bool:
    reference = values[0]
    return all(abs(value - reference) <= tolerance for value in values[1:])


def filled_band_effective_energy(
    order_parameter: Vector3 | MomentumSample,
    helicity: Helicity,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
) -> float:
    """Return the occupied filled-band energy for an order-parameter vector."""

    momentum = (
        order_parameter
        if isinstance(order_parameter[0], float)
        else _to_momentum_sample(order_parameter, radial_scale=radial_scale)  # type: ignore[arg-type]
    )
    return occupied_filled_band_energy(
        helicity,
        momentum,  # type: ignore[arg-type]
        epsilon_value=epsilon_value,
    )


def filled_band_even_radial_energy(
    order_parameter: Vector3 | MomentumSample,
    helicity: Helicity,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
) -> float:
    """Return the parity-even part of the filled-band energy."""

    momentum = (
        order_parameter
        if isinstance(order_parameter[0], float)
        else _to_momentum_sample(order_parameter, radial_scale=radial_scale)  # type: ignore[arg-type]
    )
    forward = occupied_filled_band_energy(
        helicity,
        momentum,  # type: ignore[arg-type]
        epsilon_value=epsilon_value,
    )
    backward = occupied_filled_band_energy(
        helicity,
        _negate_sample(momentum),  # type: ignore[arg-type]
        epsilon_value=epsilon_value,
    )
    return (forward + backward) / 2.0


def filled_band_odd_selector_energy(
    order_parameter: Vector3 | MomentumSample,
    helicity: Helicity,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
) -> float:
    """Return the parity-odd selector part of the filled-band energy."""

    momentum = (
        order_parameter
        if isinstance(order_parameter[0], float)
        else _to_momentum_sample(order_parameter, radial_scale=radial_scale)  # type: ignore[arg-type]
    )
    return filled_band_parity_odd_energy(
        helicity,
        momentum,  # type: ignore[arg-type]
        epsilon_value=epsilon_value,
    )


def filled_band_energy_decomposes_into_even_plus_odd(
    sample: MomentumSample = (1.0, 2.0, 3.0),
    *,
    helicity: Helicity = "right",
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = MICROSCOPIC_FILLED_BAND_ENERGY_TOLERANCE,
) -> bool:
    """Return true when ``E(h)=E_even(h)+E_odd(h)`` and antipodes split oppositely."""

    forward = filled_band_effective_energy(
        sample,
        helicity,
        epsilon_value=epsilon_value,
    )
    backward = filled_band_effective_energy(
        _negate_sample(sample),
        helicity,
        epsilon_value=epsilon_value,
    )
    even = filled_band_even_radial_energy(
        sample,
        helicity,
        epsilon_value=epsilon_value,
    )
    odd = filled_band_odd_selector_energy(
        sample,
        helicity,
        epsilon_value=epsilon_value,
    )
    return (
        abs(forward - (even + odd)) <= tolerance
        and abs(backward - (even - odd)) <= tolerance
    )


def selector_candidate_even_energies(
    helicity: Helicity,
    *,
    branch: Literal["selector", "antipode"] = "selector",
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
) -> tuple[float, ...]:
    """Return parity-even energies on selector candidates or antipodes."""

    candidates = (
        tetrahedral_selector_candidates()
        if branch == "selector"
        else tetrahedral_antipodal_controls()
    )
    return tuple(
        filled_band_even_radial_energy(
            candidate,
            helicity,
            epsilon_value=epsilon_value,
            radial_scale=radial_scale,
        )
        for candidate in candidates
    )


def even_energy_is_tetrahedral_on_candidates(
    *,
    helicity: Helicity = "right",
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    radial_scale: float = 1.0,
    tolerance: float = MICROSCOPIC_FILLED_BAND_ENERGY_TOLERANCE,
) -> bool:
    """Return true when the even energy is degenerate on selectors and antipodes."""

    selectors = selector_candidate_even_energies(
        helicity,
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    antipodes = selector_candidate_even_energies(
        helicity,
        branch="antipode",
        epsilon_value=epsilon_value,
        radial_scale=radial_scale,
    )
    return _all_close(selectors + antipodes, tolerance=tolerance)


def even_energy_permutation_symmetric(
    *,
    helicity: Helicity = "right",
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = MICROSCOPIC_FILLED_BAND_ENERGY_TOLERANCE,
) -> bool:
    """Return true when the even energy agrees on sampled coordinate permutations."""

    values = tuple(
        filled_band_even_radial_energy(
            sample,
            helicity,
            epsilon_value=epsilon_value,
        )
        for sample in MICROSCOPIC_PERMUTATION_SAMPLES
    )
    return _all_close(values, tolerance=tolerance)


def even_energy_radial_monotone_on_selector(
    *,
    helicity: Helicity = "right",
    scales: tuple[float, ...] = MICROSCOPIC_RADIAL_SCALES,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> bool:
    """Return true when the filled negative-band energy decreases with radius."""

    selector = tetrahedral_selector_candidates()[0]
    values = tuple(
        filled_band_even_radial_energy(
            selector,
            helicity,
            epsilon_value=epsilon_value,
            radial_scale=scale,
        )
        for scale in scales
    )
    return all(left > right for left, right in zip(values, values[1:], strict=False))


def odd_selector_branch_sign_stable_over_radii(
    *,
    scales: tuple[float, ...] = MICROSCOPIC_RADIAL_SCALES,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> bool:
    """Return true when V36's right-Weyl selector branch gap stays positive."""

    return all(
        filled_band_selector_branch_gap(
            "right",
            epsilon_value=epsilon_value,
            radial_scale=scale,
        )
        > FILLED_BAND_ZERO_TOLERANCE
        for scale in scales
    )


def dirac_odd_selector_cancels_on_samples(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when the Dirac/vector pair has no odd selector on samples."""

    samples = MICROSCOPIC_PERMUTATION_SAMPLES + (
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    return all(
        abs(
            filled_band_odd_selector_energy(
                sample,
                "dirac",
                epsilon_value=epsilon_value,
            )
        )
        <= tolerance
        for sample in samples
    )


def trace_polynomial_probe_remains_blind() -> bool:
    """Return true when the old scalar trace probe still misses the selector."""

    return (
        bb_scalar_h2_xyz_coefficient("right") == 0
        and bb_scalar_h2_xyz_coefficient("left") == 0
        and bb_scalar_h2_xyz_coefficient("dirac") == 0
        and not scalar_filled_band_selector_sign_passes()
    )


def v36_recovered() -> bool:
    """Return true when the V36 branch-selection gate passes."""

    return (
        filled_band_selector_potential_audit_payload().final_verdict
        == "CHIRAL_BB_BRANCH_SELECTION_PASS"
    )


@dataclass(frozen=True)
class MicroscopicFilledBandPotentialAuditPayload:
    """Verdict payload for the V37 microscopic filled-band potential gate."""

    final_verdict: str
    decomposition_identity: bool
    even_tetrahedral_on_candidates: bool
    even_permutation_symmetric: bool
    even_radial_monotone: bool
    odd_branch_sign_stable: bool
    dirac_odd_selector_cancelled: bool
    trace_probe_blind: bool
    v36_recovered: bool
    selector_even_energies: tuple[float, ...]
    antipode_even_energies: tuple[float, ...]
    right_branch_gaps_by_radius: tuple[float, ...]
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def microscopic_selector_potential_audit_payload() -> MicroscopicFilledBandPotentialAuditPayload:
    """Return the V37 microscopic filled-band potential verdict."""

    decomposition = filled_band_energy_decomposes_into_even_plus_odd()
    even_candidates = even_energy_is_tetrahedral_on_candidates()
    even_permutation = even_energy_permutation_symmetric()
    even_monotone = even_energy_radial_monotone_on_selector()
    odd_stable = odd_selector_branch_sign_stable_over_radii()
    dirac_cancelled = dirac_odd_selector_cancels_on_samples()
    trace_blind = trace_polynomial_probe_remains_blind()
    v36 = v36_recovered()
    selector_even = selector_candidate_even_energies("right")
    antipode_even = selector_candidate_even_energies("right", branch="antipode")
    gaps = tuple(
        filled_band_selector_branch_gap("right", radial_scale=scale)
        for scale in MICROSCOPIC_RADIAL_SCALES
    )

    prerequisites_pass = (
        decomposition
        and even_candidates
        and even_permutation
        and even_monotone
        and odd_stable
        and dirac_cancelled
        and trace_blind
        and v36
    )

    if prerequisites_pass:
        final_verdict = "MICROSCOPIC_FILLED_BAND_SELECTOR_POTENTIAL_PASS"
        remaining_inputs: tuple[str, ...] = ()
        interpretation = (
            "The occupied BB eigenphase energy splits into a parity-even "
            "radial filled-band contribution and the V35/V36 helicity-locked "
            "odd selector.  The odd branch sign is stable over audited radii, "
            "while the Dirac/vector pair cancels it."
        )
    else:
        final_verdict = "MICROSCOPIC_FILLED_BAND_SELECTOR_POTENTIAL_KILL"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_MICROSCOPIC_FILLED_BAND_POTENTIAL
        interpretation = (
            "The occupied BB eigenphase energy did not pass the sampled "
            "even/odd microscopic selector-potential diagnostics."
        )

    return MicroscopicFilledBandPotentialAuditPayload(
        final_verdict=final_verdict,
        decomposition_identity=decomposition,
        even_tetrahedral_on_candidates=even_candidates,
        even_permutation_symmetric=even_permutation,
        even_radial_monotone=even_monotone,
        odd_branch_sign_stable=odd_stable,
        dirac_odd_selector_cancelled=dirac_cancelled,
        trace_probe_blind=trace_blind,
        v36_recovered=v36,
        selector_even_energies=selector_even,
        antipode_even_energies=antipode_even,
        right_branch_gaps_by_radius=gaps,
        remaining_declared_inputs=remaining_inputs,
        interpretation=interpretation,
    )
