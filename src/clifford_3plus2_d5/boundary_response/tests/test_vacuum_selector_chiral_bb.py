"""Tests for the V35 chiral BB filled-band selector-sign audit."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    FILLED_BAND_NONPOLYNOMIAL_GAP,
    FILLED_BAND_RATIO_TOLERANCE,
    FILLED_BAND_ZERO_TOLERANCE,
    REMAINING_DECLARED_INPUTS_AFTER_CHIRAL_BB_SELECTOR_SIGN,
    bb_effective_hamiltonian_coefficient,
    bb_scalar_h2_quadratic_coefficients,
    bb_scalar_h2_xyz_coefficient,
    bb_trace_b3_xyz_coefficient,
    chiral_bb_selector_sign_audit_payload,
    dirac_leading_hamiltonian_matches,
    filled_band_angular_ratios,
    filled_band_dirac_cancels,
    filled_band_helicity_relation_passes,
    filled_band_parity_odd_energy,
    filled_band_selector_is_nonpolynomial_angular,
    filled_band_selector_ratio,
    filled_band_selector_sign_passes,
    filled_band_signed_orbit_is_a2u,
    filled_band_signed_orbit_ratios,
    filled_band_zero_locus_passes,
    leading_weyl_hamiltonian_matches,
    scalar_filled_band_selector_sign_passes,
    scalar_h2_xyz_is_real,
    scalar_quadratic_is_helicity_independent,
    symbolic_momentum,
    trace_b3_diagnostic_passes,
    v34_recovered,
    xyz_is_a2u_under_inversion_and_tetrahedral,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_condensation import (
    order_parameter_symbols,
    tetrahedral_cubic_polynomial,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_landau import (
    tetrahedral_rotation_group,
    transform_polynomial,
)
from clifford_3plus2_d5.spacetime_qca.bcc_weyl import expected_weyl_hamiltonian
from clifford_3plus2_d5.spacetime_qca.dirac import block_diag


def test_floquet_trace_b3_xyz_is_helicity_odd_and_dirac_cancels() -> None:
    assert bb_trace_b3_xyz_coefficient("right") == -2
    assert bb_trace_b3_xyz_coefficient("left") == 2
    assert bb_trace_b3_xyz_coefficient("dirac") == 0
    assert trace_b3_diagnostic_passes()


def test_matrix_log_leading_hamiltonians_match_weyl_convention() -> None:
    _epsilon, kx, ky, kz = symbolic_momentum()
    expected_right = expected_weyl_hamiltonian(kx, ky, kz, helicity="right")
    expected_left = expected_weyl_hamiltonian(kx, ky, kz, helicity="left")
    expected_dirac = block_diag(expected_right, expected_left)

    assert (
        bb_effective_hamiltonian_coefficient("right", 0) - expected_right
    ).applyfunc(sp.simplify) == sp.zeros(2)
    assert (
        bb_effective_hamiltonian_coefficient("left", 0) - expected_left
    ).applyfunc(sp.simplify) == sp.zeros(2)
    assert (
        bb_effective_hamiltonian_coefficient("dirac", 0) - expected_dirac
    ).applyfunc(sp.simplify) == sp.zeros(4)
    assert leading_weyl_hamiltonian_matches("right")
    assert leading_weyl_hamiltonian_matches("left")
    assert dirac_leading_hamiltonian_matches()


def test_scalar_h2_xyz_trace_probe_is_blind_negative_control() -> None:
    right = bb_scalar_h2_xyz_coefficient("right")
    left = bb_scalar_h2_xyz_coefficient("left")
    dirac = bb_scalar_h2_xyz_coefficient("dirac")

    assert right == 0
    assert left == 0
    assert dirac == 0
    assert all(scalar_h2_xyz_is_real(value) for value in (right, left, dirac))
    assert not scalar_filled_band_selector_sign_passes()


def test_filled_band_energy_has_real_helicity_locked_selector() -> None:
    momentum = (1.0, 2.0, 3.0)
    right = filled_band_parity_odd_energy("right", momentum)
    left = filled_band_parity_odd_energy("left", momentum)
    dirac = filled_band_parity_odd_energy("dirac", momentum)

    assert abs(right) > FILLED_BAND_ZERO_TOLERANCE
    assert abs(right + left) <= FILLED_BAND_ZERO_TOLERANCE
    assert abs(dirac) <= FILLED_BAND_ZERO_TOLERANCE
    assert right < 0
    assert filled_band_helicity_relation_passes()
    assert filled_band_dirac_cancels()
    assert filled_band_selector_sign_passes()


def test_filled_band_selector_is_a2u_angular_term() -> None:
    assert filled_band_zero_locus_passes()
    assert filled_band_signed_orbit_is_a2u()

    orbit_ratios = filled_band_signed_orbit_ratios()
    reference = orbit_ratios[0]
    assert reference < 0
    assert all(
        abs(ratio - reference) <= FILLED_BAND_RATIO_TOLERANCE
        for ratio in orbit_ratios[1:]
    )

    angular_ratios = filled_band_angular_ratios()
    assert max(angular_ratios) - min(angular_ratios) >= FILLED_BAND_NONPOLYNOMIAL_GAP
    assert filled_band_selector_is_nonpolynomial_angular()


def test_filled_band_selector_ratio_tracks_xyz_sign() -> None:
    positive = filled_band_selector_ratio("right", (1.0, 2.0, 3.0))
    negative = filled_band_selector_ratio("right", (-1.0, 2.0, 3.0))

    assert abs(positive - negative) <= FILLED_BAND_RATIO_TOLERANCE
    assert positive < 0


def test_scalar_quadratic_control_is_helicity_independent() -> None:
    assert bb_scalar_h2_quadratic_coefficients("right") == (0, 0, 0, 0, 0, 0)
    assert bb_scalar_h2_quadratic_coefficients("left") == (0, 0, 0, 0, 0, 0)
    assert bb_scalar_h2_quadratic_coefficients("dirac") == (0, 0, 0, 0, 0, 0)
    assert scalar_quadratic_is_helicity_independent()


def test_xyz_is_inversion_odd_and_proper_tetrahedral_invariant() -> None:
    x, y, z = order_parameter_symbols()
    cubic = x * y * z
    inverted = cubic.subs({x: -x, y: -y, z: -z}, simultaneous=True)

    assert sp.simplify(inverted + cubic) == 0
    assert all(
        sp.simplify(
            transform_polynomial(tetrahedral_cubic_polynomial(), rotation)
            - tetrahedral_cubic_polynomial()
        )
        == 0
        for rotation in tetrahedral_rotation_group()
    )
    assert xyz_is_a2u_under_inversion_and_tetrahedral()


def test_v34_regression_passes() -> None:
    assert v34_recovered()


def test_chiral_bb_selector_sign_payload_reports_filled_band_pass() -> None:
    payload = chiral_bb_selector_sign_audit_payload()

    assert payload.final_verdict == "CHIRAL_BB_FILLED_BAND_SELECTOR_SIGN_PASS"
    assert payload.trace_b3_xyz_right == -2
    assert payload.trace_b3_xyz_left == 2
    assert payload.trace_b3_xyz_dirac == 0
    assert payload.trace_b3_diagnostic_passes
    assert payload.leading_right_hamiltonian_matches
    assert payload.leading_left_hamiltonian_matches
    assert payload.leading_dirac_hamiltonian_matches
    assert payload.scalar_h2_xyz_right == 0
    assert payload.scalar_h2_xyz_left == 0
    assert payload.scalar_h2_xyz_dirac == 0
    assert payload.scalar_xyz_real
    assert not payload.scalar_selector_sign_passes
    assert payload.scalar_trace_probe_blind
    assert payload.scalar_quadratic_helicity_independent
    assert abs(payload.filled_band_selector_right) > FILLED_BAND_ZERO_TOLERANCE
    assert abs(payload.filled_band_selector_right + payload.filled_band_selector_left) <= FILLED_BAND_ZERO_TOLERANCE
    assert abs(payload.filled_band_selector_dirac) <= FILLED_BAND_ZERO_TOLERANCE
    assert payload.filled_band_right_selector_ratio < 0
    assert payload.filled_band_helicity_relation
    assert payload.filled_band_dirac_cancellation
    assert payload.filled_band_zero_locus
    assert payload.filled_band_a2u_signed_orbit
    assert payload.filled_band_nonpolynomial_angular
    assert payload.filled_band_selector_sign_passes
    assert payload.xyz_a2u_check
    assert payload.v34_recovered
    assert (
        payload.remaining_declared_inputs
        == REMAINING_DECLARED_INPUTS_AFTER_CHIRAL_BB_SELECTOR_SIGN
    )
    assert payload.remaining_declared_inputs == ()
