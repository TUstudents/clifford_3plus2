"""Session 42 finite-spacing Lorentz/rotation recovery tests."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.lorentz_recovery import (
    bcc_dirac_cosine_residual_series,
    bcc_dirac_leading_anisotropy_coefficients,
    bcc_dirac_trace_cosine,
    bcc_weyl_cosine_residual_series,
    bcc_weyl_trace_cosine,
    continuum_cosine_series,
    first_nonzero_epsilon_order,
    hypercube_energy_squared_residual_series,
    hypercube_leading_anisotropy_coefficients,
    lorentz_recovery_audit_payload,
)


def test_trace_cosine_diagnostics_are_normalized_at_origin() -> None:
    epsilon = sp.symbols("epsilon")

    assert bcc_weyl_trace_cosine(epsilon, 0, 0, 0, helicity="right") == 1
    assert bcc_weyl_trace_cosine(epsilon, 0, 0, 0, helicity="left") == 1
    assert bcc_dirac_trace_cosine(epsilon, 0, 0, 0) == 1


def test_continuum_cosine_series_is_rotationally_invariant() -> None:
    epsilon, kx, ky, kz = sp.symbols("epsilon kx ky kz")
    k2 = kx**2 + ky**2 + kz**2

    expected = 1 - epsilon**2 * k2 / 2 + epsilon**4 * k2**2 / 24
    assert sp.simplify(continuum_cosine_series(epsilon, kx, ky, kz, max_order=4) - expected) == 0


def test_weyl_residuals_have_opposite_cubic_anisotropy() -> None:
    epsilon, kx, ky, kz = sp.symbols("epsilon kx ky kz")
    quartic = kx**2 * ky**2 + kx**2 * kz**2 + ky**2 * kz**2
    right_expected = epsilon**3 * (epsilon * quartic - 6 * kx * ky * kz) / 6
    left_expected = epsilon**3 * (epsilon * quartic + 6 * kx * ky * kz) / 6

    assert sp.simplify(
        bcc_weyl_cosine_residual_series(epsilon, kx, ky, kz, helicity="right") - right_expected
    ) == 0
    assert sp.simplify(
        bcc_weyl_cosine_residual_series(epsilon, kx, ky, kz, helicity="left") - left_expected
    ) == 0


def test_dirac_pair_cancels_cubic_anisotropy() -> None:
    epsilon, kx, ky, kz = sp.symbols("epsilon kx ky kz")
    quartic = kx**2 * ky**2 + kx**2 * kz**2 + ky**2 * kz**2
    residual = bcc_dirac_cosine_residual_series(epsilon, kx, ky, kz)

    assert sp.simplify(residual - epsilon**4 * quartic / 6) == 0
    assert first_nonzero_epsilon_order(residual, epsilon) == 4


def test_bcc_dirac_leading_anisotropy_is_direction_dependent_at_order_four() -> None:
    epsilon, q = sp.symbols("epsilon q")

    assert bcc_dirac_leading_anisotropy_coefficients(epsilon, magnitude=q) == {
        "axis": 0,
        "face_diagonal": q**4 / 24,
        "body_diagonal": q**4 / 18,
    }


def test_hypercube_control_has_lower_order_directional_anisotropy() -> None:
    epsilon, kx, ky, kz, q = sp.symbols("epsilon kx ky kz q")
    residual = hypercube_energy_squared_residual_series(epsilon, kx, ky, kz)

    assert sp.expand(residual).coeff(epsilon, 2) == -(kx**4 + ky**4 + kz**4) / 3
    assert first_nonzero_epsilon_order(residual, epsilon) == 2
    assert hypercube_leading_anisotropy_coefficients(epsilon, magnitude=q) == {
        "axis": -q**4 / 3,
        "face_diagonal": -q**4 / 6,
        "body_diagonal": -q**4 / 9,
    }


def test_lorentz_recovery_audit_payload_is_stable() -> None:
    q = sp.symbols("q")
    payload = lorentz_recovery_audit_payload()

    assert payload.bcc_dirac_matches_continuum_through_order == 3
    assert payload.bcc_dirac_first_anisotropy_order == 4
    assert payload.hypercube_first_anisotropy_order == 2
    assert payload.bcc_dirac_directional_coefficients["axis"] == 0
    assert payload.bcc_dirac_directional_coefficients["body_diagonal"] == q**4 / 18
    assert payload.hypercube_directional_coefficients["axis"] == -q**4 / 3
    assert any("free-dispersion audit" in note for note in payload.notes)
