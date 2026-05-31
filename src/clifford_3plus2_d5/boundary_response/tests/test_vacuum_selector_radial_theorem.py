"""Tests for the V42 analytic radial theorem."""

from __future__ import annotations

from math import isclose

from clifford_3plus2_d5.boundary_response.vacuum_selector_radial_theorem import (
    DEFAULT_SELECTOR_SLOPE,
    analytic_branch_selection_survives_at_minimum,
    analytic_finite_nonzero_minimum_exists,
    analytic_minimum_lowers_origin,
    analytic_origin_is_destabilized,
    analytic_quartic_bounds_radius,
    analytic_radial_breaking_theorem_audit_payload,
    analytic_radial_derivative,
    analytic_radial_energy,
    analytic_radial_second_derivative,
    continuum_occupied_weyl_energy,
    massless_quartic_stationary_condition_passes,
    massless_quartic_stationary_point_is_minimum,
    massless_quartic_stationary_radius,
    v41_recovered,
)


def test_continuum_occupied_weyl_energy_destabilizes_origin() -> None:
    assert continuum_occupied_weyl_energy(0.0) == 0.0
    assert continuum_occupied_weyl_energy(0.25) < 0.0
    assert analytic_radial_energy(0.0) == 0.0
    assert analytic_radial_derivative(0.0) == -DEFAULT_SELECTOR_SLOPE
    assert analytic_origin_is_destabilized()


def test_positive_quartic_forces_finite_nonzero_minimum_condition() -> None:
    assert analytic_quartic_bounds_radius(quartic=1.0)
    assert analytic_finite_nonzero_minimum_exists(slope=1.0, quartic=1.0)
    assert not analytic_finite_nonzero_minimum_exists(slope=0.0, quartic=1.0)
    assert not analytic_finite_nonzero_minimum_exists(slope=1.0, quartic=0.0)
    assert not analytic_quartic_bounds_radius(quartic=-1.0)


def test_massless_quartic_stationary_radius_is_explicit_minimum() -> None:
    radius = massless_quartic_stationary_radius()

    assert isclose(radius, (1.0 / 4.0) ** (1.0 / 3.0), abs_tol=1.0e-12)
    assert massless_quartic_stationary_condition_passes()
    assert massless_quartic_stationary_point_is_minimum()
    assert analytic_radial_second_derivative(radius) > 0.0
    assert analytic_minimum_lowers_origin()
    assert analytic_radial_energy(radius) < analytic_radial_energy(0.0)


def test_branch_selection_survives_at_analytic_minimum() -> None:
    assert analytic_branch_selection_survives_at_minimum()


def test_analytic_radial_breaking_payload_reports_pass() -> None:
    payload = analytic_radial_breaking_theorem_audit_payload()

    assert payload.final_verdict == "ANALYTIC_RADIAL_BREAKING_THEOREM_PASS"
    assert payload.selector_slope == 1.0
    assert payload.mass_squared == 0.0
    assert payload.quartic == 1.0
    assert payload.energy_at_stationary_radius < payload.energy_at_origin
    assert payload.origin_destabilized
    assert payload.quartic_bounded
    assert payload.finite_nonzero_minimum_exists
    assert payload.stationary_condition
    assert payload.stationary_point_minimum
    assert payload.minimum_lowers_origin
    assert payload.branch_selection_survives
    assert payload.zero_slope_control_rejected
    assert payload.zero_quartic_control_rejected
    assert payload.negative_quartic_control_rejected
    assert payload.v41_recovered
    assert v41_recovered()
    assert payload.remaining_declared_inputs == (
        "positive_quartic_backreaction_bounds_selector_radius",
    )
