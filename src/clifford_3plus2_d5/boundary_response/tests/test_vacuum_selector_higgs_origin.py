"""Tests for the V40 Higgs radial-origin theorem gate."""

from __future__ import annotations

from math import isclose

from clifford_3plus2_d5.boundary_response.vacuum_selector_higgs_origin import (
    HIGGS_ORIGIN_TOLERANCE,
    REMAINING_DECLARED_INPUTS_AFTER_HIGGS_RADIAL_ORIGIN,
    anisotropic_doublet_term_fails_radial_criterion,
    charged_and_neutral_representatives_are_degenerate,
    completed_square_constant,
    completed_square_higgs_potential,
    completed_square_vev_squared,
    doublet_quadratic_is_gauge_radial,
    higgs_radial_landau_origin_audit_payload,
    quartic_has_broken_phase,
    quartic_higgs_landau_potential,
    quartic_is_bounded_below,
    radial_term_from_broken_quartic_matches_v39,
    spacetime_higgs_potential_pullback,
    spacetime_pullback_matches_v39_radial_term,
    square_completion_matches,
    v39_recovered,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_higgs_stabilizer import (
    RADIAL_STABILIZER_AUDIT_RADII,
    mexican_hat_radial_potential,
)


def test_broken_bounded_quartic_completes_to_mexican_hat() -> None:
    assert quartic_is_bounded_below(quartic=1.0)
    assert quartic_has_broken_phase(quadratic=-2.0, quartic=1.0)
    assert completed_square_vev_squared(quadratic=-2.0, quartic=1.0) == 1.0
    assert completed_square_constant(quadratic=-2.0, quartic=1.0, constant=0.0) == -1.0
    assert square_completion_matches()

    for radius in RADIAL_STABILIZER_AUDIT_RADII:
        rho = radius * radius
        assert isclose(
            quartic_higgs_landau_potential(rho),
            completed_square_higgs_potential(rho),
            abs_tol=HIGGS_ORIGIN_TOLERANCE,
        )


def test_bad_sign_controls_fail_broken_bounded_phase() -> None:
    assert not quartic_has_broken_phase(quadratic=1.0, quartic=1.0)
    assert not quartic_is_bounded_below(quartic=-1.0)
    assert not quartic_has_broken_phase(quadratic=-2.0, quartic=-1.0)


def test_spacetime_higgs_pullback_matches_v39_radial_term() -> None:
    assert radial_term_from_broken_quartic_matches_v39()
    assert spacetime_pullback_matches_v39_radial_term(component="neutral")
    assert spacetime_pullback_matches_v39_radial_term(component="charged")

    for radius in RADIAL_STABILIZER_AUDIT_RADII:
        assert isclose(
            spacetime_higgs_potential_pullback(radius, component="neutral"),
            mexican_hat_radial_potential(radius),
            abs_tol=HIGGS_ORIGIN_TOLERANCE,
        )


def test_same_norm_higgs_representatives_are_degenerate() -> None:
    assert charged_and_neutral_representatives_are_degenerate()
    assert doublet_quadratic_is_gauge_radial(charged_coefficient=1.0, neutral_coefficient=1.0)
    assert anisotropic_doublet_term_fails_radial_criterion()
    assert not doublet_quadratic_is_gauge_radial(charged_coefficient=1.0, neutral_coefficient=2.0)


def test_higgs_radial_landau_origin_payload_reports_pass() -> None:
    payload = higgs_radial_landau_origin_audit_payload()

    assert payload.final_verdict == "HIGGS_RADIAL_LANDAU_UNIQUENESS_PASS"
    assert payload.vev_squared == 1.0
    assert payload.completed_square_constant == -1.0
    assert payload.bounded_below
    assert payload.broken_phase
    assert payload.square_completion
    assert payload.radial_term_matches_v39
    assert payload.neutral_pullback_matches_v39
    assert payload.charged_pullback_matches_v39
    assert payload.same_norm_degeneracy
    assert payload.positive_quadratic_control_unbroken
    assert payload.negative_quartic_control_unbounded
    assert payload.anisotropic_control_rejected
    assert payload.v39_recovered
    assert v39_recovered()
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_HIGGS_RADIAL_ORIGIN
    assert payload.remaining_declared_inputs == (
        "higgs_backreaction_sector_enters_broken_quartic_phase",
    )
