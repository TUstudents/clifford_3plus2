"""Tests for the V34 Schur-shell origin of the tetrahedral cubic."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector_condensation import (
    norm_squared,
    order_parameter_symbols,
    tetrahedral_cubic_polynomial,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_landau import (
    transform_polynomial,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_schur_landau import (
    REMAINING_DECLARED_INPUTS_AFTER_SCHUR_SHELL_LANDAU,
    cubic_origin_matches_v33,
    eta_sign_flips_only_cubic_branch,
    inversion_flips_cubic_and_is_not_proper_tetrahedral,
    paired_unoriented_shell_cancels_cubic,
    paired_unoriented_shell_series,
    proper_tetrahedral_rotations_preserve_shell_series,
    schur_series_matches_radial_plus_cubic,
    schur_shell_cubic_coefficient,
    schur_shell_landau_audit_payload,
    schur_shell_landau_series,
    schur_shell_quadratic_coefficient,
    schur_shell_symbols,
    schur_sign_flips_all_shell_terms,
    tetrahedral_power_sum_identities_hold,
    tetrahedral_shell_couplings,
    tetrahedral_shell_power_sum,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_cubic_invariant,
    tetrahedral_selector_candidates,
)


def test_shell_couplings_are_tetrahedral_exit_projections() -> None:
    h = order_parameter_symbols()
    couplings = tetrahedral_shell_couplings(h)

    assert len(couplings) == 4
    for coupling, selector in zip(couplings, tetrahedral_selector_candidates(), strict=True):
        expected = sum(component * axis for component, axis in zip(h, selector, strict=True))
        assert sp.simplify(coupling - expected) == 0


def test_first_three_power_sums_are_exact() -> None:
    h = order_parameter_symbols()

    assert tetrahedral_shell_power_sum(1, h) == 0
    assert sp.simplify(tetrahedral_shell_power_sum(2, h) - sp.Rational(4, 3) * norm_squared(h)) == 0
    assert sp.simplify(tetrahedral_shell_power_sum(3, h) - tetrahedral_cubic_polynomial()) == 0
    assert sp.simplify(tetrahedral_shell_power_sum(3, h) - tetrahedral_cubic_invariant(h)) == 0
    assert tetrahedral_power_sum_identities_hold()


def test_schur_shell_series_is_radial_quadratic_plus_cubic() -> None:
    eta, schur_sign = schur_shell_symbols()
    h = order_parameter_symbols()
    expected = (
        schur_shell_quadratic_coefficient(eta=eta, schur_sign=schur_sign) * norm_squared(h)
        + schur_shell_cubic_coefficient(eta=eta, schur_sign=schur_sign)
        * tetrahedral_cubic_polynomial()
    )

    assert sp.simplify(schur_shell_landau_series(h, eta=eta, schur_sign=schur_sign) - expected) == 0
    assert schur_series_matches_radial_plus_cubic()


def test_cubic_origin_matches_v33_minimal_anisotropy() -> None:
    assert cubic_origin_matches_v33()


def test_eta_reversal_flips_only_cubic_branch() -> None:
    eta, schur_sign = schur_shell_symbols()

    assert sp.simplify(
        schur_shell_quadratic_coefficient(eta=-eta, schur_sign=schur_sign)
        - schur_shell_quadratic_coefficient(eta=eta, schur_sign=schur_sign)
    ) == 0
    assert sp.simplify(
        schur_shell_cubic_coefficient(eta=-eta, schur_sign=schur_sign)
        + schur_shell_cubic_coefficient(eta=eta, schur_sign=schur_sign)
    ) == 0
    assert eta_sign_flips_only_cubic_branch()


def test_schur_sign_reversal_flips_all_terms() -> None:
    eta, schur_sign = schur_shell_symbols()
    h = order_parameter_symbols()

    assert sp.simplify(
        schur_shell_landau_series(h, eta=eta, schur_sign=-schur_sign)
        + schur_shell_landau_series(h, eta=eta, schur_sign=schur_sign)
    ) == 0
    assert schur_sign_flips_all_shell_terms()


def test_paired_unoriented_shell_cancels_cubic() -> None:
    eta, schur_sign = schur_shell_symbols()
    h = order_parameter_symbols()
    expected = 2 * schur_shell_quadratic_coefficient(eta=eta, schur_sign=schur_sign) * norm_squared(h)

    assert sp.simplify(paired_unoriented_shell_series(h) - expected) == 0
    assert paired_unoriented_shell_cancels_cubic()


def test_proper_tetrahedral_rotations_preserve_shell_series() -> None:
    assert proper_tetrahedral_rotations_preserve_shell_series()


def test_inversion_flips_cubic_but_is_not_proper_tetrahedral() -> None:
    x, y, z = order_parameter_symbols()
    cubic = tetrahedral_cubic_polynomial()
    inversion = -sp.eye(3)

    assert sp.simplify(transform_polynomial(cubic, inversion) + cubic) == 0
    assert inversion_flips_cubic_and_is_not_proper_tetrahedral()


def test_schur_shell_landau_payload_reports_sign_free_pass() -> None:
    payload = schur_shell_landau_audit_payload()

    assert payload.final_verdict == "SCHUR_SHELL_TETRAHEDRAL_CUBIC_ORIGIN_PASS_SIGN_FREE"
    assert payload.p1_zero
    assert payload.p2_radial
    assert payload.p3_matches_cubic
    assert payload.series_radial_plus_cubic
    assert payload.cubic_origin_matches_v33
    assert payload.eta_sign_flips_only_cubic_branch
    assert payload.schur_sign_flips_all_terms
    assert payload.paired_unoriented_shell_cancels_cubic
    assert payload.proper_rotations_preserve_series
    assert payload.inversion_flips_cubic_and_is_not_proper
    assert payload.v33_recovered
    assert (
        payload.remaining_declared_inputs
        == REMAINING_DECLARED_INPUTS_AFTER_SCHUR_SHELL_LANDAU
    )
    assert payload.remaining_declared_inputs == (
        "oriented_boundary_shell_selects_positive_cubic_branch",
    )
