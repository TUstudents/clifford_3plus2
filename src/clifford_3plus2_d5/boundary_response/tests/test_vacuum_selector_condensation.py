"""Tests for the V32 continuous tetrahedral selector-condensation gate."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector import (
    generic_order_parameter,
    midpoint_order_parameter,
    zero_order_parameter,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_condensation import (
    REMAINING_DECLARED_INPUTS_AFTER_SELECTOR_CONDENSATION,
    axis_controls_rejected,
    coordinate_axis_controls,
    cubic_only_control_rejected,
    cubic_polynomial_matches_invariant,
    locking_potential_zero_only_at_selectors,
    radial_locking_potential,
    radial_only_control_rejected,
    selector_locking_potential,
    selector_minimizers,
    stationary_candidates_satisfy_lagrange_equations,
    symbolic_order_parameter,
    tetrahedral_cubic_polynomial,
    unit_sphere_lagrange_residual,
    unit_sphere_max_value,
    unit_sphere_stationary_candidates,
    vacuum_selector_condensation_audit_payload,
    v31_recovered,
    wrong_sign_control_rejected,
    wrong_sign_locking_potential,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_antipodal_controls,
    tetrahedral_cubic_invariant,
    tetrahedral_midpoint_controls,
    tetrahedral_selector_candidates,
)


def test_tetrahedral_cubic_invariant_has_closed_form_polynomial() -> None:
    x, y, z = symbolic_order_parameter()
    assert tetrahedral_cubic_polynomial() == sp.simplify(8 * x * y * z / sp.sqrt(3))
    assert sp.simplify(
        tetrahedral_cubic_invariant(symbolic_order_parameter())
        - tetrahedral_cubic_polynomial()
    ) == 0
    assert cubic_polynomial_matches_invariant()


def test_selector_and_antipodal_branches_have_exact_cubic_values() -> None:
    assert all(
        tetrahedral_cubic_invariant(candidate) == sp.Rational(8, 9)
        for candidate in tetrahedral_selector_candidates()
    )
    assert all(
        tetrahedral_cubic_invariant(control) == -sp.Rational(8, 9)
        for control in tetrahedral_antipodal_controls()
    )


def test_coordinate_axis_controls_have_zero_cubic_value() -> None:
    assert len(coordinate_axis_controls()) == 6
    assert all(
        tetrahedral_cubic_invariant(control) == 0
        for control in coordinate_axis_controls()
    )


def test_stationary_candidates_satisfy_unit_sphere_lagrange_equations() -> None:
    candidates = unit_sphere_stationary_candidates()
    assert len(candidates) == 14
    assert stationary_candidates_satisfy_lagrange_equations()
    assert all(unit_sphere_lagrange_residual(candidate) == (0, 0, 0) for candidate in candidates)


def test_unit_sphere_max_value_is_selector_value() -> None:
    assert unit_sphere_max_value() == sp.Rational(8, 9)


def test_selector_locking_potential_zero_set_on_audited_candidates() -> None:
    assert selector_minimizers() == (0, 1, 2, 3)
    assert all(
        selector_locking_potential(candidate) == 0
        for candidate in tetrahedral_selector_candidates()
    )
    assert locking_potential_zero_only_at_selectors()


def test_selector_locking_potential_rejects_controls() -> None:
    controls = (
        tetrahedral_antipodal_controls()
        + coordinate_axis_controls()
        + tetrahedral_midpoint_controls()
        + (midpoint_order_parameter(), zero_order_parameter(), generic_order_parameter())
    )
    assert all(sp.simplify(selector_locking_potential(control)) > 0 for control in controls)
    assert axis_controls_rejected()


def test_radial_only_cubic_only_and_wrong_sign_controls_are_rejected() -> None:
    selector = tetrahedral_selector_candidates()[0]
    antipode = tetrahedral_antipodal_controls()[0]
    axis = coordinate_axis_controls()[0]

    assert radial_locking_potential(selector) == 0
    assert radial_locking_potential(antipode) == 0
    assert radial_locking_potential(axis) == 0
    assert radial_only_control_rejected()

    assert cubic_only_control_rejected()

    assert wrong_sign_locking_potential(antipode) == 0
    assert sp.simplify(wrong_sign_locking_potential(selector)) > 0
    assert wrong_sign_control_rejected()


def test_v31_is_recovered() -> None:
    assert v31_recovered()


def test_vacuum_selector_condensation_payload_reports_pass() -> None:
    payload = vacuum_selector_condensation_audit_payload()
    assert payload.final_verdict == "CONTINUOUS_TETRAHEDRAL_SELECTOR_CONDENSATION_PASS"
    assert payload.cubic_polynomial_matches
    assert payload.unit_sphere_max_value == sp.Rational(8, 9)
    assert payload.selector_minimizers == (0, 1, 2, 3)
    assert payload.antipodal_controls_rejected
    assert payload.axis_controls_rejected
    assert payload.locking_potential_zero_only_at_selectors
    assert payload.radial_only_control_rejected
    assert payload.cubic_only_control_rejected
    assert payload.wrong_sign_control_rejected
    assert payload.v31_recovered
    assert (
        payload.remaining_declared_inputs
        == REMAINING_DECLARED_INPUTS_AFTER_SELECTOR_CONDENSATION
    )
    assert payload.remaining_declared_inputs == (
        "local_vacuum_realizes_tetrahedral_selector_locking_potential",
    )
