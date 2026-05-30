"""Tests for the V33 tetrahedral invariant Landau minimality gate."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector import (
    generic_order_parameter,
    zero_order_parameter,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_condensation import (
    coordinate_axis_controls,
    norm_squared,
    order_parameter_symbols,
    radial_only_control_rejected,
    selector_locking_potential,
    wrong_sign_control_rejected,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_landau import (
    REMAINING_DECLARED_INPUTS_AFTER_LANDAU_MINIMALITY,
    degree_three_invariant_matches_cubic,
    degree_two_invariant_is_radial,
    invariant_space_basis,
    invariant_space_dimension,
    landau_lock_matches_v32,
    landau_lock_zero_only_at_selectors,
    lowest_selector_anisotropy_degree,
    non_tetrahedral_low_order_controls_rejected,
    polynomial_is_tetrahedral_invariant,
    rotations_are_proper,
    rotations_permute_selectors,
    tetrahedral_landau_audit_payload,
    tetrahedral_landau_locking_potential,
    tetrahedral_rotation_group,
    transform_polynomial,
    v32_recovered,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_antipodal_controls,
    tetrahedral_midpoint_controls,
    tetrahedral_selector_candidates,
)


def test_tetrahedral_rotation_group_is_proper_and_permuted() -> None:
    group = tetrahedral_rotation_group()
    selectors = set(tetrahedral_selector_candidates())

    assert len(group) == 12
    assert rotations_are_proper()
    assert rotations_permute_selectors()
    for rotation in group:
        assert rotation.T * rotation == sp.eye(3)
        assert rotation.det() == 1
        rotated = {
            tuple(sp.simplify(component) for component in rotation * sp.Matrix(selector))
            for selector in selectors
        }
        assert rotated == selectors


def test_low_degree_invariant_dimensions_are_exact() -> None:
    assert invariant_space_dimension(1) == 0
    assert invariant_space_dimension(2) == 1
    assert invariant_space_dimension(3) == 1


def test_degree_two_invariant_is_radial_only() -> None:
    x, y, z = order_parameter_symbols()
    radial = norm_squared((x, y, z))

    assert degree_two_invariant_is_radial()
    assert polynomial_is_tetrahedral_invariant(radial)
    assert len(invariant_space_basis(2)) == 1


def test_degree_three_invariant_is_tetrahedral_cubic() -> None:
    x, y, z = order_parameter_symbols()

    assert degree_three_invariant_matches_cubic()
    assert polynomial_is_tetrahedral_invariant(x * y * z)
    assert len(invariant_space_basis(3)) == 1


def test_lowest_selector_anisotropy_is_cubic() -> None:
    assert lowest_selector_anisotropy_degree() == 3


def test_non_tetrahedral_low_order_controls_are_rejected() -> None:
    x, y, z = order_parameter_symbols()

    assert not polynomial_is_tetrahedral_invariant(x)
    assert not polynomial_is_tetrahedral_invariant(x**2 - y**2)
    assert polynomial_is_tetrahedral_invariant(x**2 + y**2 + z**2)
    assert non_tetrahedral_low_order_controls_rejected()


def test_transform_polynomial_acts_by_exact_substitution() -> None:
    x, y, z = order_parameter_symbols()
    polynomial = x * y * z

    assert all(
        sp.simplify(transform_polynomial(polynomial, rotation) - polynomial) == 0
        for rotation in tetrahedral_rotation_group()
    )


def test_landau_lock_recovers_v32_selector_lock() -> None:
    checks = (
        tetrahedral_selector_candidates()
        + tetrahedral_antipodal_controls()
        + coordinate_axis_controls()
        + tetrahedral_midpoint_controls()
        + (zero_order_parameter(), generic_order_parameter())
    )

    assert landau_lock_matches_v32()
    assert all(
        sp.simplify(
            tetrahedral_landau_locking_potential(candidate)
            - selector_locking_potential(candidate)
        )
        == 0
        for candidate in checks
    )


def test_landau_lock_zero_set_matches_v32_on_audited_controls() -> None:
    assert all(
        tetrahedral_landau_locking_potential(selector) == 0
        for selector in tetrahedral_selector_candidates()
    )
    assert all(
        sp.simplify(tetrahedral_landau_locking_potential(control)) > 0
        for control in (
            tetrahedral_antipodal_controls()
            + coordinate_axis_controls()
            + tetrahedral_midpoint_controls()
            + (zero_order_parameter(), generic_order_parameter())
        )
    )
    assert landau_lock_zero_only_at_selectors()


def test_landau_controls_and_v32_regression_pass() -> None:
    assert radial_only_control_rejected()
    assert wrong_sign_control_rejected()
    assert v32_recovered()


def test_tetrahedral_landau_payload_reports_pass() -> None:
    payload = tetrahedral_landau_audit_payload()

    assert payload.final_verdict == "TETRAHEDRAL_INVARIANT_LANDAU_MINIMALITY_PASS"
    assert payload.rotation_group_size == 12
    assert payload.rotations_are_proper
    assert payload.rotations_permute_selectors
    assert payload.degree_one_invariant_dimension == 0
    assert payload.degree_two_invariant_dimension == 1
    assert payload.degree_three_invariant_dimension == 1
    assert payload.degree_two_invariant_radial
    assert payload.degree_three_invariant_matches_cubic
    assert payload.lowest_selector_anisotropy_degree == 3
    assert payload.landau_lock_matches_v32
    assert payload.selector_zero_set_matches_v32
    assert payload.radial_only_control_rejected
    assert payload.cubic_only_control_rejected
    assert payload.wrong_sign_control_rejected
    assert payload.non_tetrahedral_low_order_controls_rejected
    assert payload.v32_recovered
    assert (
        payload.remaining_declared_inputs
        == REMAINING_DECLARED_INPUTS_AFTER_LANDAU_MINIMALITY
    )
    assert payload.remaining_declared_inputs == (
        "local_vacuum_enters_lowest_order_positive_tetrahedral_landau_phase",
    )
