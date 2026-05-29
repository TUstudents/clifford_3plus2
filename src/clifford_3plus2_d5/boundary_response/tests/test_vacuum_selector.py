"""Tests for the V28 vacuum-selector order-parameter gate."""

from __future__ import annotations

from itertools import permutations

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector import (
    REMAINING_DECLARED_INPUTS_AFTER_SELECTOR,
    all_selector_energy_multisets,
    energy_stabilizer_permutations,
    generic_order_parameter,
    ground_exit_indices,
    midpoint_order_parameter,
    selector_energies,
    selector_gap,
    selector_hamiltonian,
    selector_induces_residual_s3,
    selector_order_parameter,
    selector_stabilizer,
    vacuum_selector_audit_payload,
    zero_order_parameter,
)


def test_selected_exit_field_has_tetrahedral_energy_spectrum() -> None:
    h = selector_order_parameter(0)
    assert selector_energies(h) == (
        -sp.Integer(1),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
    )


def test_selected_exit_is_unique_ground_with_gap_four_thirds() -> None:
    h = selector_order_parameter(0)
    assert ground_exit_indices(h) == (0,)
    assert selector_gap(h) == sp.Rational(4, 3)


def test_selector_hamiltonian_is_diagonal_and_symmetric() -> None:
    hamiltonian = selector_hamiltonian(selector_order_parameter(0))
    assert hamiltonian == sp.diag(
        -sp.Integer(1),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
    )
    assert hamiltonian == hamiltonian.T


def test_selector_stabilizer_preserves_energy_and_induces_residual_s3() -> None:
    h = selector_order_parameter(0)
    energy_stabilizer = energy_stabilizer_permutations(h)
    expected_stabilizer = selector_stabilizer(0)
    assert len(energy_stabilizer) == 6
    assert set(energy_stabilizer) == set(expected_stabilizer)
    assert selector_induces_residual_s3(0)


def test_all_four_tetrahedral_selectors_are_conjugate() -> None:
    multisets = all_selector_energy_multisets()
    assert len(multisets) == 4
    assert all(multiset == multisets[0] for multiset in multisets)
    assert multisets[0] == (
        -sp.Integer(1),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
    )


def test_zero_order_parameter_control_is_not_a_selector() -> None:
    h = zero_order_parameter()
    assert selector_energies(h) == (0, 0, 0, 0)
    assert ground_exit_indices(h) == (0, 1, 2, 3)
    assert selector_gap(h) == 0
    assert len(energy_stabilizer_permutations(h)) == 24


def test_midpoint_order_parameter_control_has_two_ground_exits() -> None:
    h = midpoint_order_parameter(0, 1)
    assert selector_energies(h) == (
        -sp.Rational(2, 3),
        -sp.Rational(2, 3),
        sp.Rational(2, 3),
        sp.Rational(2, 3),
    )
    assert ground_exit_indices(h) == (0, 1)
    assert selector_gap(h) == 0


def test_generic_order_parameter_control_has_trivial_stabilizer() -> None:
    h = generic_order_parameter()
    assert len(ground_exit_indices(h)) == 1
    assert len(energy_stabilizer_permutations(h)) == 1
    assert energy_stabilizer_permutations(h) == (tuple(range(4)),)


def test_selector_residual_s3_is_not_generic_s4_degeneracy() -> None:
    zero_stabilizer = set(energy_stabilizer_permutations(zero_order_parameter()))
    selected_stabilizer = set(energy_stabilizer_permutations(selector_order_parameter(0)))
    all_s4 = {tuple(perm) for perm in permutations(range(4))}
    assert zero_stabilizer == all_s4
    assert selected_stabilizer < all_s4


def test_vacuum_selector_payload_reports_pass() -> None:
    payload = vacuum_selector_audit_payload()
    assert payload.final_verdict == "VACUUM_SELECTOR_ORDER_PARAMETER_PASS"
    assert payload.selected_ground_indices == (0,)
    assert payload.selected_gap == sp.Rational(4, 3)
    assert payload.selector_stabilizer_size == 6
    assert payload.selector_induces_s3
    assert payload.all_four_selectors_conjugate
    assert payload.zero_control_rejected
    assert payload.midpoint_control_rejected
    assert payload.generic_control_rejected
    assert payload.v27_applies_after_selection
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_SELECTOR
    assert payload.remaining_declared_inputs == (
        "physical_vacuum_order_parameter_exists",
        "unit_outward_causal_continuation_or_chain_normalization",
        "regular_boundary_fiber_or_max_entropy_prior",
    )
