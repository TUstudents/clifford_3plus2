"""Tests for the V31 tetrahedral vacuum-selector potential gate."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector import (
    energy_stabilizer_permutations,
    selector_energy_multiset,
    selector_induces_residual_s3,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    REMAINING_DECLARED_INPUTS_AFTER_SELECTOR_POTENTIAL,
    antipodal_control_energies,
    generic_control_rejected,
    midpoint_control_rejected,
    selector_candidate_energies,
    selector_candidates_have_residual_s3_stabilizers,
    selector_candidates_reproduce_v28_spectra,
    selector_potential_energy,
    selector_potential_gap,
    selector_potential_minimizers,
    tetrahedral_antipodal_controls,
    tetrahedral_cubic_invariant,
    tetrahedral_midpoint_controls,
    tetrahedral_selector_candidates,
    vacuum_selector_potential_audit_payload,
    wrong_sign_antipodal_minimizers,
    wrong_sign_control_rejected,
    zero_anisotropy_control_rejected,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector import zero_order_parameter


def test_selector_candidates_have_exact_cubic_invariant() -> None:
    candidates = tetrahedral_selector_candidates()
    assert len(candidates) == 4
    assert all(tetrahedral_cubic_invariant(candidate) == sp.Rational(8, 9) for candidate in candidates)
    assert selector_candidate_energies() == tuple(-sp.Rational(8, 9) for _ in range(4))
    assert selector_potential_minimizers() == (0, 1, 2, 3)


def test_selector_candidates_are_degenerate_and_v28_conjugate() -> None:
    expected = (
        -sp.Integer(1),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
    )
    assert selector_candidates_reproduce_v28_spectra()
    assert all(selector_energy_multiset(candidate) == expected for candidate in tetrahedral_selector_candidates())


def test_each_selector_minimum_has_residual_s3_stabilizer() -> None:
    assert selector_candidates_have_residual_s3_stabilizers()
    for index, candidate in enumerate(tetrahedral_selector_candidates()):
        assert len(energy_stabilizer_permutations(candidate)) == 6
        assert selector_induces_residual_s3(index)


def test_selector_gap_against_controls_is_exact() -> None:
    assert selector_potential_gap() == sp.Rational(8, 9)
    assert selector_potential_energy(zero_order_parameter()) == 0
    assert all(selector_potential_energy(control) == 0 for control in tetrahedral_midpoint_controls())
    assert all(
        selector_potential_energy(control) == sp.Rational(8, 9)
        for control in tetrahedral_antipodal_controls()
    )


def test_wrong_sign_anisotropy_selects_antipodal_controls() -> None:
    assert selector_candidate_energies(anisotropy=-1) == tuple(sp.Rational(8, 9) for _ in range(4))
    assert antipodal_control_energies(anisotropy=-1) == tuple(-sp.Rational(8, 9) for _ in range(4))
    assert wrong_sign_antipodal_minimizers() == (0, 1, 2, 3)
    assert wrong_sign_control_rejected()


def test_zero_anisotropy_control_has_no_selector_gap() -> None:
    assert selector_candidate_energies(anisotropy=0) == (0, 0, 0, 0)
    assert selector_potential_gap(anisotropy=0) == 0
    assert zero_anisotropy_control_rejected()


def test_midpoint_and_generic_controls_are_rejected() -> None:
    assert midpoint_control_rejected()
    assert generic_control_rejected()


def test_vacuum_selector_potential_payload_reports_pass() -> None:
    payload = vacuum_selector_potential_audit_payload()
    assert payload.final_verdict == "TETRAHEDRAL_SELECTOR_POTENTIAL_PASS"
    assert payload.selector_candidate_count == 4
    assert payload.selector_energy == -sp.Rational(8, 9)
    assert payload.selector_minimizers == (0, 1, 2, 3)
    assert payload.selector_gap == sp.Rational(8, 9)
    assert payload.selector_candidates_degenerate
    assert payload.selector_candidates_reproduce_v28
    assert payload.selector_stabilizers_induce_s3
    assert payload.zero_anisotropy_control_rejected
    assert payload.wrong_sign_control_rejected
    assert payload.midpoint_control_rejected
    assert payload.generic_control_rejected
    assert payload.v28_recovered
    assert (
        payload.remaining_declared_inputs
        == REMAINING_DECLARED_INPUTS_AFTER_SELECTOR_POTENTIAL
    )
    assert payload.remaining_declared_inputs == (
        "tetrahedral_selector_order_parameter_condenses",
    )
