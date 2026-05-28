"""Tests for the V16 primitive ergodicity theorem."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    full_transitive_generators,
    full_transitive_invariant_basis,
    full_transitive_symmetry_forces_flatness,
    odd_components_are_equal,
    parity_preserving_generators,
    parity_preserving_invariant_basis,
    parity_preserving_symmetry_forces_flatness,
    primitive_ergodicity_audit_payload,
    primitive_shell_amplitude,
    vector_is_invariant,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)
from clifford_3plus2_d5.boundary_response.quark_coin_rigidity import (
    isotropic_quark_phase_angle,
)


def test_parity_preserving_invariant_space_has_free_even_odd_ratio() -> None:
    basis = parity_preserving_invariant_basis()
    assert len(basis) == 2
    assert any(vector == sp.Matrix([1, 0, 0, 0, 0, 0]) for vector in basis)
    assert any(vector == sp.Matrix([0, 1, 1, 1, 1, 1]) for vector in basis)


def test_odd_shell_invariance_forces_equal_odd_components() -> None:
    for vector in parity_preserving_invariant_basis():
        assert odd_components_are_equal(vector)


def test_nonflat_ratios_still_satisfy_parity_preserving_symmetry() -> None:
    generators = parity_preserving_generators()
    for ratio in (sp.Rational(1, 2), sp.Integer(1), sp.Integer(2)):
        assert vector_is_invariant(primitive_shell_amplitude(ratio), generators)
    assert not parity_preserving_symmetry_forces_flatness()


def test_full_transitive_symmetry_has_only_flat_invariant_direction() -> None:
    basis = full_transitive_invariant_basis()
    assert len(basis) == 1
    assert basis[0] == sp.Matrix([1, 1, 1, 1, 1, 1])
    assert full_transitive_symmetry_forces_flatness()


def test_full_transitive_symmetry_rejects_nonflat_ratios() -> None:
    generators = full_transitive_generators()
    assert vector_is_invariant(primitive_shell_amplitude(1), generators)
    for ratio in (sp.Rational(1, 2), sp.Integer(2)):
        assert not vector_is_invariant(primitive_shell_amplitude(ratio), generators)


def test_only_flat_ratio_matches_v15_ckm_phase() -> None:
    assert sp.simplify(isotropic_quark_phase_angle(1) - quark_boundary_phase_angle()) == 0
    for ratio in (sp.Rational(1, 2), sp.Integer(2)):
        assert sp.simplify(isotropic_quark_phase_angle(ratio) - quark_boundary_phase_angle()) != 0


def test_primitive_ergodicity_payload_reports_no_go_pass() -> None:
    payload = primitive_ergodicity_audit_payload()
    assert payload.final_verdict == "PRIMITIVE_ERGODICITY_NO_GO_PASS"
    assert payload.parity_invariant_dimension == 2
    assert payload.full_transitive_invariant_dimension == 1
    assert payload.odd_shell_components_equal
    assert not payload.parity_preserving_symmetry_forces_flatness
    assert payload.full_transitive_symmetry_forces_flatness
    assert payload.nonflat_parity_controls_invariant
    assert payload.nonflat_full_controls_rejected
    assert payload.ckm_phase_requires_extra_ergodicity_principle
