"""Tests for the V24 regular boundary-fiber equal-degeneracy audit."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.microcanonical_reduction import (
    phase_from_degeneracies,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)
from clifford_3plus2_d5.boundary_response.regular_boundary_fiber import (
    REMAINING_DECLARED_INPUTS_AFTER_REGULAR_FIBER,
    arbitrary_degeneracies_remain_free,
    arbitrary_label_preserving_degeneracies,
    compressed_macro_degeneracy_control_rejected,
    regular_boundary_fiber_audit_payload,
    regular_fiber_degeneracies,
    regular_fiber_degeneracies_are_equal,
    regular_fiber_density_is_uniform,
    regular_fiber_ratio_and_phase,
    regular_fiber_reduced_density,
    unequal_degeneracy_control_rejected,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_regular_fiber_degeneracies_are_all_equal() -> None:
    fiber_dim = sp.symbols("D", positive=True)
    degeneracies = regular_fiber_degeneracies(fiber_dim)
    assert degeneracies == tuple(fiber_dim for _ in range(SHELL_DIMENSION))
    assert regular_fiber_degeneracies_are_equal(fiber_dim)


def test_arbitrary_conserved_label_degeneracies_remain_free() -> None:
    degeneracies = arbitrary_label_preserving_degeneracies()
    assert len(degeneracies) == SHELL_DIMENSION
    assert sp.simplify(degeneracies[0] - degeneracies[1]) != 0
    assert arbitrary_degeneracies_remain_free()


def test_regular_fiber_reduces_to_uniform_density() -> None:
    fiber_dim = sp.symbols("D", positive=True)
    _assert_matrix_equal(
        regular_fiber_reduced_density(fiber_dim),
        sp.eye(SHELL_DIMENSION) / SHELL_DIMENSION,
    )
    assert regular_fiber_density_is_uniform(fiber_dim)


def test_regular_fiber_implies_ckm_ratio_and_phase() -> None:
    fiber_dim = sp.symbols("D", positive=True)
    ratio, phase = regular_fiber_ratio_and_phase(fiber_dim)
    assert sp.simplify(ratio - 1) == 0
    assert sp.simplify(phase - quark_boundary_phase_angle()) == 0


def test_unequal_degeneracy_control_is_rejected() -> None:
    degeneracies = (2, 1, 1, 1, 1, 1)
    assert sp.simplify(
        phase_from_degeneracies(degeneracies) - quark_boundary_phase_angle()
    ) != 0
    assert unequal_degeneracy_control_rejected()


def test_compressed_macro_degeneracy_control_is_rejected() -> None:
    assert compressed_macro_degeneracy_control_rejected()


def test_regular_boundary_fiber_payload_reports_pass() -> None:
    payload = regular_boundary_fiber_audit_payload()
    assert payload.final_verdict == "REGULAR_BOUNDARY_FIBER_EQUAL_DEGENERACY_PASS"
    assert payload.regular_degeneracies_equal
    assert payload.arbitrary_degeneracies_remain_free
    assert payload.regular_density_uniform
    assert sp.simplify(payload.regular_ratio - 1) == 0
    assert sp.simplify(payload.regular_phase - quark_boundary_phase_angle()) == 0
    assert payload.unequal_degeneracy_control_rejected
    assert payload.compressed_macro_control_rejected
    assert payload.consistent_with_v22_no_go
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_REGULAR_FIBER
    assert payload.remaining_declared_inputs == (
        "vacuum_framing",
        "transfer_probe",
        "regular_boundary_fiber_or_max_entropy_prior",
    )
