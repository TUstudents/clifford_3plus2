"""Tests for the V23 equal-degeneracy microcanonical reduction theorem."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.microcanonical_reduction import (
    REMAINING_DECLARED_INPUTS_AFTER_REDUCTION,
    compressed_macro_degeneracy_control_phase,
    compressed_macro_degeneracy_control_ratio,
    equal_degeneracy_reduced_density,
    microcanonical_label_weights,
    microcanonical_reduced_density,
    microcanonical_reduction_audit_payload,
    phase_from_degeneracies,
    ratio_from_degeneracies,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_microcanonical_label_weights_normalize_symbolically() -> None:
    degeneracies = sp.symbols(f"d0:{SHELL_DIMENSION}", positive=True)
    weights = microcanonical_label_weights(degeneracies)
    assert len(weights) == SHELL_DIMENSION
    assert sp.simplify(sum(weights) - 1) == 0


def test_reduced_density_is_diagonal_in_conserved_label_projectors() -> None:
    degeneracies = (2, 3, 5, 7, 11, 13)
    density = microcanonical_reduced_density(degeneracies)
    assert density.shape == (SHELL_DIMENSION, SHELL_DIMENSION)
    for row in range(SHELL_DIMENSION):
        for column in range(SHELL_DIMENSION):
            if row != column:
                assert density[row, column] == 0
    assert sp.simplify(sp.trace(density) - 1) == 0


def test_equal_degeneracy_reduces_to_uniform_density() -> None:
    _assert_matrix_equal(equal_degeneracy_reduced_density(), sp.eye(SHELL_DIMENSION) / 6)


def test_equal_degeneracy_implies_ckm_ratio_and_phase() -> None:
    degeneracies = tuple(sp.Integer(1) for _ in range(SHELL_DIMENSION))
    assert sp.simplify(ratio_from_degeneracies(degeneracies) - 1) == 0
    assert sp.simplify(phase_from_degeneracies(degeneracies) - quark_boundary_phase_angle()) == 0


def test_unequal_degeneracy_control_is_nonuniform_and_non_ckm() -> None:
    degeneracies = (2, 1, 1, 1, 1, 1)
    density = microcanonical_reduced_density(degeneracies)
    assert density[0, 0] == sp.Rational(2, 7)
    assert not density == sp.eye(SHELL_DIMENSION) / 6
    assert sp.simplify(phase_from_degeneracies(degeneracies) - quark_boundary_phase_angle()) != 0


def test_compressed_degeneracy_control_gives_v17_branch() -> None:
    assert sp.simplify(compressed_macro_degeneracy_control_ratio() - 1 / sp.sqrt(5)) == 0
    assert sp.simplify(compressed_macro_degeneracy_control_phase() - sp.pi / 4) == 0
    assert sp.simplify(compressed_macro_degeneracy_control_phase() - quark_boundary_phase_angle()) != 0


def test_microcanonical_reduction_payload_reports_pass() -> None:
    payload = microcanonical_reduction_audit_payload()
    assert payload.final_verdict == "EQUAL_DEGENERACY_MICROCANONICAL_REDUCTION_PASS"
    assert payload.equal_degeneracy_density_uniform
    assert sp.simplify(payload.equal_degeneracy_ratio - 1) == 0
    assert sp.simplify(payload.equal_degeneracy_phase - quark_boundary_phase_angle()) == 0
    assert payload.unequal_degeneracy_control_rejected
    assert payload.compressed_degeneracy_control_rejected
    assert payload.consistent_with_v22_no_go
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_REDUCTION
    assert payload.remaining_declared_inputs == (
        "vacuum_framing",
        "transfer_probe",
        "equal_boundary_degeneracy_or_max_entropy_prior",
    )
