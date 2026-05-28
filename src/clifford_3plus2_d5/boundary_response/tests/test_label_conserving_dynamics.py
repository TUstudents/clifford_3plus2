"""Tests for the V22 label-conserving dynamics no-go."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.conserved_label_partition import (
    REMAINING_DECLARED_INPUTS,
    conserved_label_projectors,
    label_mixing_control_operator,
    scattering_preserves_conserved_labels,
)
from clifford_3plus2_d5.boundary_response.label_conserving_dynamics import (
    all_diagonal_populations_stationary,
    dephasing_preserves_populations,
    label_conserving_dynamics_audit_payload,
    label_conserving_scattering,
    label_dephasing_channel,
    label_population_density,
    label_population_vector,
    stationary_simplex_dimension,
    uniform_population_density,
    uniform_stationary,
    uniform_unique_stationary,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_generic_label_conserving_scattering_commutes_with_projectors() -> None:
    phases = sp.symbols(f"phi0:{SHELL_DIMENSION}", real=True)
    scattering = label_conserving_scattering(phases)
    assert scattering_preserves_conserved_labels(scattering)
    for projector in conserved_label_projectors():
        _assert_matrix_equal(scattering * projector, projector * scattering)


def test_generic_population_density_has_trace_one_when_probabilities_sum_to_one() -> None:
    probabilities = sp.symbols(f"p0:{SHELL_DIMENSION}")
    density = label_population_density(probabilities)
    assert sp.simplify(sp.trace(density) - sum(probabilities)) == 0
    normalized = label_population_density(
        (
            sp.Rational(1, 2),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
        )
    )
    assert sp.simplify(sp.trace(normalized) - 1) == 0


def test_label_conserving_scattering_leaves_all_diagonal_populations_stationary() -> None:
    probabilities = sp.symbols(f"p0:{SHELL_DIMENSION}")
    phases = sp.symbols(f"phi0:{SHELL_DIMENSION}", real=True)
    density = label_population_density(probabilities)
    scattering = label_conserving_scattering(phases)
    _assert_matrix_equal(scattering * density - density * scattering, sp.zeros(SHELL_DIMENSION))
    assert all_diagonal_populations_stationary()


def test_dephasing_removes_coherences_but_preserves_populations() -> None:
    matrix = sp.zeros(SHELL_DIMENSION, SHELL_DIMENSION)
    matrix[1, 3] = sp.Symbol("c")
    matrix[3, 1] = sp.Symbol("d")
    matrix[0, 0] = sp.Rational(1, 2)
    matrix[2, 2] = sp.Rational(1, 3)
    dephased = label_dephasing_channel(matrix)
    assert dephased[1, 3] == 0
    assert dephased[3, 1] == 0
    assert label_population_vector(dephased) == label_population_vector(matrix)
    assert dephasing_preserves_populations()


def test_uniform_density_is_stationary_but_not_unique() -> None:
    assert uniform_stationary()
    assert not uniform_unique_stationary()
    phases = sp.symbols(f"phi0:{SHELL_DIMENSION}", real=True)
    nonuniform = label_population_density(
        (
            sp.Rational(1, 2),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
        )
    )
    scattering = label_conserving_scattering(phases)
    _assert_matrix_equal(scattering * uniform_population_density(), uniform_population_density() * scattering)
    _assert_matrix_equal(scattering * nonuniform, nonuniform * scattering)


def test_label_mixing_control_fails_conserved_label_scattering() -> None:
    assert not scattering_preserves_conserved_labels(label_mixing_control_operator())


def test_stationary_simplex_has_five_dimensions() -> None:
    assert stationary_simplex_dimension() == 5


def test_label_conserving_dynamics_payload_reports_no_go_pass() -> None:
    payload = label_conserving_dynamics_audit_payload()
    assert payload.final_verdict == "LABEL_CONSERVING_DYNAMICS_MAX_ENTROPY_NO_GO_PASS"
    assert payload.stationary_simplex_dimension == 5
    assert payload.all_diagonal_populations_stationary
    assert payload.dephasing_preserves_populations
    assert payload.uniform_stationary
    assert not payload.uniform_unique_stationary
    assert payload.label_mixing_required_for_dynamic_uniform
    assert payload.label_mixing_violates_conservation
    assert payload.max_entropy_prior_remains_declared
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS
