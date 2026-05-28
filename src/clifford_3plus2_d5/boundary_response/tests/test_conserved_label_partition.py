"""Tests for the V21 conserved-label distinguishability theorem."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.conserved_label_partition import (
    REMAINING_DECLARED_INPUTS,
    compressed_partition_merges_conserved_labels,
    conserved_label_partition_audit_payload,
    conserved_label_partition_is_complete,
    conserved_label_projectors,
    conserved_label_scattering_operator,
    jaynes_partition_matches_conserved_labels,
    label_mixing_control_operator,
    labels_are_pairwise_distinct,
    primitive_conserved_labels,
    projectors_are_orthogonal,
    projectors_resolve_identity,
    scattering_preserves_conserved_labels,
)
from clifford_3plus2_d5.boundary_response.jaynes_primitive_ergodicity import (
    jaynes_primitive_density,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_all_primitive_channels_have_distinct_conserved_label_tuples() -> None:
    labels = primitive_conserved_labels()
    assert len(labels) == SHELL_DIMENSION
    assert labels_are_pairwise_distinct(labels)
    assert labels[0].quantum_numbers() == ("even", "none", "none")
    assert {label.color_index for label in labels[3:]} == {"red", "green", "blue"}


def test_conserved_label_projectors_are_orthogonal_and_resolve_identity() -> None:
    projectors = conserved_label_projectors()
    assert len(projectors) == SHELL_DIMENSION
    assert projectors_are_orthogonal(projectors)
    assert projectors_resolve_identity(projectors)
    assert conserved_label_partition_is_complete()
    _assert_matrix_equal(sum(projectors, sp.zeros(SHELL_DIMENSION, SHELL_DIMENSION)), sp.eye(SHELL_DIMENSION))


def test_generic_label_preserving_scattering_commutes_with_projectors() -> None:
    scattering = conserved_label_scattering_operator()
    assert scattering_preserves_conserved_labels(scattering)
    for projector in conserved_label_projectors():
        _assert_matrix_equal(scattering * projector, projector * scattering)


def test_operator_mixing_distinct_labels_fails_conservation() -> None:
    mixing = label_mixing_control_operator(left=1, right=3)
    assert not scattering_preserves_conserved_labels(mixing)


def test_compressed_partition_merges_conserved_labels_and_is_rejected() -> None:
    labels = primitive_conserved_labels()
    odd_labels = {label.quantum_numbers() for label in labels if label.parity == "odd"}
    assert len(odd_labels) == 5
    assert compressed_partition_merges_conserved_labels()


def test_v20_jaynes_uniform_density_matches_six_conserved_label_atoms() -> None:
    uniform_from_labels = sum(
        conserved_label_projectors(),
        sp.zeros(SHELL_DIMENSION, SHELL_DIMENSION),
    ) / 6
    _assert_matrix_equal(uniform_from_labels, jaynes_primitive_density(sp.Rational(1, 6)))
    assert jaynes_partition_matches_conserved_labels()


def test_conserved_label_partition_payload_reports_theorem_pass() -> None:
    payload = conserved_label_partition_audit_payload()
    assert payload.final_verdict == "CONSERVED_LABEL_PARTITION_THEOREM_PASS"
    assert payload.primitive_label_count == 6
    assert payload.labels_are_pairwise_distinct
    assert payload.projectors_are_orthogonal
    assert payload.projectors_resolve_identity
    assert payload.boundary_scattering_preserves_labels
    assert payload.mixing_control_rejected
    assert payload.compressed_partition_rejected
    assert payload.jaynes_partition_derived
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS
    assert payload.remaining_declared_inputs == (
        "vacuum_framing",
        "transfer_probe",
        "max_entropy_prior",
    )
