"""Tests for the V30 local boundary-fiber isomorphism gate."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.boundary_response.local_boundary_fiber import (
    REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER,
    all_local_fibers_pairwise_isomorphic,
    arbitrary_label_degeneracy_control_rejected,
    compressed_macro_fiber_control_rejected,
    fiber_isomorphism_witness,
    fiber_isomorphism_witness_maps_projectors,
    local_boundary_fiber_audit_payload,
    local_fiber_degeneracies,
    local_fiber_label_projectors,
    local_fiber_projector_ranks,
    local_fiber_ratio_and_phase,
    local_fiber_reduced_density,
    local_fiber_total_dimension,
    sector_dependent_control_rejected,
    sector_dependent_fiber_degeneracies,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import SHELL_DIMENSION
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_local_fiber_degeneracies_are_all_equal() -> None:
    fiber_dim = sp.symbols("D", positive=True)
    assert local_fiber_degeneracies(fiber_dim) == tuple(
        fiber_dim for _ in range(SHELL_DIMENSION)
    )


def test_local_fiber_total_dimension_is_six_times_fiber_dimension() -> None:
    fiber_dim = sp.symbols("D", positive=True)
    assert local_fiber_total_dimension(fiber_dim) == SHELL_DIMENSION * fiber_dim


def test_local_fiber_projectors_have_equal_rank() -> None:
    projectors = local_fiber_label_projectors(2)
    assert len(projectors) == SHELL_DIMENSION
    assert all(projector.shape == (12, 12) for projector in projectors)
    assert local_fiber_projector_ranks(2) == (2, 2, 2, 2, 2, 2)
    _assert_matrix_equal(sum(projectors, sp.zeros(12, 12)), sp.eye(12))


def test_projector_helper_requires_concrete_positive_integer_dimension() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        local_fiber_label_projectors(sp.symbols("D", positive=True))
    with pytest.raises(ValueError, match="positive integer"):
        local_fiber_label_projectors(0)


def test_fiber_isomorphism_witness_swaps_two_label_fibers() -> None:
    witness = fiber_isomorphism_witness(1, 4, 2)
    assert witness.shape == (12, 12)
    _assert_matrix_equal(witness.T * witness, sp.eye(12))
    assert fiber_isomorphism_witness_maps_projectors(1, 4, 2)


def test_every_pair_of_local_label_fibers_is_isomorphic() -> None:
    assert all_local_fibers_pairwise_isomorphic(2)


def test_local_fiber_reduced_density_is_uniform() -> None:
    fiber_dim = sp.symbols("D", positive=True)
    _assert_matrix_equal(
        local_fiber_reduced_density(fiber_dim),
        sp.eye(SHELL_DIMENSION) / SHELL_DIMENSION,
    )


def test_local_fiber_implies_ckm_ratio_and_phase() -> None:
    fiber_dim = sp.symbols("D", positive=True)
    ratio, phase = local_fiber_ratio_and_phase(fiber_dim)
    assert sp.simplify(ratio - 1) == 0
    assert sp.simplify(phase - quark_boundary_phase_angle()) == 0


def test_sector_dependent_fiber_control_is_rejected() -> None:
    degeneracies = sector_dependent_fiber_degeneracies(
        even_dim=2,
        bcc_dim=1,
        color_dim=3,
    )
    assert degeneracies == (2, 1, 1, 3, 3, 3)
    assert sector_dependent_control_rejected()


def test_arbitrary_and_compressed_controls_are_rejected() -> None:
    assert arbitrary_label_degeneracy_control_rejected()
    assert compressed_macro_fiber_control_rejected()


def test_local_boundary_fiber_payload_reports_pass() -> None:
    payload = local_boundary_fiber_audit_payload()
    assert payload.final_verdict == "LOCAL_BOUNDARY_FIBER_ISOMORPHISM_PASS"
    assert payload.local_degeneracies_equal
    assert payload.total_dimension == 6 * sp.symbols("D", positive=True)
    assert payload.concrete_projector_ranks == (2, 2, 2, 2, 2, 2)
    assert payload.pairwise_isomorphism_witnesses_exist
    assert payload.reduced_density_uniform
    assert sp.simplify(payload.local_ratio - 1) == 0
    assert sp.simplify(payload.local_phase - quark_boundary_phase_angle()) == 0
    assert payload.sector_dependent_control_rejected
    assert payload.arbitrary_degeneracy_control_rejected
    assert payload.compressed_macro_control_rejected
    assert payload.v21_labels_complete
    assert payload.v22_no_go_respected
    assert payload.v24_recovered
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER
    assert payload.remaining_declared_inputs == ("physical_vacuum_order_parameter_exists",)
