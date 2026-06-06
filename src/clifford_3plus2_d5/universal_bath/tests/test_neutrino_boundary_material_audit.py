"""Tests for Session 13 boundary-material origin audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.neutrino_boundary_material_audit import (
    bare_blocks_have_no_gap_parameter,
    boundary_material_audit_payload,
    emission_norm_is_same_for_retarded_and_recurrent,
    local_linear_mismatch_solution_space,
    local_mismatch_coordinate_is_unique,
    outgoing_asymptotics_not_forced_by_bare_blocks,
    positive_penalty_if_constraint_admitted,
)


def test_local_linear_mismatch_is_unique_q() -> None:
    """Single-clock linear mismatch is proportional to r1-r2."""

    basis = local_linear_mismatch_solution_space()

    assert len(basis) == 1
    assert basis[0].tolist() == [[1], [-1]]
    assert local_mismatch_coordinate_is_unique()


def test_positive_penalty_follows_if_constraint_is_admitted() -> None:
    """K^T K gives q^2, but the boundary field itself is separate."""

    assert positive_penalty_if_constraint_admitted()


def test_bare_blocks_do_not_contain_boundary_material_gap() -> None:
    """The BB amplitudes contain no stiffness parameter."""

    assert bare_blocks_have_no_gap_parameter()
    assert emission_norm_is_same_for_retarded_and_recurrent()


def test_outgoing_asymptotics_are_not_selected_by_local_emission_norm() -> None:
    """Retarded and recurrent completions share emission data but differ."""

    assert outgoing_asymptotics_not_forced_by_bare_blocks()


def test_session_13_boundary_material_payload() -> None:
    """Session 13 records a real no-derivation boundary."""

    payload = boundary_material_audit_payload()

    assert payload.final_verdict == "NEUTRINO_BOUNDARY_MATERIAL_ORIGIN_NOT_DERIVED_AUDIT"
    assert payload.local_mismatch_coordinate_unique
    assert payload.positive_penalty_if_constraint_admitted
    assert payload.bare_blocks_have_no_gap_parameter
    assert payload.retarded_completion_uses_same_emission_norm
    assert payload.recurrent_completion_uses_same_emission_norm
    assert payload.retarded_and_recurrent_visible_powers_differ
    assert payload.recurrent_two_step_return_nonzero
    assert not payload.outgoing_asymptotics_forced_by_bare_blocks
    assert not payload.positive_locking_forced_by_bare_blocks
    assert payload.session_12_conditional_graph_passes
    assert "boundary-material model" in payload.next_required_object
