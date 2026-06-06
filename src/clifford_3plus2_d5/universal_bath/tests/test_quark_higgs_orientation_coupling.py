"""Tests for Session 24 quark Higgs-door orientation-coupling audit."""

import sympy as sp

from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_laplacian_from_nilpotent,
    nilpotent_flag_operator,
)
from clifford_3plus2_d5.universal_bath.quark_higgs_orientation_coupling import (
    HERMITIAN_CLOSURE_READOUT,
    RETARDED_FLAG_READOUT,
    assignment_allowed_by_available_constraints,
    available_constraints_select_unique_assignment,
    declared_assignment,
    endpoint_reflection_involutive,
    endpoint_reflection_operator,
    hermitian_closure_reflection_invariant,
    hermitian_closure_requires_extra_pairing,
    higgs_conjugation_supplies_reversal_not_closure,
    quark_higgs_orientation_coupling_payload,
    reflection_maps_flag_to_closure,
    reflection_maps_flag_to_reverse_flag,
    swapped_assignment,
)
from clifford_3plus2_d5.universal_bath.quark_height_orientation_bridge import (
    ORIENTATION_COUPLING_PREMISE,
)


def test_endpoint_reflection_reverses_flag_not_closure() -> None:
    reflection = endpoint_reflection_operator()
    flag = nilpotent_flag_operator()
    closure = flag_laplacian_from_nilpotent(flag)

    assert reflection == sp.Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert endpoint_reflection_involutive()
    assert reflection * flag * reflection == flag.T
    assert reflection_maps_flag_to_reverse_flag()
    assert reflection * closure * reflection == closure
    assert hermitian_closure_reflection_invariant()
    assert not reflection_maps_flag_to_closure()
    assert hermitian_closure_requires_extra_pairing()
    assert higgs_conjugation_supplies_reversal_not_closure()


def test_declared_and_swapped_assignments_both_survive_available_constraints() -> None:
    declared = declared_assignment()
    swapped = swapped_assignment()

    assert declared.up_readout == RETARDED_FLAG_READOUT
    assert declared.down_readout == HERMITIAN_CLOSURE_READOUT
    assert swapped.up_readout == HERMITIAN_CLOSURE_READOUT
    assert swapped.down_readout == RETARDED_FLAG_READOUT
    assert assignment_allowed_by_available_constraints(declared)
    assert assignment_allowed_by_available_constraints(swapped)
    assert not available_constraints_select_unique_assignment()


def test_session_24_payload_reports_not_derived_audit() -> None:
    payload = quark_higgs_orientation_coupling_payload()

    assert payload.final_verdict == "QUARK_HIGGS_ORIENTATION_COUPLING_NOT_DERIVED_AUDIT"
    assert payload.height_orientation_bridge_pass
    assert payload.current_parity_selector_pass
    assert payload.down_identity_veto_pass
    assert payload.hypercharge_forces_doors
    assert payload.neutral_higgs_components
    assert payload.endpoint_reflection_involutive
    assert payload.reflection_maps_flag_to_reverse_flag
    assert payload.hermitian_closure_reflection_invariant
    assert not payload.reflection_maps_flag_to_closure
    assert payload.hermitian_closure_requires_extra_pairing
    assert assignment_allowed_by_available_constraints(payload.declared_assignment)
    assert assignment_allowed_by_available_constraints(payload.swapped_assignment)
    assert not payload.available_constraints_select_unique_assignment
    assert payload.swapped_assignment_survives_controls
    assert payload.higgs_conjugation_supplies_reversal_not_closure
    assert not payload.higgs_door_orientation_coupling_derived
    assert payload.remaining_orientation_premise == ORIENTATION_COUPLING_PREMISE
