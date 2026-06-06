"""Tests for Session 20 quark height-orientation bridge audit."""

from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_laplacian_from_nilpotent,
    nilpotent_flag_operator,
)
from clifford_3plus2_d5.depth_scar.successor_certificate import (
    allowed_successors_from_certificate,
)
from clifford_3plus2_d5.universal_bath.quark_height_door import (
    down_repair_operator,
    up_repair_operator,
)
from clifford_3plus2_d5.universal_bath.quark_height_orientation_bridge import (
    ORIENTATION_COUPLING_PREMISE,
    down_readout_is_closure_of_up_flag,
    hermitian_closure_matches_down_repair,
    higgs_door_orientation_coupling_derived,
    oriented_nilpotent_matches_up_repair,
    oriented_successors_match_flag,
    quark_height_orientation_bridge_payload,
)


def test_depth_scar_successor_certificate_supplies_oriented_flag() -> None:
    assert allowed_successors_from_certificate() == {"a": ("u",), "b": ("a",)}
    assert oriented_successors_match_flag()


def test_up_and_down_repairs_are_two_readouts_of_one_flag() -> None:
    assert nilpotent_flag_operator() == up_repair_operator()
    assert flag_laplacian_from_nilpotent(nilpotent_flag_operator()) == down_repair_operator()
    assert oriented_nilpotent_matches_up_repair()
    assert hermitian_closure_matches_down_repair()
    assert down_readout_is_closure_of_up_flag()


def test_higgs_door_orientation_coupling_is_not_derived() -> None:
    assert not higgs_door_orientation_coupling_derived()


def test_session_20_payload_reports_orientation_bridge_audit() -> None:
    payload = quark_height_orientation_bridge_payload()

    assert payload.final_verdict == "QUARK_HEIGHT_ORIENTATION_BRIDGE_NOT_DERIVED_AUDIT"
    assert payload.height_door_audit_pass
    assert payload.successor_certificate_pass
    assert payload.nilpotent_flag_pass
    assert payload.allowed_successors == {"a": ("u",), "b": ("a",)}
    assert payload.oriented_successors_match_flag
    assert payload.oriented_nilpotent_matches_up_repair
    assert payload.hermitian_closure_matches_down_repair
    assert payload.down_readout_is_closure_of_up_flag
    assert payload.hypercharge_forces_doors
    assert payload.neutral_higgs_components
    assert payload.swapped_assignment_hypercharge_allowed
    assert not payload.higgs_door_orientation_coupling_derived
    assert payload.remaining_orientation_premise == ORIENTATION_COUPLING_PREMISE
    assert payload.quark_sources_still_unfrozen
