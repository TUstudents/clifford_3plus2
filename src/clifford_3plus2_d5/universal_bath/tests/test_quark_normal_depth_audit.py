"""Tests for Session 16 quark normal-depth placement audit."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.quark_normal_depth_audit import (
    depth_operator_not_diagonal_in_port_basis,
    doubled_port_heights,
    port_height_filtration,
    port_heights_are_not_depth_spectrum,
    quark_normal_depth_audit_payload,
)


def test_depth_scar_height_filtration_is_not_the_normal_mode_spectrum() -> None:
    assert port_height_filtration() == {"u": 0, "a": 1, "b": 2}
    assert doubled_port_heights() == (0, 2, 4)
    assert port_heights_are_not_depth_spectrum()


def test_depth_operator_is_not_a_hand_written_port_diagonal() -> None:
    payload = quark_normal_depth_audit_payload()

    assert payload.depth_operator_port_basis == sp.Matrix(
        [
            [2, -2, 0],
            [-2, 4, -2],
            [0, -2, 2],
        ]
    )
    assert payload.hand_written_diagonal_control == sp.diag(0, 2, 6)
    assert depth_operator_not_diagonal_in_port_basis()


def test_quark_normal_depth_audit_reports_source_placement_not_derived() -> None:
    payload = quark_normal_depth_audit_payload()

    assert payload.final_verdict == "QUARK_NORMAL_DEPTH_PLACEMENT_NOT_DERIVED_AUDIT"
    assert payload.quark_source_assembly_pass
    assert payload.nilpotent_flag_pass
    assert payload.microscopic_locality_pass
    assert not payload.port_height_filtration_microscopically_derived
    assert not payload.one_tick_geometry_microscopically_derived
    assert payload.defect_mode_depths == (0, 2, 6)
    assert payload.defect_mode_depths_match_scar
    assert payload.depth_operator_not_diagonal_in_port_basis
    assert payload.port_heights_are_not_depth_spectrum
    assert payload.up_dictionary_normal_depth is None
    assert payload.down_dictionary_normal_depth is None
    assert payload.dictionary_depths_still_unfrozen
    assert payload.graph_depths_do_not_freeze_source_depths
    assert payload.normal_depth_premise_still_open
