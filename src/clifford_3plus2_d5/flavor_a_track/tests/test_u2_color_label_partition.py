"""Tests for U2 — sector difference is the color quantum number."""

from __future__ import annotations

from clifford_3plus2_d5.flavor_a_track.u2_color_label_partition import (
    LEPTON_RESIDUAL_SPLIT,
    color_label_partition_audit_payload,
    color_label_partition_verdict,
    conserved_label_color_split,
    lepton_residual_dimension,
    noncolor_cores_match,
    quark_color_dimension,
    quark_noncolor_core_split,
    quark_shell_is_lepton_core_plus_color,
)


def test_noncolor_cores_match_1_plus_2() -> None:
    assert quark_noncolor_core_split() == {"trivial": 1, "doublet": 2}
    assert LEPTON_RESIDUAL_SPLIT == {"trivial": 1, "doublet": 2}
    assert noncolor_cores_match()


def test_quark_adds_three_color_ports() -> None:
    assert quark_color_dimension() == 3
    assert lepton_residual_dimension() == 3
    assert quark_shell_is_lepton_core_plus_color()


def test_conserved_label_color_split_is_three_three() -> None:
    assert conserved_label_color_split() == (3, 3)


def test_payload_pass() -> None:
    payload = color_label_partition_audit_payload()
    assert payload.final_verdict == "SECTOR_DIFFERENCE_IS_COLOR_LABEL"
    assert payload.noncolor_cores_match
    assert payload.label_partition_complete
    assert payload.quark_is_core_plus_color


def test_verdict_kills_when_any_check_fails() -> None:
    # Decisive negative control: each failing check must force the KILL string,
    # proving the gate can fail rather than always passing.
    assert (
        color_label_partition_verdict(True, True, True, True)
        == "SECTOR_DIFFERENCE_IS_COLOR_LABEL"
    )
    assert color_label_partition_verdict(False, True, True, True) == "SHELLS_INDEPENDENT_KILL"
    assert color_label_partition_verdict(True, False, True, True) == "SHELLS_INDEPENDENT_KILL"
    assert color_label_partition_verdict(True, True, False, True) == "SHELLS_INDEPENDENT_KILL"
    assert color_label_partition_verdict(True, True, True, False) == "SHELLS_INDEPENDENT_KILL"
