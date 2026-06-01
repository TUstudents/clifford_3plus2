"""V12 tests for leakage and shortcut controls."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.successor_certificate import (
    FailureClass,
    Verdict,
    Veto,
    certificate_implies_v11_no_leakage,
    external_leakage_controls_rejected,
    shortcut_repair_control_rejected,
    transition_certificate_row,
    unique_successor_enumeration_certificate_pass,
)


def test_external_leakage_controls_are_rejected_by_named_vetoes() -> None:
    assert transition_certificate_row("a", "bulk_u").failure_class == FailureClass.EXTERNAL_LEAKAGE
    assert Veto.BOUNDARY_SECTOR in transition_certificate_row("a", "bulk_u").vetoes
    assert Veto.COLOR in transition_certificate_row("a", "wrong_color_u").vetoes
    assert Veto.WEYL in transition_certificate_row("b", "wrong_weyl_a").vetoes
    assert Veto.SUPERSELECTION in transition_certificate_row("b", "orthogonal_coin_a").vetoes
    assert external_leakage_controls_rejected()


def test_shortcut_repair_control_is_rejected_separately_from_external_leakage() -> None:
    row = transition_certificate_row("b", "u")

    assert row.verdict == Verdict.FORBID
    assert row.failure_class == FailureClass.SHORTCUT_REPAIR
    assert shortcut_repair_control_rejected()


def test_certificate_implies_v11_no_leakage_condition() -> None:
    assert certificate_implies_v11_no_leakage()
    assert unique_successor_enumeration_certificate_pass()
