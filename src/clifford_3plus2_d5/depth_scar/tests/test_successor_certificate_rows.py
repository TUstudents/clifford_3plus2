"""V12 tests for transition-certificate rows and vetoes."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.successor_certificate import (
    FailureClass,
    Verdict,
    Veto,
    all_forbidden_rows_have_vetoes,
    allowed_successors_from_certificate,
    certificate_targets_pass,
    transition_certificate_row,
    transition_vetoes,
    transition_verdict,
)


def test_allowed_successors_are_exactly_u_and_a() -> None:
    assert transition_verdict("a", "u") == Verdict.ALLOW
    assert transition_vetoes("a", "u") == ()
    assert transition_verdict("b", "a") == Verdict.ALLOW
    assert transition_vetoes("b", "a") == ()
    assert allowed_successors_from_certificate() == {"a": ("u",), "b": ("a",)}
    assert certificate_targets_pass()


def test_every_forbidden_transition_has_at_least_one_veto() -> None:
    assert all_forbidden_rows_have_vetoes()


def test_shortcut_repair_has_locality_height_and_parity_vetoes() -> None:
    row = transition_certificate_row("b", "u")

    assert row.verdict == Verdict.FORBID
    assert row.failure_class == FailureClass.SHORTCUT_REPAIR
    assert Veto.LOCALITY in row.vetoes
    assert Veto.HEIGHT in row.vetoes
    assert Veto.BCC_PARITY in row.vetoes


def test_forbidden_repaired_range_control_is_not_external_leakage() -> None:
    row = transition_certificate_row("a", "a")

    assert row.verdict == Verdict.FORBID
    assert row.failure_class == FailureClass.FORBIDDEN_REPAIRED_RANGE
    assert Veto.HEIGHT in row.vetoes
