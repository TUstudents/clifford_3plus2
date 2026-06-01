"""V12 tests for the finite successor-certificate basis."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.successor_certificate import (
    ACTIVE_SOURCES,
    certificate_is_complete,
    local_boundary_basis,
    state_by_label,
    transition_certificate,
)


def test_local_boundary_basis_contains_residual_ports_and_controls() -> None:
    labels = tuple(state.label for state in local_boundary_basis())

    assert labels[:3] == ("u", "a", "b")
    assert "bulk_u" in labels
    assert "spectator_u" in labels
    assert "wrong_color_u" in labels
    assert "wrong_weyl_a" in labels
    assert "orthogonal_coin_a" in labels
    assert len(labels) == len(set(labels)) == 12


def test_residual_port_signatures_are_expected() -> None:
    assert state_by_label("u").height == 0
    assert state_by_label("a").height == 1
    assert state_by_label("b").height == 2
    assert state_by_label("u").bcc_parity == state_by_label("b").bcc_parity == 0
    assert state_by_label("a").bcc_parity == 1


def test_transition_certificate_is_complete_over_active_sources() -> None:
    assert ACTIVE_SOURCES == ("a", "b")
    assert len(transition_certificate()) == 2 * len(local_boundary_basis())
    assert certificate_is_complete()
