"""Tests for Session 10 selected neutrino family-port graph."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.universal_bath.neutrino_family_port_graph import (
    active_neutrino_projector,
    family_graph_cross_moments,
    family_graph_diagonal_differences,
    k3_control_diagonal_differences,
    product_identity_hamiltonian,
    radial_active_cross_moments,
    radial_family_projector,
    selected_family_graph_hamiltonian,
    selected_family_graph_payload,
)


def test_active_and_radial_projectors_resolve_family_space() -> None:
    """The selected family graph splits active u/b plane from radial a."""

    active = active_neutrino_projector()
    radial = radial_family_projector()

    assert active.rank() == 2
    assert radial.rank() == 1
    assert active * radial == sp.zeros(3)
    assert sp.simplify(active + radial - sp.eye(3)) == sp.zeros(3)


def test_selected_graph_is_not_full_product_identity() -> None:
    """The selected graph is not H_chain tensor I_family."""

    shells = 4
    assert selected_family_graph_hamiltonian(shells) != product_identity_hamiltonian(shells)


def test_family_port_cross_moments_vanish_directly() -> None:
    """The graph itself gives zero u/b moments and equal u/b returns."""

    shells = 5
    powers = (0, 1, 2, 3, 4)

    assert family_graph_cross_moments(shells, powers) == (0, 0, 0, 0, 0)
    assert family_graph_diagonal_differences(shells, powers) == (0, 0, 0, 0, 0)
    assert radial_active_cross_moments(shells, powers) == (0, 0, 0, 0, 0)


def test_k3_control_does_not_have_equal_u_b_returns() -> None:
    """The full K3 residual graph is the wrong microscopic family graph."""

    differences = k3_control_diagonal_differences(5, (0, 1, 2, 3, 4))

    assert differences[0] == 0
    assert any(sp.simplify(value) != 0 for value in differences[1:])


def test_session_10_family_port_graph_payload() -> None:
    """Session 10 upgrades the internal neutrino graph, not all sectors."""

    payload = selected_family_graph_payload()

    assert payload.final_verdict == "NEUTRINO_FAMILY_PORT_GRAPH_INTERNAL_PASS"
    assert payload.cross_moments_zero
    assert payload.diagonal_moments_equal
    assert payload.radial_mode_separated
    assert payload.graph_differs_from_product_identity
    assert payload.tail_response_matches_target
    assert payload.k3_control_rejected
    assert payload.full_product_control_has_radial_mode
    assert payload.rank_one_control_has_cross_return
    assert payload.alternate_tail_control_rejected
    assert payload.can_upgrade_neutrino_core
