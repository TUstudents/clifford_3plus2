"""Tests for Session 09 microscopic BCC neutrino moment audit."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.universal_bath.neutrino_bcc_moments import (
    bb_edge_blocks,
    missing_family_labels,
    mixed_normal_norm,
    neutrino_bcc_moment_audit_payload,
    same_normal_norm,
)


def test_exact_bb_edge_norm_split() -> None:
    """The microscopic BB edge has the expected q=0/leakage split."""

    blocks = bb_edge_blocks()
    same = same_normal_norm(blocks)
    mixed = mixed_normal_norm(blocks)

    assert same == sp.eye(2) / 2
    assert mixed == sp.eye(2) / 2
    assert sp.simplify(same + mixed - sp.eye(2)) == sp.zeros(2)


def test_missing_family_labels_are_the_neutrino_gate() -> None:
    """The microscopic edge graph still lacks u/b family-port labels."""

    assert missing_family_labels() == ("family_port_u", "family_port_b")


def test_session_09_blocks_neutrino_upgrade() -> None:
    """The product bath remains internal until BCC family moments are defined."""

    payload = neutrino_bcc_moment_audit_payload()

    assert payload.final_verdict == "NEUTRINO_BCC_MOMENT_GRAPH_NOT_DERIVED_AUDIT"
    assert payload.product_internal_pass
    assert payload.product_ansatz_cross_moments_zero
    assert payload.product_ansatz_diagonal_equal
    assert payload.product_ansatz_is_only_family_factor
    assert not payload.microscopic_family_port_graph_available
    assert not payload.bcc_family_cross_moments_defined
    assert not payload.can_upgrade_neutrino_core
    assert payload.rank_one_control_has_cross_return
