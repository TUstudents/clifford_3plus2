"""Tests for A3-4 — combined unified-transfer-boundary verdict + A3b deferral."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.flavor_a_track.a32_quark_transfer_schur import (
    quark_transfer_from_common_chain,
)
from clifford_3plus2_d5.flavor_a_track.reuse import (
    quark_transition_amplitude,
    residual_graph_decaying_factor,
)
from clifford_3plus2_d5.flavor_a_track.unified_boundary_audit import (
    REMAINING_DECLARED_INPUTS,
    unified_boundary_audit_payload,
)


def test_combined_pass() -> None:
    payload = unified_boundary_audit_payload()
    assert payload.final_verdict == "UNIFIED_TRANSFER_BOUNDARY_PASS"
    assert payload.transfer_boundary_unified


def test_subpayloads_pass() -> None:
    payload = unified_boundary_audit_payload()
    assert payload.common_hq.final_verdict == "LEPTON_SIGMA_FROM_COMMON_HQ"
    assert payload.quark_transfer_schur.final_verdict == "QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR"
    assert payload.coupling_structure.final_verdict == "SECTOR_STRUCTURE_IN_COUPLINGS"


def test_full_flavor_pattern_deferred_to_a3b() -> None:
    payload = unified_boundary_audit_payload()
    assert payload.full_flavor_pattern_proven is False
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS
    assert payload.remaining_declared_inputs == (
        "quark_family_depths_0_2_6_derived",
        "clebsch_and_coin_factors_derived_from_chiral16",
    )


def test_gate_can_fail_decisive_negative_control() -> None:
    # The quark amplitudes match the common (K3) chain factor's powers, but NOT a
    # different chain (K2 / golden root). If the quark transfer used a different
    # chain, A3-2 would return TRANSFER_NOT_UNIFIABLE_KILL.
    common = quark_transfer_from_common_chain()
    golden = residual_graph_decaying_factor(2)
    for pair in ((1, 2), (2, 3), (1, 3)):
        assert sp.simplify(common[pair] - quark_transition_amplitude(*pair)) == 0
        depth = {1: 0, 2: 2, 3: 6}
        mismatched = golden ** abs(depth[pair[1]] - depth[pair[0]])
        assert sp.simplify(mismatched - quark_transition_amplitude(*pair)) != 0
