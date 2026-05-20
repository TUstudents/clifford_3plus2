"""Tests for ``strong_cp_audit.py`` (aggregates SC-1..SC-5).

Slow: aggregates the H^(2) BCH-heavy SC-3 and SC-5 sub-phases; each
test triggers the full ~30s symbolic computation chain.
"""

from __future__ import annotations

import pytest

from clifford_3plus2_d5.strongcp.strong_cp_audit import strong_cp_audit_payload
from clifford_3plus2_d5.strongcp.theta_bar_constraint import (
    is_safe_at_order,
    theta_bar_constraint_payload,
)

pytestmark = pytest.mark.slow


def test_audit_sc1_consistent() -> None:
    payload = strong_cp_audit_payload()
    assert payload.sc1_degree3_harmonics_consistent


def test_audit_sc2_centrosymmetric() -> None:
    payload = strong_cp_audit_payload()
    assert payload.sc2_centrosymmetry.all_centrosymmetric


def test_audit_sc3_selection_rule_applies() -> None:
    payload = strong_cp_audit_payload()
    assert payload.sc3_higher_order_parity.selection_rule_applies


def test_audit_sc5_no_direct_theta_shift() -> None:
    payload = strong_cp_audit_payload()
    assert payload.sc5_chiral_anomaly.no_direct_theta_shift_at_o_eps_eps2


def test_audit_structural_argument_complete() -> None:
    payload = strong_cp_audit_payload()
    assert payload.structural_argument_complete


def test_audit_final_verdict_trivial_at_o_eps_eps2() -> None:
    payload = strong_cp_audit_payload()
    assert "TRIVIAL" in payload.final_verdict
    assert "O(ε)" in payload.final_verdict
    assert "SAFE" in payload.final_verdict


def test_sc4_direct_computation_confirms() -> None:
    payload = strong_cp_audit_payload()
    assert payload.direct_computation_confirms
    assert payload.sc4_lattice_topological_charge.plaquette_rep_is_parity_even
    assert payload.sc4_lattice_topological_charge.a2u_projection_of_pair_tensor_is_zero
    assert payload.sc4_lattice_topological_charge.spatial_only_q_dimensionally_trivial


def test_sc4_gauge_independence_holds_for_pati_salam() -> None:
    payload = strong_cp_audit_payload()
    sc4 = payload.sc4_lattice_topological_charge
    assert sc4.gauge_independence_su2_l
    assert sc4.gauge_independence_su2_r
    assert sc4.gauge_independence_su4_ps


def test_audit_verdict_mentions_sc4_confirmation() -> None:
    payload = strong_cp_audit_payload()
    assert "SC-4" in payload.final_verdict
    assert "CONFIRMS" in payload.final_verdict


def test_theta_bound_safe_at_all_orders_in_eps() -> None:
    # With ε ≲ 2 × 10⁻³³ m and Λ_QCD⁻¹ ~ 1 fm, any ε^n suppression
    # for n ≥ 1 is at least (2×10⁻¹⁸)^n — well below 10⁻¹⁰.
    for n in (1, 2, 3, 5, 10):
        assert is_safe_at_order(n)


def test_theta_bar_constraint_payload_all_safe() -> None:
    payload = theta_bar_constraint_payload()
    assert payload.safe_at_order_one
    assert payload.safe_at_order_two
    assert payload.safe_at_order_three
    assert "STRONG-CP SAFE" in payload.interpretation
