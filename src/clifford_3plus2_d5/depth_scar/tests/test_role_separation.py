"""Tests for V2 role-separation verdict."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.role_separation import (
    depth_scar_prediction_ledger_payload,
)


def test_depth_scar_prediction_ledger_passes_without_overclaiming() -> None:
    payload = depth_scar_prediction_ledger_payload()
    assert payload.final_verdict == "DEPTH_SCAR_PREDICTION_LEDGER_PASS"
    assert payload.projectors_resolve_identity
    assert payload.transfer_kernel_matches_v1
    assert payload.port_transfer_relations_hold
    assert payload.endpoint_parity_blocks_even_odd
    assert payload.leading_kernel_is_democratic_rank_one
    assert payload.pure_path_has_no_intrinsic_cp_holonomy
    assert payload.restored_triangle_has_one_loop
    assert payload.ckm_lambda_exponents == {(1, 2): 1, (2, 3): 2, (1, 3): 3}
    assert payload.two_sided_mass_depth_semigroup == (0, 2, 4, 6, 8, 12)
    assert payload.two_sided_lambda_power_semigroup == (0, 1, 2, 3, 4, 6)
    assert payload.p3_is_mass_model_by_default is False
    assert payload.p3_is_cp_source_by_default is False
    assert payload.p3_is_universal_lepton_scar_by_default is False

