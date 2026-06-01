"""Aggregate payload tests for V5 nilpotent flag scar origin."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.nilpotent_flag import nilpotent_flag_scar_payload


def test_nilpotent_flag_scar_payload_passes() -> None:
    payload = nilpotent_flag_scar_payload()

    assert payload.final_verdict == "NILPOTENT_FLAG_SCAR_ORIGIN_PASS"
    assert payload.nilpotent_order_pass
    assert payload.laplacian_matches_path
    assert payload.spectrum_pass
    assert payload.transfer_matches_v1
    assert payload.rank_one_control_rejected
    assert payload.cyclic_closure_control_rejected
    assert payload.weighted_target_conditions_pass
    assert payload.equal_unit_flag_steps_microscopically_derived is False
