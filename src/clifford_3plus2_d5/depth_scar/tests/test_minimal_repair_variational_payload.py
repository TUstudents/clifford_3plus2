"""Aggregate payload tests for V8 minimal causal repair."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.minimal_repair_variational import (
    minimal_causal_repair_variational_payload,
)


def test_minimal_causal_repair_variational_payload_passes() -> None:
    payload = minimal_causal_repair_variational_payload()

    assert payload.final_verdict == "MINIMAL_CAUSAL_REPAIR_VARIATIONAL_PASS"
    assert payload.feasible_support_count == 12
    assert payload.minimal_cost == 2
    assert payload.minimizer_count == 6
    assert payload.minimizer_orbit_count == 1
    assert payload.minimizers_equivalent_to_flag
    assert payload.minimizer_spectra_pass
    assert payload.shortcuts_excluded_by_cost
    assert payload.rank_one_relaxed_control_pass
    assert payload.cycle_relaxed_control_pass
    assert payload.constant_cost_control_pass
    assert payload.microscopic_cost_principle_derived is False
