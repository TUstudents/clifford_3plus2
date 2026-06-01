"""Aggregate payload tests for V3 edge-weight scar selection."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.edge_weight_potential import (
    edge_weight_scar_potential_payload,
    path_scar_weight_vectors,
)


def test_edge_weight_scar_potential_payload_passes() -> None:
    payload = edge_weight_scar_potential_payload()
    assert payload.final_verdict == "EDGE_WEIGHT_SCAR_POTENTIAL_PASS"
    assert set(payload.minima) == set(path_scar_weight_vectors())
    assert payload.scar_potential_symmetric
    assert payload.path_scar_minima_pass
    assert payload.path_scar_spectra_pass
    assert payload.defect_term_load_bearing
    assert payload.symmetric_triangle_control_rejected
    assert payload.scar_dynamically_derived_from_qca is False

