"""Aggregate payload tests for V9 microscopic locality."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.microscopic_locality import (
    PATH_REPAIR_EDGES,
    SHORTCUT_EDGE,
    microscopic_locality_minimality_pass,
    microscopic_locality_payload,
)


def test_microscopic_locality_payload_passes_with_conditional_normalization() -> None:
    payload = microscopic_locality_payload()

    assert payload.final_verdict == "MICROSCOPIC_LOCALITY_MINIMALITY_CONDITIONAL_PASS"
    assert payload.support_verdict == "V9A_MICROSCOPIC_SUPPORT_MINIMALITY_PASS"
    assert (
        payload.normalization_verdict
        == "V9B_MICROSCOPIC_NORMALIZATION_MINIMALITY_CONDITIONAL_PASS"
    )
    assert payload.monotone_edges == (*PATH_REPAIR_EDGES, SHORTCUT_EDGE)
    assert payload.one_tick_edges == PATH_REPAIR_EDGES
    assert payload.shortcut_forbidden_by_locality
    assert payload.local_rank_complete_support_count == 1
    assert payload.relaxed_rank_complete_support_count == 2
    assert payload.shortcut_admitted_when_locality_relaxed
    assert payload.local_support_induces_path_laplacian
    assert payload.weighted_path_formula_pass
    assert payload.target_spectrum_forces_unit_weights
    assert payload.partial_isometry_saturation_conditional_pass
    assert payload.height_filtration_microscopically_derived is False
    assert payload.one_tick_boundary_geometry_microscopically_derived is False
    assert microscopic_locality_minimality_pass()
