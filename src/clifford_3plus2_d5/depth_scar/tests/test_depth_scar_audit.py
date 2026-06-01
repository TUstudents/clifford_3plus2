"""Aggregate verdict tests for the depth_scar sidecar."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.audit import depth_scar_audit_payload


def test_depth_scar_audit_passes_operator_origin_gate() -> None:
    payload = depth_scar_audit_payload()
    assert payload.final_verdict == "PATH_DEFECT_LAPLACIAN_DEPTH_PASS"
    assert payload.depth_spectrum == (sp.Integer(0), sp.Integer(2), sp.Integer(6))
    assert payload.mode_depths == payload.depth_spectrum
    assert payload.transition_depths == {(1, 2): 2, (2, 3): 4, (1, 3): 6}
    assert payload.path_spectrum_passes
    assert payload.k3_control_rejected
    assert payload.diagonal_control_rejected
    assert payload.weighted_controls_pass
    assert payload.scar_dynamically_derived is False

