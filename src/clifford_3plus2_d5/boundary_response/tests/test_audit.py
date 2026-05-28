"""Tests for the boundary-response combined audit."""

from __future__ import annotations

import clifford_3plus2_d5.boundary_response as boundary_response
from clifford_3plus2_d5.boundary_response.audit import boundary_core_audit_payload
from clifford_3plus2_d5.boundary_response.explicit_hq import explicit_hq_audit_payload
from clifford_3plus2_d5.boundary_response.framed_sterile import framed_sterile_audit_payload
from clifford_3plus2_d5.boundary_response.impedance import impedance_audit_payload


def test_audit_reports_transfer_pass() -> None:
    payload = boundary_core_audit_payload()
    assert payload.transfer_verdict == "TRANSFER_PASS"


def test_audit_reports_unbroken_k3_tail_kill() -> None:
    payload = boundary_core_audit_payload()
    assert payload.symmetry_verdict == "S3_KILL"
    assert payload.tail_verdict == "K3_TAIL_KILL"
    assert payload.final_verdict == "BOUNDARY_CORE_KILL_UNBROKEN_K3"


def test_audit_keeps_pmns_ckm_parked() -> None:
    payload = boundary_core_audit_payload()
    assert payload.pmns_ckm_parked
    assert "PMNS/CKM" in payload.interpretation


def test_v1_kill_and_v2_effective_pass_are_distinct_gates() -> None:
    v1 = boundary_core_audit_payload()
    v2 = framed_sterile_audit_payload()
    assert v1.final_verdict == "BOUNDARY_CORE_KILL_UNBROKEN_K3"
    assert v2.final_verdict == "FRAMED_STERILE_EFFECTIVE_PASS"
    assert v2.pmns_ckm_parked


def test_v3_explicit_hq_is_convergence_only_gate() -> None:
    payload = explicit_hq_audit_payload()
    assert payload.final_verdict == "EXPLICIT_HQ_CONVERGENCE_ONLY"
    assert payload.pmns_ckm_parked


def test_v4_impedance_catalog_is_free_parameter_gate() -> None:
    payload = impedance_audit_payload()
    assert payload.final_verdict == "IMPEDANCE_FREE_PARAMETER"
    assert payload.pmns_ckm_parked


def test_package_exports_conditional_pmns_and_ckm_apis() -> None:
    exported = set(boundary_response.__all__)
    assert "conditional_pmns_matrix" in exported
    assert "pmns_conditional_audit_payload" in exported
    assert "conditional_ckm_matrix" in exported
    assert "ckm_conditional_audit_payload" in exported
