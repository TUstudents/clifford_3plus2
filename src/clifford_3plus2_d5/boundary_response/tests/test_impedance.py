"""Tests for the V4 endpoint impedance-matching catalog."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import diagnose_framed_response
from clifford_3plus2_d5.boundary_response.framed_sterile import framed_sterile_effective_response
from clifford_3plus2_d5.boundary_response.impedance import (
    endpoint_catalog,
    evaluate_endpoint_candidate,
    impedance_audit_payload,
    impedance_hq_for_candidate,
    impedance_response_for_candidate,
    inferred_collective_return,
    matched_load_edge_energy,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_selected_s2_invariant,
    residual_basis_matrix,
)
from clifford_3plus2_d5.boundary_response.schur import matrix_equal
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def _candidate_by_name(name: str):
    candidates = {candidate.name: candidate for candidate in endpoint_catalog(10)}
    return candidates[name]


def test_endpoint_catalog_names_are_deterministic() -> None:
    assert tuple(candidate.name for candidate in endpoint_catalog(10)) == (
        "bare_site",
        "self_energy_stub",
        "two_site_dimer",
        "mirrored_one_step_tail",
        "symmetric_matched_load",
    )


def test_each_candidate_builds_valid_hq_and_coupling() -> None:
    for candidate in endpoint_catalog(6):
        h_q, coupling = impedance_hq_for_candidate(6, candidate)
        assert h_q.rows == h_q.cols
        assert coupling.rows == h_q.rows
        assert coupling.cols == 3
        assert matrix_equal(h_q, h_q.T)


def test_each_catalog_response_preserves_selected_s2() -> None:
    for candidate in endpoint_catalog(6):
        response = impedance_response_for_candidate(6, candidate)
        assert is_selected_s2_invariant(response)


def test_catalog_candidates_have_no_direct_radial_coupling() -> None:
    basis = residual_basis_matrix(("a", "u", "b"))
    for candidate in endpoint_catalog(6):
        _, coupling = impedance_hq_for_candidate(6, candidate)
        in_residual_basis = (coupling * basis).applyfunc(sp.simplify)
        assert all(sp.simplify(entry) == 0 for entry in in_residual_basis[:, 0])


def test_bare_site_reproduces_known_unequal_return_failure() -> None:
    result = evaluate_endpoint_candidate(10, _candidate_by_name("bare_site"))
    assert result.verdict == "IMPEDANCE_KILL"
    assert "UNEQUAL_RETURN" in result.failure_reasons


def test_matched_load_solves_exact_return_relation_but_is_tuned() -> None:
    candidate = _candidate_by_name("symmetric_matched_load")
    result = evaluate_endpoint_candidate(10, candidate)
    assert result.verdict == "IMPEDANCE_FREE_PARAMETER"
    assert result.failure_reasons == ()
    assert result.sector_specific_tuning
    assert result.tuning_relation == "edge_energy = z_transfer - 1 / inferred_collective_return"


def test_matched_load_energy_reproduces_inferred_collective_return() -> None:
    z = 2 * sp.sqrt(2)
    energy = matched_load_edge_energy(10)
    edge_return = sp.simplify(1 / (z - energy))
    assert sp.simplify(edge_return - inferred_collective_return(10)) == 0


def test_cross_coupled_control_triggers_cross_return() -> None:
    candidate = _candidate_by_name("bare_site")
    h_q, coupling = impedance_hq_for_candidate(6, candidate, include_cross_coupling=True)
    from clifford_3plus2_d5.boundary_response.schur import self_energy

    response = self_energy(2 * sp.sqrt(2), h_q, coupling)
    diag = diagnose_framed_response(response, transfer_amplitude=epsilon())
    assert "CROSS_RETURN" in diag.failure_reasons


def test_radial_leak_control_triggers_radial_leakage() -> None:
    candidate = _candidate_by_name("bare_site")
    h_q, coupling = impedance_hq_for_candidate(6, candidate, include_radial_leakage=True)
    from clifford_3plus2_d5.boundary_response.schur import self_energy

    response = self_energy(2 * sp.sqrt(2), h_q, coupling)
    diag = diagnose_framed_response(response, transfer_amplitude=epsilon())
    assert "RADIAL_LEAKAGE" in diag.failure_reasons


def test_parameterized_catalog_verdict_is_free_parameter_not_theorem() -> None:
    payload = impedance_audit_payload(shells=10)
    assert payload.final_verdict == "IMPEDANCE_FREE_PARAMETER"
    assert payload.best_candidate == "symmetric_matched_load"
    assert payload.pmns_ckm_parked


def test_unequal_return_control_still_detected() -> None:
    response = framed_sterile_effective_response(collective_return=sp.Integer(2))
    diag = diagnose_framed_response(response, transfer_amplitude=epsilon())
    assert "UNEQUAL_RETURN" in diag.failure_reasons
