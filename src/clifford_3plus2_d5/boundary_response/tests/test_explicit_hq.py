"""Tests for V3 explicit finite-shell sterile boundary diagnostics."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import (
    diagnose_framed_response,
    explicit_hq_audit_payload,
    explicit_hq_block_hamiltonian,
    explicit_hq_coupling_matrix,
    explicit_hq_self_energy,
    finite_shell_effective_response,
    finite_transfer_chain_hamiltonian,
    green_transfer_amplitude,
    safe_explicit_hq_self_energy,
    transfer_amplitude_errors,
    transfer_probe,
)
from clifford_3plus2_d5.boundary_response.framed_sterile import (
    framed_sterile_effective_response,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_s3_invariant,
    is_selected_s2_invariant,
    residual_basis_matrix,
)
from clifford_3plus2_d5.boundary_response.schur import matrix_equal
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def test_transfer_chain_hamiltonian_shape_and_symmetry() -> None:
    h_q = finite_transfer_chain_hamiltonian(5)
    assert h_q.shape == (5, 5)
    assert matrix_equal(h_q, h_q.T)


def test_explicit_hq_shapes() -> None:
    h_q = explicit_hq_block_hamiltonian(4)
    coupling = explicit_hq_coupling_matrix(4)
    assert h_q.shape == (5, 5)
    assert coupling.shape == (5, 3)


def test_explicit_hq_coupling_has_no_radial_component_by_default() -> None:
    coupling = explicit_hq_coupling_matrix(4)
    basis = residual_basis_matrix(("a", "u", "b"))
    in_residual_basis = (coupling * basis).applyfunc(sp.simplify)
    assert all(sp.simplify(entry) == 0 for entry in in_residual_basis[:, 0])


def test_green_transfer_amplitude_converges_monotonically_to_epsilon() -> None:
    errors = transfer_amplitude_errors(10)
    assert all(float(sp.N(err)) > 0 for err in errors)
    assert all(
        abs(float(sp.N(right))) <= abs(float(sp.N(left)))
        for left, right in zip(errors, errors[1:], strict=False)
    )
    assert abs(float(sp.N(errors[-1]))) < 1e-6


def test_finite_shell_effective_response_has_no_cross_terms() -> None:
    response = finite_shell_effective_response(8)
    basis = residual_basis_matrix(("a", "u", "b"))
    residual = (basis.T * response * basis).applyfunc(sp.simplify)
    assert residual[0, 0] == 0
    assert all(residual[row, col] == 0 for row in range(3) for col in range(3) if row != col)
    assert is_selected_s2_invariant(response)
    assert not is_s3_invariant(response)


def test_bad_depth_variant_triggers_wrong_transfer_ratio() -> None:
    response = finite_shell_effective_response(8, collective_depth=0)
    diag = diagnose_framed_response(response, transfer_amplitude=sp.Integer(1))
    assert "WRONG_TRANSFER_RATIO" in diag.failure_reasons


def test_artificial_radial_leakage_is_diagnosed() -> None:
    response = explicit_hq_self_energy(4, include_radial_leakage=True)
    amp = green_transfer_amplitude(4)
    diag = diagnose_framed_response(response, transfer_amplitude=amp)
    assert "RADIAL_LEAKAGE" in diag.failure_reasons


def test_cross_return_is_diagnosed() -> None:
    response = framed_sterile_effective_response(cross_return=sp.Integer(1))
    diag = diagnose_framed_response(response, transfer_amplitude=epsilon())
    assert "CROSS_RETURN" in diag.failure_reasons


def test_unequal_return_is_diagnosed() -> None:
    response = framed_sterile_effective_response(collective_return=sp.Integer(2))
    diag = diagnose_framed_response(response, transfer_amplitude=epsilon())
    assert "UNEQUAL_RETURN" in diag.failure_reasons


def test_singular_probe_is_reported_without_uncaught_inverse_error() -> None:
    response, singular = safe_explicit_hq_self_energy(2, z_probe=sp.Integer(0))
    assert response is None
    assert singular


def test_raw_shell_schur_response_is_diagnosed_separately() -> None:
    response = explicit_hq_self_energy(8, z_probe=transfer_probe())
    amp = green_transfer_amplitude(8)
    diag = diagnose_framed_response(response, transfer_amplitude=amp)
    assert "UNEQUAL_RETURN" in diag.failure_reasons


def test_explicit_hq_audit_reports_convergence_only() -> None:
    payload = explicit_hq_audit_payload(shells=10)
    assert payload.final_verdict == "EXPLICIT_HQ_CONVERGENCE_ONLY"
    assert payload.transfer_errors_decrease
    assert abs(float(sp.N(payload.transfer_amplitude_error))) < 1e-6
    assert payload.raw_diagnostics is not None
    assert "UNEQUAL_RETURN" in payload.raw_diagnostics.failure_reasons
    assert payload.pmns_ckm_parked
