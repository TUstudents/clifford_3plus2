"""V3 explicit finite-shell sterile boundary diagnostics.

V2 still assumed equal diagonal sterile returns.  This module adds an explicit
finite transfer-chain ``H_Q`` and asks what it can actually support.

The chain is the standard nearest-neighbor path adjacency evaluated at the
transfer probe ``z = 2 sqrt(2)``.  The endpoint Green-function ratio

    G(depth, 0) / G(0, 0)

converges to ``epsilon = sqrt(2)-1``.  This derives the transfer-depth
amplitude from a finite resolvent rather than inserting it as a coupling.

The module deliberately keeps two checks separate:

* the Green ratio supports the V2 transfer-depth input;
* the raw shell-coupled Schur response is diagnosed independently and is not
  claimed to equal the target unless it actually does.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.matrices.exceptions import NonInvertibleMatrixError

from clifford_3plus2_d5.boundary_response.framed_sterile import (
    collective_tail_channel,
    opposite_edge_channel,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_s3_invariant,
    is_selected_s2_invariant,
    residual_basis_matrix,
    residual_projectors,
    residual_vectors,
)
from clifford_3plus2_d5.boundary_response.schur import self_energy
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def transfer_probe() -> sp.Expr:
    """Return the natural resolvent probe for the transfer-chain audit."""

    return 2 * sp.sqrt(2)


def finite_transfer_chain_hamiltonian(shells: int) -> sp.Matrix:
    """Return the finite nearest-neighbor transfer-chain adjacency."""

    if shells < 1:
        raise ValueError("shells must be positive")
    h_q = sp.zeros(shells, shells)
    for idx in range(shells - 1):
        h_q[idx, idx + 1] = 1
        h_q[idx + 1, idx] = 1
    return h_q


def transfer_chain_resolvent(shells: int, z_probe: sp.Expr | None = None) -> sp.Matrix:
    """Return ``(z I - H_chain)^-1`` for the finite transfer chain."""

    z = transfer_probe() if z_probe is None else z_probe
    h_q = finite_transfer_chain_hamiltonian(shells)
    return ((z * sp.eye(shells) - h_q).inv()).applyfunc(sp.simplify)


def green_transfer_amplitude(
    shells: int,
    *,
    depth: int = 1,
    z_probe: sp.Expr | None = None,
) -> sp.Expr:
    """Return the finite Green-function transfer amplitude to ``depth``."""

    if depth < 0:
        raise ValueError("depth must be non-negative")
    if shells <= depth:
        raise ValueError("shells must exceed the requested depth")
    if depth == 0:
        return sp.Integer(1)
    resolvent = transfer_chain_resolvent(shells, z_probe=z_probe)
    return sp.simplify(resolvent[depth, 0] / resolvent[0, 0])


def transfer_amplitude_errors(max_shells: int, *, depth: int = 1) -> tuple[sp.Expr, ...]:
    """Return exact errors ``epsilon - amp_N`` for shell sizes ``depth+1..N``."""

    if max_shells <= depth:
        raise ValueError("max_shells must exceed depth")
    return tuple(
        sp.simplify(epsilon() - green_transfer_amplitude(shells, depth=depth))
        for shells in range(depth + 1, max_shells + 1)
    )


def errors_decrease_monotonically(errors: tuple[sp.Expr, ...]) -> bool:
    """Return true when absolute numeric errors decrease monotonically."""

    floats = [abs(float(sp.N(err))) for err in errors]
    return all(right <= left for left, right in zip(floats, floats[1:], strict=False))


def finite_shell_effective_response(
    shells: int,
    *,
    collective_depth: int = 1,
    edge_depth: int = 0,
) -> sp.Matrix:
    """Return the V2-style response using amplitudes derived from the chain."""

    projectors = residual_projectors()
    amp_u = green_transfer_amplitude(shells, depth=collective_depth)
    amp_b = green_transfer_amplitude(shells, depth=edge_depth)
    return sp.simplify(amp_u**2 * projectors["u"] + amp_b**2 * projectors["b"])


def explicit_hq_block_hamiltonian(shells: int, *, edge_energy: sp.Expr = sp.Integer(0)) -> sp.Matrix:
    """Return a block ``H_Q``: transfer chain plus one opposite-edge site."""

    chain = finite_transfer_chain_hamiltonian(shells)
    h_q = sp.zeros(shells + 1, shells + 1)
    h_q[:shells, :shells] = chain
    h_q[shells, shells] = edge_energy
    return h_q


def explicit_hq_coupling_matrix(
    shells: int,
    *,
    collective_shell: int = 1,
    include_radial_leakage: bool = False,
) -> sp.Matrix:
    """Return raw shell couplings from projected family channels to ``Q``."""

    if shells <= collective_shell:
        raise ValueError("shells must exceed collective_shell")
    coupling = sp.zeros(shells + 1, 3)
    coupling[collective_shell, :] = collective_tail_channel().T
    coupling[shells, :] = opposite_edge_channel().T
    if include_radial_leakage:
        coupling[shells, :] += residual_vectors()["a"].T
    return coupling.applyfunc(sp.simplify)


def explicit_hq_self_energy(
    shells: int,
    *,
    z_probe: sp.Expr | None = None,
    collective_shell: int = 1,
    edge_energy: sp.Expr = sp.Integer(0),
    include_radial_leakage: bool = False,
) -> sp.Matrix:
    """Return the raw Schur self-energy for the finite explicit ``H_Q``."""

    z = transfer_probe() if z_probe is None else z_probe
    h_q = explicit_hq_block_hamiltonian(shells, edge_energy=edge_energy)
    coupling = explicit_hq_coupling_matrix(
        shells,
        collective_shell=collective_shell,
        include_radial_leakage=include_radial_leakage,
    )
    return self_energy(z, h_q, coupling)


@dataclass(frozen=True)
class ResponseDiagnostics:
    """Failure diagnostics for a framed response matrix."""

    lambda_a: sp.Expr
    lambda_u: sp.Expr
    lambda_b: sp.Expr
    inferred_collective_return: sp.Expr | None
    inferred_edge_return: sp.Expr
    failure_reasons: tuple[str, ...]
    selected_s2_invariant: bool
    full_s3_invariant: bool


def diagnose_framed_response(
    response: sp.Matrix,
    *,
    transfer_amplitude: sp.Expr,
) -> ResponseDiagnostics:
    """Classify deviations from ``epsilon^2 P_u + P_b`` structure."""

    basis = residual_basis_matrix(("a", "u", "b"))
    in_residual_basis = (basis.T * response * basis).applyfunc(sp.simplify)
    lambda_a = sp.simplify(in_residual_basis[0, 0])
    lambda_u = sp.simplify(in_residual_basis[1, 1])
    lambda_b = sp.simplify(in_residual_basis[2, 2])

    reasons: list[str] = []
    if any(
        sp.simplify(in_residual_basis[row, col]) != 0
        for row in range(3)
        for col in range(3)
        if row != col
    ):
        reasons.append("CROSS_RETURN")
    if lambda_a != 0 or any(
        sp.simplify(in_residual_basis[idx, 0]) != 0
        or sp.simplify(in_residual_basis[0, idx]) != 0
        for idx in (1, 2)
    ):
        reasons.append("RADIAL_LEAKAGE")

    if sp.simplify(transfer_amplitude - epsilon()) != 0:
        reasons.append("WRONG_TRANSFER_RATIO")

    inferred_edge_return = lambda_b
    inferred_collective_return = None
    if transfer_amplitude != 0:
        inferred_collective_return = sp.simplify(lambda_u / (transfer_amplitude**2))
        if sp.simplify(inferred_collective_return - inferred_edge_return) != 0:
            reasons.append("UNEQUAL_RETURN")

    return ResponseDiagnostics(
        lambda_a=lambda_a,
        lambda_u=lambda_u,
        lambda_b=lambda_b,
        inferred_collective_return=inferred_collective_return,
        inferred_edge_return=inferred_edge_return,
        failure_reasons=tuple(dict.fromkeys(reasons)),
        selected_s2_invariant=is_selected_s2_invariant(response),
        full_s3_invariant=is_s3_invariant(response),
    )


def safe_explicit_hq_self_energy(
    shells: int,
    *,
    z_probe: sp.Expr,
) -> tuple[sp.Matrix | None, bool]:
    """Return ``(self_energy, singular_probe)`` without uncaught inverse errors."""

    try:
        return explicit_hq_self_energy(shells, z_probe=z_probe), False
    except NonInvertibleMatrixError:
        return None, True


@dataclass(frozen=True)
class ExplicitHqAuditPayload:
    """V3 explicit finite-shell audit payload."""

    final_verdict: str
    shells: int
    transfer_amplitude: sp.Expr
    transfer_amplitude_error: sp.Expr
    transfer_errors_decrease: bool
    effective_diagnostics: ResponseDiagnostics
    raw_diagnostics: ResponseDiagnostics | None
    raw_singular_probe: bool
    pmns_ckm_parked: bool
    interpretation: str


def explicit_hq_audit_payload(shells: int = 10) -> ExplicitHqAuditPayload:
    """Return the V3 explicit finite-shell audit verdict."""

    amp = green_transfer_amplitude(shells, depth=1)
    errors = transfer_amplitude_errors(shells)
    effective_response = finite_shell_effective_response(shells)
    effective_diag = diagnose_framed_response(effective_response, transfer_amplitude=amp)

    raw_response, raw_singular = safe_explicit_hq_self_energy(shells, z_probe=transfer_probe())
    raw_diag = (
        None
        if raw_response is None
        else diagnose_framed_response(raw_response, transfer_amplitude=amp)
    )

    transfer_error = sp.simplify(epsilon() - amp)
    errors_decrease = errors_decrease_monotonically(errors)
    transfer_close = abs(float(sp.N(transfer_error))) < 1e-6

    if raw_diag is not None and not raw_diag.failure_reasons:
        final_verdict = "EXPLICIT_HQ_PASS"
        interpretation = (
            "The raw finite H_Q Schur response matched the framed sterile "
            "target diagnostics. PMNS/CKM still remain parked until charged "
            "lepton and quark boundary Hamiltonians are derived."
        )
    elif (
        errors_decrease
        and transfer_close
        and set(effective_diag.failure_reasons).issubset({"WRONG_TRANSFER_RATIO"})
    ):
        final_verdict = "EXPLICIT_HQ_CONVERGENCE_ONLY"
        interpretation = (
            "The explicit transfer-chain H_Q derives the epsilon transfer "
            "amplitude by Green-function convergence, and the derived finite "
            "effective response has the expected framed structure. The raw "
            "shell-coupled Schur response does not yet match the target, so "
            "the equal-return layer has not been fully replaced."
        )
    else:
        final_verdict = "EXPLICIT_HQ_KILL"
        interpretation = (
            "The explicit finite H_Q candidate failed transfer convergence or "
            "framed response diagnostics. PMNS/CKM remain parked."
        )

    return ExplicitHqAuditPayload(
        final_verdict=final_verdict,
        shells=shells,
        transfer_amplitude=amp,
        transfer_amplitude_error=transfer_error,
        transfer_errors_decrease=errors_decrease,
        effective_diagnostics=effective_diag,
        raw_diagnostics=raw_diag,
        raw_singular_probe=raw_singular,
        pmns_ckm_parked=True,
        interpretation=interpretation,
    )
