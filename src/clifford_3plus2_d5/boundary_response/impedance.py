"""V4 endpoint impedance-matching catalog.

V3 showed that the finite transfer chain derives the ``epsilon`` transfer
amplitude but that the raw shell-coupled Schur response has the wrong endpoint
impedance.  This module audits small local endpoint loads for the opposite-edge
sterile return.

The key distinction is structural:

* an untuned local load that matches the inferred collective return would be a
  theorem candidate;
* a load that matches only after solving for a sector-specific scalar remains a
  free-parameter result.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import (
    ResponseDiagnostics,
    diagnose_framed_response,
    finite_transfer_chain_hamiltonian,
    green_transfer_amplitude,
    transfer_probe,
)
from clifford_3plus2_d5.boundary_response.framed_sterile import (
    collective_tail_channel,
    opposite_edge_channel,
)
from clifford_3plus2_d5.boundary_response.schur import self_energy
from clifford_3plus2_d5.boundary_response.transfer import epsilon


@dataclass(frozen=True)
class EndpointCandidate:
    """A local opposite-edge endpoint load."""

    name: str
    h_edge: sp.Matrix
    edge_coupling: sp.Matrix
    free_parameter_count: int
    sector_specific_tuning: bool
    graph_degree: int
    description: str
    tuning_relation: str | None = None


@dataclass(frozen=True)
class ImpedanceCandidateResult:
    """Diagnostic result for one endpoint candidate."""

    name: str
    verdict: str
    diagnostics: ResponseDiagnostics
    unresolved_sites: int
    free_parameter_count: int
    graph_degree: int
    sector_specific_tuning: bool
    tuning_relation: str | None
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ImpedanceAuditPayload:
    """Combined V4 endpoint-catalog verdict."""

    final_verdict: str
    shells: int
    best_candidate: str | None
    candidate_results: tuple[ImpedanceCandidateResult, ...]
    pmns_ckm_parked: bool
    interpretation: str


def _path_adjacency(sites: int) -> sp.Matrix:
    if sites < 1:
        raise ValueError("sites must be positive")
    h_q = sp.zeros(sites, sites)
    for idx in range(sites - 1):
        h_q[idx, idx + 1] = 1
        h_q[idx + 1, idx] = 1
    return h_q


def inferred_collective_return(shells: int) -> sp.Expr:
    """Return the collective endpoint impedance inferred from the chain."""

    z = transfer_probe()
    chain = finite_transfer_chain_hamiltonian(shells)
    resolvent = (z * sp.eye(shells) - chain).inv()
    amp = green_transfer_amplitude(shells)
    return sp.simplify(resolvent[1, 1] / (amp**2))


def matched_load_edge_energy(shells: int) -> sp.Expr:
    """Return the one-site energy that exactly matches the inferred return."""

    z = transfer_probe()
    target_return = inferred_collective_return(shells)
    return sp.simplify(z - 1 / target_return)


def endpoint_catalog(shells: int) -> tuple[EndpointCandidate, ...]:
    """Return the deterministic minimal endpoint catalog."""

    edge = sp.Matrix([1])
    matched_energy = matched_load_edge_energy(shells)
    return (
        EndpointCandidate(
            name="bare_site",
            h_edge=sp.Matrix([[0]]),
            edge_coupling=edge,
            free_parameter_count=0,
            sector_specific_tuning=False,
            graph_degree=0,
            description="One isolated opposite-edge sterile site.",
        ),
        EndpointCandidate(
            name="self_energy_stub",
            h_edge=sp.Matrix([[0, 1], [1, 0]]),
            edge_coupling=sp.Matrix([1, 0]),
            free_parameter_count=0,
            sector_specific_tuning=False,
            graph_degree=1,
            description="Endpoint site connected to one zero-energy stub.",
        ),
        EndpointCandidate(
            name="two_site_dimer",
            h_edge=sp.Matrix([[0, 1], [1, 1]]),
            edge_coupling=sp.Matrix([1, 0]),
            free_parameter_count=0,
            sector_specific_tuning=False,
            graph_degree=1,
            description="Endpoint site connected to one integer-offset dimer partner.",
        ),
        EndpointCandidate(
            name="mirrored_one_step_tail",
            h_edge=_path_adjacency(shells),
            edge_coupling=sp.Matrix([0, 1, *([0] * (shells - 2))]),
            free_parameter_count=0,
            sector_specific_tuning=False,
            graph_degree=2 if shells > 2 else 1,
            description="Opposite-edge sector coupled to shell one of a mirrored path tail.",
        ),
        EndpointCandidate(
            name="symmetric_matched_load",
            h_edge=sp.Matrix([[matched_energy]]),
            edge_coupling=edge,
            free_parameter_count=1,
            sector_specific_tuning=True,
            graph_degree=0,
            description="One-site load with energy solved to match collective return.",
            tuning_relation="edge_energy = z_transfer - 1 / inferred_collective_return",
        ),
    )


def impedance_hq_for_candidate(
    shells: int,
    candidate: EndpointCandidate,
    *,
    collective_shell: int = 1,
    include_radial_leakage: bool = False,
    include_cross_coupling: bool = False,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Build block ``H_Q`` and coupling matrix for one endpoint candidate."""

    if shells <= collective_shell:
        raise ValueError("shells must exceed collective_shell")
    chain = finite_transfer_chain_hamiltonian(shells)
    edge_dim = candidate.h_edge.rows
    h_q = sp.zeros(shells + edge_dim, shells + edge_dim)
    h_q[:shells, :shells] = chain
    h_q[shells : shells + edge_dim, shells : shells + edge_dim] = candidate.h_edge

    coupling = sp.zeros(shells + edge_dim, 3)
    coupling[collective_shell, :] = collective_tail_channel().T
    for idx, coeff in enumerate(candidate.edge_coupling):
        coupling[shells + idx, :] += coeff * opposite_edge_channel().T
        if include_cross_coupling:
            coupling[shells + idx, :] += coeff * collective_tail_channel().T
        if include_radial_leakage:
            from clifford_3plus2_d5.boundary_response.residual_basis import residual_vectors

            coupling[shells + idx, :] += coeff * residual_vectors()["a"].T
    return h_q.applyfunc(sp.simplify), coupling.applyfunc(sp.simplify)


def impedance_response_for_candidate(shells: int, candidate: EndpointCandidate) -> sp.Matrix:
    """Return raw Schur response for one endpoint candidate."""

    h_q, coupling = impedance_hq_for_candidate(shells, candidate)
    return self_energy(transfer_probe(), h_q, coupling)


def _filtered_failures(
    diagnostics: ResponseDiagnostics,
    *,
    transfer_error: sp.Expr,
    transfer_tolerance: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    for reason in diagnostics.failure_reasons:
        if reason == "WRONG_TRANSFER_RATIO" and abs(float(sp.N(transfer_error))) <= transfer_tolerance:
            continue
        failures.append(reason)
    return tuple(failures)


def evaluate_endpoint_candidate(
    shells: int,
    candidate: EndpointCandidate,
    *,
    transfer_tolerance: float = 1e-6,
) -> ImpedanceCandidateResult:
    """Evaluate one endpoint candidate."""

    response = impedance_response_for_candidate(shells, candidate)
    amp = green_transfer_amplitude(shells)
    diagnostics = diagnose_framed_response(response, transfer_amplitude=amp)
    failures = _filtered_failures(
        diagnostics,
        transfer_error=sp.simplify(amp - epsilon()),
        transfer_tolerance=transfer_tolerance,
    )

    if not failures and not candidate.sector_specific_tuning:
        verdict = "IMPEDANCE_MATCH_PASS"
    elif not failures and candidate.sector_specific_tuning:
        verdict = "IMPEDANCE_FREE_PARAMETER"
    else:
        verdict = "IMPEDANCE_KILL"

    return ImpedanceCandidateResult(
        name=candidate.name,
        verdict=verdict,
        diagnostics=diagnostics,
        unresolved_sites=shells + candidate.h_edge.rows,
        free_parameter_count=candidate.free_parameter_count,
        graph_degree=candidate.graph_degree,
        sector_specific_tuning=candidate.sector_specific_tuning,
        tuning_relation=candidate.tuning_relation,
        failure_reasons=failures,
    )


def impedance_audit_payload(shells: int = 10) -> ImpedanceAuditPayload:
    """Run the V4 endpoint impedance catalog audit."""

    results = tuple(evaluate_endpoint_candidate(shells, candidate) for candidate in endpoint_catalog(shells))
    structural_passes = tuple(r for r in results if r.verdict == "IMPEDANCE_MATCH_PASS")
    tuned_passes = tuple(r for r in results if r.verdict == "IMPEDANCE_FREE_PARAMETER")

    if structural_passes:
        best = min(
            structural_passes,
            key=lambda r: (r.free_parameter_count, r.unresolved_sites, r.graph_degree, r.name),
        )
        final = "IMPEDANCE_MATCH_PASS"
        interpretation = (
            f"Endpoint catalog found an untuned impedance match: {best.name}. "
            "This is a theorem candidate for replacing the V2 equal-return ansatz."
        )
    elif tuned_passes:
        best = min(
            tuned_passes,
            key=lambda r: (r.free_parameter_count, r.unresolved_sites, r.graph_degree, r.name),
        )
        final = "IMPEDANCE_FREE_PARAMETER"
        interpretation = (
            f"Endpoint catalog matches only through the tuned candidate {best.name}. "
            "The equality of u/b returns remains a free endpoint impedance parameter."
        )
    else:
        best = None
        final = "IMPEDANCE_KILL_MINIMAL_CATALOG"
        interpretation = (
            "No minimal endpoint catalog candidate matched the u/b return impedance. "
            "The simple sterile-edge picture needs a stronger symmetry, a larger "
            "boundary, or a semi-infinite Weyl-function model."
        )

    return ImpedanceAuditPayload(
        final_verdict=final,
        shells=shells,
        best_candidate=None if best is None else best.name,
        candidate_results=results,
        pmns_ckm_parked=True,
        interpretation=interpretation,
    )
