"""V2 framed sterile-boundary effective audit.

This module tests the corrected non-tautological standard at the effective
boundary level.  It does *not* accept a prebuilt coupling of the form

    epsilon |s_u><u| + |s_b><b|.

Instead it derives the two channel directions from local incidence vectors in
the original residual basis:

    collective tail:      (1, 1, 1) -> u
    opposite edge current:(0, 1, -1) -> b

and derives the relative channel amplitude from transfer depth:

    amp(depth=1) / amp(depth=0) = epsilon.

The remaining assumption is explicit: the two sterile return Green functions
have equal low-energy normalization and no cross-return.  This is a stronger
effective result than V1, but still not a full microscopic ``H_Q`` derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_selected_s2_invariant,
    projector,
    residual_projectors,
    residual_vectors,
)
from clifford_3plus2_d5.boundary_response.schur import matrix_equal
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def _normalize(vector: sp.Matrix) -> sp.Matrix:
    norm_sq = sp.simplify((vector.T * vector)[0])
    if norm_sq == 0:
        raise ValueError("cannot normalize the zero vector")
    return (vector / sp.sqrt(norm_sq)).applyfunc(sp.simplify)


def collective_tail_incidence_raw() -> sp.Matrix:
    """Return the local incidence vector for the residual collective tail."""

    return sp.Matrix([1, 1, 1])


def opposite_edge_incidence_raw() -> sp.Matrix:
    """Return the oriented incidence vector for the opposite residual edge."""

    return sp.Matrix([0, 1, -1])


def collective_tail_channel() -> sp.Matrix:
    """Return the normalized collective channel derived from local incidence."""

    return _normalize(collective_tail_incidence_raw())


def opposite_edge_channel() -> sp.Matrix:
    """Return the normalized opposite-edge channel derived from local incidence."""

    return _normalize(opposite_edge_incidence_raw())


def transfer_depth_amplitude(depth: int) -> sp.Expr:
    """Return the residual transfer amplitude for an integer transfer depth."""

    if depth < 0:
        raise ValueError("transfer depth must be non-negative")
    return sp.simplify(epsilon() ** depth)


def framed_sterile_coupling_matrix(
    *,
    collective_depth: int = 1,
    edge_depth: int = 0,
) -> sp.Matrix:
    """Return the derived two-channel coupling matrix ``V``.

    Rows are unresolved sterile channels and columns are projected family
    channels.  The directions come from local incidence; the relative magnitude
    comes from transfer depth.
    """

    collective = transfer_depth_amplitude(collective_depth) * collective_tail_channel().T
    edge = transfer_depth_amplitude(edge_depth) * opposite_edge_channel().T
    return sp.Matrix.vstack(collective, edge).applyfunc(sp.simplify)


def framed_sterile_effective_response(
    *,
    collective_return: sp.Expr = sp.Integer(1),
    edge_return: sp.Expr = sp.Integer(1),
    cross_return: sp.Expr = sp.Integer(0),
    collective_depth: int = 1,
    edge_depth: int = 0,
) -> sp.Matrix:
    """Return the low-energy framed sterile self-energy response.

    ``collective_return`` and ``edge_return`` model the low-energy diagonal
    sterile Green functions.  ``cross_return`` defaults to zero; nonzero values
    deliberately test failure of the diagonal-return assumption.
    """

    coupling = framed_sterile_coupling_matrix(
        collective_depth=collective_depth,
        edge_depth=edge_depth,
    )
    returns = sp.Matrix(
        [
            [collective_return, cross_return],
            [cross_return, edge_return],
        ]
    )
    return (coupling.T * returns * coupling).applyfunc(sp.simplify)


def framed_sterile_target() -> sp.Matrix:
    """Return the target response derived by the effective framed ansatz."""

    projectors = residual_projectors()
    return sp.simplify(epsilon() ** 2 * projectors["u"] + projectors["b"])


@dataclass(frozen=True)
class FramedSterileAuditPayload:
    """Verdict payload for the V2 framed sterile effective audit."""

    incidence_verdict: str
    transfer_depth_verdict: str
    response_verdict: str
    final_verdict: str
    collective_selects_u: bool
    edge_selects_b: bool
    radial_mode_absent: bool
    transfer_ratio: sp.Expr
    response_matches_target: bool
    broken_equal_return_matches_target: bool
    pmns_ckm_parked: bool
    interpretation: str


def framed_sterile_audit_payload() -> FramedSterileAuditPayload:
    """Return the V2 framed sterile effective verdict."""

    vectors = residual_vectors()
    projectors = residual_projectors()
    collective = collective_tail_channel()
    edge = opposite_edge_channel()

    collective_selects_u = matrix_equal(projector(collective), projectors["u"])
    edge_selects_b = matrix_equal(projector(edge), projectors["b"])
    radial_mode_absent = (
        sp.simplify((collective.T * vectors["a"])[0]) == 0
        and sp.simplify((edge.T * vectors["a"])[0]) == 0
    )
    incidence_pass = collective_selects_u and edge_selects_b and radial_mode_absent

    ratio = sp.simplify(transfer_depth_amplitude(1) / transfer_depth_amplitude(0))
    transfer_pass = sp.simplify(ratio - epsilon()) == 0

    response = framed_sterile_effective_response()
    target = framed_sterile_target()
    response_matches = matrix_equal(response, target) and is_selected_s2_invariant(response)
    broken_matches = matrix_equal(
        framed_sterile_effective_response(collective_return=sp.Integer(2)),
        target,
    )

    if incidence_pass:
        incidence_verdict = "FRAMED_INCIDENCE_PASS"
    else:
        incidence_verdict = "FRAMED_INCIDENCE_KILL"

    if transfer_pass:
        transfer_depth_verdict = "TRANSFER_DEPTH_PASS"
    else:
        transfer_depth_verdict = "TRANSFER_DEPTH_KILL"

    if response_matches and not broken_matches:
        response_verdict = "FRAMED_STERILE_EFFECTIVE_PASS"
    else:
        response_verdict = "FRAMED_STERILE_KILL"

    if incidence_pass and transfer_pass and response_verdict == "FRAMED_STERILE_EFFECTIVE_PASS":
        final_verdict = "FRAMED_STERILE_EFFECTIVE_PASS"
        interpretation = (
            "The framed effective sterile ansatz derives the u and b channels "
            "from local incidence vectors, derives g_u/g_b = epsilon from one "
            "extra transfer depth, and yields K_nu = epsilon^2 P_u + P_b when "
            "the two sterile return Green functions have equal low-energy "
            "normalization and zero cross-return.  This reopens only the "
            "neutrino-core gate; PMNS/CKM remain parked until explicit "
            "charged-lepton and quark boundary Hamiltonians are derived."
        )
    else:
        final_verdict = "FRAMED_STERILE_KILL"
        interpretation = (
            "The framed sterile effective ansatz failed one of the required "
            "incidence, transfer-depth, or response checks.  PMNS/CKM remain "
            "parked."
        )

    return FramedSterileAuditPayload(
        incidence_verdict=incidence_verdict,
        transfer_depth_verdict=transfer_depth_verdict,
        response_verdict=response_verdict,
        final_verdict=final_verdict,
        collective_selects_u=collective_selects_u,
        edge_selects_b=edge_selects_b,
        radial_mode_absent=radial_mode_absent,
        transfer_ratio=ratio,
        response_matches_target=response_matches,
        broken_equal_return_matches_target=broken_matches,
        pmns_ckm_parked=True,
        interpretation=interpretation,
    )
