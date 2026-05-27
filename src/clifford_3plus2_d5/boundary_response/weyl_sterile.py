"""V6 exact semi-infinite Weyl-function sterile theorem.

V5 proves the product sterile-tail mechanism at finite shell depth:

    H_Q^(N) = H_chain^(N) ⊗ I_family.

The finite transfer amplitude converges to ``epsilon``.  This module replaces
that convergence audit with the semi-infinite chain Weyl function.  For the
unit nearest-neighbor half-line adjacency, the head Green function is the
decaying solution of

    m(z) = 1 / (z - m(z)),

namely

    m(z) = (z - sqrt(z^2 - 4)) / 2,

where the branch is fixed by ``m(z) ~ 1/z`` at infinity.  At the transfer probe
``z = 2 sqrt(2)``, this equals ``epsilon = sqrt(2)-1`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import transfer_probe
from clifford_3plus2_d5.boundary_response.product_sterile import (
    normalized_rank_one_negative_control,
    product_sterile_normalized_response,
    product_sterile_transfer_amplitude,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    k_nu_operator,
    residual_basis_matrix,
    residual_projectors,
)
from clifford_3plus2_d5.boundary_response.schur import matrix_equal
from clifford_3plus2_d5.boundary_response.transfer import epsilon, epsilon_fourth, epsilon_squared


def semi_infinite_weyl_function(z: sp.Expr) -> sp.Expr:
    """Return the decaying Weyl function for the semi-infinite unit chain."""

    return sp.simplify((z - sp.sqrt(z**2 - 4)) / 2)


def weyl_quadratic_residual(z: sp.Expr) -> sp.Expr:
    """Return the residual of ``m(z)^2 - z m(z) + 1 = 0``."""

    m_z = semi_infinite_weyl_function(z)
    return sp.simplify(m_z**2 - z * m_z + 1)


def weyl_fixed_point_residual(z: sp.Expr) -> sp.Expr:
    """Return the residual of ``m(z) = 1 / (z - m(z))``."""

    m_z = semi_infinite_weyl_function(z)
    return sp.simplify(m_z - 1 / (z - m_z))


def weyl_branch_limit(z: sp.Symbol) -> sp.Expr:
    """Return ``lim_{z -> oo} z m(z)`` for the selected branch."""

    return sp.limit(z * semi_infinite_weyl_function(z), z, sp.oo)


def weyl_transfer_amplitude(z: sp.Expr) -> sp.Expr:
    """Return the exact semi-infinite transfer amplitude.

    For the half-line chain, the resolvent equation at the head gives
    ``G(1,0) / G(0,0) = m(z)`` once ``m(z)`` satisfies the Weyl fixed point.
    """

    return semi_infinite_weyl_function(z)


def weyl_product_sterile_response(z: sp.Expr | None = None) -> sp.Matrix:
    """Return the exact product sterile self-energy ``Sigma(z)``."""

    probe = transfer_probe() if z is None else z
    m_z = semi_infinite_weyl_function(probe)
    projectors = residual_projectors()
    normalized = m_z**2 * projectors["u"] + projectors["b"]
    return sp.simplify(m_z * normalized)


def weyl_product_sterile_normalized_response(z: sp.Expr | None = None) -> sp.Matrix:
    """Return the exact normalized product response ``m(z)^2 P_u + P_b``."""

    probe = transfer_probe() if z is None else z
    m_z = semi_infinite_weyl_function(probe)
    projectors = residual_projectors()
    return sp.simplify(m_z**2 * projectors["u"] + projectors["b"])


@dataclass(frozen=True)
class WeylSterileAuditPayload:
    """Verdict payload for the V6 exact Weyl-function theorem."""

    final_verdict: str
    z_probe: sp.Expr
    weyl_value: sp.Expr
    fixed_point_residual: sp.Expr
    quadratic_residual: sp.Expr
    branch_limit: sp.Expr
    mass_ratio: sp.Expr
    mass_squared_ratio: sp.Expr
    response_matches_target: bool
    finite_v5_transfer_error: sp.Expr
    finite_v5_response_error_norm: sp.Expr
    negative_control_has_cross_return: bool
    pmns_ckm_parked: bool
    interpretation: str


def weyl_sterile_audit_payload(*, finite_shells: int = 10) -> WeylSterileAuditPayload:
    """Return the exact V6 Weyl-function theorem verdict."""

    z = transfer_probe()
    m_z = semi_infinite_weyl_function(z)
    normalized = weyl_product_sterile_normalized_response(z)
    target = k_nu_operator()
    response_matches = matrix_equal(normalized, target)

    z_symbol = sp.symbols("z", positive=True)
    fixed_residual = weyl_fixed_point_residual(z_symbol)
    quadratic_residual = weyl_quadratic_residual(z_symbol)
    branch_limit = weyl_branch_limit(z_symbol)

    finite_transfer_error = sp.simplify(product_sterile_transfer_amplitude(finite_shells) - m_z)
    finite_response = product_sterile_normalized_response(finite_shells)
    finite_response_error = finite_response - normalized
    finite_response_error_norm = sp.simplify(
        sum(entry**2 for entry in finite_response_error)
    )

    basis = residual_basis_matrix(("a", "u", "b"))
    negative = normalized_rank_one_negative_control(finite_shells)
    negative_in_basis = (basis.T * negative * basis).applyfunc(sp.simplify)
    negative_cross = sp.simplify(negative_in_basis[1, 2]) != 0 or sp.simplify(negative_in_basis[2, 1]) != 0

    if (
        sp.simplify(m_z - epsilon()) == 0
        and fixed_residual == 0
        and quadratic_residual == 0
        and branch_limit == 1
        and response_matches
        and negative_cross
    ):
        final_verdict = "PRODUCT_STERILE_LIMIT_PASS"
        interpretation = (
            "The semi-infinite unit-chain Weyl function fixes the V5 product "
            "sterile limit exactly. At z = 2 sqrt(2), m(z) = epsilon, so the "
            "normalized product response is epsilon^2 P_u + P_b with no "
            "endpoint load tuning. PMNS/CKM remain parked until explicit "
            "charged-lepton and quark boundary shells are derived."
        )
    else:
        final_verdict = "PRODUCT_STERILE_LIMIT_KILL"
        interpretation = (
            "The Weyl-function theorem failed the branch, fixed-point, target "
            "response, or negative-control check. PMNS/CKM remain parked."
        )

    return WeylSterileAuditPayload(
        final_verdict=final_verdict,
        z_probe=z,
        weyl_value=m_z,
        fixed_point_residual=fixed_residual,
        quadratic_residual=quadratic_residual,
        branch_limit=branch_limit,
        mass_ratio=epsilon_squared(),
        mass_squared_ratio=epsilon_fourth(),
        response_matches_target=response_matches,
        finite_v5_transfer_error=finite_transfer_error,
        finite_v5_response_error_norm=finite_response_error_norm,
        negative_control_has_cross_return=negative_cross,
        pmns_ckm_parked=True,
        interpretation=interpretation,
    )
