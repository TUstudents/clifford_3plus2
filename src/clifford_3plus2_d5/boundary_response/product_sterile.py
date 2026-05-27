"""V5 product sterile-tail boundary response audit.

V4 found that endpoint impedance matching stays a free parameter when the
collective and opposite-edge returns are modeled as distinct local loads.  V5
tests the cleaner tensor mechanism:

    H_Q = H_chain ⊗ I_family.

The sterile dynamics is one transfer chain, while the residual family label
lives inside the unresolved sector.  This makes the ``u`` and ``b`` sterile
returns equal by tensor structure and makes their cross-return vanish by
orthogonality in the family factor.  The only asymmetry is the derived transfer
depth amplitude

    amp_N(z) = G_chain[1, 0] / G_chain[0, 0],

which converges to ``epsilon = sqrt(2)-1``.

This module deliberately includes a negative control: if the family factor is
removed and both couplings hit the same sterile head, the response is rank-one
and contains ``u``/``b`` cross terms.  That is the caveat the product model must
avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import (
    finite_transfer_chain_hamiltonian,
    green_transfer_amplitude,
    transfer_chain_resolvent,
    transfer_probe,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    residual_basis_matrix,
    residual_projectors,
    residual_vectors,
)
from clifford_3plus2_d5.boundary_response.schur import matrix_equal, self_energy
from clifford_3plus2_d5.boundary_response.transfer import epsilon, epsilon_fourth, epsilon_squared


def _shell_family_state(shells: int, shell: int, family_vector: sp.Matrix) -> sp.Matrix:
    """Return ``|shell> ⊗ |family_vector>`` in shell-major order."""

    if shells < 1:
        raise ValueError("shells must be positive")
    if shell < 0 or shell >= shells:
        raise ValueError("shell index out of range")
    if family_vector.shape != (3, 1):
        raise ValueError("family_vector must be a 3x1 column")

    state = sp.zeros(3 * shells, 1)
    start = 3 * shell
    state[start : start + 3, 0] = family_vector
    return state.applyfunc(sp.simplify)


def product_sterile_hamiltonian(shells: int) -> sp.Matrix:
    """Return the finite product bath ``H_chain ⊗ I_family``."""

    return sp.kronecker_product(finite_transfer_chain_hamiltonian(shells), sp.eye(3))


def product_sterile_resolvent(shells: int, z_probe: sp.Expr | None = None) -> sp.Matrix:
    """Return ``(z I - H_product)^-1`` for the finite product bath."""

    z = transfer_probe() if z_probe is None else z_probe
    h_q = product_sterile_hamiltonian(shells)
    return ((z * sp.eye(h_q.rows) - h_q).inv()).applyfunc(sp.simplify)


def product_sterile_factorized_resolvent(
    shells: int,
    z_probe: sp.Expr | None = None,
) -> sp.Matrix:
    """Return the product resolvent using the exact tensor factorization."""

    return sp.kronecker_product(transfer_chain_resolvent(shells, z_probe=z_probe), sp.eye(3))


def product_sterile_transfer_amplitude(
    shells: int,
    z_probe: sp.Expr | None = None,
) -> sp.Expr:
    """Return the derived one-step transfer amplitude for the product model."""

    return green_transfer_amplitude(shells, depth=1, z_probe=z_probe)


def product_sterile_head_return(shells: int, z_probe: sp.Expr | None = None) -> sp.Expr:
    """Return ``<head|(z-H_chain)^-1|head>`` for the sterile chain."""

    resolvent = transfer_chain_resolvent(shells, z_probe=z_probe)
    return sp.simplify(resolvent[0, 0])


def product_sterile_coupling_matrix(shells: int, z_probe: sp.Expr | None = None) -> sp.Matrix:
    """Return the derived product-model coupling matrix.

    Both response channels couple to the same sterile head, but to orthogonal
    family states inside ``Q``.  The collective ``u`` channel carries the
    finite-chain transfer amplitude derived from the Green function.
    """

    vectors = residual_vectors()
    amp = product_sterile_transfer_amplitude(shells, z_probe=z_probe)
    q_u = _shell_family_state(shells, 0, vectors["u"])
    q_b = _shell_family_state(shells, 0, vectors["b"])
    coupling = amp * q_u * vectors["u"].T + q_b * vectors["b"].T
    return coupling.applyfunc(sp.simplify)


def product_sterile_return_matrix(shells: int, z_probe: sp.Expr | None = None) -> sp.Matrix:
    """Return sterile returns on ``(|head,u>, |head,b>)``."""

    vectors = residual_vectors()
    q_u = _shell_family_state(shells, 0, vectors["u"])
    q_b = _shell_family_state(shells, 0, vectors["b"])
    resolvent = product_sterile_factorized_resolvent(shells, z_probe=z_probe)
    states = (q_u, q_b)
    return sp.Matrix(
        [
            [sp.simplify((left.T * resolvent * right)[0]) for right in states]
            for left in states
        ]
    )


def product_sterile_effective_response(shells: int = 10, z_probe: sp.Expr | None = None) -> sp.Matrix:
    """Return ``Sigma_N(z)`` for the product sterile model."""

    resolvent = product_sterile_factorized_resolvent(shells, z_probe=z_probe)
    coupling = product_sterile_coupling_matrix(shells, z_probe=z_probe)
    return (coupling.T * resolvent * coupling).applyfunc(sp.simplify)


def product_sterile_normalized_response(
    shells: int = 10,
    z_probe: sp.Expr | None = None,
) -> sp.Matrix:
    """Return the product response divided by the common head return."""

    response = product_sterile_effective_response(shells, z_probe=z_probe)
    head_return = product_sterile_head_return(shells, z_probe=z_probe)
    return (response / head_return).applyfunc(sp.simplify)


def product_sterile_finite_target(shells: int = 10, z_probe: sp.Expr | None = None) -> sp.Matrix:
    """Return ``amp_N^2 P_u + P_b`` for the finite product model."""

    amp = product_sterile_transfer_amplitude(shells, z_probe=z_probe)
    projectors = residual_projectors()
    return sp.simplify(amp**2 * projectors["u"] + projectors["b"])


def rank_one_sterile_negative_control(
    shells: int = 10,
    z_probe: sp.Expr | None = None,
) -> sp.Matrix:
    """Return the response when the family factor is incorrectly removed."""

    amp = product_sterile_transfer_amplitude(shells, z_probe=z_probe)
    vectors = residual_vectors()
    coupling_direction = amp * vectors["u"] + vectors["b"]
    coupling = sp.zeros(shells, 3)
    coupling[0, :] = coupling_direction.T
    z = transfer_probe() if z_probe is None else z_probe
    return self_energy(z, finite_transfer_chain_hamiltonian(shells), coupling)


def normalized_rank_one_negative_control(
    shells: int = 10,
    z_probe: sp.Expr | None = None,
) -> sp.Matrix:
    """Return the rank-one negative control divided by the head return."""

    response = rank_one_sterile_negative_control(shells, z_probe=z_probe)
    head_return = product_sterile_head_return(shells, z_probe=z_probe)
    return (response / head_return).applyfunc(sp.simplify)


@dataclass(frozen=True)
class ProductSterileAuditPayload:
    """Verdict payload for the V5 product sterile-tail audit."""

    final_verdict: str
    shells: int
    transfer_amplitude: sp.Expr
    transfer_amplitude_error: sp.Expr
    mass_ratio: sp.Expr
    mass_squared_ratio: sp.Expr
    epsilon_limit_mass_ratio: sp.Expr
    epsilon_limit_mass_squared_ratio: sp.Expr
    equal_returns: bool
    cross_return_zero: bool
    radial_mode_absent: bool
    response_matches_finite_target: bool
    negative_control_has_cross_return: bool
    negative_control_rank: int
    pmns_ckm_parked: bool
    interpretation: str


def product_sterile_audit_payload(
    shells: int = 10,
    *,
    transfer_tolerance: float = 1e-6,
) -> ProductSterileAuditPayload:
    """Return the V5 product sterile-tail verdict."""

    amp = product_sterile_transfer_amplitude(shells)
    transfer_error = sp.simplify(epsilon() - amp)
    normalized = product_sterile_normalized_response(shells)
    finite_target = product_sterile_finite_target(shells)
    response_matches = matrix_equal(normalized, finite_target)

    return_matrix = product_sterile_return_matrix(shells)
    equal_returns = sp.simplify(return_matrix[0, 0] - return_matrix[1, 1]) == 0
    cross_return_zero = sp.simplify(return_matrix[0, 1]) == 0 and sp.simplify(return_matrix[1, 0]) == 0

    basis = residual_basis_matrix(("a", "u", "b"))
    normalized_in_basis = (basis.T * normalized * basis).applyfunc(sp.simplify)
    radial_mode_absent = all(
        sp.simplify(normalized_in_basis[row, col]) == 0
        for row in range(3)
        for col in range(3)
        if row == 0 or col == 0
    )

    negative = normalized_rank_one_negative_control(shells)
    negative_in_basis = (basis.T * negative * basis).applyfunc(sp.simplify)
    negative_cross = sp.simplify(negative_in_basis[1, 2]) != 0 or sp.simplify(negative_in_basis[2, 1]) != 0
    negative_rank = negative.rank()

    converged = abs(float(sp.N(transfer_error))) <= transfer_tolerance
    if (
        equal_returns
        and cross_return_zero
        and radial_mode_absent
        and response_matches
        and negative_cross
        and negative_rank == 1
        and converged
    ):
        final_verdict = "PRODUCT_STERILE_CONVERGENCE_PASS"
        interpretation = (
            "The product sterile bath H_Q = H_chain ⊗ I_family makes the "
            "u/b sterile returns equal and the cross-return vanish by tensor "
            "structure. The finite transfer amplitude is derived from the "
            "chain Green function and converges to epsilon, so the normalized "
            "response converges to epsilon^2 P_u + P_b. PMNS/CKM remain "
            "parked until charged-lepton and quark boundary shells are "
            "derived."
        )
    else:
        final_verdict = "PRODUCT_STERILE_KILL"
        interpretation = (
            "The product sterile bath failed return equality, cross-return "
            "cancellation, radial exclusion, finite target matching, the "
            "rank-one negative control, or transfer convergence. PMNS/CKM "
            "remain parked."
        )

    return ProductSterileAuditPayload(
        final_verdict=final_verdict,
        shells=shells,
        transfer_amplitude=amp,
        transfer_amplitude_error=transfer_error,
        mass_ratio=sp.simplify(amp**2),
        mass_squared_ratio=sp.simplify(amp**4),
        epsilon_limit_mass_ratio=epsilon_squared(),
        epsilon_limit_mass_squared_ratio=epsilon_fourth(),
        equal_returns=equal_returns,
        cross_return_zero=cross_return_zero,
        radial_mode_absent=radial_mode_absent,
        response_matches_finite_target=response_matches,
        negative_control_has_cross_return=negative_cross,
        negative_control_rank=negative_rank,
        pmns_ckm_parked=True,
        interpretation=interpretation,
    )
