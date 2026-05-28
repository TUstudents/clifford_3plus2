"""V17 chiral boundary normalization no-go.

The tempting chiral-boundary route is to exchange the even primitive channel
with the normalized odd collective mode

    |O> = (|o_1> + ... + |o_5>) / sqrt(5).

This is a legitimate orthogonal involution compatible with odd-shell ``S_5``
symmetry.  But its collective-plane eigenvectors are ``|e> ± |O>``.  In the
V15 primitive-channel convention

    psi(r) ∝ |e> + r (|o_1> + ... + |o_5>),

those eigenvectors have ``r = ±1/sqrt(5)``, not ``r = ±1``.  Therefore the
orthogonal chiral swap gives phase ``pi/4`` on the positive branch, not the CKM
phase ``atan(sqrt(5))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
    parity_preserving_generators,
    primitive_even_vector,
    primitive_odd_sum_vector,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)
from clifford_3plus2_d5.boundary_response.quark_coin_rigidity import (
    isotropic_quark_phase_angle,
)


def normalized_odd_collective_vector() -> sp.Matrix:
    """Return ``|O> = (sum_A |o_A>) / sqrt(5)``."""

    return primitive_odd_sum_vector() / sp.sqrt(5)


def even_projector() -> sp.Matrix:
    """Return the even-channel projector."""

    even = primitive_even_vector()
    return even * even.T


def odd_collective_projector() -> sp.Matrix:
    """Return the normalized odd-collective projector."""

    odd_collective = normalized_odd_collective_vector()
    return odd_collective * odd_collective.T


def odd_projector() -> sp.Matrix:
    """Return the five-dimensional odd-shell projector."""

    return sp.diag(0, 1, 1, 1, 1, 1)


def odd_perp_projector() -> sp.Matrix:
    """Return the odd-shell subspace orthogonal to the collective mode."""

    return sp.simplify(odd_projector() - odd_collective_projector())


def orthogonal_chiral_swap() -> sp.Matrix:
    """Return the orthogonal involution exchanging ``|e>`` and ``|O>``."""

    even = primitive_even_vector()
    odd_collective = normalized_odd_collective_vector()
    return sp.simplify(even * odd_collective.T + odd_collective * even.T - odd_perp_projector())


def collective_positive_eigenvector() -> sp.Matrix:
    """Return the ``+1`` collective-plane eigenvector of the chiral swap."""

    return primitive_even_vector() + normalized_odd_collective_vector()


def collective_negative_eigenvector() -> sp.Matrix:
    """Return the ``-1`` collective-plane eigenvector of the chiral swap."""

    return primitive_even_vector() - normalized_odd_collective_vector()


def primitive_ratio_from_collective_vector(vector: sp.Matrix) -> sp.Expr:
    """Return the V15 primitive-channel ratio ``r`` for a collective vector."""

    even_component = vector[0, 0]
    if sp.simplify(even_component) == 0:
        raise ValueError("collective vector must have a nonzero even component")
    odd_component = vector[1, 0]
    return sp.simplify(odd_component / even_component)


def orthogonal_chiral_swap_forced_ratio() -> sp.Expr:
    """Return the positive-branch ratio forced by the orthogonal swap."""

    return primitive_ratio_from_collective_vector(collective_positive_eigenvector())


def orthogonal_chiral_swap_phase_angle() -> sp.Expr:
    """Return the positive-branch V15 phase induced by the orthogonal swap."""

    return sp.simplify(isotropic_quark_phase_angle(orthogonal_chiral_swap_forced_ratio()))


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices agree exactly."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def _commutes_with_odd_s5(matrix: sp.Matrix) -> bool:
    """Return true when ``matrix`` commutes with odd-shell ``S_5`` generators."""

    return all(_matrix_equal(generator * matrix, matrix * generator) for generator in parity_preserving_generators())


@dataclass(frozen=True)
class ChiralBoundaryNormalizationAuditPayload:
    """Verdict payload for the V17 chiral boundary normalization no-go."""

    final_verdict: str
    orthogonal_chiral_swap_forced_ratio: sp.Expr
    orthogonal_chiral_swap_phase: sp.Expr
    ckm_flat_ratio: sp.Expr
    ckm_phase: sp.Expr
    orthogonal_chiral_swap_derives_ckm_phase: bool
    interpretation: str


def chiral_boundary_normalization_audit_payload() -> ChiralBoundaryNormalizationAuditPayload:
    """Return the V17 chiral boundary normalization no-go verdict."""

    odd_collective = normalized_odd_collective_vector()
    sigma = orthogonal_chiral_swap()
    identity = sp.eye(SHELL_DIMENSION)
    positive_ratio = orthogonal_chiral_swap_forced_ratio()
    positive_phase = orthogonal_chiral_swap_phase_angle()
    ckm_phase = quark_boundary_phase_angle()

    checks_pass = (
        sp.simplify((odd_collective.T * odd_collective)[0, 0] - 1) == 0
        and _matrix_equal(sigma * sigma, identity)
        and _matrix_equal(sigma.T * sigma, identity)
        and _matrix_equal(sigma * primitive_even_vector(), odd_collective)
        and _matrix_equal(sigma * odd_collective, primitive_even_vector())
        and _commutes_with_odd_s5(sigma)
        and sp.simplify(positive_ratio - 1 / sp.sqrt(5)) == 0
        and sp.simplify(positive_phase - sp.pi / 4) == 0
        and sp.simplify(positive_phase - ckm_phase) != 0
    )

    if checks_pass:
        final_verdict = "CHIRAL_BOUNDARY_NORMALIZATION_NO_GO_PASS"
        interpretation = (
            "The orthogonal chiral swap exchanging the even channel with the "
            "normalized odd collective mode is mathematically consistent and "
            "S5-compatible, but its eigenstates force primitive-channel ratio "
            "r=1/sqrt(5), giving phase pi/4. It does not derive the CKM "
            "flatness ratio r=1."
        )
    else:
        final_verdict = "CHIRAL_BOUNDARY_NORMALIZATION_NO_GO_KILL"
        interpretation = (
            "The chiral swap was not orthogonal/involutive/S5-compatible, or "
            "the induced ratio/phase diagnostics failed."
        )

    return ChiralBoundaryNormalizationAuditPayload(
        final_verdict=final_verdict,
        orthogonal_chiral_swap_forced_ratio=positive_ratio,
        orthogonal_chiral_swap_phase=positive_phase,
        ckm_flat_ratio=sp.Integer(1),
        ckm_phase=ckm_phase,
        orthogonal_chiral_swap_derives_ckm_phase=sp.simplify(positive_phase - ckm_phase) == 0,
        interpretation=interpretation,
    )
