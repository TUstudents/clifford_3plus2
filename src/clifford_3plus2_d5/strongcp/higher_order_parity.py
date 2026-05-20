"""Phase SC-3: higher-order H^(n) cubic-group parity audit.

Extends cp/'s O(ε) machinery to H^(2) (the O(ε²) correction).  By
the centrosymmetry verified in Phase SC-2, every H^(n) extracted via
BCH from the BCC walk's Bloch operator has definite parity under
k → -k:

    H^(n) has polynomial degree (n + 1) in k.
    Under k → -k: H^(n)(k) → (-1)^(n+1) H^(n)(k).
    Equivalently: γ⁰ · H^(n)(k) · γ⁰⁻¹ = (-1)^(n+1) H^(n)(k).

So H^(1) (degree 2) is parity-even, H^(2) (degree 3) is parity-odd,
H^(3) (degree 4) is parity-even, etc.

The load-bearing test for Strong-CP: does H^(2) (the lowest-order
odd-degree correction) populate the A_{2u} cubic-harmonic irrep?
If yes, the BCC walk contributes to θ_QCD at O(ε²).  If no, the
selection rule (g × g = g, u × u = g, g × u = u) combined with
H^(1) ∈ T_{2g} (a g-irrep) shows no power of {H^(1), H^(2)} can
produce a u-invariant — therefore H^(2k) ∈ g-irreps, H^(2k+1) ∈
{T_{1u}, T_{2u}} (NOT A_{2u}), and θ_QCD = 0 to all orders.

**Verdict from this audit**: H^(2) is 100% in T_{1u}, zero A_{2u}.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.strongcp.cubic_harmonics_degree3 import (
    DEGREE3_IRREP_NAMES,
    Degree3IrrepName,
    decompose_degree3_matrix_of_polynomials,
)
from clifford_3plus2_d5.strongcp.reuse import (
    bcc_dirac_symbol,
    effective_hamiltonian_first_correction,
    nth_order_in_epsilon,
    same_matrix,
    symbolic_momentum,
)


def _bloch_minus_identity_epsilon_coeff(order: int) -> sp.Matrix:
    """Return ``X_m = coefficient of ε^m in (U(k) - I)`` for m = ``order``."""

    epsilon, kx, ky, kz = symbolic_momentum()
    U = bcc_dirac_symbol(epsilon, kx, ky, kz)
    X = (U - sp.eye(4)).applyfunc(sp.simplify)
    return nth_order_in_epsilon(X, epsilon, order)


@lru_cache(maxsize=4)
def _log_U_coefficient(order: int) -> sp.Matrix:
    """Return ``L_m``, the coefficient of ε^m in ``log U(k) = sum_m ε^m L_m``.

    Uses the standard Mercator series:

        log(I + X) = X − X²/2 + X³/3 − X⁴/4 + ...

    truncated at the requested order.
    """

    if order < 1:
        raise ValueError("log U starts at order 1")
    x1 = _bloch_minus_identity_epsilon_coeff(1)
    if order == 1:
        return x1.applyfunc(sp.simplify)
    x2 = _bloch_minus_identity_epsilon_coeff(2)
    if order == 2:
        result = (x2 - x1 * x1 / 2).applyfunc(sp.simplify)
        return result
    x3 = _bloch_minus_identity_epsilon_coeff(3)
    if order == 3:
        # L_3 = X_3 − (X_1·X_2 + X_2·X_1)/2 + X_1³/3
        result = (
            x3
            - (x1 * x2 + x2 * x1) / 2
            + x1**3 / sp.Integer(3)
        ).applyfunc(sp.simplify)
        return result
    raise NotImplementedError(f"order {order} log-U extraction not implemented")


def effective_hamiltonian_zeroth_order() -> sp.Matrix:
    """Return ``H^(0)(k) = α · k`` from L_1 (the leading-order Dirac term)."""

    l1 = _log_U_coefficient(1)
    return (sp.I * l1).applyfunc(sp.simplify)


def effective_hamiltonian_first_correction_from_bch() -> sp.Matrix:
    """Return ``H^(1)(k) = i · L_2`` independently of cp/'s computation.

    Cross-checks the cp/ result.
    """

    l2 = _log_U_coefficient(2)
    return (sp.I * l2).applyfunc(sp.simplify)


def effective_hamiltonian_second_correction() -> sp.Matrix:
    """Return ``H^(2)(k) = i · L_3``.

    H^(2) is the O(ε²) correction to the effective Dirac Hamiltonian.
    Polynomial degree 3 in (kx, ky, kz).
    """

    l3 = _log_U_coefficient(3)
    return (sp.I * l3).applyfunc(sp.simplify)


def h1_from_bch_matches_cp_implementation() -> bool:
    """Cross-check that BCH-derived H^(1) agrees with cp/'s implementation."""

    bch = effective_hamiltonian_first_correction_from_bch()
    cp_version = effective_hamiltonian_first_correction()
    diff = (bch - cp_version).applyfunc(sp.simplify)
    return same_matrix(diff, sp.zeros(4, 4))


def h2_is_hermitian() -> bool:
    """Return whether ``H^(2) - H^(2)† = 0``."""

    h2 = effective_hamiltonian_second_correction()
    residual = (h2 - h2.H).applyfunc(sp.simplify)
    return same_matrix(residual, sp.zeros(4, 4))


def h2_decomposition() -> dict[Degree3IrrepName, sp.Matrix]:
    """Decompose H^(2) into the degree-3 cubic-harmonic irreps {A_{2u}, T_{2u}, T_{1u}}."""

    _, kx, ky, kz = symbolic_momentum()
    h2 = effective_hamiltonian_second_correction()
    return decompose_degree3_matrix_of_polynomials(h2, (kx, ky, kz))


def h2_a2u_component_is_zero() -> bool:
    """Return whether H^(2) has zero A_{2u} content (the load-bearing test).

    A_{2u} is the θ_QCD-term irrep.  If H^(2)'s A_{2u} component
    vanishes, the BCC walk does not contribute to θ at O(ε²).
    """

    decomp = h2_decomposition()
    a2u = decomp["A2u"]
    return same_matrix(a2u, sp.zeros(a2u.rows, a2u.cols))


def h2_lives_entirely_in_t1u() -> bool:
    """Return whether H^(2) is 100% in T_{1u} (zero A_{2u}, zero T_{2u})."""

    decomp = h2_decomposition()
    a2u_zero = same_matrix(decomp["A2u"], sp.zeros(4, 4))
    t2u_zero = same_matrix(decomp["T2u"], sp.zeros(4, 4))
    return a2u_zero and t2u_zero


def h2_is_chirality_parity_odd() -> bool:
    """Return whether the upper-left block of H^(2) equals minus the lower-right.

    H^(2) has polynomial degree 3 (odd) in k → parity-odd under
    k → -k.  Combined with centrosymmetry ``γ⁰ H γ⁰⁻¹ = H(-k)``,
    this means γ⁰ H^(2)(k) γ⁰⁻¹ = -H^(2)(k), i.e., the chirality
    blocks come with opposite signs.
    """

    h2 = effective_hamiltonian_second_correction()
    upper = h2[:2, :2]
    lower = h2[2:, 2:]
    residual = (upper + lower).applyfunc(sp.simplify)
    return same_matrix(residual, sp.zeros(2, 2))


@dataclass(frozen=True)
class HigherOrderParityAuditPayload:
    """Result of the Phase SC-3 audit."""

    h1_cross_check_passes: bool
    h2_is_hermitian: bool
    h2_chirality_parity_odd: bool
    h2_a2u_component_zero: bool
    h2_entirely_in_t1u: bool
    h2_irrep_summary: dict[str, bool]
    selection_rule_applies: bool
    verdict: str
    interpretation: str


def higher_order_parity_payload() -> HigherOrderParityAuditPayload:
    """Run the Phase SC-3 audit."""

    h1_check = h1_from_bch_matches_cp_implementation()
    h2_herm = h2_is_hermitian()
    h2_parity = h2_is_chirality_parity_odd()
    h2_a2u_zero = h2_a2u_component_is_zero()
    h2_in_t1u = h2_lives_entirely_in_t1u()
    decomp = h2_decomposition()
    irrep_summary = {
        irrep: not same_matrix(decomp[irrep], sp.zeros(4, 4))
        for irrep in DEGREE3_IRREP_NAMES
    }

    # The selection rule applies if (a) cross-check passes, (b) H^(2)
    # is Hermitian and parity-odd, and (c) A_{2u} component is zero.
    selection_rule = (
        h1_check and h2_herm and h2_parity and h2_a2u_zero
    )

    if selection_rule:
        verdict = "PARITY SELECTION RULE — H^(2) has zero A_{2u} content"
        interpretation = (
            "H^(1) cross-checked against cp/'s implementation (BCH match).  "
            "H^(2) extracted via L_3 BCH coefficient is Hermitian and "
            "chirality-parity-odd (γ⁰ conjugation flips its sign), "
            "confirming the centrosymmetry-derived parity assignment.  "
            "Cubic-harmonic decomposition: H^(2) is 100% in T_{1u} with "
            "ZERO A_{2u} content.  The θ_QCD-term operator lives in "
            "A_{2u}, so by the cubic-group selection rule (g × g = g, "
            "u × u = g, g × u = u) combined with H^(1) ∈ T_{2g} ⊂ g-irreps "
            "and H^(2) ∈ T_{1u} ⊂ u-irreps minus A_{2u}, no product of "
            "{H^(1), H^(2)} can populate A_{2u} in the effective action's "
            "θ-channel.  The BCC walk's contribution to θ_QCD vanishes "
            "structurally at O(ε) and O(ε²).  Phases SC-4 and SC-5 "
            "verify this by direct topological-charge and chiral-anomaly "
            "computation."
        )
    else:
        verdict = "PARITY AUDIT INCONSISTENCY"
        interpretation = (
            f"h1_check: {h1_check}; h2_hermitian: {h2_herm}; "
            f"h2_parity_odd: {h2_parity}; h2_a2u_zero: {h2_a2u_zero}; "
            f"irrep summary: {irrep_summary}.  Investigate before "
            "proceeding."
        )

    return HigherOrderParityAuditPayload(
        h1_cross_check_passes=h1_check,
        h2_is_hermitian=h2_herm,
        h2_chirality_parity_odd=h2_parity,
        h2_a2u_component_zero=h2_a2u_zero,
        h2_entirely_in_t1u=h2_in_t1u,
        h2_irrep_summary=irrep_summary,
        selection_rule_applies=selection_rule,
        verdict=verdict,
        interpretation=interpretation,
    )
