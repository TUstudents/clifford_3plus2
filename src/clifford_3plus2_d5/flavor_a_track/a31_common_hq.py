"""A3-1 — one common H_Q (sterile chain); the lepton Sigma is its Schur complement.

The common boundary is the semi-infinite unit sterile chain. Its decaying Weyl
transfer factor at the transfer probe ``z = 2 sqrt(2)`` is

    m(z) = semi_infinite_weyl_function(2 sqrt(2)) = sqrt(2) - 1 = epsilon,

and the lepton/neutrino core is its (normalized) Schur self-energy

    Sigma_lepton = m(z)^2 P_u + P_b = epsilon^2 P_u + P_b = K_nu.

This gate fixes that common ``H_Q`` and re-confirms the lepton sector as its
Schur complement, so the quark sector (A3-2) can be attached to the *same*
chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.flavor_a_track.reuse import (
    epsilon,
    k_nu_operator,
    semi_infinite_weyl_function,
    transfer_probe,
    weyl_product_sterile_normalized_response,
)


def common_chain_transfer_factor() -> sp.Expr:
    """Return the common sterile-chain Weyl transfer factor m(z) at z = 2 sqrt(2)."""

    return sp.simplify(semi_infinite_weyl_function(transfer_probe()))


def lepton_self_energy() -> sp.Matrix:
    """Return the normalized lepton Schur self-energy from the common chain."""

    return weyl_product_sterile_normalized_response()


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    if left.shape != right.shape:
        return False
    return all(sp.simplify(entry) == 0 for entry in left - right)


def lepton_sigma_matches_k_nu() -> bool:
    """Return true when the chain Schur self-energy equals K_nu = eps^2 P_u + P_b."""

    return _matrix_equal(lepton_self_energy(), k_nu_operator())


def transfer_factor_is_epsilon() -> bool:
    """Return true when the common chain's transfer factor equals epsilon."""

    return sp.simplify(common_chain_transfer_factor() - epsilon()) == 0


def common_hq_verdict(transfer_factor_is_epsilon: bool, sigma_matches_k_nu: bool) -> str:
    """Return the A3-1 verdict from the two checks. Pure decision (KILL-testable)."""

    if transfer_factor_is_epsilon and sigma_matches_k_nu:
        return "LEPTON_SIGMA_FROM_COMMON_HQ"
    return "LEPTON_SIGMA_FROM_COMMON_HQ_KILL"


@dataclass(frozen=True)
class CommonHqAuditPayload:
    """Verdict payload for the A3-1 common-H_Q gate."""

    final_verdict: str
    common_chain_transfer_factor: sp.Expr
    transfer_factor_is_epsilon: bool
    lepton_self_energy: sp.Matrix
    lepton_sigma_matches_k_nu: bool
    interpretation: str


def common_hq_audit_payload() -> CommonHqAuditPayload:
    """Return the A3-1 common-H_Q verdict."""

    factor_ok = transfer_factor_is_epsilon()
    sigma_ok = lepton_sigma_matches_k_nu()

    final_verdict = common_hq_verdict(factor_ok, sigma_ok)

    if final_verdict == "LEPTON_SIGMA_FROM_COMMON_HQ":
        interpretation = (
            "The common boundary H_Q is the semi-infinite sterile chain; its "
            "Weyl transfer factor at z = 2 sqrt(2) is sqrt(2)-1 = epsilon, and "
            "its normalized Schur self-energy is exactly K_nu = epsilon^2 P_u + "
            "P_b. The lepton sector is a Schur complement of this one H_Q; the "
            "quark sector is attached to the same chain in A3-2."
        )
    else:
        interpretation = (
            "The common chain transfer factor is not epsilon, or its Schur "
            "self-energy does not equal K_nu. The common H_Q is not established."
        )

    return CommonHqAuditPayload(
        final_verdict=final_verdict,
        common_chain_transfer_factor=common_chain_transfer_factor(),
        transfer_factor_is_epsilon=factor_ok,
        lepton_self_energy=lepton_self_energy(),
        lepton_sigma_matches_k_nu=sigma_ok,
        interpretation=interpretation,
    )
