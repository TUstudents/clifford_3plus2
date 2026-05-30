"""A3-2 — the quark transfer hierarchy is the *same*-chain Schur/Green structure.

The quark transfer amplitudes are currently *assembled* as ``epsilon^depth`` for
depths {1<->2: 2, 2<->3: 4, 1<->3: 6}.  This gate shows they are powers of the
*same* sterile-chain Weyl factor that drives the lepton core (A3-1): coupling the
three quark families to the common chain at depths {0,2,6} makes the transition
amplitudes the chain resolvent / Green-function ratios

    A_ij = m(z)^{|d_i - d_j|} = epsilon^{|d_i - d_j|},

and the finite-shell chain Green ratio ``green_transfer_amplitude(depth)``
converges to the same value.  So the quark transfer sector is a Schur complement
of the one common ``H_Q``, not an independent assembly.

Controls (the gate must be able to fail): an odd transfer depth gives a power not
in the even quark hierarchy, and a *different* chain (e.g. the K_2 / golden root)
gives different amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.unified_boundary.reuse import (
    CKM_TRANSITIONS,
    green_transfer_amplitude,
    quark_transition_amplitude,
    quark_transition_depths,
    residual_graph_decaying_factor,
    semi_infinite_weyl_function,
    transfer_probe,
)

FINITE_SHELLS = 12
CONVERGENCE_TOLERANCE = 1.0e-6


def common_chain_transfer_factor() -> sp.Expr:
    """Return the common sterile-chain Weyl transfer factor (same as A3-1)."""

    return sp.simplify(semi_infinite_weyl_function(transfer_probe()))


def quark_transfer_from_common_chain() -> dict[tuple[int, int], sp.Expr]:
    """Return each quark transition amplitude as (common chain factor)^depth."""

    factor = common_chain_transfer_factor()
    depths = quark_transition_depths()
    return {pair: sp.simplify(factor ** depths[pair]) for pair in CKM_TRANSITIONS}


def quark_transfer_matches_common_chain() -> bool:
    """Return true when the assembled quark amplitudes equal the common-chain powers."""

    predicted = quark_transfer_from_common_chain()
    return all(
        sp.simplify(predicted[pair] - quark_transition_amplitude(*pair)) == 0
        for pair in CKM_TRANSITIONS
    )


def finite_shell_green_ratios_converge() -> bool:
    """Return true when finite-chain Green ratios converge to the quark amplitudes."""

    depths = quark_transition_depths()
    for pair in CKM_TRANSITIONS:
        green = green_transfer_amplitude(FINITE_SHELLS, depth=depths[pair])
        target = quark_transition_amplitude(*pair)
        if abs(float(sp.N(green - target))) > CONVERGENCE_TOLERANCE:
            return False
    return True


def odd_depth_control_distinct() -> bool:
    """Return true when an odd-depth power is not in the even quark hierarchy."""

    factor = common_chain_transfer_factor()
    odd_amplitude = sp.simplify(factor ** 3)
    return all(
        sp.simplify(odd_amplitude - quark_transition_amplitude(*pair)) != 0
        for pair in CKM_TRANSITIONS
    )


def scaled_chain_control_distinct() -> bool:
    """Return true when a different chain (K_2 / golden root) changes the amplitudes."""

    golden = sp.simplify(residual_graph_decaying_factor(2))
    depths = quark_transition_depths()
    return all(
        sp.simplify(golden ** depths[pair] - quark_transition_amplitude(*pair)) != 0
        for pair in CKM_TRANSITIONS
    )


@dataclass(frozen=True)
class QuarkTransferSchurAuditPayload:
    """Verdict payload for the A3-2 quark-transfer-as-Schur gate."""

    final_verdict: str
    common_chain_transfer_factor: sp.Expr
    quark_transition_depths: dict[tuple[int, int], int]
    quark_transfer_from_common_chain: dict[tuple[int, int], sp.Expr]
    transfer_matches_common_chain: bool
    finite_shell_converges: bool
    odd_depth_control_distinct: bool
    scaled_chain_control_distinct: bool
    interpretation: str


def quark_transfer_schur_audit_payload() -> QuarkTransferSchurAuditPayload:
    """Return the A3-2 verdict."""

    matches = quark_transfer_matches_common_chain()
    converges = finite_shell_green_ratios_converge()
    odd_distinct = odd_depth_control_distinct()
    scaled_distinct = scaled_chain_control_distinct()

    checks_pass = matches and converges and odd_distinct and scaled_distinct

    if checks_pass:
        final_verdict = "QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR"
        interpretation = (
            "The quark transfer amplitudes epsilon^2, epsilon^4, epsilon^6 are "
            "powers of the same sterile-chain Weyl factor that drives the "
            "lepton core: coupling the three families at depths {0,2,6} makes "
            "them chain Green-function ratios (finite-shell ratios converge to "
            "the same values). The quark transfer sector is a Schur complement "
            "of the one common H_Q. Odd-depth and scaled-chain controls change "
            "the amplitudes, so the match is chain-specific. Deriving the depths "
            "{0,2,6} from the chiral-16 is deferred to A3b."
        )
    else:
        final_verdict = "TRANSFER_NOT_UNIFIABLE_KILL"
        interpretation = (
            "The quark transfer amplitudes are not powers of the common sterile "
            "chain factor (or a control failed). The quark transfer sector is "
            "not a Schur complement of the lepton H_Q; the transfer boundary "
            "does not unify."
        )

    return QuarkTransferSchurAuditPayload(
        final_verdict=final_verdict,
        common_chain_transfer_factor=common_chain_transfer_factor(),
        quark_transition_depths=quark_transition_depths(),
        quark_transfer_from_common_chain=quark_transfer_from_common_chain(),
        transfer_matches_common_chain=matches,
        finite_shell_converges=converges,
        odd_depth_control_distinct=odd_distinct,
        scaled_chain_control_distinct=scaled_distinct,
        interpretation=interpretation,
    )
