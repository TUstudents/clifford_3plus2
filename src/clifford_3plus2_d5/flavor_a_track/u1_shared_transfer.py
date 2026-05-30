"""U1 — shared transfer invariant across sectors.

Universality requires that every sector's flavor structure is driven by the
*same* transfer factor ``epsilon``, and that this factor is the decaying root of
the residual ``K_3`` graph (degree 2) rather than three independent coincidences.

Each sector exposes an ``epsilon``-power:

    neutrino mass ratio      : epsilon^4   (boundary_response.transfer.epsilon_fourth)
    charged-lepton leakage   : epsilon^2   (depth-2 Weyl transfer)
    quark 1<->2 / 2<->3 / 1<->3 : epsilon^2 / epsilon^4 / epsilon^6

If universality holds, all of these equal ``rho^power`` for the single residual
graph root ``rho = residual_graph_decaying_factor(3) = sqrt(2)-1``.  Two controls
make the check non-trivial:

* graph-tracking: ``K_2`` (golden) and ``K_4`` roots differ from ``K_3``, so the
  shared root is graph-specific;
* negative control: an independent-``epsilon`` sector (one built from the ``K_4``
  root) is detected — it does not match the universal prediction, so a genuinely
  sector-dependent ``epsilon`` would force ``INDEPENDENT_EPSILON_KILL``.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.flavor_a_track.reuse import (
    charged_lepton_leakage_depth_amplitude,
    epsilon_fourth,
    quark_transition_amplitude,
    residual_graph_decaying_factor,
)

K3_GRAPH_SIZE = 3


def sector_epsilon_powers() -> dict[str, int]:
    """Return the integer ``epsilon``-power exposed by each sector."""

    return {
        "neutrino_mass_ratio": 4,
        "charged_lepton_leakage": 2,
        "quark_12": 2,
        "quark_23": 4,
        "quark_13": 6,
    }


def sector_actual_values() -> dict[str, sp.Expr]:
    """Return each sector's ``epsilon``-dependent quantity from its own module."""

    return {
        "neutrino_mass_ratio": sp.simplify(epsilon_fourth()),
        "charged_lepton_leakage": sp.simplify(charged_lepton_leakage_depth_amplitude(2)),
        "quark_12": sp.simplify(quark_transition_amplitude(1, 2)),
        "quark_23": sp.simplify(quark_transition_amplitude(2, 3)),
        "quark_13": sp.simplify(quark_transition_amplitude(1, 3)),
    }


def residual_graph_root(size: int = K3_GRAPH_SIZE) -> sp.Expr:
    """Return the decaying transfer root of the residual complete graph ``K_size``."""

    return sp.simplify(residual_graph_decaying_factor(size))


def predicted_from_root(root: sp.Expr) -> dict[str, sp.Expr]:
    """Return each sector's quantity predicted by a single transfer root."""

    powers = sector_epsilon_powers()
    return {name: sp.simplify(root ** power) for name, power in powers.items()}


def sectors_match_root(root: sp.Expr) -> bool:
    """Return true when every sector's actual value equals ``root^power``."""

    actual = sector_actual_values()
    predicted = predicted_from_root(root)
    return all(sp.simplify(actual[name] - predicted[name]) == 0 for name in actual)


def graph_roots_distinct() -> bool:
    """Return true when the ``K_2``, ``K_3``, ``K_4`` roots are all different."""

    k2 = residual_graph_root(2)
    k3 = residual_graph_root(3)
    k4 = residual_graph_root(4)
    return (
        sp.simplify(k2 - k3) != 0
        and sp.simplify(k4 - k3) != 0
        and sp.simplify(k2 - k4) != 0
    )


def independent_epsilon_is_detected() -> bool:
    """Return true when a ``K_4``-rooted sector fails the universal prediction.

    This is the negative control: if any real sector were built from a different
    graph root, ``sectors_match_root(K_3 root)`` would be false.  We confirm that
    a ``K_4``-based quark amplitude does not equal the universal ``K_3``
    prediction, so the gate is sensitive to a sector-dependent ``epsilon``.
    """

    k3 = residual_graph_root(3)
    k4 = residual_graph_root(4)
    tampered_quark_12 = sp.simplify(k4 ** 2)
    universal_quark_12 = sp.simplify(k3 ** 2)
    return sp.simplify(tampered_quark_12 - universal_quark_12) != 0


@dataclass(frozen=True)
class SharedTransferAuditPayload:
    """Verdict payload for the U1 shared-transfer-invariant gate."""

    final_verdict: str
    residual_graph_root: sp.Expr
    sector_powers: dict[str, int]
    sector_actual_values: dict[str, sp.Expr]
    sectors_match_k3_root: bool
    graph_roots_distinct: bool
    independent_epsilon_detected: bool
    k2_root: sp.Expr
    k4_root: sp.Expr
    interpretation: str


def shared_transfer_audit_payload() -> SharedTransferAuditPayload:
    """Return the U1 shared-transfer-invariant verdict."""

    k3_root = residual_graph_root(3)
    match = sectors_match_root(k3_root)
    distinct = graph_roots_distinct()
    detected = independent_epsilon_is_detected()

    checks_pass = match and distinct and detected

    if checks_pass:
        final_verdict = "SHARED_TRANSFER_INVARIANT"
        interpretation = (
            "Every sector's epsilon-dependent quantity equals rho^power for the "
            "single residual K3 graph root rho = sqrt(2)-1: neutrino ratio "
            "rho^4, charged-lepton leakage rho^2, quark transitions rho^2, "
            "rho^4, rho^6. The K2 and K4 roots differ from K3, so the shared "
            "invariant is graph-specific, and a K4-rooted sector is detected as "
            "non-universal. The transfer invariant is shared, not three "
            "coincidences."
        )
    else:
        final_verdict = "INDEPENDENT_EPSILON_KILL"
        interpretation = (
            "The sectors do not all track the single residual K3 graph root "
            "(or the graph-tracking / independence controls failed). The "
            "transfer invariant is sector-dependent; universality is dead."
        )

    return SharedTransferAuditPayload(
        final_verdict=final_verdict,
        residual_graph_root=k3_root,
        sector_powers=sector_epsilon_powers(),
        sector_actual_values=sector_actual_values(),
        sectors_match_k3_root=match,
        graph_roots_distinct=distinct,
        independent_epsilon_detected=detected,
        k2_root=residual_graph_root(2),
        k4_root=residual_graph_root(4),
        interpretation=interpretation,
    )
