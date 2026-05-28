"""V12 quark transfer-depth hierarchy gate.

V11 verifies the primitive quark shell and flat coin.  V12 audits only the next
quark-sector assumption: the raw boundary transfer depths

    1 <-> 2 : epsilon^2
    2 <-> 3 : epsilon^4
    1 <-> 3 : epsilon^6.

Color-return and BCC Clebsch prefactors remain parked for Q3.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_shell_audit_payload,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon

QUARK_FAMILIES = (1, 2, 3)
CKM_TRANSITIONS = ((1, 2), (2, 3), (1, 3))
EXPECTED_TRANSITION_DEPTHS = {
    (1, 2): 2,
    (2, 3): 4,
    (1, 3): 6,
}


def quark_family_depths() -> dict[int, int]:
    """Return the minimal ordered quark boundary-depth embedding."""

    return {1: 0, 2: 2, 3: 6}


def _normalized_pair(left: int, right: int) -> tuple[int, int]:
    """Return the normalized family transition pair."""

    if left == right:
        raise ValueError("transition must use distinct families")
    pair = tuple(sorted((left, right)))
    if pair not in CKM_TRANSITIONS:
        raise ValueError(f"unknown quark family transition: {(left, right)}")
    return pair


def quark_transition_depths(
    family_depths: dict[int, int] | None = None,
) -> dict[tuple[int, int], int]:
    """Return pairwise boundary transfer depths."""

    depths = quark_family_depths() if family_depths is None else family_depths
    return {
        pair: abs(depths[pair[1]] - depths[pair[0]])
        for pair in CKM_TRANSITIONS
    }


def quark_transition_amplitude(
    left: int,
    right: int,
    family_depths: dict[int, int] | None = None,
) -> sp.Expr:
    """Return the raw transfer amplitude for a quark family transition."""

    pair = _normalized_pair(left, right)
    depth = quark_transition_depths(family_depths)[pair]
    return sp.simplify(epsilon() ** depth)


def transition_depths_are_even(depths: dict[tuple[int, int], int]) -> bool:
    """Return true when all transition depths are even."""

    return all(depth % 2 == 0 for depth in depths.values())


def transition_depths_are_additive(depths: dict[tuple[int, int], int]) -> bool:
    """Return true when ``depth_13 = depth_12 + depth_23``."""

    return depths[(1, 3)] == depths[(1, 2)] + depths[(2, 3)]


def matches_ckm_depth_ordering(depths: dict[tuple[int, int], int]) -> bool:
    """Return true when depths match the Q2 CKM hierarchy."""

    return depths == EXPECTED_TRANSITION_DEPTHS


def _expr_equal(left: sp.Expr, right: sp.Expr) -> bool:
    """Return true when two exact expressions simplify to the same value."""

    return sp.simplify(left - right) == 0


def odd_depth_family_embedding_control() -> dict[int, int]:
    """Return a bad embedding with odd transfer depths."""

    return {1: 0, 2: 1, 3: 5}


def nonadditive_transition_depth_control() -> dict[tuple[int, int], int]:
    """Return a bad direct transition-depth assignment."""

    return {
        (1, 2): 2,
        (2, 3): 4,
        (1, 3): 8,
    }


def permuted_family_embedding_control() -> dict[int, int]:
    """Return a bad embedding with family labels swapped out of CKM order."""

    return {1: 0, 2: 6, 3: 2}


@dataclass(frozen=True)
class QuarkTransferHierarchyAuditPayload:
    """Verdict payload for the V12 quark transfer hierarchy gate."""

    final_verdict: str
    family_depths: dict[int, int]
    transition_depths: dict[tuple[int, int], int]
    transition_amplitudes: dict[tuple[int, int], sp.Expr]
    even_depths: bool
    additive_depths: bool
    ckm_ordering_matches: bool
    odd_depth_control_rejected: bool
    nonadditive_control_rejected: bool
    permuted_label_control_rejected: bool
    ckm_parked: bool
    interpretation: str


def quark_transfer_hierarchy_audit_payload() -> QuarkTransferHierarchyAuditPayload:
    """Return the V12 quark transfer hierarchy verdict."""

    v11 = quark_boundary_shell_audit_payload()
    depths = quark_transition_depths()
    amplitudes = {
        pair: quark_transition_amplitude(*pair)
        for pair in CKM_TRANSITIONS
    }

    even_depths = transition_depths_are_even(depths)
    additive_depths = transition_depths_are_additive(depths)
    ckm_ordering_matches = matches_ckm_depth_ordering(depths)

    odd_depths = quark_transition_depths(odd_depth_family_embedding_control())
    odd_depth_rejected = not transition_depths_are_even(odd_depths)
    nonadditive_rejected = not transition_depths_are_additive(nonadditive_transition_depth_control())
    permuted_depths = quark_transition_depths(permuted_family_embedding_control())
    permuted_rejected = not matches_ckm_depth_ordering(permuted_depths)

    checks_pass = (
        v11.final_verdict == "QUARK_BOUNDARY_SHELL_Q1_PASS"
        and depths == EXPECTED_TRANSITION_DEPTHS
        and _expr_equal(amplitudes[(1, 2)], epsilon() ** 2)
        and _expr_equal(amplitudes[(2, 3)], epsilon() ** 4)
        and _expr_equal(amplitudes[(1, 3)], epsilon() ** 6)
        and even_depths
        and additive_depths
        and ckm_ordering_matches
        and odd_depth_rejected
        and nonadditive_rejected
        and permuted_rejected
    )

    if checks_pass:
        final_verdict = "QUARK_TRANSFER_HIERARCHY_Q2_PASS"
        interpretation = (
            "The ordered quark boundary-depth embedding {1:0, 2:2, 3:6} "
            "gives raw transfer depths 2, 4, and 6, hence amplitudes "
            "epsilon^2, epsilon^4, and epsilon^6. Odd-depth, non-additive, "
            "and permuted-label controls are rejected. CKM magnitudes remain "
            "parked until Q3 supplies color and BCC Clebsches."
        )
    else:
        final_verdict = "QUARK_TRANSFER_HIERARCHY_Q2_KILL"
        interpretation = (
            "The quark boundary-depth hierarchy, V11 prerequisite, or one of "
            "the negative controls failed. CKM magnitudes remain parked."
        )

    return QuarkTransferHierarchyAuditPayload(
        final_verdict=final_verdict,
        family_depths=quark_family_depths(),
        transition_depths=depths,
        transition_amplitudes=amplitudes,
        even_depths=even_depths,
        additive_depths=additive_depths,
        ckm_ordering_matches=ckm_ordering_matches,
        odd_depth_control_rejected=odd_depth_rejected,
        nonadditive_control_rejected=nonadditive_rejected,
        permuted_label_control_rejected=permuted_rejected,
        ckm_parked=True,
        interpretation=interpretation,
    )
