"""Session 09 microscopic BCC neutrino moment audit.

Session 03 checked the neutrino product bath

    H_Q = H_chain tensor I_family.

That is an internal consistency certificate, not a microscopic BCC derivation:
the ``u``/``b`` cross moments vanish because the family factor is an inserted
identity.  This module audits the currently available microscopic BB edge
update and asks whether it is already rich enough to compute

    <u|H_BCC^k|b>

without adding the product family factor.  The honest answer is no: the exact
BB edge graph contains spinor/q-depth structure and leakage channels, but no
family-port graph carrying ``u`` and ``b``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp

from clifford_3plus2_d5.universal_bath.neutrino_product import (
    cross_moments,
    diagonal_moment_differences,
    neutrino_product_bath_payload,
    rank_one_control_has_cross_return,
)

IMAGINARY_UNIT = sp.I


@dataclass(frozen=True)
class BBEdgeBlocks:
    """Exact same-normal and mixed-normal BB edge blocks."""

    b_plus: sp.Matrix
    b_minus: sp.Matrix
    m_plus2: sp.Matrix
    m_minus2: sp.Matrix


@dataclass(frozen=True)
class NeutrinoBCCMomentAuditPayload:
    """Session 09 verdict for the microscopic neutrino moment gate."""

    final_verdict: str
    product_internal_pass: bool
    checked_moment_powers: tuple[int, ...]
    same_normal_norm: sp.Matrix
    mixed_normal_norm: sp.Matrix
    total_norm_is_identity: bool
    q0_scar_block_available: bool
    leakage_block_available: bool
    microscopic_edge_labels: tuple[str, ...]
    missing_family_labels: tuple[str, ...]
    microscopic_family_port_graph_available: bool
    product_ansatz_cross_moments_zero: bool
    product_ansatz_diagonal_equal: bool
    product_ansatz_is_only_family_factor: bool
    bcc_family_cross_moments_defined: bool
    can_upgrade_neutrino_core: bool
    rank_one_control_has_cross_return: bool
    next_required_object: str
    interpretation: str


def canonical_hops() -> dict[tuple[int, int, int], sp.Matrix]:
    """Return the pinned BB Weyl hop matrices by body-diagonal direction."""

    q_plus = (1 + IMAGINARY_UNIT) / 4
    q_minus = (1 - IMAGINARY_UNIT) / 4
    p1 = sp.Matrix([[1, 0], [1, 0]])
    p2 = sp.Matrix([[0, 1], [0, 1]])
    p3 = sp.Matrix([[1, 0], [-1, 0]])
    p4 = sp.Matrix([[0, -1], [0, 1]])
    hop_list = [
        q_plus * p1,
        q_minus * p2,
        q_minus * p1,
        q_plus * p2,
        q_minus * p3,
        q_plus * p4,
        q_plus * p3,
        q_minus * p4,
    ]
    directions = list(product((1, -1), repeat=3))
    return {direction: hop for direction, hop in zip(directions, hop_list, strict=True)}


def block_sum(
    hops: dict[tuple[int, int, int], sp.Matrix],
    directions: tuple[tuple[int, int, int], ...],
) -> sp.Matrix:
    """Sum hop blocks over the supplied BCC directions."""

    return sum((hops[direction] for direction in directions), sp.zeros(2)).applyfunc(
        sp.simplify
    )


def bb_edge_blocks() -> BBEdgeBlocks:
    """Return the exact q=0 same-normal and q=+-2 mixed-normal edge blocks."""

    hops = canonical_hops()
    return BBEdgeBlocks(
        b_plus=block_sum(hops, ((1, 1, 1), (1, 1, -1))),
        b_minus=block_sum(hops, ((-1, -1, 1), (-1, -1, -1))),
        m_plus2=block_sum(hops, ((1, -1, 1), (1, -1, -1))),
        m_minus2=block_sum(hops, ((-1, 1, 1), (-1, 1, -1))),
    )


def same_normal_norm(blocks: BBEdgeBlocks | None = None) -> sp.Matrix:
    """Return the q=0 same-normal survival norm."""

    edge = bb_edge_blocks() if blocks is None else blocks
    return sp.simplify(edge.b_plus.H * edge.b_plus + edge.b_minus.H * edge.b_minus)


def mixed_normal_norm(blocks: BBEdgeBlocks | None = None) -> sp.Matrix:
    """Return the q=+-2 mixed-normal leakage norm."""

    edge = bb_edge_blocks() if blocks is None else blocks
    return sp.simplify(edge.m_plus2.H * edge.m_plus2 + edge.m_minus2.H * edge.m_minus2)


def microscopic_edge_labels() -> tuple[str, ...]:
    """Return the labels actually present in the current BB edge graph."""

    return ("spinor", "q0_same_normal", "q_plus2_leakage", "q_minus2_leakage")


def missing_family_labels() -> tuple[str, ...]:
    """Return the family-port labels needed for the neutrino cross-moment test."""

    return ("family_port_u", "family_port_b")


def product_ansatz_cross_moments_zero(
    powers: tuple[int, ...],
    *,
    shells: int,
) -> bool:
    """Return whether the Session 03 product cross moments vanish."""

    return all(sp.simplify(moment) == 0 for moment in cross_moments(shells, powers))


def product_ansatz_diagonal_equal(
    powers: tuple[int, ...],
    *,
    shells: int,
) -> bool:
    """Return whether the Session 03 product diagonal moments are equal."""

    return all(
        sp.simplify(difference) == 0
        for difference in diagonal_moment_differences(shells, powers)
    )


def neutrino_bcc_moment_audit_payload(
    *,
    shells: int = 6,
    powers: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> NeutrinoBCCMomentAuditPayload:
    """Return the Session 09 microscopic BCC neutrino moment audit."""

    product_payload = neutrino_product_bath_payload(shells=shells, powers=powers)
    product_internal = product_payload.final_verdict == "NEUTRINO_PRODUCT_BATH_INTERNAL_PASS"
    blocks = bb_edge_blocks()
    same_norm = same_normal_norm(blocks)
    mixed_norm = mixed_normal_norm(blocks)
    total_norm_ok = sp.simplify(same_norm + mixed_norm - sp.eye(2)) == sp.zeros(2)
    same_norm_ok = same_norm == sp.eye(2) / 2
    mixed_norm_ok = mixed_norm == sp.eye(2) / 2
    family_graph_available = False
    bcc_cross_defined = family_graph_available
    can_upgrade = product_internal and bcc_cross_defined
    product_cross_zero = product_ansatz_cross_moments_zero(powers, shells=shells)
    product_diag_equal = product_ansatz_diagonal_equal(powers, shells=shells)
    rank_one_cross = rank_one_control_has_cross_return()

    checks_pass = (
        product_internal
        and same_norm_ok
        and mixed_norm_ok
        and total_norm_ok
        and product_cross_zero
        and product_diag_equal
        and rank_one_cross
        and not family_graph_available
        and not can_upgrade
    )

    if checks_pass:
        final_verdict = "NEUTRINO_BCC_MOMENT_GRAPH_NOT_DERIVED_AUDIT"
        interpretation = (
            "The exact microscopic BB edge blocks provide the q=0 scar branch "
            "and the q=+-2 leakage branch with the expected 1/2 + 1/2 norm "
            "split.  Session 03 product-bath cross moments still vanish, but "
            "that vanishing uses an inserted family identity.  The current "
            "microscopic edge graph has spinor/q-depth labels only and no "
            "u/b family-port nodes, so <u|H_BCC^k|b> is not yet defined as a "
            "BCC walk-counting observable.  The neutrino epsilon^4 result "
            "therefore remains product-ansatz protected until a microscopic "
            "family-port boundary graph is supplied."
        )
    else:
        final_verdict = "NEUTRINO_BCC_MOMENT_AUDIT_KILL"
        interpretation = (
            "The Session 09 audit failed either the product internal "
            "prerequisite, the exact BB q=0/leakage norm split, or the "
            "negative control."
        )

    return NeutrinoBCCMomentAuditPayload(
        final_verdict=final_verdict,
        product_internal_pass=product_internal,
        checked_moment_powers=powers,
        same_normal_norm=same_norm,
        mixed_normal_norm=mixed_norm,
        total_norm_is_identity=total_norm_ok,
        q0_scar_block_available=same_norm_ok,
        leakage_block_available=mixed_norm_ok,
        microscopic_edge_labels=microscopic_edge_labels(),
        missing_family_labels=missing_family_labels(),
        microscopic_family_port_graph_available=family_graph_available,
        product_ansatz_cross_moments_zero=product_cross_zero,
        product_ansatz_diagonal_equal=product_diag_equal,
        product_ansatz_is_only_family_factor=True,
        bcc_family_cross_moments_defined=bcc_cross_defined,
        can_upgrade_neutrino_core=can_upgrade,
        rank_one_control_has_cross_return=rank_one_cross,
        next_required_object="microscopic BCC family-port boundary graph for u/b",
        interpretation=interpretation,
    )
