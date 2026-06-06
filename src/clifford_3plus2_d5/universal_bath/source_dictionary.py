"""Write-once source dictionary gate for the universal bath sidecar.

The dictionary freezes only source anchors that are fixed upstream by charge
and boundary geometry.  It records quark sources as unresolved until a
mass-independent BCC source vector exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import sympy as sp

from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    selected_charged_lepton_port,
)
from clifford_3plus2_d5.boundary_response.framed_sterile import (
    collective_tail_channel,
    opposite_edge_channel,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    residual_basis_matrix,
    standard_basis,
)
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind

IMAG = sp.I


class SourceStatus(StrEnum):
    """Source-freezing status."""

    FROZEN = "frozen"
    UNRESOLVED = "unresolved_not_frozen"


@dataclass(frozen=True)
class SourceAnchor:
    """One source entry in the write-once dictionary."""

    label: str
    sector: str
    reduction: ReductionKind
    charge_anchor: str
    port_vector: sp.Matrix | None
    residual_components: dict[str, sp.Expr]
    normal_depth: int | None
    certainty: str
    status: SourceStatus
    upstream: str
    interpretation: str
    uses_flavor_data: bool = False


@dataclass(frozen=True)
class SourceDictionaryPayload:
    """Session 02 verdict payload."""

    final_verdict: str
    frozen_sources: tuple[SourceAnchor, ...]
    unresolved_sources: tuple[SourceAnchor, ...]
    survival_operator: sp.Matrix
    survival_operator_universal: bool
    frozen_survival_weights: dict[str, sp.Expr]
    all_frozen_sources_survive: bool
    no_flavor_data_used: bool
    all_physical_sources_frozen: bool
    interpretation: str


def bb_same_normal_blocks() -> tuple[sp.Matrix, sp.Matrix]:
    """Return the pinned BB same-normal edge blocks."""

    b_plus = sp.Matrix(
        [
            [sp.Rational(1, 2), IMAG / 2],
            [0, 0],
        ]
    )
    b_minus = sp.Matrix(
        [
            [0, 0],
            [IMAG / 2, sp.Rational(1, 2)],
        ]
    )
    return b_plus, b_minus


def bb_first_hop_survival_operator() -> sp.Matrix:
    """Return ``B_+^* B_+ + B_-^* B_-`` for the same-normal BB branch."""

    b_plus, b_minus = bb_same_normal_blocks()
    return sp.simplify(b_plus.H * b_plus + b_minus.H * b_minus)


def residual_components(vector: sp.Matrix) -> dict[str, sp.Expr]:
    """Return components of a standard-port vector in the ``(a,u,b)`` basis."""

    basis = residual_basis_matrix(("a", "u", "b"))
    components = (basis.T * vector).applyfunc(sp.simplify)
    return {
        "a": components[0, 0],
        "u": components[1, 0],
        "b": components[2, 0],
    }


def vector_norm_squared(vector: sp.Matrix) -> sp.Expr:
    """Return the exact Euclidean norm squared of a real port vector."""

    return sp.simplify((vector.T * vector)[0])


def first_hop_survival_weight(anchor: SourceAnchor) -> sp.Expr | None:
    """Return the radial first-hop survival weight for a frozen source."""

    if anchor.port_vector is None:
        return None
    return sp.simplify(vector_norm_squared(anchor.port_vector) / 2)


def frozen_source_anchors() -> tuple[SourceAnchor, ...]:
    """Return source anchors fixed without flavor data."""

    e1, _, _ = standard_basis()
    return (
        SourceAnchor(
            label="neutrino_collective_u",
            sector="neutrino",
            reduction=ReductionKind.POSITIVE_JACOBI,
            charge_anchor=(
                "neutral lepton / sterile Majorana return channel; residual "
                "family label carried by the product sterile bath"
            ),
            port_vector=collective_tail_channel(),
            residual_components=residual_components(collective_tail_channel()),
            normal_depth=1,
            certainty="C:7",
            status=SourceStatus.FROZEN,
            upstream="boundary_response.framed_sterile: collective_tail_channel, depth 1",
            interpretation=(
                "Local collective incidence (1,1,1) selects the residual singlet "
                "u and carries one extra transfer depth."
            ),
        ),
        SourceAnchor(
            label="neutrino_edge_b",
            sector="neutrino",
            reduction=ReductionKind.POSITIVE_JACOBI,
            charge_anchor=(
                "neutral lepton / sterile Majorana return channel; opposite-edge "
                "S2 residual family label"
            ),
            port_vector=opposite_edge_channel(),
            residual_components=residual_components(opposite_edge_channel()),
            normal_depth=0,
            certainty="C:7",
            status=SourceStatus.FROZEN,
            upstream="boundary_response.framed_sterile: opposite_edge_channel, depth 0",
            interpretation=(
                "Local opposite-edge incidence (0,1,-1) selects the residual b "
                "channel with no extra transfer depth."
            ),
        ),
        SourceAnchor(
            label="charged_lepton_active_e1",
            sector="charged_lepton",
            reduction=ReductionKind.CMV_OPUC,
            charge_anchor=(
                "L -> e_R via the direct Higgs doublet H; selected colorless "
                "active boundary port"
            ),
            port_vector=selected_charged_lepton_port(),
            residual_components=residual_components(e1),
            normal_depth=2,
            certainty="C:6",
            status=SourceStatus.FROZEN,
            upstream="boundary_response.charged_lepton_leakage: selected e1, depth 2",
            interpretation=(
                "The selected active port is e1 = sqrt(2/3) a + 1/sqrt(3) u, "
                "with zero b component.  The two-step depth is the existing "
                "charged-lepton leakage assumption, not a charged-lepton mass fit."
            ),
        ),
    )


def unresolved_source_anchors() -> tuple[SourceAnchor, ...]:
    """Return source anchors whose charge is known but BCC source is not frozen."""

    return (
        SourceAnchor(
            label="up_quark_boundary_source",
            sector="up_quark",
            reduction=ReductionKind.CMV_OPUC,
            charge_anchor="Q_L -> u_R via H_tilde, color triplet, hypercharge conserving",
            port_vector=None,
            residual_components={},
            normal_depth=None,
            certainty="C:3",
            status=SourceStatus.UNRESOLVED,
            upstream=(
                "boundary_response quark Q1-Q3 gates give shell/depth/prefactors, "
                "but not a universal-bath source vector"
            ),
            interpretation=(
                "The quark sidecar supplies conditional shell and CKM-depth "
                "structure.  It does not yet freeze the source vector V_f from "
                "charge plus BCC boundary geometry, so this dictionary refuses "
                "to invent one."
            ),
        ),
        SourceAnchor(
            label="down_quark_boundary_source",
            sector="down_quark",
            reduction=ReductionKind.INDEFINITE_LOOKAHEAD_JACOBI,
            charge_anchor="Q_L -> d_R via H, color triplet, hypercharge conserving",
            port_vector=None,
            residual_components={},
            normal_depth=None,
            certainty="C:3",
            status=SourceStatus.UNRESOLVED,
            upstream=(
                "boundary_response quark Q1-Q3 gates give shell/depth/prefactors, "
                "but not a universal-bath source vector"
            ),
            interpretation=(
                "The down-quark reduction class is known, but the real symmetric "
                "bath source is not fixed as a port/depth vector.  It remains a "
                "Session 07 object, not a Session 02 assumption."
            ),
        ),
    )


def source_dictionary_payload() -> SourceDictionaryPayload:
    """Return the Session 02 source-dictionary verdict."""

    frozen = frozen_source_anchors()
    unresolved = unresolved_source_anchors()
    survival = bb_first_hop_survival_operator()
    survival_universal = survival == sp.eye(2) / 2
    weights = {
        anchor.label: first_hop_survival_weight(anchor)
        for anchor in frozen
    }
    frozen_survive = all(sp.simplify(weight - sp.Rational(1, 2)) == 0 for weight in weights.values())
    no_flavor_data = not any(anchor.uses_flavor_data for anchor in (*frozen, *unresolved))
    all_physical_sources_frozen = not unresolved

    if survival_universal and frozen_survive and no_flavor_data and unresolved:
        final_verdict = "SOURCE_DICTIONARY_CORE_PASS"
        interpretation = (
            "Session 02 freezes the lepton-side source anchors supported by "
            "existing boundary-response gates and certifies the universal BB "
            "first-hop survival weight 1/2 for each normalized q=0 source. "
            "Up/down quark charge anchors are recorded, but their BCC bath "
            "source vectors are not frozen; this is intentional anti-fitting, "
            "not an implementation gap."
        )
    elif survival_universal and frozen_survive and no_flavor_data and all_physical_sources_frozen:
        final_verdict = "SOURCE_DICTIONARY_FROZEN_PASS"
        interpretation = (
            "All physical source anchors are frozen and share the universal BB "
            "first-hop survival branch."
        )
    else:
        final_verdict = "SOURCE_DICTIONARY_KILL"
        interpretation = (
            "The source dictionary failed its survival, normalization, or "
            "no-flavor-data gate."
        )

    return SourceDictionaryPayload(
        final_verdict=final_verdict,
        frozen_sources=frozen,
        unresolved_sources=unresolved,
        survival_operator=survival,
        survival_operator_universal=survival_universal,
        frozen_survival_weights=weights,
        all_frozen_sources_survive=frozen_survive,
        no_flavor_data_used=no_flavor_data,
        all_physical_sources_frozen=all_physical_sources_frozen,
        interpretation=interpretation,
    )


def source_dictionary_scaffold_payload() -> SourceDictionaryPayload:
    """Compatibility wrapper for older Session 01 callers."""

    return source_dictionary_payload()
