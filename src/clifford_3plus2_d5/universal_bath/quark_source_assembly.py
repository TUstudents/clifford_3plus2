"""Session 15 quark source assembly and freeze audit.

The up/down finite heads exist only as conditional objects until the actual
microscopic quark source vectors are frozen.  This session assembles the
available quark-side certificates into one dependency graph and refuses the
source freeze unless the remaining microscopic inputs are selected without
flavor data.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.universal_bath.down_quark_indefinite_jacobi import (
    down_quark_indefinite_jacobi_payload,
)
from clifford_3plus2_d5.universal_bath.neutrino_active_plane import (
    active_plane_incidence_payload,
)
from clifford_3plus2_d5.universal_bath.quark_color_lift import (
    quark_color_lift_payload,
)
from clifford_3plus2_d5.universal_bath.quark_height_door import (
    quark_height_door_payload,
)
from clifford_3plus2_d5.universal_bath.source_dictionary import (
    SourceAnchor,
    SourceStatus,
    source_dictionary_payload,
)
from clifford_3plus2_d5.universal_bath.up_quark_nilpotent_cmv import (
    up_quark_nilpotent_cmv_payload,
)


HEIGHT_DYNAMICS_PREMISE = "height_dynamics_selects_up_nilpotent_down_hermitian"
ACTIVE_COLOR_RETURN_PREMISE = "microscopic_active_hidden_color_return_selects_regular_s3_shell"
BOTTOM_RANK_FIVE_PREMISE = "boundary_dynamics_selects_or_kills_down_rank_five_line"
QUARK_NORMAL_DEPTH_PREMISE = "quark_normal_depth_placements_on_bcc_scar_are_frozen"


@dataclass(frozen=True)
class QuarkSourceRequirement:
    """One requirement needed before freezing a quark source vector."""

    label: str
    derived: bool
    evidence: str
    consequence_if_missing: str


@dataclass(frozen=True)
class QuarkSourceAssemblyPayload:
    """Session 15 quark source assembly verdict."""

    final_verdict: str
    source_dictionary_pass: bool
    up_source_unresolved: bool
    down_source_unresolved: bool
    up_missing_source_fields: tuple[str, ...]
    down_missing_source_fields: tuple[str, ...]
    common_family_incidence_pass: bool
    height_door_pass: bool
    color_lift_pass: bool
    up_head_pass: bool
    down_head_pass: bool
    conditional_heads_assembled: bool
    up_conditional_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    down_spectator_baseline_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    down_active_baseline_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    down_active_candidate_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    height_mode_selected_by_dynamics: bool
    active_color_return_selected: bool
    bottom_rank_five_decided: bool
    quark_normal_depths_frozen: bool
    requirements: tuple[QuarkSourceRequirement, ...]
    unresolved_premises: tuple[str, ...]
    source_freeze_ready: bool
    quark_sources_remain_unfrozen: bool
    interpretation: str


def unresolved_quark_sources() -> dict[str, SourceAnchor]:
    """Return unresolved up/down quark anchors from the write-once dictionary."""

    payload = source_dictionary_payload()
    return {
        anchor.label: anchor
        for anchor in payload.unresolved_sources
        if anchor.sector in {"up_quark", "down_quark"}
    }


def missing_source_fields(anchor: SourceAnchor) -> tuple[str, ...]:
    """Return source fields that are still missing for a dictionary anchor."""

    fields: list[str] = []
    if anchor.port_vector is None:
        fields.append("port_vector")
    if not anchor.residual_components:
        fields.append("residual_components")
    if anchor.normal_depth is None:
        fields.append("normal_depth")
    return tuple(fields)


def quark_source_requirements() -> tuple[QuarkSourceRequirement, ...]:
    """Return the requirement ledger for a microscopic quark source freeze."""

    source_payload = source_dictionary_payload()
    incidence = active_plane_incidence_payload()
    height = quark_height_door_payload()
    color = quark_color_lift_payload()
    down = down_quark_indefinite_jacobi_payload()
    sources = unresolved_quark_sources()
    up_source = sources["up_quark_boundary_source"]
    down_source = sources["down_quark_boundary_source"]
    normal_depths_frozen = (
        up_source.normal_depth is not None
        and down_source.normal_depth is not None
    )

    height_selected = not height.repair_mode_not_forced_by_hypercharge
    active_selected = color.gauge_alone_selects_active and color.source_freeze_ready
    rank_five_decided = (
        down.regular_candidate_forced_by_s3_alone
        or not down.regular_candidate_available
    )

    return (
        QuarkSourceRequirement(
            label="write_once_quark_charge_anchors",
            derived=(
                source_payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
                and up_source.status == SourceStatus.UNRESOLVED
                and down_source.status == SourceStatus.UNRESOLVED
            ),
            evidence=(
                "Session 02 records SM quark charge anchors and intentionally "
                "keeps V_u,V_d unfrozen."
            ),
            consequence_if_missing="the quark source dictionary would be circular",
        ),
        QuarkSourceRequirement(
            label="common_residual_family_incidence",
            derived=incidence.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS",
            evidence="Session 11 fixes the residual (u,a,b) incidence basis from selected e1.",
            consequence_if_missing="there is no universal family-port basis to reuse for quarks",
        ),
        QuarkSourceRequirement(
            label=HEIGHT_DYNAMICS_PREMISE,
            derived=height_selected,
            evidence=(
                "Session 08A proves H_tilde/H charge doors but shows the "
                "nilpotent-vs-Hermitian repair assignment is not forced by "
                "hypercharge."
            ),
            consequence_if_missing="up/down repair modes cannot be read from the SM doors alone",
        ),
        QuarkSourceRequirement(
            label=ACTIVE_COLOR_RETURN_PREMISE,
            derived=active_selected,
            evidence=(
                "Session 08B shows active hidden color return is gauge-compatible "
                "but not selected over the spectator shell."
            ),
            consequence_if_missing="the down hidden shell is ambiguous between 3-port and regular S3",
        ),
        QuarkSourceRequirement(
            label=BOTTOM_RANK_FIVE_PREMISE,
            derived=rank_five_decided,
            evidence=(
                "Session 07 makes the rank-five bottom line available in the "
                "regular S3 shell but not forced by S3 alone."
            ),
            consequence_if_missing="the physical down bottom line remains unselected",
        ),
        QuarkSourceRequirement(
            label=QUARK_NORMAL_DEPTH_PREMISE,
            derived=normal_depths_frozen,
            evidence="Session 02 still has normal_depth=None for both quark source anchors.",
            consequence_if_missing="the quark sources are not microscopic BCC boundary placements",
        ),
    )


def unresolved_premises(
    requirements: tuple[QuarkSourceRequirement, ...] | None = None,
) -> tuple[str, ...]:
    """Return labels of unresolved requirements."""

    selected = quark_source_requirements() if requirements is None else requirements
    return tuple(requirement.label for requirement in selected if not requirement.derived)


def quark_source_assembly_payload() -> QuarkSourceAssemblyPayload:
    """Return the Session 15 quark source assembly audit payload."""

    source_payload = source_dictionary_payload()
    sources = unresolved_quark_sources()
    up_source = sources["up_quark_boundary_source"]
    down_source = sources["down_quark_boundary_source"]
    incidence = active_plane_incidence_payload()
    height = quark_height_door_payload()
    color = quark_color_lift_payload()
    up = up_quark_nilpotent_cmv_payload()
    down = down_quark_indefinite_jacobi_payload()
    requirements = quark_source_requirements()
    unresolved = unresolved_premises(requirements)

    source_pass = source_payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
    up_unresolved = up_source.status == SourceStatus.UNRESOLVED
    down_unresolved = down_source.status == SourceStatus.UNRESOLVED
    incidence_pass = incidence.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS"
    height_pass = height.final_verdict == "QUARK_HEIGHT_DOOR_NO_DERIVATION_AUDIT"
    color_pass = color.final_verdict == "QUARK_COLOR_LIFT_NO_SELECTION_AUDIT"
    up_pass = up.final_verdict == "UP_NILPOTENT_HEAD_CONDITIONAL_PASS"
    down_pass = down.final_verdict == "DOWN_HEAD_FORK_LOCALIZED_PASS"

    conditional_heads = (
        up_pass
        and down_pass
        and up.taylor_profile == (sp.Rational(1, 4), 1 / sp.sqrt(2), sp.Integer(1))
        and color.spectator_embedding.down_head is not None
        and color.active_embedding.down_head is not None
        and color.active_embedding.candidate_head is not None
    )

    height_selected = not height.repair_mode_not_forced_by_hypercharge
    active_selected = color.gauge_alone_selects_active and color.source_freeze_ready
    bottom_decided = (
        down.regular_candidate_forced_by_s3_alone
        or not down.regular_candidate_available
    )
    up_missing = missing_source_fields(up_source)
    down_missing = missing_source_fields(down_source)
    depths_frozen = "normal_depth" not in up_missing and "normal_depth" not in down_missing
    freeze_ready = (
        source_pass
        and up_unresolved is False
        and down_unresolved is False
        and incidence_pass
        and height_selected
        and active_selected
        and bottom_decided
        and depths_frozen
    )

    checks_pass = (
        source_pass
        and up_unresolved
        and down_unresolved
        and incidence_pass
        and height_pass
        and color_pass
        and conditional_heads
        and not height_selected
        and not active_selected
        and not bottom_decided
        and not depths_frozen
        and not freeze_ready
        and unresolved == (
            HEIGHT_DYNAMICS_PREMISE,
            ACTIVE_COLOR_RETURN_PREMISE,
            BOTTOM_RANK_FIVE_PREMISE,
            QUARK_NORMAL_DEPTH_PREMISE,
        )
    )

    if checks_pass:
        final_verdict = "QUARK_SOURCE_FREEZE_NOT_DERIVED_AUDIT"
        interpretation = (
            "The quark source dependency graph is now assembled.  The common "
            "residual family incidence basis, SM charge doors, conditional up "
            "nilpotent head, and conditional down real-symmetric heads are all "
            "available.  But a microscopic BCC source freeze still requires "
            "four missing inputs: the height-dynamics rule, active hidden "
            "color-return selection, the down rank-five decision, and actual "
            "normal-depth placements for V_u,V_d.  Therefore the quark heads "
            "remain conditional and must not be promoted to complete "
            "microscopic family-port boundary graphs."
        )
    else:
        final_verdict = "QUARK_SOURCE_ASSEMBLY_AUDIT_KILL"
        interpretation = (
            "The quark source assembly audit failed either a prerequisite "
            "certificate or the strict non-freeze controls."
        )

    spectator_head = color.spectator_embedding.down_head
    active_head = color.active_embedding.down_head
    candidate_head = color.active_embedding.candidate_head
    if spectator_head is None or active_head is None or candidate_head is None:
        raise ValueError("quark color-lift heads are unexpectedly missing")

    return QuarkSourceAssemblyPayload(
        final_verdict=final_verdict,
        source_dictionary_pass=source_pass,
        up_source_unresolved=up_unresolved,
        down_source_unresolved=down_unresolved,
        up_missing_source_fields=up_missing,
        down_missing_source_fields=down_missing,
        common_family_incidence_pass=incidence_pass,
        height_door_pass=height_pass,
        color_lift_pass=color_pass,
        up_head_pass=up_pass,
        down_head_pass=down_pass,
        conditional_heads_assembled=conditional_heads,
        up_conditional_profile=up.taylor_profile,
        down_spectator_baseline_profile=spectator_head.clebsch_vector,
        down_active_baseline_profile=active_head.clebsch_vector,
        down_active_candidate_profile=candidate_head.clebsch_vector,
        height_mode_selected_by_dynamics=height_selected,
        active_color_return_selected=active_selected,
        bottom_rank_five_decided=bottom_decided,
        quark_normal_depths_frozen=depths_frozen,
        requirements=requirements,
        unresolved_premises=unresolved,
        source_freeze_ready=freeze_ready,
        quark_sources_remain_unfrozen=up_unresolved and down_unresolved,
        interpretation=interpretation,
    )
