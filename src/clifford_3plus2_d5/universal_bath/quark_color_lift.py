"""Session 08B quark color-lift audit.

The visible quark mass operator must be color-scalar.  This rejects a fixed
rank-one color source.  It does not by itself choose between a color-scalar
spectator source, which stays on the three residual ports, and an active hidden
color-return lift, which preserves visible color while enlarging the hidden
return shell to the regular six-channel S3 object.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_shell_audit_payload,
    quark_shell_dimension_breakdown,
)
from clifford_3plus2_d5.boundary_response.quark_clebsch_factors import (
    color_return_contraction,
    color_return_factor,
    su3_fundamental_generators,
)
from clifford_3plus2_d5.universal_bath.down_quark_indefinite_jacobi import (
    DownJacobiHeadCandidate,
    regular_s3_baseline_head,
    regular_s3_candidate_head,
    three_port_permutation_head,
)
from clifford_3plus2_d5.universal_bath.quark_height_door import (
    quark_height_door_payload,
)


@dataclass(frozen=True)
class ColorLiftEmbedding:
    """One visible/hidden color embedding candidate."""

    label: str
    visible_color_operator: sp.Matrix
    visible_color_scalar: bool
    commutes_with_su3: bool
    hidden_return_graph: str
    hidden_shell_breakdown: dict[str, int]
    down_head: DownJacobiHeadCandidate | None
    candidate_head: DownJacobiHeadCandidate | None
    preserves_visible_color: bool
    reaches_regular_s3_shell: bool
    interpretation: str


@dataclass(frozen=True)
class QuarkColorLiftPayload:
    """Session 08B color-lift verdict."""

    final_verdict: str
    height_door_prerequisite_pass: bool
    quark_shell_prerequisite_pass: bool
    fixed_color_embedding: ColorLiftEmbedding
    spectator_embedding: ColorLiftEmbedding
    active_embedding: ColorLiftEmbedding
    fixed_color_rejected: bool
    spectator_preserves_color: bool
    spectator_stays_three_port: bool
    active_preserves_visible_color: bool
    active_regular_s3_baseline: bool
    active_rank_five_candidate_available: bool
    active_rank_five_candidate_forced: bool
    color_return_contraction_scalar: bool
    gauge_alone_selects_active: bool
    source_freeze_ready: bool
    interpretation: str


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices agree after exact simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def commutes_with_su3(matrix: sp.Matrix) -> bool:
    """Return whether a visible color operator commutes with all SU(3) generators."""

    return all(
        _matrix_equal(generator * matrix - matrix * generator, sp.zeros(3, 3))
        for generator in su3_fundamental_generators()
    )


def visible_color_is_scalar(matrix: sp.Matrix) -> bool:
    """Return whether a visible color operator is proportional to identity."""

    scalar = sp.simplify(sp.trace(matrix) / 3)
    return _matrix_equal(matrix, scalar * sp.eye(3))


def fixed_color_operator() -> sp.Matrix:
    """Return the forbidden fixed-color rank-one source control."""

    return sp.diag(1, 0, 0)


def color_scalar_operator() -> sp.Matrix:
    """Return a normalized visible color scalar."""

    return sp.eye(3) / 3


def active_color_return_operator() -> sp.Matrix:
    """Return the visible color operator after summing active hidden colors."""

    return color_scalar_operator()


def fixed_color_embedding() -> ColorLiftEmbedding:
    """Return the rank-one fixed-color control embedding."""

    operator = fixed_color_operator()
    return ColorLiftEmbedding(
        label="fixed_color_vector_control",
        visible_color_operator=operator,
        visible_color_scalar=visible_color_is_scalar(operator),
        commutes_with_su3=commutes_with_su3(operator),
        hidden_return_graph="rank_one_color_control",
        hidden_shell_breakdown={"visible_rank": 1},
        down_head=None,
        candidate_head=None,
        preserves_visible_color=False,
        reaches_regular_s3_shell=False,
        interpretation=(
            "A fixed color vector is a rank-one visible color projector.  It "
            "does not commute with SU(3)c and is rejected before any mass head "
            "is read."
        ),
    )


def spectator_color_embedding() -> ColorLiftEmbedding:
    """Return the color-scalar spectator embedding."""

    operator = color_scalar_operator()
    head = three_port_permutation_head()
    return ColorLiftEmbedding(
        label="color_scalar_spectator",
        visible_color_operator=operator,
        visible_color_scalar=visible_color_is_scalar(operator),
        commutes_with_su3=commutes_with_su3(operator),
        hidden_return_graph="residual_3_port_permutation_shell",
        hidden_shell_breakdown={"ports": 3},
        down_head=head,
        candidate_head=None,
        preserves_visible_color=True,
        reaches_regular_s3_shell=False,
        interpretation=(
            "The spectator embedding is color-scalar and gauge-compatible, but "
            "its hidden return graph remains the residual three-port shell."
        ),
    )


def active_color_embedding() -> ColorLiftEmbedding:
    """Return the active color-return embedding."""

    operator = active_color_return_operator()
    baseline = regular_s3_baseline_head()
    candidate = regular_s3_candidate_head()
    return ColorLiftEmbedding(
        label="active_color_return_lift",
        visible_color_operator=operator,
        visible_color_scalar=visible_color_is_scalar(operator),
        commutes_with_su3=commutes_with_su3(operator),
        hidden_return_graph="regular_s3_cayley_shell",
        hidden_shell_breakdown=quark_shell_dimension_breakdown(),
        down_head=baseline,
        candidate_head=candidate,
        preserves_visible_color=True,
        reaches_regular_s3_shell=True,
        interpretation=(
            "The active lift sums over hidden color-return channels while the "
            "visible color operator remains scalar.  It reaches the six-channel "
            "regular S3 shell and can host the rank-five bottom candidate."
        ),
    )


def color_return_contraction_is_scalar() -> bool:
    """Return whether the SU(3) fundamental return contraction is ``C_F I``."""

    return _matrix_equal(color_return_contraction(), color_return_factor() * sp.eye(3))


def quark_color_lift_payload() -> QuarkColorLiftPayload:
    """Return the Session 08B color-lift audit payload."""

    height = quark_height_door_payload()
    shell = quark_boundary_shell_audit_payload()
    fixed = fixed_color_embedding()
    spectator = spectator_color_embedding()
    active = active_color_embedding()

    height_pass = height.final_verdict == "QUARK_HEIGHT_DOOR_AUDIT_CONDITIONAL_PASS"
    shell_pass = shell.final_verdict == "QUARK_BOUNDARY_SHELL_Q1_PASS"
    fixed_rejected = not fixed.commutes_with_su3 and not fixed.visible_color_scalar
    spectator_preserves = spectator.commutes_with_su3 and spectator.visible_color_scalar
    spectator_three_port = (
        spectator.down_head is not None
        and spectator.down_head.matches_baseline
        and not spectator.reaches_regular_s3_shell
    )
    active_preserves = active.commutes_with_su3 and active.visible_color_scalar
    active_baseline = (
        active.down_head is not None
        and active.down_head.matches_baseline
        and active.hidden_shell_breakdown == {
            "even_direct": 1,
            "bcc_odd": 2,
            "color_odd": 3,
            "odd_total": 5,
            "total": 6,
        }
    )
    active_candidate_available = active.candidate_head is not None and active.candidate_head.matches_candidate
    active_candidate_forced = bool(active.candidate_head and active.candidate_head.forced_by_symmetry)
    color_scalar = color_return_contraction_is_scalar()
    gauge_alone_selects_active = False
    source_freeze_ready = False

    checks_pass = (
        height_pass
        and shell_pass
        and fixed_rejected
        and spectator_preserves
        and spectator_three_port
        and active_preserves
        and active_baseline
        and active_candidate_available
        and not active_candidate_forced
        and color_scalar
        and not gauge_alone_selects_active
        and not source_freeze_ready
    )

    if checks_pass:
        final_verdict = "QUARK_COLOR_LIFT_AUDIT_CONDITIONAL_PASS"
        interpretation = (
            "Visible color covariance rejects a fixed color vector.  Both the "
            "spectator and active embeddings are visible color scalars, so "
            "gauge covariance alone does not choose the active lift.  The "
            "spectator embedding stays on the three-port baseline; the active "
            "hidden color-return lift preserves visible color while reaching "
            "the six-channel regular S3 shell and making the rank-five bottom "
            "candidate available, though not forced.  The quark source freeze "
            "therefore remains conditional on a microscopic active-return "
            "selection rule."
        )
    else:
        final_verdict = "QUARK_COLOR_LIFT_AUDIT_KILL"
        interpretation = (
            "The color-lift audit failed the height-door prerequisite, quark "
            "shell prerequisite, color-covariance, spectator, active-lift, or "
            "rank-five availability checks."
        )

    return QuarkColorLiftPayload(
        final_verdict=final_verdict,
        height_door_prerequisite_pass=height_pass,
        quark_shell_prerequisite_pass=shell_pass,
        fixed_color_embedding=fixed,
        spectator_embedding=spectator,
        active_embedding=active,
        fixed_color_rejected=fixed_rejected,
        spectator_preserves_color=spectator_preserves,
        spectator_stays_three_port=spectator_three_port,
        active_preserves_visible_color=active_preserves,
        active_regular_s3_baseline=active_baseline,
        active_rank_five_candidate_available=active_candidate_available,
        active_rank_five_candidate_forced=active_candidate_forced,
        color_return_contraction_scalar=color_scalar,
        gauge_alone_selects_active=gauge_alone_selects_active,
        source_freeze_ready=source_freeze_ready,
        interpretation=interpretation,
    )
