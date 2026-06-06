"""Session 17 active hidden color-return microcanonical audit.

Session 08B showed that visible color covariance does not choose between a
three-port spectator shell and the active six-channel hidden color-return
shell.  Boundary-response V23 supplies a sharper conditional statement: if the
unresolved primitive quark boundary shell is microcanonical with equal bath
degeneracy per primitive label, tracing out the bath gives the uniform six-label
state.  That selects the active shell inside the primitive-shell model, but it
does not derive the equal-degeneracy / max-entropy prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.microcanonical_reduction import (
    REMAINING_DECLARED_INPUTS_AFTER_REDUCTION,
    compressed_macro_degeneracy_control_phase,
    compressed_macro_degeneracy_control_ratio,
    equal_degeneracy_reduced_density,
    microcanonical_reduction_audit_payload,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import SHELL_DIMENSION
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_shell_audit_payload,
    quark_primitive_channels,
    quark_shell_dimension_breakdown,
)
from clifford_3plus2_d5.universal_bath.quark_color_lift import (
    quark_color_lift_payload,
)
from clifford_3plus2_d5.universal_bath.quark_source_assembly import (
    ACTIVE_COLOR_RETURN_PREMISE,
    quark_source_assembly_payload,
)


MICROCANONICAL_ACTIVE_RETURN_PREMISE = "equal_boundary_degeneracy_or_max_entropy_prior"


@dataclass(frozen=True)
class QuarkActiveColorMicrocanonicalPayload:
    """Session 17 active hidden color-return verdict."""

    final_verdict: str
    quark_source_assembly_pass: bool
    color_lift_pass: bool
    primitive_shell_pass: bool
    microcanonical_reduction_pass: bool
    primitive_channel_names: tuple[str, ...]
    primitive_shell_breakdown: dict[str, int]
    active_shell_breakdown: dict[str, int]
    active_shell_matches_primitive_shell: bool
    spectator_shell_dimension: int
    spectator_is_compressed_control: bool
    equal_degeneracy_density_uniform: bool
    equal_degeneracy_weights: tuple[sp.Expr, ...]
    equal_weights_cover_all_primitive_labels: bool
    compressed_macro_ratio: sp.Expr
    compressed_macro_phase: sp.Expr
    compressed_macro_control_rejected: bool
    active_return_selected_inside_microcanonical_shell: bool
    gauge_alone_selected_active: bool
    remaining_declared_inputs_after_reduction: tuple[str, ...]
    active_color_blocker_reduced_to_microcanonical_prior: bool
    source_freeze_ready: bool
    interpretation: str


def primitive_channel_names() -> tuple[str, ...]:
    """Return primitive quark channel names in shell order."""

    return tuple(channel.name for channel in quark_primitive_channels())


def equal_degeneracy_label_weights() -> tuple[sp.Expr, ...]:
    """Return the label weights of the equal-degeneracy reduced density."""

    density = equal_degeneracy_reduced_density()
    return tuple(sp.simplify(density[index, index]) for index in range(SHELL_DIMENSION))


def equal_weights_cover_all_primitive_labels() -> bool:
    """Return whether every primitive label has equal nonzero weight."""

    weights = equal_degeneracy_label_weights()
    return len(weights) == SHELL_DIMENSION and all(weight == sp.Rational(1, 6) for weight in weights)


def active_shell_matches_primitive_shell() -> bool:
    """Return whether the active hidden-color lift reaches the primitive shell."""

    color = quark_color_lift_payload()
    return color.active_embedding.hidden_shell_breakdown == quark_shell_dimension_breakdown()


def spectator_is_compressed_control() -> bool:
    """Return whether the spectator shell has fewer labels than the primitive shell."""

    color = quark_color_lift_payload()
    return color.spectator_embedding.hidden_shell_breakdown == {"ports": 3}


def quark_active_color_microcanonical_payload() -> QuarkActiveColorMicrocanonicalPayload:
    """Return the Session 17 active color-return microcanonical audit payload."""

    source = quark_source_assembly_payload()
    color = quark_color_lift_payload()
    shell = quark_boundary_shell_audit_payload()
    microcanonical = microcanonical_reduction_audit_payload()
    names = primitive_channel_names()
    weights = equal_degeneracy_label_weights()
    primitive_breakdown = quark_shell_dimension_breakdown()
    active_breakdown = color.active_embedding.hidden_shell_breakdown
    spectator_dimension = color.spectator_embedding.hidden_shell_breakdown["ports"]

    source_pass = source.final_verdict == "QUARK_SOURCE_FREEZE_NOT_DERIVED_AUDIT"
    color_pass = color.final_verdict == "QUARK_COLOR_LIFT_NO_SELECTION_AUDIT"
    shell_pass = shell.final_verdict == "QUARK_BOUNDARY_SHELL_Q1_PASS"
    microcanonical_pass = (
        microcanonical.final_verdict == "EQUAL_DEGENERACY_MICROCANONICAL_REDUCTION_PASS"
    )
    active_matches = active_shell_matches_primitive_shell()
    spectator_compressed = spectator_is_compressed_control()
    equal_density = microcanonical.equal_degeneracy_density_uniform
    equal_covers = equal_weights_cover_all_primitive_labels()
    compressed_rejected = microcanonical.compressed_degeneracy_control_rejected
    active_selected = (
        shell_pass
        and microcanonical_pass
        and active_matches
        and equal_density
        and equal_covers
        and spectator_compressed
        and compressed_rejected
    )
    gauge_selected = color.gauge_alone_selects_active
    reduced_to_prior = (
        active_selected
        and not gauge_selected
        and MICROCANONICAL_ACTIVE_RETURN_PREMISE in microcanonical.remaining_declared_inputs
        and ACTIVE_COLOR_RETURN_PREMISE in source.unresolved_premises
    )
    source_freeze_ready = False

    checks_pass = (
        source_pass
        and color_pass
        and shell_pass
        and microcanonical_pass
        and len(names) == SHELL_DIMENSION
        and primitive_breakdown == {
            "even_direct": 1,
            "bcc_odd": 2,
            "color_odd": 3,
            "odd_total": 5,
            "total": 6,
        }
        and active_breakdown == primitive_breakdown
        and spectator_dimension == 3
        and equal_covers
        and active_selected
        and not gauge_selected
        and reduced_to_prior
        and not source_freeze_ready
    )

    if checks_pass:
        final_verdict = "QUARK_ACTIVE_COLOR_RETURN_MICROCANONICAL_CONDITIONAL_PASS"
        interpretation = (
            "Inside the primitive-shell model, equal-degeneracy microcanonical "
            "reduction gives the uniform six-label density and therefore "
            "selects the active hidden color-return shell over the compressed "
            "three-port spectator shell.  This reduces the active-color "
            "blocker to the physical microcanonical/equal-degeneracy prior; "
            "it is still not a gauge-covariance theorem and does not by "
            "itself freeze V_u,V_d."
        )
    else:
        final_verdict = "QUARK_ACTIVE_COLOR_MICROCANONICAL_AUDIT_KILL"
        interpretation = (
            "The primitive-shell prerequisite, color-lift prerequisite, "
            "microcanonical reduction, active-shell match, or compressed "
            "control failed."
        )

    return QuarkActiveColorMicrocanonicalPayload(
        final_verdict=final_verdict,
        quark_source_assembly_pass=source_pass,
        color_lift_pass=color_pass,
        primitive_shell_pass=shell_pass,
        microcanonical_reduction_pass=microcanonical_pass,
        primitive_channel_names=names,
        primitive_shell_breakdown=primitive_breakdown,
        active_shell_breakdown=active_breakdown,
        active_shell_matches_primitive_shell=active_matches,
        spectator_shell_dimension=spectator_dimension,
        spectator_is_compressed_control=spectator_compressed,
        equal_degeneracy_density_uniform=equal_density,
        equal_degeneracy_weights=weights,
        equal_weights_cover_all_primitive_labels=equal_covers,
        compressed_macro_ratio=compressed_macro_degeneracy_control_ratio(),
        compressed_macro_phase=compressed_macro_degeneracy_control_phase(),
        compressed_macro_control_rejected=compressed_rejected,
        active_return_selected_inside_microcanonical_shell=active_selected,
        gauge_alone_selected_active=gauge_selected,
        remaining_declared_inputs_after_reduction=REMAINING_DECLARED_INPUTS_AFTER_REDUCTION,
        active_color_blocker_reduced_to_microcanonical_prior=reduced_to_prior,
        source_freeze_ready=source_freeze_ready,
        interpretation=interpretation,
    )
