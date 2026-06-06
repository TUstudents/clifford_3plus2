"""Session 11 selected active-plane incidence certificate.

Session 10 supplied the internal family-port graph

    H_fam = H_chain tensor (P_u + P_b) + I tensor Lambda P_a.

This module audits the missing projector selection.  The selected BCC residual
port is the first standard family port ``e1``.  Its collective component is
the ``S3`` singlet ``u``.  Removing that collective trace leaves the selected
traceless radial line ``a``:

    e1 - <u,e1> u  ->  a.

The active neutrino plane is therefore the orthogonal complement of this
radial line.  Equivalently, it is spanned by the collective incidence channel
``u`` and the unique opposite-edge current ``b``.  This proves the Session 10
projector from selected-port incidence data, while still leaving a physical
BB/QCA gate open: the microscopic update must identify the detraced selected
line ``a`` with the q-mismatched / penalized radial boundary mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.framed_sterile import (
    collective_tail_channel,
    opposite_edge_channel,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_s3_invariant,
    is_selected_s2_invariant,
    projector,
    residual_basis_matrix,
    residual_projectors,
    residual_vectors,
    standard_basis,
)
from clifford_3plus2_d5.universal_bath.neutrino_family_port_graph import (
    active_neutrino_projector,
    matrix_equal,
    selected_family_graph_payload,
)

REMAINING_DECLARED_INPUTS_AFTER_ACTIVE_PLANE = (
    "bb_q_mismatch_penalizes_the_detraced_selected_port_radial_line",
    "retarded_outgoing_boundary_condition_closes_only_the_active_incidence_plane",
)


@dataclass(frozen=True)
class ActivePlaneIncidencePayload:
    """Session 11 active-plane incidence verdict."""

    final_verdict: str
    selected_port: sp.Matrix
    selected_port_components_aub: tuple[sp.Expr, sp.Expr, sp.Expr]
    detraced_radial_line: sp.Matrix
    opposite_edge_line: sp.Matrix
    active_projector: sp.Matrix
    radial_projector: sp.Matrix
    active_projector_rank: int
    radial_projector_rank: int
    detraced_line_matches_a: bool
    opposite_edge_line_matches_b: bool
    active_projector_matches_session_10: bool
    active_radial_resolution_identity: bool
    active_projector_selected_s2_invariant: bool
    active_projector_not_full_s3_invariant: bool
    selected_s2_symmetry_alone_not_sufficient: bool
    raw_selected_port_line_control_rejected: bool
    session_10_family_moment_gate_passes: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def _normalize(vector: sp.Matrix) -> sp.Matrix:
    """Return an exact normalized column vector."""

    norm_sq = sp.simplify((vector.T * vector)[0])
    if norm_sq == 0:
        raise ValueError("cannot normalize zero vector")
    return (vector / sp.sqrt(norm_sq)).applyfunc(sp.simplify)


def selected_boundary_port(index: int = 0) -> sp.Matrix:
    """Return the selected residual standard port ``e_index``."""

    ports = standard_basis()
    if index < 0 or index >= len(ports):
        raise ValueError("selected port index out of range")
    return ports[index]


def selected_port_components_aub(index: int = 0) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return selected-port components in the ``(a,u,b)`` residual basis."""

    basis = residual_basis_matrix(("a", "u", "b"))
    components = (basis.T * selected_boundary_port(index)).applyfunc(sp.simplify)
    return (components[0, 0], components[1, 0], components[2, 0])


def detraced_selected_radial_line(index: int = 0) -> sp.Matrix:
    """Return the selected traceless radial line from ``e_i``.

    The trace is the collective ``u`` component.  Removing it turns the selected
    port into the selected doublet/radial line.
    """

    port = selected_boundary_port(index)
    collective = collective_tail_channel()
    trace_component = sp.simplify((collective.T * port)[0])
    return _normalize(port - trace_component * collective)


def opposite_edge_line_from_incidence(index: int = 0) -> sp.Matrix:
    """Return the unique current orthogonal to collective and radial lines."""

    collective = collective_tail_channel()
    radial = detraced_selected_radial_line(index)
    current = collective.cross(radial)
    current = _normalize(current)

    if index == 0:
        reference = opposite_edge_channel()
        overlap = sp.simplify((reference.T * current)[0])
        if sp.simplify(overlap + 1) == 0:
            current = -current
    return current.applyfunc(sp.simplify)


def radial_projector_from_incidence(index: int = 0) -> sp.Matrix:
    """Return the radial projector selected by detraced incidence."""

    return projector(detraced_selected_radial_line(index))


def active_projector_from_incidence(index: int = 0) -> sp.Matrix:
    """Return the active projector orthogonal to the selected radial line."""

    return sp.simplify(sp.eye(3) - radial_projector_from_incidence(index))


def active_projector_from_channels(index: int = 0) -> sp.Matrix:
    """Return the active projector from collective plus opposite-edge channels."""

    collective = collective_tail_channel()
    edge = opposite_edge_line_from_incidence(index)
    return sp.simplify(projector(collective) + projector(edge))


def raw_selected_port_active_control(index: int = 0) -> sp.Matrix:
    """Return the wrong active plane obtained by treating ``e_i`` as radial."""

    return sp.simplify(sp.eye(3) - projector(selected_boundary_port(index)))


def selected_s2_symmetry_only_counterexample() -> sp.Matrix:
    """Return an ``S2``-invariant operator that still mixes radial/active."""

    return projector(selected_boundary_port(0))


def _has_radial_active_mixing(operator: sp.Matrix, radial: sp.Matrix, active: sp.Matrix) -> bool:
    """Return true when ``operator`` has nonzero radial-active blocks."""

    left = (radial * operator * active).applyfunc(sp.simplify)
    right = (active * operator * radial).applyfunc(sp.simplify)
    return any(entry != 0 for entry in (*left, *right))


def selected_s2_symmetry_alone_not_sufficient() -> bool:
    """Return true when selected ``S2`` leaves an active-plane ambiguity."""

    counterexample = selected_s2_symmetry_only_counterexample()
    radial = residual_projectors()["a"]
    active = active_neutrino_projector()
    return (
        is_selected_s2_invariant(counterexample)
        and _has_radial_active_mixing(counterexample, radial, active)
    )


def active_plane_incidence_payload() -> ActivePlaneIncidencePayload:
    """Return the Session 11 active-plane incidence certificate."""

    vectors = residual_vectors()
    projectors = residual_projectors()
    components = selected_port_components_aub()
    radial_line = detraced_selected_radial_line()
    edge_line = opposite_edge_line_from_incidence()
    radial_projector = radial_projector_from_incidence()
    active_projector = active_projector_from_incidence()
    channel_projector = active_projector_from_channels()
    session_10_active = active_neutrino_projector()

    detraced_matches_a = matrix_equal(projector(radial_line), projectors["a"])
    edge_matches_b = matrix_equal(projector(edge_line), projectors["b"])
    active_matches = (
        matrix_equal(active_projector, session_10_active)
        and matrix_equal(channel_projector, session_10_active)
    )
    resolution = matrix_equal(active_projector + radial_projector, sp.eye(3))
    raw_control_rejected = not matrix_equal(raw_selected_port_active_control(), session_10_active)
    s2_not_enough = selected_s2_symmetry_alone_not_sufficient()
    session_10 = selected_family_graph_payload()
    session_10_pass = session_10.final_verdict == "NEUTRINO_FAMILY_PORT_GRAPH_INTERNAL_PASS"

    checks_pass = (
        sp.simplify(components[0] - sp.sqrt(sp.Rational(2, 3))) == 0
        and sp.simplify(components[1] - 1 / sp.sqrt(3)) == 0
        and components[2] == 0
        and detraced_matches_a
        and edge_matches_b
        and active_matches
        and active_projector.rank() == 2
        and radial_projector.rank() == 1
        and resolution
        and is_selected_s2_invariant(active_projector)
        and not is_s3_invariant(active_projector)
        and s2_not_enough
        and raw_control_rejected
        and session_10_pass
        and sp.simplify((radial_line.T * vectors["u"])[0]) == 0
        and sp.simplify((radial_line.T * vectors["b"])[0]) == 0
    )

    if checks_pass:
        final_verdict = "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS"
        interpretation = (
            "The selected residual port e1 decomposes as sqrt(2/3) a + "
            "1/sqrt(3) u.  Removing the collective trace fixes the selected "
            "traceless radial line a, and its orthogonal complement is exactly "
            "the active u/b incidence plane used in Session 10.  The unique "
            "current perpendicular to the collective and radial lines is b.  "
            "Selected-S2 symmetry alone is not sufficient: the S2-invariant "
            "projector P_e1 still mixes radial and active components.  Thus "
            "the active plane is derived from selected-port incidence plus "
            "detracing, while the remaining physical BB/QCA input is that the "
            "q-mismatched boundary dynamics penalizes this radial line and "
            "the retarded outgoing tail closes only on its orthogonal active "
            "plane."
        )
    else:
        final_verdict = "NEUTRINO_ACTIVE_PLANE_INCIDENCE_KILL"
        interpretation = (
            "The selected-port decomposition, detraced radial line, opposite "
            "edge current, active projector, controls, or Session 10 moment "
            "gate failed."
        )

    return ActivePlaneIncidencePayload(
        final_verdict=final_verdict,
        selected_port=selected_boundary_port(),
        selected_port_components_aub=components,
        detraced_radial_line=radial_line,
        opposite_edge_line=edge_line,
        active_projector=active_projector,
        radial_projector=radial_projector,
        active_projector_rank=active_projector.rank(),
        radial_projector_rank=radial_projector.rank(),
        detraced_line_matches_a=detraced_matches_a,
        opposite_edge_line_matches_b=edge_matches_b,
        active_projector_matches_session_10=active_matches,
        active_radial_resolution_identity=resolution,
        active_projector_selected_s2_invariant=is_selected_s2_invariant(active_projector),
        active_projector_not_full_s3_invariant=not is_s3_invariant(active_projector),
        selected_s2_symmetry_alone_not_sufficient=s2_not_enough,
        raw_selected_port_line_control_rejected=raw_control_rejected,
        session_10_family_moment_gate_passes=session_10_pass,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_ACTIVE_PLANE,
        interpretation=interpretation,
    )
