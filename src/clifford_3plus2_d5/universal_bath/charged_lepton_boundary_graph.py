"""Session 14 charged-lepton minimal family-port boundary graph.

The charged-lepton branch is not a positive one-sided self-energy like the
neutrino branch.  A charged-lepton Yukawa is a chiral two-sided Schur kernel:

    B_e(z) = V_R^T (z-H_Q)^-1 V_L.

This module implements the minimal colorless active family-port graph described
in the synthesis.  It has two coherent trace-return states and two residual
plane states.  Its pole residue is

    sqrt(2) P_u + R_theta P_perp,

with ``theta = -2*pi/3 - 2/9`` supplied by the existing holonomy/torsion
premises.  Acting on the selected port ``e1`` gives the Koide
trace/traceless equipartition exactly.

The session is deliberately scoped: it proves the exact minimal graph once the
two trace paths and theta are supplied.  It does not derive the microscopic
origin of the trace paths or the active torsion dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.leptonic_boundary_holonomy import (
    leptonic_boundary_holonomy_audit_payload,
)
from clifford_3plus2_d5.boundary_response.residual_basis import residual_basis_matrix
from clifford_3plus2_d5.universal_bath.charged_lepton_cmv import (
    charged_lepton_cmv_head_payload,
)
from clifford_3plus2_d5.universal_bath.charged_lepton_torsion import (
    charged_lepton_torsion_payload,
    charged_lepton_torsion_weight,
)
from clifford_3plus2_d5.universal_bath.source_dictionary import (
    source_dictionary_payload,
)

REMAINING_DECLARED_INPUTS_AFTER_CHARGED_LEPTON_GRAPH = (
    "microscopic_colorless_bcc_higgs_boundary_derives_two_coherent_trace_paths",
    "active_cmv_torsion_angle_2_over_9_is_generated_by_boundary_dynamics",
    "charged_lepton_overall_scale_rho_or_mu_e",
)


@dataclass(frozen=True)
class ChargedLeptonBoundaryGraphPayload:
    """Session 14 charged-lepton minimal boundary graph verdict."""

    final_verdict: str
    source_dictionary_pass: bool
    cmv_head_pass: bool
    torsion_gate_pass: bool
    holonomy_gate_pass: bool
    theta_base: sp.Expr
    torsion_angle: sp.Expr
    theta: sp.Expr
    left_coupling: sp.Matrix
    right_coupling: sp.Matrix
    residue: sp.Matrix
    target_residue: sp.Matrix
    residue_matches_target: bool
    selected_port_uab: sp.Matrix
    residue_on_selected_port: sp.Matrix
    normalized_square_root_vector_uab: sp.Matrix
    normalized_square_root_vector_standard: sp.Matrix
    trace_weight: sp.Expr
    traceless_weight: sp.Expr
    trace_traceless_equipartition: bool
    koide_parameter: sp.Expr
    koide_parameter_is_two_thirds: bool
    one_trace_path_control_rejected: bool
    one_sided_hermitian_control_rejected: bool
    torsion_angle_not_derived_by_graph: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def selected_port_uab() -> sp.Matrix:
    """Return ``e1`` in the residual coordinate order ``(u,a,b)``."""

    return sp.Matrix([1 / sp.sqrt(3), sp.sqrt(sp.Rational(2, 3)), 0])


def charged_lepton_base_angle() -> sp.Expr:
    """Return the residual-plane base angle ``-2*pi/3``."""

    holonomy = leptonic_boundary_holonomy_audit_payload()
    parent_a3_angle = sp.pi / 4
    return sp.simplify(sp.pi * holonomy.principal_angle - parent_a3_angle)


def charged_lepton_torsion_angle() -> sp.Expr:
    """Return the active torsion angle supplied by the Session 05 gate."""

    return charged_lepton_torsion_weight()


def charged_lepton_boundary_angle() -> sp.Expr:
    """Return ``theta = -2*pi/3 - 2/9``."""

    return sp.simplify(charged_lepton_base_angle() - charged_lepton_torsion_angle())


def rotation_plane(theta: sp.Expr) -> sp.Matrix:
    """Return the real rotation on the ``(a,b)`` plane."""

    return sp.Matrix(
        [
            [sp.cos(theta), -sp.sin(theta)],
            [sp.sin(theta), sp.cos(theta)],
        ]
    )


def target_residue(theta: sp.Expr | None = None, rho: sp.Expr = sp.Integer(1)) -> sp.Matrix:
    """Return ``rho (sqrt(2) P_u + R_theta P_perp)`` in ``(u,a,b)`` order."""

    angle = charged_lepton_boundary_angle() if theta is None else theta
    plane = rotation_plane(angle)
    return sp.Matrix(
        [
            [rho * sp.sqrt(2), 0, 0],
            [0, rho * plane[0, 0], rho * plane[0, 1]],
            [0, rho * plane[1, 0], rho * plane[1, 1]],
        ]
    ).applyfunc(sp.simplify)


def left_coupling_matrix() -> sp.Matrix:
    """Return ``V_L`` for ``Q=(t_+,t_-,p_a,p_b)`` and ``P=(u,a,b)``."""

    return sp.Matrix(
        [
            [1 / sp.sqrt(2), 0, 0],
            [1 / sp.sqrt(2), 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )


def right_coupling_matrix(theta: sp.Expr | None = None, rho: sp.Expr = sp.Integer(1)) -> sp.Matrix:
    """Return ``V_R`` for the minimal charged-lepton boundary pole graph."""

    angle = charged_lepton_boundary_angle() if theta is None else theta
    return sp.Matrix(
        [
            [rho, 0, 0],
            [rho, 0, 0],
            [0, rho * sp.cos(angle), rho * sp.sin(angle)],
            [0, -rho * sp.sin(angle), rho * sp.cos(angle)],
        ]
    ).applyfunc(sp.simplify)


def pole_residue(theta: sp.Expr | None = None, rho: sp.Expr = sp.Integer(1)) -> sp.Matrix:
    """Return the two-sided pole residue ``V_R^T V_L``."""

    return (right_coupling_matrix(theta, rho).T * left_coupling_matrix()).applyfunc(
        sp.simplify
    )


def residue_on_selected_port(theta: sp.Expr | None = None) -> sp.Matrix:
    """Return the unnormalized residue amplitude on ``e1``."""

    return (pole_residue(theta) * selected_port_uab()).applyfunc(sp.simplify)


def normalized_square_root_vector_uab(theta: sp.Expr | None = None) -> sp.Matrix:
    """Return the normalized charged-lepton square-root vector in ``(u,a,b)``."""

    vector = residue_on_selected_port(theta)
    norm_sq = sp.trigsimp(sp.simplify((vector.T * vector)[0]))
    return (vector / sp.sqrt(norm_sq)).applyfunc(lambda entry: sp.trigsimp(sp.simplify(entry)))


def normalized_square_root_vector_standard(theta: sp.Expr | None = None) -> sp.Matrix:
    """Return the normalized square-root vector in standard family coordinates."""

    basis = residual_basis_matrix(("u", "a", "b"))
    return (basis * normalized_square_root_vector_uab(theta)).applyfunc(
        lambda entry: sp.trigsimp(sp.simplify(entry))
    )


def trace_weight(theta: sp.Expr | None = None) -> sp.Expr:
    """Return the squared collective trace weight."""

    vector = normalized_square_root_vector_uab(theta)
    return sp.simplify(vector[0] ** 2)


def traceless_weight(theta: sp.Expr | None = None) -> sp.Expr:
    """Return the total squared residual-plane weight."""

    vector = normalized_square_root_vector_uab(theta)
    return sp.trigsimp(sp.simplify(vector[1] ** 2 + vector[2] ** 2))


def koide_parameter(theta: sp.Expr | None = None) -> sp.Expr:
    """Return Koide's ``sum(w_i^2)/(sum(w_i))^2`` for the graph vector."""

    vector = normalized_square_root_vector_standard(theta)
    numerator = sp.trigsimp(sp.simplify((vector.T * vector)[0]))
    denominator = sp.trigsimp(sp.simplify(sum(vector, sp.Integer(0)) ** 2))
    return sp.trigsimp(sp.simplify(numerator / denominator))


def one_trace_path_residue(theta: sp.Expr | None = None) -> sp.Matrix:
    """Return the control residue with only one trace path."""

    angle = charged_lepton_boundary_angle() if theta is None else theta
    plane = rotation_plane(angle)
    return sp.Matrix(
        [
            [1, 0, 0],
            [0, plane[0, 0], plane[0, 1]],
            [0, plane[1, 0], plane[1, 1]],
        ]
    ).applyfunc(sp.simplify)


def one_trace_path_control_rejected() -> bool:
    """Return whether one trace path fails trace/traceless equipartition."""

    vector = one_trace_path_residue() * selected_port_uab()
    trace = sp.simplify(vector[0] ** 2)
    plane = sp.trigsimp(sp.simplify(vector[1] ** 2 + vector[2] ** 2))
    return sp.simplify(trace - plane) != 0


def one_sided_hermitian_control_rejected(theta: sp.Expr | None = None) -> bool:
    """Return whether a positive one-sided residue can be the target rotation."""

    residue = target_residue(theta)
    antisymmetric = (residue - residue.T).applyfunc(sp.simplify)
    return any(abs(float(sp.N(entry))) > 1e-12 for entry in antisymmetric)


def charged_lepton_boundary_graph_payload() -> ChargedLeptonBoundaryGraphPayload:
    """Return the Session 14 charged-lepton minimal graph certificate."""

    source_payload = source_dictionary_payload()
    cmv_payload = charged_lepton_cmv_head_payload()
    torsion_payload = charged_lepton_torsion_payload()
    holonomy_payload = leptonic_boundary_holonomy_audit_payload()

    source_pass = source_payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
    cmv_pass = cmv_payload.final_verdict == "CHARGED_LEPTON_CMV_HEAD_PACKAGING_PASS"
    torsion_pass = torsion_payload.final_verdict == "CHARGED_LEPTON_2_OVER_9_OCCUPATION_PASS"
    holonomy_pass = holonomy_payload.final_verdict == "LEPTONIC_PHASE_WORD_DERIVED_PASS"

    theta_base = charged_lepton_base_angle()
    torsion = charged_lepton_torsion_angle()
    theta = charged_lepton_boundary_angle()
    left = left_coupling_matrix()
    right = right_coupling_matrix(theta)
    residue = pole_residue(theta)
    target = target_residue(theta)
    residue_matches = residue == target
    selected = selected_port_uab()
    action = residue_on_selected_port(theta)
    normalized_uab = normalized_square_root_vector_uab(theta)
    normalized_standard = normalized_square_root_vector_standard(theta)
    trace = trace_weight(theta)
    plane = traceless_weight(theta)
    equipartition = sp.simplify(trace - sp.Rational(1, 2)) == 0 and sp.simplify(
        plane - sp.Rational(1, 2)
    ) == 0
    koide = koide_parameter(theta)
    koide_ok = sp.simplify(koide - sp.Rational(2, 3)) == 0
    one_trace_rejected = one_trace_path_control_rejected()
    hermitian_rejected = one_sided_hermitian_control_rejected(theta)

    checks_pass = (
        source_pass
        and cmv_pass
        and torsion_pass
        and holonomy_pass
        and sp.simplify(theta_base + 2 * sp.pi / 3) == 0
        and sp.simplify(torsion - sp.Rational(2, 9)) == 0
        and sp.simplify(theta + 2 * sp.pi / 3 + sp.Rational(2, 9)) == 0
        and residue_matches
        and sp.simplify(action[0] - sp.sqrt(sp.Rational(2, 3))) == 0
        and sp.trigsimp(sp.simplify(action[1] ** 2 + action[2] ** 2 - sp.Rational(2, 3))) == 0
        and equipartition
        and koide_ok
        and one_trace_rejected
        and hermitian_rejected
    )

    if checks_pass:
        final_verdict = "CHARGED_LEPTON_MINIMAL_BOUNDARY_GRAPH_PASS"
        interpretation = (
            "The minimal colorless active two-sided boundary graph realizes "
            "the charged-lepton residue sqrt(2) P_u + R_theta P_perp exactly. "
            "The two coherent trace paths produce the sqrt(2) trace "
            "enhancement, and acting on the selected port e1 gives exact "
            "trace/traceless equipartition and Koide K=2/3.  A one-trace-path "
            "control fails equipartition, and a one-sided Hermitian self-energy "
            "control cannot produce the nontrivial plane rotation.  This is a "
            "minimal graph realization: the microscopic BCC/Higgs origin of "
            "the two trace paths and the active 2/9 torsion dynamics remain "
            "declared inputs."
        )
    else:
        final_verdict = "CHARGED_LEPTON_MINIMAL_BOUNDARY_GRAPH_KILL"
        interpretation = (
            "The charged-lepton minimal graph failed source, CMV, torsion, "
            "holonomy, residue, equipartition, Koide, or control checks."
        )

    return ChargedLeptonBoundaryGraphPayload(
        final_verdict=final_verdict,
        source_dictionary_pass=source_pass,
        cmv_head_pass=cmv_pass,
        torsion_gate_pass=torsion_pass,
        holonomy_gate_pass=holonomy_pass,
        theta_base=theta_base,
        torsion_angle=torsion,
        theta=theta,
        left_coupling=left,
        right_coupling=right,
        residue=residue,
        target_residue=target,
        residue_matches_target=residue_matches,
        selected_port_uab=selected,
        residue_on_selected_port=action,
        normalized_square_root_vector_uab=normalized_uab,
        normalized_square_root_vector_standard=normalized_standard,
        trace_weight=trace,
        traceless_weight=plane,
        trace_traceless_equipartition=equipartition,
        koide_parameter=koide,
        koide_parameter_is_two_thirds=koide_ok,
        one_trace_path_control_rejected=one_trace_rejected,
        one_sided_hermitian_control_rejected=hermitian_rejected,
        torsion_angle_not_derived_by_graph=True,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_CHARGED_LEPTON_GRAPH,
        interpretation=interpretation,
    )
