"""Session 19 charged-lepton trace-path and torsion-origin audit.

Session 14 built the minimal charged-lepton two-sided boundary graph once two
coherent trace paths and the active angle

    theta = -2*pi/3 - 2/9

were supplied.  This session audits whether those two supplied ingredients are
already derived by the current universal-bath machinery.

The audit separates two facts:

* trace/traceless equipartition forces exactly two coherent trace paths inside
  the minimal graph ansatz;
* the ``2/9`` number is an occupation weight of the frozen source, but the
  present graph still inserts it as a rotation angle.

Thus the session reduces the charged-lepton microscopic gap to two sharply
named inputs rather than promoting the minimal graph to a derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.leptonic_boundary_holonomy import (
    leptonic_boundary_holonomy_audit_payload,
)
from clifford_3plus2_d5.universal_bath.charged_lepton_boundary_graph import (
    charged_lepton_base_angle,
    charged_lepton_boundary_angle,
    charged_lepton_boundary_graph_payload,
    charged_lepton_torsion_angle,
    koide_parameter,
    left_coupling_matrix,
    right_coupling_matrix,
    trace_weight,
    traceless_weight,
)
from clifford_3plus2_d5.universal_bath.charged_lepton_torsion import (
    charged_lepton_torsion_payload,
    charged_lepton_torsion_weight,
    source_occupation_weights,
)

TRACE_PATH_ORIGIN_PREMISE = (
    "microscopic_colorless_bcc_higgs_boundary_derives_two_coherent_trace_paths"
)
TORSION_DYNAMICS_PREMISE = (
    "active_cmv_torsion_angle_2_over_9_is_generated_by_boundary_dynamics"
)


@dataclass(frozen=True)
class ChargedLeptonTraceTorsionOriginPayload:
    """Session 19 charged-lepton trace/torsion origin verdict."""

    final_verdict: str
    boundary_graph_pass: bool
    torsion_gate_pass: bool
    holonomy_gate_pass: bool
    source_occupation_weights: dict[str, sp.Expr]
    trace_weight_formula: sp.Expr
    traceless_weight_formula: sp.Expr
    required_trace_path_count: int
    supplied_trace_path_count: int
    supplied_trace_paths_match_required_count: bool
    two_trace_paths_force_equipartition: bool
    one_trace_path_control_rejected: bool
    three_trace_path_control_rejected: bool
    trace_path_origin_derived_from_bcc_higgs: bool
    torsion_occupation_weight: sp.Expr
    torsion_used_as_rotation_angle: sp.Expr
    torsion_weight_inserted_into_theta: bool
    occupation_to_angle_map_derived: bool
    cmv_phase_word_pass: bool
    cmv_phase_word_is_not_torsion_angle: bool
    minimal_graph_accepts_supplied_theta: bool
    koide_still_exact_with_supplied_theta: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def trace_weight_for_path_count(path_count: int) -> sp.Expr:
    """Return the collective trace weight for ``path_count`` coherent trace paths.

    The selected source has ``|u|^2=1/3`` and residual-plane weight ``2/3``.
    With ``n`` coherent trace returns the trace amplitude is enhanced by
    ``sqrt(n)`` while the residual plane is not.  After normalization the trace
    weight is ``n/(n+2)``.
    """

    if path_count <= 0:
        raise ValueError("path_count must be positive")
    n = sp.Integer(path_count)
    return sp.simplify(n / (n + 2))


def traceless_weight_for_path_count(path_count: int) -> sp.Expr:
    """Return the residual-plane weight for ``path_count`` coherent trace paths."""

    if path_count <= 0:
        raise ValueError("path_count must be positive")
    n = sp.Integer(path_count)
    return sp.simplify(sp.Integer(2) / (n + 2))


def required_trace_path_count_for_equipartition() -> int:
    """Return the positive integer trace-path count that gives equipartition."""

    for count in range(1, 8):
        if sp.simplify(trace_weight_for_path_count(count) - sp.Rational(1, 2)) == 0:
            return count
    raise ValueError("no positive trace-path count gave equipartition in the checked range")


def supplied_trace_path_count() -> int:
    """Count trace-only pole rows in the Session 14 minimal graph."""

    left = left_coupling_matrix()
    right = right_coupling_matrix()
    count = 0
    for row in range(left.rows):
        trace_coupled = left[row, 0] != 0 and right[row, 0] != 0
        plane_decoupled = all(
            sp.simplify(entry) == 0
            for entry in (left[row, 1], left[row, 2], right[row, 1], right[row, 2])
        )
        if trace_coupled and plane_decoupled:
            count += 1
    return count


def two_trace_paths_force_equipartition() -> bool:
    """Return whether exactly two paths give trace/traceless equipartition."""

    return (
        required_trace_path_count_for_equipartition() == 2
        and sp.simplify(trace_weight_for_path_count(2) - sp.Rational(1, 2)) == 0
        and sp.simplify(traceless_weight_for_path_count(2) - sp.Rational(1, 2)) == 0
    )


def one_trace_path_control_rejected() -> bool:
    """Return whether one coherent trace path fails equipartition."""

    return sp.simplify(trace_weight_for_path_count(1) - sp.Rational(1, 2)) != 0


def three_trace_path_control_rejected() -> bool:
    """Return whether three coherent trace paths fail equipartition."""

    return sp.simplify(trace_weight_for_path_count(3) - sp.Rational(1, 2)) != 0


def torsion_weight_inserted_into_theta() -> bool:
    """Return whether Session 14 uses the occupation weight as a plane angle."""

    return (
        sp.simplify(charged_lepton_torsion_angle() - charged_lepton_torsion_weight()) == 0
        and sp.simplify(
            charged_lepton_boundary_angle()
            - (charged_lepton_base_angle() - charged_lepton_torsion_weight())
        )
        == 0
    )


def cmv_phase_word_is_not_torsion_angle() -> bool:
    """Return whether the holonomy phase word is distinct from the ``2/9`` torsion."""

    holonomy = leptonic_boundary_holonomy_audit_payload()
    return sp.simplify(sp.pi * holonomy.principal_angle - charged_lepton_torsion_weight()) != 0


def charged_lepton_trace_torsion_origin_payload() -> ChargedLeptonTraceTorsionOriginPayload:
    """Return the Session 19 charged-lepton trace/torsion origin audit."""

    graph = charged_lepton_boundary_graph_payload()
    torsion = charged_lepton_torsion_payload()
    holonomy = leptonic_boundary_holonomy_audit_payload()

    graph_pass = graph.final_verdict == "CHARGED_LEPTON_MINIMAL_BOUNDARY_GRAPH_PASS"
    torsion_pass = torsion.final_verdict == "CHARGED_LEPTON_2_OVER_9_OCCUPATION_PASS"
    holonomy_pass = holonomy.final_verdict == "LEPTONIC_PHASE_WORD_DERIVED_PASS"

    required_count = required_trace_path_count_for_equipartition()
    supplied_count = supplied_trace_path_count()
    supplied_matches_required = supplied_count == required_count
    two_paths = two_trace_paths_force_equipartition()
    one_rejected = one_trace_path_control_rejected()
    three_rejected = three_trace_path_control_rejected()
    theta_uses_torsion = torsion_weight_inserted_into_theta()
    phase_distinct = cmv_phase_word_is_not_torsion_angle()
    koide_exact = sp.simplify(koide_parameter() - sp.Rational(2, 3)) == 0
    graph_theta_exact = sp.simplify(
        charged_lepton_boundary_angle() + 2 * sp.pi / 3 + sp.Rational(2, 9)
    ) == 0

    checks_pass = (
        graph_pass
        and torsion_pass
        and holonomy_pass
        and source_occupation_weights() == {"a": sp.Rational(2, 3), "u": sp.Rational(1, 3), "b": 0}
        and two_paths
        and supplied_matches_required
        and one_rejected
        and three_rejected
        and theta_uses_torsion
        and phase_distinct
        and graph_theta_exact
        and koide_exact
    )

    if checks_pass:
        final_verdict = "CHARGED_LEPTON_TRACE_TORSION_ORIGIN_NOT_DERIVED_AUDIT"
        interpretation = (
            "Trace/traceless equipartition in the minimal charged-lepton graph "
            "forces exactly two coherent trace paths: n trace paths give trace "
            "weight n/(n+2), so n=2 is the unique positive integer solution. "
            "The Session 14 graph supplies exactly two trace-only pole rows, "
            "and one-path/three-path controls fail.  However the current repo "
            "does not derive those two rows from a microscopic colorless "
            "BCC/Higgs boundary.  Likewise 2/9 is derived as the frozen-source "
            "occupation weight p_a p_u, but Session 14 still inserts that "
            "number as a rotation angle in theta.  No occupation-to-angle "
            "dynamics is derived here."
        )
    else:
        final_verdict = "CHARGED_LEPTON_TRACE_TORSION_ORIGIN_KILL"
        interpretation = (
            "The charged-lepton trace/torsion audit failed graph, torsion, "
            "holonomy, trace-count, negative-control, theta-insertion, or "
            "Koide consistency checks."
        )

    return ChargedLeptonTraceTorsionOriginPayload(
        final_verdict=final_verdict,
        boundary_graph_pass=graph_pass,
        torsion_gate_pass=torsion_pass,
        holonomy_gate_pass=holonomy_pass,
        source_occupation_weights=source_occupation_weights(),
        trace_weight_formula=trace_weight_for_path_count(required_count),
        traceless_weight_formula=traceless_weight_for_path_count(required_count),
        required_trace_path_count=required_count,
        supplied_trace_path_count=supplied_count,
        supplied_trace_paths_match_required_count=supplied_matches_required,
        two_trace_paths_force_equipartition=two_paths,
        one_trace_path_control_rejected=one_rejected,
        three_trace_path_control_rejected=three_rejected,
        trace_path_origin_derived_from_bcc_higgs=False,
        torsion_occupation_weight=charged_lepton_torsion_weight(),
        torsion_used_as_rotation_angle=charged_lepton_torsion_angle(),
        torsion_weight_inserted_into_theta=theta_uses_torsion,
        occupation_to_angle_map_derived=False,
        cmv_phase_word_pass=holonomy_pass,
        cmv_phase_word_is_not_torsion_angle=phase_distinct,
        minimal_graph_accepts_supplied_theta=graph_theta_exact,
        koide_still_exact_with_supplied_theta=trace_weight() == sp.Rational(1, 2)
        and traceless_weight() == sp.Rational(1, 2)
        and koide_exact,
        remaining_declared_inputs=(TRACE_PATH_ORIGIN_PREMISE, TORSION_DYNAMICS_PREMISE),
        interpretation=interpretation,
    )
