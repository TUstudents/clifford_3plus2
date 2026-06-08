"""Run MicroCUSP Session A: microscopic q-stiffness material audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import microscopic_q_stiffness_audit


def main() -> None:
    """Print the microscopic q-stiffness audit."""

    audit = microscopic_q_stiffness_audit()
    material = audit.material
    controls = audit.controls
    print("stiffness matrix =", material.stiffness_matrix)
    print("microscopic action =", material.microscopic_action)
    print("effective q-action =", material.action_in_q)
    print("q-reflection even =", material.q_reflection_even)
    print("no linear term =", material.no_linear_term)
    print("curvature =", material.curvature)
    print("trace mode ungapped =", material.trace_mode_ungapped)
    print("adjacent leakage gap =", material.adjacent_leakage_gap)
    print(
        "positive semidefinite rank-one stiffness =",
        material.stiffness_matrix_positive_semidefinite,
    )
    print(
        "matches Session 05 finite locking =",
        audit.effective_action_matches_finite_locking,
    )
    print("gap matches Session 05 =", audit.adjacent_gap_matches_finite_locking)
    print(
        "hard-gap suppresses mixed feedback =",
        audit.hard_gap_limit_suppresses_mixed_feedback,
    )
    print("zero control rejected =", controls.zero_control_rejected)
    print("linear control mirror defect =", controls.linear_control_mirror_defect)
    print("linear control rejected =", controls.linear_control_rejected)
    print("absolute control analytic at zero =", controls.absolute_control_analytic_at_zero)
    print("absolute control rejected =", controls.absolute_control_rejected)
    print("controls rejected =", controls.controls_rejected)
    print("session A pass =", audit.session_a_pass)
    print("remaining microscopic gates =", audit.microscopic_gate_remaining)


if __name__ == "__main__":
    main()
