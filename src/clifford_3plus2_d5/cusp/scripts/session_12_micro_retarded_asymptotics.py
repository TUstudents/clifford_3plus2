"""Run MicroCUSP Session B: no-incoming retarded boundary audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import microscopic_retarded_boundary_audit


def main() -> None:
    """Print the microscopic retarded-boundary audit."""

    audit = microscopic_retarded_boundary_audit()
    print("local unitary dilation exists =", audit.local_unitary_dilation_exists)
    print("mixed emission M =", audit.mixed_emission)
    print("mixed return G =", audit.mixed_return)
    print("no-incoming feedback =", audit.no_incoming_feedback)
    print("no-incoming feedback zero =", audit.no_incoming_feedback_zero)
    print("visible powers match q=0 survival =", audit.no_incoming_visible_powers_match)
    print(
        "visible Schur kernel has no recurrent mixed return =",
        audit.visible_schur_kernel_has_no_recurrent_mixed_return,
    )
    print("recurrent feedback =", audit.recurrent_feedback)
    print("recurrent feedback nonzero =", audit.recurrent_feedback_nonzero)
    print("hard-wall feedback =", audit.hard_wall_feedback)
    print("hard-wall feedback nonzero =", audit.hard_wall_feedback_nonzero)
    print("symmetric feedback =", audit.symmetric_feedback)
    print("symmetric feedback nonzero =", audit.symmetric_feedback_nonzero)
    print("controls rejected =", audit.controls_rejected)
    print("finite cusp graph preserved =", audit.finite_cusp_graph_preserved)
    print("session B pass =", audit.session_b_pass)
    print("remaining microscopic gates =", audit.microscopic_gate_remaining)


if __name__ == "__main__":
    main()
