"""Run the finite local boundary-dilation audit for the cusp graph."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import cusp_boundary_dilation_audit, target_a_payload


def main() -> None:
    """Print the Session 04 boundary-dilation payload."""

    audit = cusp_boundary_dilation_audit()
    payload = target_a_payload()
    print("local isometry shape =", audit.local_isometry.shape)
    print("local isometry norm =", audit.local_isometry_norm)
    print("local isometry pass =", audit.local_isometry_pass)
    print("local input dimension =", audit.local_input_dimension)
    print("local output dimension =", audit.local_output_dimension)
    print("unitary completion nullity =", audit.unitary_completion_nullity)
    print("unitary dilation exists =", audit.unitary_dilation_exists_pass)
    print("finite gap feedback scalar =", audit.finite_gap_feedback_scalar)
    print("hard-gap feedback zero =", audit.hard_gap_feedback_zero)
    print("retarded feedback limit zero =", audit.retarded_feedback_limit_zero)
    print(
        "retarded visible powers match survival =",
        audit.retarded_visible_powers_match_survival,
    )
    print("recurrent wedge return =", audit.recurrent_wedge_return)
    print("recurrent wedge return nonzero =", audit.recurrent_wedge_return_nonzero)
    print("recurrent visible powers differ =", audit.recurrent_visible_powers_differ)
    print("outgoing leakage pass =", audit.outgoing_leakage_pass)
    print("boundary dilation pass =", audit.boundary_dilation_pass)
    print("remaining declared inputs =", audit.remaining_declared_inputs)
    print("Target A pass =", payload.target_a_pass)


if __name__ == "__main__":
    main()
