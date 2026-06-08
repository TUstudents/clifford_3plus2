"""Run the finite boundary-material origin audit for the cusp graph."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import (
    cusp_boundary_material_origin_audit,
    target_a_payload,
)


def main() -> None:
    """Print the Session 05 boundary-material origin payload."""

    audit = cusp_boundary_material_origin_audit()
    payload = target_a_payload()
    print("q-locking polynomial =", audit.q_locking.selected_polynomial)
    print("q-locking values on (0,-2,+2) =", audit.q_locking.selected_penalty_values)
    print("q-locking gap =", audit.q_locking.selected_gap)
    print("q-locking uniqueness solution =", audit.q_locking.uniqueness_solution)
    print("linear control values =", audit.q_locking.linear_control_values)
    print("constant control values =", audit.q_locking.constant_control_values)
    print("linear control rejected =", audit.q_locking.linear_control_rejected)
    print("constant control rejected =", audit.q_locking.constant_control_rejected)
    print("q-locking origin pass =", audit.q_locking.q_locking_origin_pass)
    print("outgoing causal rule =", audit.outgoing.causal_rule)
    print(
        "retarded powers match survival =",
        audit.outgoing.retarded_visible_powers_match_survival,
    )
    print("recurrent wedge return =", audit.outgoing.recurrent_wedge_return)
    print("recurrent return nonzero =", audit.outgoing.recurrent_wedge_return_nonzero)
    print("recurrent powers differ =", audit.outgoing.recurrent_visible_powers_differ)
    print("outgoing origin pass =", audit.outgoing.outgoing_origin_pass)
    print("material origin pass =", audit.material_origin_pass)
    print("remaining physical axioms =", audit.remaining_physical_axioms)
    print("Target A pass =", payload.target_a_pass)


if __name__ == "__main__":
    main()
