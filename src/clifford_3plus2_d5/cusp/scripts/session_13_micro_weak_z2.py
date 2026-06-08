"""Run MicroCUSP Session C: weak/BCC Z2 recirculation audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import microscopic_weak_z2_recirculation_audit


def main() -> None:
    """Print the weak/BCC Z2 recirculation audit."""

    audit = microscopic_weak_z2_recirculation_audit()
    print("same-normal branch common signs =", audit.branch_common_signs)
    print("branch parity group =", audit.branch_parity_group)
    print("branch parity group closed =", audit.branch_parity_group_closed)
    print("charge table through six =", audit.charge_table_through_six)
    print("sign table through six =", audit.sign_table_through_six)
    print("primitive return length =", audit.primitive_return_length)
    print("channel count =", audit.channel_count)
    print("one-step forbidden =", audit.one_step_forbidden)
    print("automaton matches branch parity =", audit.automaton_matches_branch_parity)
    print("trivial charge control =", audit.trivial_charge_control_first_values)
    print("order-one control =", audit.order_one_control_first_values)
    print("weak-only control =", audit.weak_only_control_first_values)
    print("controls rejected =", audit.controls_rejected)
    print("session C pass =", audit.session_c_pass)
    print("remaining microscopic gates =", audit.microscopic_gate_remaining)


if __name__ == "__main__":
    main()
