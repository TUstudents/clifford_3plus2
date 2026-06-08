"""Run MicroCUSP Session D: color-center Z3 recirculation audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import microscopic_color_z3_recirculation_audit


def main() -> None:
    """Print the color Z3 recirculation audit."""

    audit = microscopic_color_z3_recirculation_audit()
    print("center phase =", audit.center_phase)
    print("charge table through six =", audit.charge_table_through_six)
    print("holonomy table through three =", audit.holonomy_table_through_three)
    print("open holonomies nontrivial =", audit.open_holonomies_nontrivial)
    print("closed holonomy neutral =", audit.closed_holonomy_neutral)
    print("primitive return length =", audit.primitive_return_length)
    print("channel count =", audit.channel_count)
    print("one-step forbidden =", audit.one_step_forbidden)
    print("automaton matches center holonomy =", audit.automaton_matches_center_holonomy)
    print("wrong color length 2 control =", audit.wrong_color_length_two_first_values)
    print("wrong color length 4 control =", audit.wrong_color_length_four_first_values)
    print("spectator color control =", audit.spectator_color_control_first_values)
    print(
        "gauged-away open phase control =",
        audit.gauged_away_open_phase_control_first_values,
    )
    print("controls rejected =", audit.controls_rejected)
    print("combined cusp graph preserved =", audit.combined_cusp_graph_preserved)
    print("session D pass =", audit.session_d_pass)
    print("remaining microscopic gates =", audit.microscopic_gate_remaining)


if __name__ == "__main__":
    main()
