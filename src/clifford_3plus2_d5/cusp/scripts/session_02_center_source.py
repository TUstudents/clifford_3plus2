"""Run the local center-charge recirculation source certificate."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import target_a_payload


def main() -> None:
    """Print the Session 02 Target A source payload."""

    payload = target_a_payload()
    print("closure sources =", tuple(source.name for source in payload.closure_sources))
    print("source center orders =", payload.closure_source_orders)
    print("source one-step forbidden =", payload.closure_source_one_step_forbidden)
    print(
        "material closure sources =",
        tuple(source.name for source in payload.material_closure_sources),
    )
    print("material same-normal norm =", payload.material_same_normal_norm)
    print("material mixed-normal norm =", payload.material_mixed_normal_norm)
    print("material norm split pass =", payload.material_norm_split_pass)
    print(
        "material weak source from BCC pass =",
        payload.material_weak_source_from_bcc_pass,
    )
    print("material source pass =", payload.material_source_pass)
    print("automaton return orders =", payload.automaton_return_orders)
    print("automaton one-step forbidden =", payload.automaton_one_step_forbidden)
    print("automaton charge tables through 6 =", payload.automaton_charge_tables_through_six)
    print("automaton return ticks through 6 =", payload.automaton_return_ticks_through_six)
    print("automaton graph edge count =", len(payload.automaton_graph_edges))
    print("closed-walk counts through 6 =", payload.closed_walk_counts_through_six)
    print("primitive closures =", payload.graph_primitive_closures)
    print("first valuations =", payload.first_three_valuations)
    print("graph controls =", payload.graph_controls)
    print("automaton graph pass =", payload.automaton_graph_pass)
    print("Target A pass =", payload.target_a_pass)


if __name__ == "__main__":
    main()
