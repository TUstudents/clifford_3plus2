"""Run the finite BCC/material source certificate for the cusp graph."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import cusp_boundary_material_source, target_a_payload


def main() -> None:
    """Print the Session 03 boundary-material source payload."""

    material = cusp_boundary_material_source()
    payload = target_a_payload()
    print("q-clock same-normal count =", material.q_clock.same_normal_count)
    print("q-clock mixed-normal count =", material.q_clock.mixed_normal_count)
    print("same-normal norm =", material.same_normal_norm)
    print("mixed-normal norm =", material.mixed_normal_norm)
    print("total norm identity =", material.total_norm_is_identity)
    print("norm split pass =", material.norm_split_pass)
    print("weak source =", material.weak_source)
    print("color source =", material.color_source)
    print("weak source from BCC pass =", material.weak_source_from_bcc_pass)
    print("color source pass =", material.color_source_pass)
    print("material source pass =", material.material_source_pass)
    print("automaton return orders =", payload.automaton_return_orders)
    print("closed-walk counts through 6 =", payload.closed_walk_counts_through_six)
    print("first valuations =", payload.first_three_valuations)
    print("Target A pass =", payload.target_a_pass)


if __name__ == "__main__":
    main()
