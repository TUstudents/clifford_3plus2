"""Run the Session 02 D3 clock source certificate."""

from __future__ import annotations

from clifford_3plus2_d5.threeclocks.d3_clock import d3_clock_source_payload


def main() -> None:
    """Print Session 02 payload."""

    payload = d3_clock_source_payload()
    print("D3 relations pass =", payload.d3_relations_pass)
    print("selected tooth components (u,a,b) =", payload.selected_tooth_components)
    print("selected tooth has no tangent component =", payload.selected_tooth_has_no_tangent)
    print("tangent current is b =", payload.tangent_current_is_b)
    print("radial second difference is a =", payload.radial_second_difference_is_a)
    print("quark source b derived =", payload.quark_source_b_derived)
    print("three-port Laplacian spectrum =", payload.three_port_laplacian_spectrum)
    print(
        "three-port Laplacian degenerate control =",
        payload.three_port_laplacian_is_degenerate_control,
    )
    print("literal port cut equals repair flag =", payload.literal_cut_equals_repair_flag)
    print("repair flag nilpotent target pass =", payload.repair_flag_nilpotent_target_pass)
    print("first-return orders target (u,a,b) =", payload.first_return_orders_target)
    print("conditional up profile =", payload.up_profile_conditional)
    print("conditional down rank-five profile =", payload.down_rank_five_profile_conditional)
    print("contact-return control profile =", payload.down_contact_allowed_profile_control)
    print("open theorem targets:")
    for target in payload.open_theorem_targets:
        print(" -", target)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
