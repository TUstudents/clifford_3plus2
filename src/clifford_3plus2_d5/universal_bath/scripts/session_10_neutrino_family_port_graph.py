"""Run the Session 10 selected neutrino family-port graph certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.neutrino_family_port_graph import (
    selected_family_graph_payload,
)


def main() -> None:
    """Print Session 10 payload."""

    payload = selected_family_graph_payload()
    print("shells =", payload.shells)
    print("checked moment powers =", payload.checked_moment_powers)
    print("active projector =")
    print(payload.active_projector)
    print("radial projector =")
    print(payload.radial_projector)
    print("active projector rank =", payload.active_projector_rank)
    print("radial projector rank =", payload.radial_projector_rank)
    print("active + radial = I =", payload.active_radial_resolution_identity)
    print("graph differs from product identity =", payload.graph_differs_from_product_identity)
    print("graph is selected-S2 native =", payload.graph_is_selected_s2_native)
    print("cross moments zero =", payload.cross_moments_zero)
    print("diagonal moments equal =", payload.diagonal_moments_equal)
    print("radial mode separated =", payload.radial_mode_separated)
    print("tail response matches target =", payload.tail_response_matches_target)
    print("K3 control rejected =", payload.k3_control_rejected)
    print("full product control has radial mode =", payload.full_product_control_has_radial_mode)
    print("rank-one control has cross return =", payload.rank_one_control_has_cross_return)
    print("alternate tail control rejected =", payload.alternate_tail_control_rejected)
    print("can upgrade neutrino core =", payload.can_upgrade_neutrino_core)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
