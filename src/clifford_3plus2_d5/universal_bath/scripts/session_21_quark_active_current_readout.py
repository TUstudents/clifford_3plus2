"""Run the Session 21 quark active-current readout certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_active_current_readout import (
    quark_active_current_readout_payload,
)


def main() -> None:
    """Print Session 21 payload."""

    payload = quark_active_current_readout_payload()
    print("active plane pass =", payload.active_plane_pass)
    print("height bridge pass =", payload.height_bridge_pass)
    print("up head pass =", payload.up_head_pass)
    print("down odd shell pass =", payload.down_odd_shell_pass)
    print("current line residual (u,a,b) =", payload.current_line_residual_uab)
    print("active current is b =", payload.active_current_is_b)
    print("active current unique non-scalar line =", payload.active_current_unique_non_scalar_line)
    print("first return orders =", payload.first_return_orders)
    print("first return orders light-to-heavy =", payload.first_return_orders_light_to_heavy)
    print("up radial depths =", payload.up_radial_depths)
    print("down radial depths =", payload.down_radial_depths)
    print("up coherent profile =", payload.up_coherent_profile)
    print("up profile matches conditional head =", payload.up_profile_matches_conditional_head)
    print("up geometric control rejected =", payload.up_geometric_control_rejected)
    print("down readout is covariance =", payload.down_readout_is_covariance)
    print("down baseline counts =", payload.down_baseline_counts)
    print("down baseline profile =", payload.down_baseline_profile)
    print("down odd-shell counts =", payload.down_odd_shell_counts)
    print("down odd-shell profile =", payload.down_odd_shell_profile)
    print(
        "identity veto selects odd shell if assumed =",
        payload.down_identity_veto_selects_odd_shell_if_assumed,
    )
    print(
        "identity veto microscopically derived =",
        payload.down_identity_veto_microscopically_derived,
    )
    print("source freeze candidate =", payload.source_freeze_candidate)
    print("source freeze ready =", payload.source_freeze_ready)
    print("remaining physical inputs =", payload.remaining_physical_inputs)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
