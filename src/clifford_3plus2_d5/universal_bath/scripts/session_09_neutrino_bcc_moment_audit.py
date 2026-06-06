"""Run the Session 09 microscopic BCC neutrino moment audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.neutrino_bcc_moments import (
    neutrino_bcc_moment_audit_payload,
)


def main() -> None:
    """Print Session 09 payload."""

    payload = neutrino_bcc_moment_audit_payload()
    print("product internal pass =", payload.product_internal_pass)
    print("checked moment powers =", payload.checked_moment_powers)
    print("same-normal norm =")
    print(payload.same_normal_norm)
    print("mixed-normal norm =")
    print(payload.mixed_normal_norm)
    print("total norm is identity =", payload.total_norm_is_identity)
    print("q=0 scar block available =", payload.q0_scar_block_available)
    print("leakage block available =", payload.leakage_block_available)
    print("microscopic edge labels =", payload.microscopic_edge_labels)
    print("missing family labels =", payload.missing_family_labels)
    print(
        "microscopic family-port graph available =",
        payload.microscopic_family_port_graph_available,
    )
    print("product ansatz cross moments zero =", payload.product_ansatz_cross_moments_zero)
    print("product ansatz diagonal equal =", payload.product_ansatz_diagonal_equal)
    print("product ansatz is only family factor =", payload.product_ansatz_is_only_family_factor)
    print("BCC family cross moments defined =", payload.bcc_family_cross_moments_defined)
    print("can upgrade neutrino core =", payload.can_upgrade_neutrino_core)
    print("rank-one control has cross return =", payload.rank_one_control_has_cross_return)
    print("next required object =", payload.next_required_object)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
