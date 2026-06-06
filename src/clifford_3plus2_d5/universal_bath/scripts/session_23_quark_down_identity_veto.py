"""Run the Session 23 down identity-return veto certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_down_identity_veto import (
    quark_down_identity_veto_payload,
)


def main() -> None:
    """Print Session 23 payload."""

    payload = quark_down_identity_veto_payload()
    print("active current readout pass =", payload.active_current_readout_pass)
    print("current parity selector pass =", payload.current_parity_selector_pass)
    print("odd shell pass =", payload.odd_shell_pass)
    print("direct identity names =", payload.direct_identity_names)
    print("retarded allowed names =", payload.retarded_allowed_names)
    print("retarded rejected names =", payload.retarded_rejected_names)
    print("direct identity is unique =", payload.direct_identity_is_unique)
    print("direct identity vetoed =", payload.direct_identity_vetoed)
    print("all allowed returns are odd =", payload.all_allowed_returns_are_odd)
    print("allowed return count =", payload.allowed_return_count)
    print("allowed return breakdown =", payload.allowed_return_breakdown)
    print("retarded counts =", payload.retarded_counts)
    print("retarded profile =", payload.retarded_profile)
    print("baseline counts =", payload.baseline_counts)
    print("baseline profile =", payload.baseline_profile)
    print(
        "baseline control rejected by retarded predicate =",
        payload.baseline_control_rejected_by_retarded_predicate,
    )
    print(
        "rank five selected inside retarded model =",
        payload.rank_five_selected_inside_retarded_model,
    )
    print("down identity veto premise reduced =", payload.down_identity_veto_premise_reduced)
    print("microscopic bare BB derivation =", payload.microscopic_bare_bcc_derivation)
    print("remaining physical premises =", payload.remaining_physical_premises)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
