"""Run the Session 01 universal-bath spine certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.audit import universal_bath_audit_payload
from clifford_3plus2_d5.universal_bath.tail import silver_tail_payload


def main() -> None:
    """Print Session 01 payload."""

    tail = silver_tail_payload()
    payload = universal_bath_audit_payload()
    print("selected z =", tail.selected_z)
    print("epsilon =", tail.epsilon)
    print("tail(selected z) =", tail.tail_value)
    print("tail fixed-point residual =", tail.fixed_point_residual)
    print("toy measure positive =", payload.toy_measure_positive)
    print("moment round trip =", payload.moment_round_trip)
    print("response round trip =", payload.response_round_trip)
    print("finite-head Schur match =", payload.finite_head_schur_match)
    print("alternate tail changes response =", payload.alternate_tail_changes_response)
    print("reduction taxonomy pass =", payload.reduction_taxonomy_pass)
    print("OPUC free tail pass =", payload.opuc_free_tail_pass)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
