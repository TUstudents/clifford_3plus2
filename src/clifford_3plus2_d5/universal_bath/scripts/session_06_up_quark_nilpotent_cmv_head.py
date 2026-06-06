"""Run the Session 06 up-quark nilpotent CMV-head certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.up_quark_nilpotent_cmv import (
    up_quark_nilpotent_cmv_payload,
)


def main() -> None:
    """Print Session 06 payload."""

    payload = up_quark_nilpotent_cmv_payload()
    print("source dictionary pass =", payload.source_dictionary_pass)
    print("quark source unresolved =", payload.quark_source_unresolved)
    print("source label =", payload.source_label)
    print("source reduction =", payload.source_reduction)
    print("survival operator =")
    print(payload.survival_operator)
    print("survival weight =", payload.survival_weight)
    print("injection amplitude =", payload.injection_amplitude)
    print("scalar Clebsch prerequisite pass =", payload.scalar_clebsch_prerequisite_pass)
    print("up stacking prerequisite pass =", payload.up_stacking_prerequisite_pass)
    print("nilpotent flag =")
    print(payload.nilpotent_flag)
    print("nilpotent order three =", payload.nilpotent_order_three)
    print("Taylor kernel =")
    print(payload.taylor_kernel)
    print("Taylor profile =", payload.taylor_profile)
    print("expected profile =", payload.expected_profile)
    print("Taylor profile matches =", payload.taylor_profile_matches)
    print("geometric control profile =", payload.geometric_control_profile)
    print("geometric control rejected =", payload.geometric_control_rejected)
    print("old sqrt2 control =", payload.old_sqrt2_control)
    print("old sqrt2 control rejected =", payload.old_sqrt2_control_rejected)
    print("finite Verblunsky head =", payload.finite_verblunsky_head)
    print("finite head inside unit disk =", payload.finite_head_inside_unit_disk)
    print("free tail after head =", payload.free_tail_after_head)
    print("full quark source not derived =", payload.full_quark_source_not_derived)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
