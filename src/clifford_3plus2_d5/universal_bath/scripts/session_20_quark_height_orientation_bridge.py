"""Run the Session 20 quark height-orientation bridge audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_height_orientation_bridge import (
    quark_height_orientation_bridge_payload,
)


def main() -> None:
    """Print Session 20 payload."""

    payload = quark_height_orientation_bridge_payload()
    print("height-door audit pass =", payload.height_door_audit_pass)
    print("successor certificate pass =", payload.successor_certificate_pass)
    print("nilpotent flag pass =", payload.nilpotent_flag_pass)
    print("allowed successors =", payload.allowed_successors)
    print("successors match flag =", payload.oriented_successors_match_flag)
    print("oriented nilpotent matches up repair =", payload.oriented_nilpotent_matches_up_repair)
    print("Hermitian closure matches down repair =", payload.hermitian_closure_matches_down_repair)
    print("down readout is closure of up flag =", payload.down_readout_is_closure_of_up_flag)
    print("hypercharge forces doors =", payload.hypercharge_forces_doors)
    print("neutral Higgs components =", payload.neutral_higgs_components)
    print("swapped assignment hypercharge allowed =", payload.swapped_assignment_hypercharge_allowed)
    print(
        "Higgs-door orientation coupling derived =",
        payload.higgs_door_orientation_coupling_derived,
    )
    print("remaining orientation premise =", payload.remaining_orientation_premise)
    print("quark sources still unfrozen =", payload.quark_sources_still_unfrozen)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
