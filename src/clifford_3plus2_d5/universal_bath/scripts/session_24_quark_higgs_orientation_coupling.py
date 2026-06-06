"""Run the Session 24 quark Higgs-door orientation-coupling audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_higgs_orientation_coupling import (
    quark_higgs_orientation_coupling_payload,
)


def main() -> None:
    """Print Session 24 payload."""

    payload = quark_higgs_orientation_coupling_payload()
    print("height orientation bridge pass =", payload.height_orientation_bridge_pass)
    print("current parity selector pass =", payload.current_parity_selector_pass)
    print("down identity veto pass =", payload.down_identity_veto_pass)
    print("hypercharge forces doors =", payload.hypercharge_forces_doors)
    print("neutral Higgs components =", payload.neutral_higgs_components)
    print("endpoint reflection =", payload.endpoint_reflection)
    print("endpoint reflection involutive =", payload.endpoint_reflection_involutive)
    print("reflection maps flag to reverse flag =", payload.reflection_maps_flag_to_reverse_flag)
    print("Hermitian closure reflection invariant =", payload.hermitian_closure_reflection_invariant)
    print("reflection maps flag to closure =", payload.reflection_maps_flag_to_closure)
    print("Hermitian closure requires extra pairing =", payload.hermitian_closure_requires_extra_pairing)
    print("declared assignment =", payload.declared_assignment)
    print("swapped assignment =", payload.swapped_assignment)
    print(
        "available constraints select unique assignment =",
        payload.available_constraints_select_unique_assignment,
    )
    print("swapped assignment survives controls =", payload.swapped_assignment_survives_controls)
    print(
        "Higgs conjugation supplies reversal not closure =",
        payload.higgs_conjugation_supplies_reversal_not_closure,
    )
    print(
        "Higgs-door orientation coupling derived =",
        payload.higgs_door_orientation_coupling_derived,
    )
    print("remaining orientation premise =", payload.remaining_orientation_premise)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
