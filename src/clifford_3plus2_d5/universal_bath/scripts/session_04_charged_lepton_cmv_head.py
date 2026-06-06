"""Run the Session 04 charged-lepton CMV finite-head certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.charged_lepton_cmv import (
    charged_lepton_cmv_head_payload,
)


def main() -> None:
    """Print Session 04 payload."""

    payload = charged_lepton_cmv_head_payload()
    print("source dictionary pass =", payload.source_dictionary_pass)
    print("holonomy prerequisite pass =", payload.holonomy_prerequisite_pass)
    print("source label =", payload.source_label)
    print("source reduction =", payload.source_reduction)
    print("source depth =", payload.source_depth)
    print("residual components =", payload.residual_components)
    print("two-step leakage =", payload.two_step_leakage)
    print("rotation sine =", payload.rotation_sine)
    print("rotation sine squared =", payload.rotation_sine_squared)
    print("phase angle / pi =", payload.phase_angle)
    print("phase =", payload.phase)
    print("alpha_e =", payload.alpha)
    print("|alpha_e|^2 =", payload.alpha_modulus_squared)
    print("alpha inside unit disk =", payload.alpha_inside_unit_disk)
    print("CMV/Givens head =")
    print(payload.cmv_head)
    print("CMV/Givens head unitary =", payload.cmv_head_unitary)
    print("Verblunsky coefficients =", payload.verblunsky_coefficients)
    print("free tail after head =", payload.free_tail_after_head)
    print("depth-one control rejected =", payload.depth_one_control_rejected)
    print("depth-three control rejected =", payload.depth_three_control_rejected)
    print("b-leakage control rejected =", payload.b_leakage_control_rejected)
    print("holonomy controls rejected =", payload.holonomy_controls_rejected)
    print("PMNS parked =", payload.pmns_parked)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
