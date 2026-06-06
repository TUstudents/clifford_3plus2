"""Run the Session 05 charged-lepton ``2/9`` torsion certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.charged_lepton_torsion import (
    charged_lepton_torsion_payload,
)


def main() -> None:
    """Print Session 05 payload."""

    payload = charged_lepton_torsion_payload()
    print("source dictionary pass =", payload.source_dictionary_pass)
    print("source label =", payload.source_label)
    print("source reduction =", payload.source_reduction)
    print("residual components =", payload.residual_components)
    print("occupation weights =", payload.occupation_weights)
    print("occupation weights normalized =", payload.occupation_weights_normalized)
    print("b occupation zero =", payload.b_occupation_zero)
    print("torsion transition weight =", payload.torsion_transition_weight)
    print("expected torsion weight =", payload.expected_torsion_weight)
    print("coherent transition amplitude =", payload.coherent_transition_amplitude)
    print("coherent amplitude rejected =", payload.coherent_amplitude_rejected)
    print("equal-weight control =", payload.equal_weight_control)
    print("equal-weight control rejected =", payload.equal_weight_control_rejected)
    print("one-port controls rejected =", payload.one_port_controls_rejected)
    print("CMV phase not rederived =", payload.cmv_phase_not_rederived)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
