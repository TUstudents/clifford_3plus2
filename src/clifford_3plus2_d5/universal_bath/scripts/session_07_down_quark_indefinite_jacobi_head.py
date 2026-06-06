"""Run the Session 07 down-quark indefinite Jacobi-head certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.down_quark_indefinite_jacobi import (
    down_quark_indefinite_jacobi_payload,
)


def main() -> None:
    """Print Session 07 payload."""

    payload = down_quark_indefinite_jacobi_payload()
    print("source dictionary pass =", payload.source_dictionary_pass)
    print("quark source unresolved =", payload.quark_source_unresolved)
    print("source label =", payload.source_label)
    print("source reduction =", payload.source_reduction)
    print("subset prerequisite pass =", payload.subset_prerequisite_pass)
    print("S3 projector prerequisite pass =", payload.s3_projector_prerequisite_pass)
    print("three-port head =", payload.three_port_head)
    print("regular baseline head =", payload.regular_baseline_head)
    print("regular candidate head =", payload.regular_candidate_head)
    print("three-port cannot host candidate =", payload.three_port_cannot_host_candidate)
    print("regular candidate available =", payload.regular_candidate_available)
    print("regular candidate forced by S3 alone =", payload.regular_candidate_forced_by_s3_alone)
    print("rank two requires defect polarization =", payload.rank_two_requires_defect_polarization)
    print("rank five not unique =", payload.rank_five_not_unique)
    print("signature breakdown control detected =", payload.signature_breakdown_control_detected)
    print("selected physical head =", payload.selected_physical_head)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
