"""Run the Session 18 down odd-shell rank-five audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_down_odd_shell import (
    quark_down_odd_shell_payload,
)


def main() -> None:
    """Print Session 18 payload."""

    payload = quark_down_odd_shell_payload()
    print("active color microcanonical pass =", payload.active_color_microcanonical_pass)
    print("down subset pass =", payload.down_subset_pass)
    print("S3 projector pass =", payload.s3_projector_pass)
    print("S3 rank five ambiguous =", payload.s3_rank_five_ambiguous)
    print("full shell count =", payload.full_shell_subset.count)
    print("BCC odd count =", payload.bcc_odd_subset.count)
    print("odd shell count =", payload.odd_shell_subset.count)
    print("primitive counts =", payload.primitive_counts)
    print("candidate counts =", payload.candidate_counts)
    print("candidate Clebsch vector =", payload.candidate_clebsch_vector)
    print("odd shell rank five =", payload.odd_shell_rank_five)
    print(
        "odd shell complement of even direct =",
        payload.odd_shell_is_complement_of_even_direct,
    )
    print("BCC middle rank two =", payload.bcc_middle_rank_two)
    print("color-only middle control rejected =", payload.color_only_middle_control_rejected)
    print(
        "compressed parity control rejected for middle =",
        payload.compressed_parity_control_rejected_for_middle,
    )
    print("primitive parity selects rank-five line =", payload.primitive_parity_selects_rank_five_line)
    print("remaining readout premise =", payload.remaining_readout_premise)
    print("source freeze ready =", payload.source_freeze_ready)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
