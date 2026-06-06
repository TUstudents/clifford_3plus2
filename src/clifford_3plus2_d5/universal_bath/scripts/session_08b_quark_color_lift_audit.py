"""Run the Session 08B quark color-lift audit certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_color_lift import (
    quark_color_lift_payload,
)


def main() -> None:
    """Print Session 08B payload."""

    payload = quark_color_lift_payload()
    print("height-door prerequisite pass =", payload.height_door_prerequisite_pass)
    print("quark shell prerequisite pass =", payload.quark_shell_prerequisite_pass)
    print("fixed color embedding =", payload.fixed_color_embedding)
    print("spectator embedding =", payload.spectator_embedding)
    print("active embedding =", payload.active_embedding)
    print("fixed color rejected =", payload.fixed_color_rejected)
    print("spectator preserves color =", payload.spectator_preserves_color)
    print("spectator stays three-port =", payload.spectator_stays_three_port)
    print("active preserves visible color =", payload.active_preserves_visible_color)
    print("active regular S3 baseline =", payload.active_regular_s3_baseline)
    print("active rank-five candidate available =", payload.active_rank_five_candidate_available)
    print("active rank-five candidate forced =", payload.active_rank_five_candidate_forced)
    print("color return contraction scalar =", payload.color_return_contraction_scalar)
    print("gauge alone selects active =", payload.gauge_alone_selects_active)
    print("source freeze ready =", payload.source_freeze_ready)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
