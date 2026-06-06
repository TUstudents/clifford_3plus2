"""Run the Session 12 q-mismatch and retarded-compression certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.neutrino_q_mismatch_retarded import (
    q_mismatch_retarded_payload,
)


def main() -> None:
    """Print Session 12 payload."""

    payload = q_mismatch_retarded_payload()
    print("same-normal direction count =", payload.same_normal_direction_count)
    print("mixed-normal direction count =", payload.mixed_normal_direction_count)
    print("same-normal norm =")
    print(payload.same_normal_norm)
    print("mixed-normal norm =")
    print(payload.mixed_normal_norm)
    print("total norm is identity =", payload.total_norm_is_identity)
    print("family radial projector matches incidence =", payload.family_radial_projector_matches_incidence)
    print("family active projector matches incidence =", payload.family_active_projector_matches_incidence)
    print("leakage gap =", payload.leakage_gap)
    print("mixed Schur feedback =")
    print(payload.mixed_schur_feedback)
    print("hard-gap feedback zero =", payload.hard_gap_feedback_zero)
    print("finite-gap feedback scalar =", payload.finite_gap_feedback_scalar)
    print("retarded Weyl equation passes =", payload.retarded_weyl_equation_passes)
    print("retarded Weyl normalization passes =", payload.retarded_weyl_normalization_passes)
    print("retarded feedback limit zero =", payload.retarded_feedback_limit_zero)
    print("retarded visible powers match survival =", payload.retarded_visible_powers_match_survival)
    print("recurrent wedge return nonzero =", payload.recurrent_wedge_return_nonzero)
    print("recurrent visible powers differ =", payload.recurrent_visible_powers_differ)
    print("Session 10 family graph passes =", payload.session_10_family_graph_passes)
    print("Session 11 active plane passes =", payload.session_11_active_plane_passes)
    print("remaining declared inputs =", payload.remaining_declared_inputs)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
