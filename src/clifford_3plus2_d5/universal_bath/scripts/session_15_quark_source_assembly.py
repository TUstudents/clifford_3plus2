"""Run the Session 15 quark source assembly audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_source_assembly import (
    quark_source_assembly_payload,
)


def main() -> None:
    """Print Session 15 payload."""

    payload = quark_source_assembly_payload()
    print("source dictionary pass =", payload.source_dictionary_pass)
    print("up source unresolved =", payload.up_source_unresolved)
    print("down source unresolved =", payload.down_source_unresolved)
    print("up missing source fields =", payload.up_missing_source_fields)
    print("down missing source fields =", payload.down_missing_source_fields)
    print("common family incidence pass =", payload.common_family_incidence_pass)
    print("height door pass =", payload.height_door_pass)
    print("color lift pass =", payload.color_lift_pass)
    print("up head pass =", payload.up_head_pass)
    print("down head pass =", payload.down_head_pass)
    print("conditional heads assembled =", payload.conditional_heads_assembled)
    print("up conditional profile =", payload.up_conditional_profile)
    print("down spectator baseline profile =", payload.down_spectator_baseline_profile)
    print("down active baseline profile =", payload.down_active_baseline_profile)
    print("down active candidate profile =", payload.down_active_candidate_profile)
    print("height mode selected by dynamics =", payload.height_mode_selected_by_dynamics)
    print("active color return selected =", payload.active_color_return_selected)
    print("bottom rank-five decided =", payload.bottom_rank_five_decided)
    print("quark normal depths frozen =", payload.quark_normal_depths_frozen)
    print("unresolved premises =", payload.unresolved_premises)
    print("source freeze ready =", payload.source_freeze_ready)
    print("quark sources remain unfrozen =", payload.quark_sources_remain_unfrozen)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
