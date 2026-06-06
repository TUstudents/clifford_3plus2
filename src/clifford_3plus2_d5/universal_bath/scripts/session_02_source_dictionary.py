"""Run the Session 02 source-dictionary certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.source_dictionary import source_dictionary_payload


def main() -> None:
    """Print Session 02 payload."""

    payload = source_dictionary_payload()
    print("survival operator universal =", payload.survival_operator_universal)
    print("survival operator =")
    print(payload.survival_operator)
    print("frozen source survival weights =")
    for label, weight in payload.frozen_survival_weights.items():
        print(f"  {label}: {weight}")
    print("frozen sources =")
    for anchor in payload.frozen_sources:
        print(
            f"  {anchor.label}: sector={anchor.sector}, "
            f"reduction={anchor.reduction}, depth={anchor.normal_depth}, "
            f"certainty={anchor.certainty}"
        )
    print("unresolved sources =")
    for anchor in payload.unresolved_sources:
        print(
            f"  {anchor.label}: sector={anchor.sector}, "
            f"reduction={anchor.reduction}, certainty={anchor.certainty}"
        )
    print("all frozen sources survive =", payload.all_frozen_sources_survive)
    print("no flavor data used =", payload.no_flavor_data_used)
    print("all physical sources frozen =", payload.all_physical_sources_frozen)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
