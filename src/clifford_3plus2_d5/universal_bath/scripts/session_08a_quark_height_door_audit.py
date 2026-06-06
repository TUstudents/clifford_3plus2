"""Run the Session 08A quark height-door audit certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_height_door import (
    quark_height_door_payload,
)


def main() -> None:
    """Print Session 08A payload."""

    payload = quark_height_door_payload()
    print("source dictionary pass =", payload.source_dictionary_pass)
    print("up source unresolved =", payload.up_source_unresolved)
    print("down source unresolved =", payload.down_source_unresolved)
    print("up door =", payload.up_door)
    print("down door =", payload.down_door)
    print("hypercharge forces Higgs doors =", payload.hypercharge_forces_higgs_doors)
    print("neutral Higgs components =", payload.neutral_higgs_components)
    print("up operator nilpotent =", payload.up_operator_nilpotent)
    print("up operator non-Hermitian =", payload.up_operator_non_hermitian)
    print("down operator Hermitian =", payload.down_operator_hermitian)
    print("down operator not nilpotent =", payload.down_operator_not_nilpotent)
    print("down Laplacian spectrum =", payload.down_laplacian_spectrum)
    print("down spectrum matches path =", payload.down_laplacian_spectrum_matches_path)
    print(
        "swapped repair hypercharge allowed =",
        payload.swapped_repair_assignment_hypercharge_allowed,
    )
    print(
        "swapped repair rejected by height premise =",
        payload.swapped_repair_assignment_rejected_by_height_premise,
    )
    print("repair mode not forced by hypercharge =", payload.repair_mode_not_forced_by_hypercharge)
    print("quark sources still unfrozen =", payload.quark_sources_still_unfrozen)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
