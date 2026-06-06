"""Session 20 quark height-orientation bridge audit.

Session 08A showed that electroweak hypercharge forces the Higgs doors
``H_tilde`` and ``H`` but does not force the repair-mode assignment

    up   -> oriented nilpotent
    down -> Hermitian closure.

The depth-scar sidecar contains a stronger fact than Session 08A used: the
finite successor certificate selects a unique oriented repair flag

    a -> u,  b -> a.

This session imports that certificate and checks whether the quark height-door
premise can be reduced.  It can: the up nilpotent and down Hermitian closure
are two readouts of the same certified flag.  It still cannot derive which SM
Higgs door couples to which readout.  The remaining premise is therefore an
orientation-coupling rule, not the existence of the repair flag itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_laplacian_from_nilpotent,
    nilpotent_flag_operator,
    nilpotent_flag_scar_payload,
    nilpotent_order_pass,
)
from clifford_3plus2_d5.depth_scar.successor_certificate import (
    allowed_successors_from_certificate,
    successor_certificate_payload,
)
from clifford_3plus2_d5.universal_bath.quark_height_door import (
    down_higgs_door,
    down_repair_operator,
    hypercharge_forces_higgs_doors,
    neutral_higgs_components,
    quark_height_door_payload,
    swapped_repair_assignment_hypercharge_allowed,
    up_higgs_door,
    up_repair_operator,
)

ORIENTATION_COUPLING_PREMISE = (
    "higgs_door_orientation_couples_H_tilde_to_retarded_flag_and_H_to_flag_closure"
)


@dataclass(frozen=True)
class QuarkHeightOrientationBridgePayload:
    """Session 20 quark height-orientation bridge verdict."""

    final_verdict: str
    height_door_audit_pass: bool
    successor_certificate_pass: bool
    nilpotent_flag_pass: bool
    allowed_successors: dict[str, tuple[str, ...]]
    oriented_successors_match_flag: bool
    oriented_nilpotent: sp.Matrix
    up_repair: sp.Matrix
    oriented_nilpotent_matches_up_repair: bool
    hermitian_closure: sp.Matrix
    down_repair: sp.Matrix
    hermitian_closure_matches_down_repair: bool
    down_readout_is_closure_of_up_flag: bool
    hypercharge_forces_doors: bool
    neutral_higgs_components: bool
    swapped_assignment_hypercharge_allowed: bool
    higgs_door_orientation_coupling_derived: bool
    remaining_orientation_premise: str
    quark_sources_still_unfrozen: bool
    interpretation: str


def oriented_successors_match_flag() -> bool:
    """Return whether the V12 successor certificate encodes ``a->u,b->a``."""

    return allowed_successors_from_certificate() == {"a": ("u",), "b": ("a",)}


def oriented_nilpotent_matches_up_repair() -> bool:
    """Return whether the certified nilpotent is the declared up repair."""

    return nilpotent_flag_operator() == up_repair_operator()


def hermitian_closure_matches_down_repair() -> bool:
    """Return whether the flag Laplacian is the declared down repair."""

    return flag_laplacian_from_nilpotent(nilpotent_flag_operator()) == down_repair_operator()


def down_readout_is_closure_of_up_flag() -> bool:
    """Return whether the down operator is the Hermitian closure of the up flag."""

    flag = up_repair_operator()
    return (
        nilpotent_order_pass(flag)
        and flag_laplacian_from_nilpotent(flag) == down_repair_operator()
    )


def higgs_door_orientation_coupling_derived() -> bool:
    """Return whether current inputs derive the Higgs-door orientation rule."""

    height = quark_height_door_payload()
    # Session 08A proves the opposite: hypercharge allows a swapped repair
    # assignment.  The successor certificate derives the flag, not its coupling
    # to H_tilde/H.
    return not height.repair_mode_not_forced_by_hypercharge


def quark_height_orientation_bridge_payload() -> QuarkHeightOrientationBridgePayload:
    """Return the Session 20 quark height-orientation bridge audit."""

    height = quark_height_door_payload()
    successor = successor_certificate_payload()
    flag = nilpotent_flag_scar_payload()

    height_pass = height.final_verdict == "QUARK_HEIGHT_DOOR_NO_DERIVATION_AUDIT"
    successor_pass = successor.final_verdict == "V12_UNIQUE_SUCCESSOR_ENUMERATION_CERTIFICATE_PASS"
    flag_pass = flag.final_verdict == "NILPOTENT_FLAG_SCAR_ORIGIN_PASS"
    successors_match = oriented_successors_match_flag()
    nilpotent_matches = oriented_nilpotent_matches_up_repair()
    closure_matches = hermitian_closure_matches_down_repair()
    closure_of_flag = down_readout_is_closure_of_up_flag()
    hypercharge = hypercharge_forces_higgs_doors()
    neutral = neutral_higgs_components()
    swapped_allowed = swapped_repair_assignment_hypercharge_allowed()
    orientation_derived = higgs_door_orientation_coupling_derived()

    checks_pass = (
        height_pass
        and successor_pass
        and flag_pass
        and successors_match
        and nilpotent_matches
        and closure_matches
        and closure_of_flag
        and hypercharge
        and neutral
        and swapped_allowed
        and not orientation_derived
        and up_higgs_door().door == "H_tilde"
        and down_higgs_door().door == "H"
    )

    if checks_pass:
        final_verdict = "QUARK_HEIGHT_ORIENTATION_BRIDGE_NOT_DERIVED_AUDIT"
        interpretation = (
            "The depth-scar successor certificate supplies the oriented repair "
            "flag a->u, b->a.  The declared up repair is exactly this "
            "nilpotent flag, and the declared down repair is exactly its "
            "Hermitian Laplacian closure.  Thus the two quark repair objects "
            "are not independent assumptions.  However hypercharge still "
            "allows a swapped repair assignment, so the missing microscopic "
            "input is the orientation-coupling rule that attaches H_tilde to "
            "the retarded flag and H to the flag closure."
        )
    else:
        final_verdict = "QUARK_HEIGHT_ORIENTATION_BRIDGE_KILL"
        interpretation = (
            "The quark height-orientation bridge failed the height-door, "
            "successor-certificate, nilpotent-flag, closure, hypercharge, or "
            "negative-control checks."
        )

    return QuarkHeightOrientationBridgePayload(
        final_verdict=final_verdict,
        height_door_audit_pass=height_pass,
        successor_certificate_pass=successor_pass,
        nilpotent_flag_pass=flag_pass,
        allowed_successors=allowed_successors_from_certificate(),
        oriented_successors_match_flag=successors_match,
        oriented_nilpotent=nilpotent_flag_operator(),
        up_repair=up_repair_operator(),
        oriented_nilpotent_matches_up_repair=nilpotent_matches,
        hermitian_closure=flag_laplacian_from_nilpotent(nilpotent_flag_operator()),
        down_repair=down_repair_operator(),
        hermitian_closure_matches_down_repair=closure_matches,
        down_readout_is_closure_of_up_flag=closure_of_flag,
        hypercharge_forces_doors=hypercharge,
        neutral_higgs_components=neutral,
        swapped_assignment_hypercharge_allowed=swapped_allowed,
        higgs_door_orientation_coupling_derived=orientation_derived,
        remaining_orientation_premise=ORIENTATION_COUPLING_PREMISE,
        quark_sources_still_unfrozen=True,
        interpretation=interpretation,
    )
