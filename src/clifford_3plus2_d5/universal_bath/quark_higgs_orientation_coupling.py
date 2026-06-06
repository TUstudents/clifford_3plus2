"""Session 24 quark Higgs-door orientation-coupling audit.

The quark mass head now has three strong pieces:

* the SM charges force the Higgs doors ``H_tilde`` and ``H``;
* the depth-scar successor certificate supplies one oriented flag
  ``N: b -> a -> u``;
* the selected-port current theorem selects the odd current source line ``b``.

The remaining keystone is the coupling rule

    H_tilde -> retarded flag N,
    H       -> Hermitian closure Delta_N.

This session tests whether that rule follows from the currently available
geometry.  It does not.  Endpoint reflection implements orientation reversal

    R N R = N.T,

while the down operator is the paired Hermitian closure

    Delta_N = N N.T + N.T N - (N + N.T).

The closure is invariant under endpoint reflection, but it is not produced by
Higgs conjugation/reversal alone.  Consequently both the declared assignment
and the swapped assignment remain constructible under the available gauge,
current, and flag constraints.  The rule is still a dynamical Higgs-boundary
coupling premise.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_laplacian_from_nilpotent,
    nilpotent_flag_operator,
)
from clifford_3plus2_d5.universal_bath.quark_current_parity_selector import (
    quark_current_parity_selector_payload,
)
from clifford_3plus2_d5.universal_bath.quark_down_identity_veto import (
    quark_down_identity_veto_payload,
)
from clifford_3plus2_d5.universal_bath.quark_height_door import (
    DOWN_QUARK_DOOR,
    UP_QUARK_DOOR,
    down_higgs_door,
    hypercharge_forces_higgs_doors,
    neutral_higgs_components,
    swapped_repair_assignment_hypercharge_allowed,
    up_higgs_door,
)
from clifford_3plus2_d5.universal_bath.quark_height_orientation_bridge import (
    ORIENTATION_COUPLING_PREMISE,
    quark_height_orientation_bridge_payload,
)

RETARDED_FLAG_READOUT = "retarded_oriented_flag_N"
HERMITIAN_CLOSURE_READOUT = "hermitian_flag_laplacian_closure"


@dataclass(frozen=True)
class HiggsOrientationAssignment:
    """One candidate assignment of SM Higgs doors to repair readouts."""

    name: str
    up_door: str
    down_door: str
    up_readout: str
    down_readout: str
    hypercharge_allowed: bool
    neutral_components: bool
    odd_current_source_compatible: bool
    readout_objects_available: bool
    constructed_by_lookup: bool


@dataclass(frozen=True)
class QuarkHiggsOrientationCouplingPayload:
    """Session 24 Higgs-door orientation-coupling verdict."""

    final_verdict: str
    height_orientation_bridge_pass: bool
    current_parity_selector_pass: bool
    down_identity_veto_pass: bool
    hypercharge_forces_doors: bool
    neutral_higgs_components: bool
    endpoint_reflection: sp.Matrix
    endpoint_reflection_involutive: bool
    reflection_maps_flag_to_reverse_flag: bool
    hermitian_closure_reflection_invariant: bool
    reflection_maps_flag_to_closure: bool
    hermitian_closure_requires_extra_pairing: bool
    declared_assignment: HiggsOrientationAssignment
    swapped_assignment: HiggsOrientationAssignment
    available_constraints_select_unique_assignment: bool
    swapped_assignment_survives_controls: bool
    higgs_conjugation_supplies_reversal_not_closure: bool
    higgs_door_orientation_coupling_derived: bool
    remaining_orientation_premise: str
    interpretation: str


def matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices are exactly equal after simplification."""

    return all(sp.simplify(entry) == 0 for entry in left - right)


def endpoint_reflection_operator() -> sp.Matrix:
    """Return endpoint reflection on residual order ``(u,a,b)``."""

    return sp.Matrix(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )


def endpoint_reflection_involutive() -> bool:
    """Return whether endpoint reflection squares to the identity."""

    reflection = endpoint_reflection_operator()
    return matrix_equal(reflection * reflection, sp.eye(3))


def reflection_maps_flag_to_reverse_flag() -> bool:
    """Return whether endpoint reflection maps ``N`` to ``N.T``."""

    flag = nilpotent_flag_operator()
    reflection = endpoint_reflection_operator()
    return matrix_equal(reflection * flag * reflection, flag.T)


def hermitian_closure_reflection_invariant() -> bool:
    """Return whether the Hermitian flag closure is reflection-invariant."""

    closure = flag_laplacian_from_nilpotent(nilpotent_flag_operator())
    reflection = endpoint_reflection_operator()
    return matrix_equal(reflection * closure * reflection, closure)


def reflection_maps_flag_to_closure() -> bool:
    """Return whether endpoint reflection alone produces the Hermitian closure."""

    flag = nilpotent_flag_operator()
    closure = flag_laplacian_from_nilpotent(flag)
    reflection = endpoint_reflection_operator()
    return matrix_equal(reflection * flag * reflection, closure)


def hermitian_closure_requires_extra_pairing() -> bool:
    """Return whether the closure is not one of the oriented flag branches."""

    flag = nilpotent_flag_operator()
    closure = flag_laplacian_from_nilpotent(flag)
    return (
        not matrix_equal(closure, flag)
        and not matrix_equal(closure, flag.T)
        and matrix_equal(closure, flag * flag.T + flag.T * flag - (flag + flag.T))
    )


def readout_objects_available() -> bool:
    """Return whether the flag and closure readouts are already certified."""

    bridge = quark_height_orientation_bridge_payload()
    return (
        bridge.final_verdict == "QUARK_HEIGHT_ORIENTATION_BRIDGE_NOT_DERIVED_AUDIT"
        and bridge.oriented_nilpotent_matches_up_repair
        and bridge.hermitian_closure_matches_down_repair
        and bridge.down_readout_is_closure_of_up_flag
    )


def _assignment(name: str, up_readout: str, down_readout: str) -> HiggsOrientationAssignment:
    """Return one candidate Higgs-door assignment."""

    current = quark_current_parity_selector_payload()
    return HiggsOrientationAssignment(
        name=name,
        up_door=up_higgs_door().door,
        down_door=down_higgs_door().door,
        up_readout=up_readout,
        down_readout=down_readout,
        hypercharge_allowed=hypercharge_forces_higgs_doors(),
        neutral_components=neutral_higgs_components(),
        odd_current_source_compatible=(
            current.final_verdict == "QUARK_CURRENT_PARITY_SELECTOR_PASS"
            and current.current_parity_selects_b
        ),
        readout_objects_available=readout_objects_available(),
        constructed_by_lookup=True,
    )


def declared_assignment() -> HiggsOrientationAssignment:
    """Return the desired quark Higgs-door orientation assignment."""

    return _assignment(
        "declared",
        up_readout=RETARDED_FLAG_READOUT,
        down_readout=HERMITIAN_CLOSURE_READOUT,
    )


def swapped_assignment() -> HiggsOrientationAssignment:
    """Return the swapped repair-readout control assignment."""

    return _assignment(
        "swapped_control",
        up_readout=HERMITIAN_CLOSURE_READOUT,
        down_readout=RETARDED_FLAG_READOUT,
    )


def assignment_allowed_by_available_constraints(assignment: HiggsOrientationAssignment) -> bool:
    """Return whether current constraints allow an assignment."""

    return (
        assignment.up_door == UP_QUARK_DOOR
        and assignment.down_door == DOWN_QUARK_DOOR
        and assignment.hypercharge_allowed
        and assignment.neutral_components
        and assignment.odd_current_source_compatible
        and assignment.readout_objects_available
    )


def available_constraints_select_unique_assignment() -> bool:
    """Return whether current constraints select only the declared assignment."""

    assignments = (declared_assignment(), swapped_assignment())
    allowed = tuple(assignment for assignment in assignments if assignment_allowed_by_available_constraints(assignment))
    return len(allowed) == 1 and allowed[0].name == "declared"


def higgs_conjugation_supplies_reversal_not_closure() -> bool:
    """Return whether Higgs conjugation geometry supplies reversal but not closure."""

    return (
        reflection_maps_flag_to_reverse_flag()
        and hermitian_closure_reflection_invariant()
        and not reflection_maps_flag_to_closure()
        and hermitian_closure_requires_extra_pairing()
    )


def quark_higgs_orientation_coupling_payload() -> QuarkHiggsOrientationCouplingPayload:
    """Return the Session 24 Higgs-door orientation-coupling audit payload."""

    bridge = quark_height_orientation_bridge_payload()
    current = quark_current_parity_selector_payload()
    down = quark_down_identity_veto_payload()
    declared = declared_assignment()
    swapped = swapped_assignment()

    bridge_pass = bridge.final_verdict == "QUARK_HEIGHT_ORIENTATION_BRIDGE_NOT_DERIVED_AUDIT"
    current_pass = current.final_verdict == "QUARK_CURRENT_PARITY_SELECTOR_PASS"
    down_pass = down.final_verdict == "DOWN_IDENTITY_RETURN_VETO_RANK_FIVE_CONDITIONAL_PASS"
    hypercharge = hypercharge_forces_higgs_doors()
    neutral = neutral_higgs_components()
    involutive = endpoint_reflection_involutive()
    reverses_flag = reflection_maps_flag_to_reverse_flag()
    closure_invariant = hermitian_closure_reflection_invariant()
    reflection_to_closure = reflection_maps_flag_to_closure()
    closure_requires_pairing = hermitian_closure_requires_extra_pairing()
    declared_allowed = assignment_allowed_by_available_constraints(declared)
    swapped_allowed = assignment_allowed_by_available_constraints(swapped)
    unique = available_constraints_select_unique_assignment()
    reversal_not_closure = higgs_conjugation_supplies_reversal_not_closure()
    orientation_derived = unique and declared_allowed and not swapped_allowed

    checks_pass = (
        bridge_pass
        and current_pass
        and down_pass
        and hypercharge
        and neutral
        and swapped_repair_assignment_hypercharge_allowed()
        and involutive
        and reverses_flag
        and closure_invariant
        and not reflection_to_closure
        and closure_requires_pairing
        and declared_allowed
        and swapped_allowed
        and not unique
        and reversal_not_closure
        and not orientation_derived
        and bridge.remaining_orientation_premise == ORIENTATION_COUPLING_PREMISE
    )

    if checks_pass:
        final_verdict = "QUARK_HIGGS_ORIENTATION_COUPLING_NOT_DERIVED_AUDIT"
        interpretation = (
            "The SM charges force H_tilde/H door labels, the selected-S2 odd "
            "current theorem supplies the b source, and the depth-scar "
            "certificate supplies both the retarded flag and its Hermitian "
            "closure.  Endpoint reflection maps N to N.T and leaves the "
            "Hermitian closure invariant, but it does not turn N into the "
            "closure.  The swapped assignment therefore survives the same "
            "available constraints.  The Higgs-door orientation coupling "
            "remains a dynamical boundary premise."
        )
    else:
        final_verdict = "QUARK_HIGGS_ORIENTATION_COUPLING_AUDIT_KILL"
        interpretation = (
            "The Higgs-door, current-source, down-veto, reflection, assignment, "
            "or negative-control checks failed."
        )

    return QuarkHiggsOrientationCouplingPayload(
        final_verdict=final_verdict,
        height_orientation_bridge_pass=bridge_pass,
        current_parity_selector_pass=current_pass,
        down_identity_veto_pass=down_pass,
        hypercharge_forces_doors=hypercharge,
        neutral_higgs_components=neutral,
        endpoint_reflection=endpoint_reflection_operator(),
        endpoint_reflection_involutive=involutive,
        reflection_maps_flag_to_reverse_flag=reverses_flag,
        hermitian_closure_reflection_invariant=closure_invariant,
        reflection_maps_flag_to_closure=reflection_to_closure,
        hermitian_closure_requires_extra_pairing=closure_requires_pairing,
        declared_assignment=declared,
        swapped_assignment=swapped,
        available_constraints_select_unique_assignment=unique,
        swapped_assignment_survives_controls=swapped_allowed,
        higgs_conjugation_supplies_reversal_not_closure=reversal_not_closure,
        higgs_door_orientation_coupling_derived=orientation_derived,
        remaining_orientation_premise=ORIENTATION_COUPLING_PREMISE,
        interpretation=interpretation,
    )
