"""Session 16 quark normal-depth placement audit.

The depth-scar sidecar supplies a graph-native quark depth operator with
spectrum ``{0,2,6}``.  That is not the same datum as the universal-bath source
dictionary field ``normal_depth`` for ``V_u`` and ``V_d``.  This session keeps
those two notions separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.microscopic_locality import (
    HEIGHTS,
    PORTS,
    microscopic_locality_payload,
)
from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    nilpotent_flag_operator,
    nilpotent_flag_scar_payload,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    defect_mode_depths,
    depth_scar_operator,
    hand_written_diagonal_depth_operator,
)
from clifford_3plus2_d5.universal_bath.quark_source_assembly import (
    QUARK_NORMAL_DEPTH_PREMISE,
    quark_source_assembly_payload,
    unresolved_quark_sources,
)


@dataclass(frozen=True)
class QuarkNormalDepthAuditPayload:
    """Session 16 normal-depth placement verdict."""

    final_verdict: str
    quark_source_assembly_pass: bool
    nilpotent_flag_pass: bool
    microscopic_locality_pass: bool
    port_height_filtration: dict[str, int]
    port_height_filtration_microscopically_derived: bool
    one_tick_geometry_microscopically_derived: bool
    repair_flag: sp.Matrix
    depth_operator_port_basis: sp.Matrix
    hand_written_diagonal_control: sp.Matrix
    depth_operator_not_diagonal_in_port_basis: bool
    defect_mode_depths: tuple[sp.Expr, ...]
    defect_mode_depths_match_scar: bool
    doubled_port_heights: tuple[int, ...]
    port_heights_are_not_depth_spectrum: bool
    up_dictionary_normal_depth: int | None
    down_dictionary_normal_depth: int | None
    dictionary_depths_still_unfrozen: bool
    graph_depths_do_not_freeze_source_depths: bool
    normal_depth_premise_still_open: bool
    interpretation: str


def port_height_filtration() -> dict[str, int]:
    """Return the depth-scar port-height filtration."""

    return dict(zip(PORTS, HEIGHTS, strict=True))


def doubled_port_heights() -> tuple[int, ...]:
    """Return BCC-doubled port heights in ``(u,a,b)`` order."""

    return tuple(2 * height for height in HEIGHTS)


def depth_operator_not_diagonal_in_port_basis() -> bool:
    """Return whether the path-scar depth is not a hand-written port diagonal."""

    return sp.simplify(depth_scar_operator() - hand_written_diagonal_depth_operator()) != sp.zeros(3, 3)


def port_heights_are_not_depth_spectrum() -> bool:
    """Return whether port heights should not be confused with normal-mode depths."""

    return doubled_port_heights() != tuple(int(depth) for depth in EXPECTED_DEPTH_SPECTRUM)


def quark_normal_depth_audit_payload() -> QuarkNormalDepthAuditPayload:
    """Return the Session 16 quark normal-depth placement audit payload."""

    source = quark_source_assembly_payload()
    flag = nilpotent_flag_scar_payload()
    locality = microscopic_locality_payload()
    sources = unresolved_quark_sources()
    up_source = sources["up_quark_boundary_source"]
    down_source = sources["down_quark_boundary_source"]
    mode_depths = defect_mode_depths()

    source_pass = source.final_verdict == "QUARK_SOURCE_FREEZE_NOT_DERIVED_AUDIT"
    flag_pass = flag.final_verdict == "NILPOTENT_FLAG_SCAR_ORIGIN_PASS"
    locality_pass = locality.final_verdict == "MICROSCOPIC_LOCALITY_MINIMALITY_CONDITIONAL_PASS"
    mode_depths_match = mode_depths == EXPECTED_DEPTH_SPECTRUM
    not_diagonal = depth_operator_not_diagonal_in_port_basis()
    heights_not_spectrum = port_heights_are_not_depth_spectrum()
    dictionary_unfrozen = up_source.normal_depth is None and down_source.normal_depth is None
    graph_depths_not_sources = (
        mode_depths_match
        and not_diagonal
        and heights_not_spectrum
        and dictionary_unfrozen
    )
    premise_open = QUARK_NORMAL_DEPTH_PREMISE in source.unresolved_premises

    checks_pass = (
        source_pass
        and flag_pass
        and locality_pass
        and port_height_filtration() == {"u": 0, "a": 1, "b": 2}
        and not locality.height_filtration_microscopically_derived
        and not locality.one_tick_boundary_geometry_microscopically_derived
        and nilpotent_flag_operator() == sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        and mode_depths_match
        and not_diagonal
        and heights_not_spectrum
        and dictionary_unfrozen
        and graph_depths_not_sources
        and premise_open
    )

    if checks_pass:
        final_verdict = "QUARK_NORMAL_DEPTH_PLACEMENT_NOT_DERIVED_AUDIT"
        interpretation = (
            "The depth-scar theorem supplies the conditional height filtration "
            "h(u,a,b)=(0,1,2), the length-3 nilpotent flag b->a->u, and the "
            "normal-mode depth spectrum {0,2,6}.  But these are graph data, "
            "not source dictionary placements.  The depth operator is not a "
            "diagonal port-depth assignment, the doubled port heights are "
            "(0,2,4) rather than {0,2,6}, and the quark dictionary still has "
            "normal_depth=None for both V_u and V_d.  Therefore the normal "
            "depth blocker remains open, now sharpened to a source-placement "
            "problem rather than a depth-scar algebra problem."
        )
    else:
        final_verdict = "QUARK_NORMAL_DEPTH_AUDIT_KILL"
        interpretation = (
            "The quark normal-depth audit failed a source-assembly, depth-scar, "
            "locality, spectrum, or non-freeze control."
        )

    return QuarkNormalDepthAuditPayload(
        final_verdict=final_verdict,
        quark_source_assembly_pass=source_pass,
        nilpotent_flag_pass=flag_pass,
        microscopic_locality_pass=locality_pass,
        port_height_filtration=port_height_filtration(),
        port_height_filtration_microscopically_derived=locality.height_filtration_microscopically_derived,
        one_tick_geometry_microscopically_derived=locality.one_tick_boundary_geometry_microscopically_derived,
        repair_flag=nilpotent_flag_operator(),
        depth_operator_port_basis=depth_scar_operator(),
        hand_written_diagonal_control=hand_written_diagonal_depth_operator(),
        depth_operator_not_diagonal_in_port_basis=not_diagonal,
        defect_mode_depths=mode_depths,
        defect_mode_depths_match_scar=mode_depths_match,
        doubled_port_heights=doubled_port_heights(),
        port_heights_are_not_depth_spectrum=heights_not_spectrum,
        up_dictionary_normal_depth=up_source.normal_depth,
        down_dictionary_normal_depth=down_source.normal_depth,
        dictionary_depths_still_unfrozen=dictionary_unfrozen,
        graph_depths_do_not_freeze_source_depths=graph_depths_not_sources,
        normal_depth_premise_still_open=premise_open,
        interpretation=interpretation,
    )
