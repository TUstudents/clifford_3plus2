"""Scan runners for the leptonic bridge laboratory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from clifford_3plus2_d5.lepton.bloch import iter_lab_a_bloch_candidates
from clifford_3plus2_d5.lepton.lab_b import (
    iter_lab_b_domain_wall_candidates,
    iter_lab_b_physical_domain_wall_candidates,
    iter_lab_b_strict_candidates,
    iter_lab_b_structural_wall_candidates,
)
from clifford_3plus2_d5.lepton.lab_a_wall import iter_lab_a_wall_candidates
from clifford_3plus2_d5.lepton.primitives import iter_lab_a_onsite_candidates
from clifford_3plus2_d5.lepton.profiles import (
    lab_a_profile,
    lab_b_domain_wall_profile,
    lab_b_physical_domain_wall_profile,
    lab_b_structural_profile,
    lab_b_structural_wall_profile,
)
from clifford_3plus2_d5.lepton.verdict import RuleToVerdictV2Result, rule_to_verdict_v2


@dataclass(frozen=True)
class LabAScanRow:
    rule_name: str
    verdict: str
    reason: str
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    center_dimension: int
    central_j_candidate_count: int
    primitive_classes: tuple[str, ...]
    block_dimensions: tuple[int, ...]
    block_commutativity: tuple[bool, ...]
    load_bearing_qca_bridge: bool
    mechanism: str = "on_site"
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LabBScanRow:
    rule_name: str
    verdict: str
    reason: str
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    center_dimension: int
    central_j_candidate_count: int
    idempotent_verdict: str
    commutant_verdict: str
    block_dimensions: tuple[int, ...]
    block_commutativity: tuple[bool, ...]
    load_bearing_qca_bridge: bool
    load_bearing_domain_wall_candidate: bool
    mechanism: str
    metadata: tuple[tuple[str, str], ...] = ()


def lab_a_scan_row(
    rule_name: str,
    result: RuleToVerdictV2Result,
    *,
    mechanism: str = "on_site",
    metadata: Sequence[tuple[str, str]] = (),
) -> LabAScanRow:
    return LabAScanRow(
        rule_name=rule_name,
        verdict=result.verdict.value,
        reason=result.reason,
        generated_algebra_dimension=result.generated_algebra_dimension,
        generated_algebra_closed=result.generated_algebra_closed,
        center_dimension=result.center_dimension,
        central_j_candidate_count=len(result.central_j_candidates),
        primitive_classes=tuple(item.value for item in result.primitive_classes),
        block_dimensions=result.block_dimensions,
        block_commutativity=result.block_commutativity,
        load_bearing_qca_bridge=result.load_bearing_qca_bridge,
        mechanism=mechanism,
        metadata=tuple(metadata),
    )


def lab_b_scan_row(
    rule_name: str,
    result: RuleToVerdictV2Result,
    *,
    mechanism: str,
    metadata: Sequence[tuple[str, str]] = (),
) -> LabBScanRow:
    return LabBScanRow(
        rule_name=rule_name,
        verdict=result.verdict.value,
        reason=result.reason,
        generated_algebra_dimension=result.generated_algebra_dimension,
        generated_algebra_closed=result.generated_algebra_closed,
        center_dimension=result.center_dimension,
        central_j_candidate_count=len(result.central_j_candidates),
        idempotent_verdict=result.idempotent_verdict.value,
        commutant_verdict=result.commutant_verdict.value,
        block_dimensions=result.block_dimensions,
        block_commutativity=result.block_commutativity,
        load_bearing_qca_bridge=result.load_bearing_qca_bridge,
        load_bearing_domain_wall_candidate=result.load_bearing_domain_wall_candidate,
        mechanism=mechanism,
        metadata=tuple(metadata),
    )


def run_lab_a_onsite_scan(
    *,
    max_candidates: int | None = None,
    max_depth: int = 2,
    angle_orders: Sequence[int] = (2, 3, 4),
) -> tuple[LabAScanRow, ...]:
    profile = lab_a_profile()
    rows = []
    for candidate in iter_lab_a_onsite_candidates(
        max_depth=max_depth,
        angle_orders=angle_orders,
        max_candidates=max_candidates,
    ):
        result = rule_to_verdict_v2(candidate.layers, profile)
        rows.append(lab_a_scan_row(candidate.name, result, mechanism="on_site"))
    return tuple(rows)


def run_lab_b_strict_scan(
    *,
    max_candidates: int | None = None,
) -> tuple[LabBScanRow, ...]:
    profile = lab_b_structural_profile()
    rows = []
    for candidate in iter_lab_b_strict_candidates(max_candidates=max_candidates):
        result = rule_to_verdict_v2(candidate.layers, profile)
        rows.append(
            lab_b_scan_row(
                candidate.name,
                result,
                mechanism="strict",
                metadata=candidate.metadata,
            )
        )
    return tuple(rows)


def run_lab_b_structural_wall_scan(
    *,
    max_candidates: int | None = None,
    max_pairs: int | None = None,
    max_transitions_per_pair: int = 2,
    include_generic: bool = False,
) -> tuple[LabBScanRow, ...]:
    profile = lab_b_structural_wall_profile()
    rows = []
    for candidate in iter_lab_b_structural_wall_candidates(
        max_candidates=max_candidates,
        max_pairs=max_pairs,
        max_transitions_per_pair=max_transitions_per_pair,
        include_generic=include_generic,
    ):
        result = rule_to_verdict_v2(
            candidate.layers,
            profile,
            wall_context=candidate.wall_context,
        )
        rows.append(
            lab_b_scan_row(
                candidate.name,
                result,
                mechanism="structural_wall",
                metadata=candidate.metadata,
            )
        )
    return tuple(rows)


def run_lab_b_domain_wall_scan(
    *,
    max_candidates: int | None = None,
    max_pairs: int | None = None,
    max_transitions_per_pair: int = 2,
    include_generic: bool = False,
    verify_center_exact: bool = True,
) -> tuple[LabBScanRow, ...]:
    profile = lab_b_domain_wall_profile()
    if not verify_center_exact:
        profile = replace(profile, verify_known_center_basis_exact=False)
    rows = []
    for candidate in iter_lab_b_domain_wall_candidates(
        max_candidates=max_candidates,
        max_pairs=max_pairs,
        max_transitions_per_pair=max_transitions_per_pair,
        include_generic=include_generic,
    ):
        result = rule_to_verdict_v2(
            candidate.layers,
            profile,
            wall_context=candidate.wall_context,
        )
        rows.append(
            lab_b_scan_row(
                candidate.name,
                result,
                mechanism="domain_wall",
                metadata=candidate.metadata,
            )
        )
    return tuple(rows)


def run_lab_b_physical_domain_wall_scan(
    *,
    max_candidates: int | None = None,
    max_pairs: int | None = None,
    max_transitions_per_pair: int = 2,
    include_generic: bool = False,
) -> tuple[LabBScanRow, ...]:
    profile = lab_b_physical_domain_wall_profile()
    rows = []
    for candidate in iter_lab_b_physical_domain_wall_candidates(
        max_candidates=max_candidates,
        max_pairs=max_pairs,
        max_transitions_per_pair=max_transitions_per_pair,
        include_generic=include_generic,
    ):
        result = rule_to_verdict_v2(
            candidate.layers,
            profile,
            wall_context=candidate.wall_context,
        )
        rows.append(
            lab_b_scan_row(
                candidate.name,
                result,
                mechanism="physical_domain_wall",
                metadata=candidate.metadata,
            )
        )
    return tuple(rows)


def run_lab_a_bloch_scan(
    *,
    max_candidates: int | None = None,
    periods: Sequence[int] = (3, 4, 6),
    winding_values: Sequence[int] = (-1, 1, 2),
    onsite_kinds: Sequence[str] = ("identity", "mode_swap"),
) -> tuple[LabAScanRow, ...]:
    profile = lab_a_profile()
    rows = []
    for candidate in iter_lab_a_bloch_candidates(
        periods=periods,
        winding_values=winding_values,
        onsite_kinds=onsite_kinds,
        max_candidates=max_candidates,
    ):
        result = rule_to_verdict_v2(candidate.layers, profile)
        rows.append(
            lab_a_scan_row(
                candidate.name,
                result,
                mechanism="bloch",
                metadata=candidate.metadata,
            )
        )
    return tuple(rows)


def run_lab_a_wall_scan(
    *,
    max_candidates: int | None = None,
    max_pairs: int | None = None,
    max_transitions_per_pair: int = 2,
    angle_orders: Sequence[int] = (2, 3, 4),
    include_generic: bool = False,
) -> tuple[LabAScanRow, ...]:
    profile = lab_a_profile()
    rows = []
    for candidate in iter_lab_a_wall_candidates(
        angle_orders=angle_orders,
        max_pairs=max_pairs,
        max_transitions_per_pair=max_transitions_per_pair,
        max_candidates=max_candidates,
        include_generic=include_generic,
    ):
        result = rule_to_verdict_v2(candidate.layers, profile)
        rows.append(
            lab_a_scan_row(
                candidate.name,
                result,
                mechanism="wall_same_spectrum",
                metadata=candidate.metadata,
            )
        )
    return tuple(rows)
