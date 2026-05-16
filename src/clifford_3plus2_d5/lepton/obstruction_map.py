"""Consolidated obstruction map for the lepton laboratory.

The entries in this module are intentionally static. They summarize the
session reports and stable lepton-local tests without rerunning expensive
SymPy scans or wall audits.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum


class ObstructionStatus(StrEnum):
    CONTROL_POSITIVE = "control_positive"
    TOY_POSITIVE = "toy_positive"
    DOMAIN_WALL_POSITIVE = "domain_wall_positive"
    PHYSICAL_NEGATIVE = "physical_negative"
    SEARCH_NEGATIVE = "search_negative"
    OPEN = "open"


class ObstructionMode(StrEnum):
    CENTER_TOO_SMALL = "center_too_small"
    CENTER_TOO_LARGE = "center_too_large"
    RANK_ONE_LOCKING = "rank_one_locking"
    IRREDUCIBLE_NO_SPLIT = "irreducible_no_split"
    MULTIPLE_J_ORBITS = "multiple_j_orbits"
    SYNTHETIC_ONLY = "synthetic_only"
    PHYSICAL_PROMOTION_EXTRA_CENTER = "physical_promotion_extra_center"
    NOT_TESTED = "not_tested"
    NONE_CONTROL = "none_control"
    NONE_TOY = "none_toy"


@dataclass(frozen=True)
class ObstructionEntry:
    name: str
    frame: str
    dimension: str
    target: str
    status: ObstructionStatus
    obstruction: ObstructionMode
    caveat: str
    load_bearing_qca_bridge: bool = False
    load_bearing_domain_wall_candidate: bool = False


def obstruction_entries() -> tuple[ObstructionEntry, ...]:
    return (
        ObstructionEntry(
            name="lab_a_clock_plane_r4",
            frame="real_carrier_clock_plane",
            dimension="R4",
            target="M_2(C) closure with central clock J",
            status=ObstructionStatus.TOY_POSITIVE,
            obstruction=ObstructionMode.NONE_TOY,
            caveat="Positive is clock-plane-aware, not J-blind; primitives encode the clock planes.",
        ),
        ObstructionEntry(
            name="lab_b_strict_r6",
            frame="real_carrier_fixed_split",
            dimension="R6",
            target="global (2,4) split with unique +/-J",
            status=ObstructionStatus.SEARCH_NEGATIVE,
            obstruction=ObstructionMode.MULTIPLE_J_ORBITS,
            caveat="Reproduces Route 1: center C + C gives four central J candidates, not a forced pair.",
        ),
        ObstructionEntry(
            name="lab_b_structural_wall_r6",
            frame="real_carrier_structural_wall",
            dimension="R6",
            target="global (2,4) split with wall layer",
            status=ObstructionStatus.SEARCH_NEGATIVE,
            obstruction=ObstructionMode.MULTIPLE_J_ORBITS,
            caveat="Non-translation-invariant wall preserving the global split still reproduces the multiple-J outcome.",
        ),
        ObstructionEntry(
            name="lab_b_internal_domain_wall_r6",
            frame="real_carrier_internal_domain_wall",
            dimension="R6",
            target="side-local sign lock with central R-complex structure",
            status=ObstructionStatus.DOMAIN_WALL_POSITIVE,
            obstruction=ObstructionMode.NONE_TOY,
            caveat="Internal single-carrier toy candidate; not the original global bridge contract.",
            load_bearing_domain_wall_candidate=True,
        ),
        ObstructionEntry(
            name="lab_b_physical_domain_wall_r12",
            frame="real_carrier_physical_domain_wall",
            dimension="R12",
            target="two-site physical promotion of the domain-wall sign lock",
            status=ObstructionStatus.PHYSICAL_NEGATIVE,
            obstruction=ObstructionMode.PHYSICAL_PROMOTION_EXTRA_CENTER,
            caveat="Exact audit finds center dimension 8 rather than the intended dimension 2.",
        ),
        ObstructionEntry(
            name="complex_c3_synthetic_split",
            frame="complex_split_first_fixed",
            dimension="C3",
            target="C + M_2(C) split",
            status=ObstructionStatus.CONTROL_POSITIVE,
            obstruction=ObstructionMode.NONE_CONTROL,
            caveat="Synthetic checker control; proves the complex verdict distinguishes split, locked, and irreducible cases.",
        ),
        ObstructionEntry(
            name="complex_c5_synthetic_split",
            frame="complex_split_first_fixed",
            dimension="C5",
            target="M_3(C) + M_2(C) fixed canonical split",
            status=ObstructionStatus.CONTROL_POSITIVE,
            obstruction=ObstructionMode.NONE_CONTROL,
            caveat="Synthetic fixed-frame witness; not a discovered mechanism.",
        ),
        ObstructionEntry(
            name="complex_c5_discovered_synthetic_split",
            frame="complex_split_first_discovered",
            dimension="C5",
            target="unique discovered rank (3,2) central split",
            status=ObstructionStatus.CONTROL_POSITIVE,
            obstruction=ObstructionMode.SYNTHETIC_ONLY,
            caveat="Canonical and conjugated synthetic witnesses pass; non-synthetic evidence is separate.",
        ),
        ObstructionEntry(
            name="complex_c5_phase_permutation_search",
            frame="complex_split_first_discovered",
            dimension="C5",
            target="non-synthetic phase-permutation discovered (3,2) split",
            status=ObstructionStatus.SEARCH_NEGATIVE,
            obstruction=ObstructionMode.CENTER_TOO_LARGE,
            caveat="Small order-2 panel gives not_solved center/idempotent cases plus irreducible no-split.",
        ),
        ObstructionEntry(
            name="complex_c5_monomial_search",
            frame="complex_split_first_discovered",
            dimension="C5",
            target="non-synthetic monomial discovered (3,2) split",
            status=ObstructionStatus.SEARCH_NEGATIVE,
            obstruction=ObstructionMode.IRREDUCIBLE_NO_SPLIT,
            caveat="Panel alternates between full irreducible behavior and rank-one locking; no split candidate.",
        ),
        ObstructionEntry(
            name="complex_c5_finite_order_search",
            frame="complex_split_first_discovered",
            dimension="C5",
            target="non-synthetic finite-order discovered (3,2) split",
            status=ObstructionStatus.SEARCH_NEGATIVE,
            obstruction=ObstructionMode.RANK_ONE_LOCKING,
            caveat="Finite-order permutation pairs mostly produce central rank-one locks; no split candidate.",
        ),
        ObstructionEntry(
            name="complex_c5_dense_conjugated_control",
            frame="complex_split_first_discovered_dense",
            dimension="C5",
            target="dense non-coordinate discovered (3,2) split",
            status=ObstructionStatus.CONTROL_POSITIVE,
            obstruction=ObstructionMode.SYNTHETIC_ONLY,
            caveat="Dense Householder conjugated synthetic control passes and fixed C5 rejects it; still engineered.",
        ),
        ObstructionEntry(
            name="complex_c5_dense_hadamard_search",
            frame="complex_split_first_discovered_dense",
            dimension="C5",
            target="non-synthetic dense sign-conjugate discovered (3,2) split",
            status=ObstructionStatus.SEARCH_NEGATIVE,
            obstruction=ObstructionMode.CENTER_TOO_LARGE,
            caveat="Dense sign-conjugate panel leaves centers too large; no split candidate.",
        ),
        ObstructionEntry(
            name="complex_c5_dense_householder_search",
            frame="complex_split_first_discovered_dense",
            dimension="C5",
            target="non-synthetic dense reflection discovered (3,2) split",
            status=ObstructionStatus.SEARCH_NEGATIVE,
            obstruction=ObstructionMode.CENTER_TOO_LARGE,
            caveat="Householder reflection pairs leave center dimension 10 in the small panel.",
        ),
        ObstructionEntry(
            name="complex_c5_dense_fourier_lite_search",
            frame="complex_split_first_discovered_dense",
            dimension="C5",
            target="non-synthetic dense permutation-conjugate discovered (3,2) split",
            status=ObstructionStatus.SEARCH_NEGATIVE,
            obstruction=ObstructionMode.RANK_ONE_LOCKING,
            caveat="Dense Fourier-lite panel reproduces irreducible/no-split or rank-one locking behavior.",
        ),
        ObstructionEntry(
            name="designed_locality_aware_complex_qca",
            frame="complex_split_first_locality_aware",
            dimension="C5 on finite lattice",
            target="derive (3,2) split from transport locality or gauge-bundle wall structure",
            status=ObstructionStatus.OPEN,
            obstruction=ObstructionMode.NOT_TESTED,
            caveat="Requires a designed QCA primitive; random finite matrix panels are not the right test.",
        ),
    )


def obstruction_status_counts() -> tuple[tuple[str, int], ...]:
    counts = Counter(entry.status.value for entry in obstruction_entries())
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def obstruction_mode_counts() -> tuple[tuple[str, int], ...]:
    counts = Counter(entry.obstruction.value for entry in obstruction_entries())
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def obstruction_summary() -> dict[str, object]:
    entries = obstruction_entries()
    return {
        "entry_count": len(entries),
        "status_counts": obstruction_status_counts(),
        "obstruction_counts": obstruction_mode_counts(),
        "load_bearing_qca_bridge_count": sum(
            1 for entry in entries if entry.load_bearing_qca_bridge
        ),
        "domain_wall_candidate_count": sum(
            1 for entry in entries if entry.load_bearing_domain_wall_candidate
        ),
        "open_entries": tuple(
            entry.name for entry in entries if entry.status == ObstructionStatus.OPEN
        ),
    }
