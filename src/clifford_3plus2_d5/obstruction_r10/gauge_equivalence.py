"""Gauge-equivalence checks for Route-1 compatible J sign choices.

This module checks the fixed-SM-data version of the relaxed standard. It does
not certify a broader equivalence that uses an outer Spin(10) automorphism or a
microscopic charge-conjugation primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from clifford_3plus2_d5.branching import BranchingSector, branching_table
from clifford_3plus2_d5.obstruction_r10.qca.floquet_alpha_noncommuting import (
    FloquetAlphaNoncommutingCandidate,
    floquet_alpha_noncommuting_candidates,
    floquet_alpha_noncommuting_completion_j_signs,
)


@dataclass(frozen=True, order=True)
class BranchingSignatureSector:
    n3: int
    n2: int
    multiplicity: int
    hypercharge: Fraction
    label: str


@dataclass(frozen=True)
class GaugeEquivalentJPattern:
    index: int
    pair_orientation_signs: tuple[int, ...]
    alpha_flip: int
    eta_flip: int
    global_pm_class: tuple[int, int]
    direct_hodge_preserves_even_chirality: bool
    direct_fixed_hypercharge_preserved: bool
    global_pm_fixed_hypercharge_preserved: bool
    intrinsic_branching_signature: tuple[BranchingSignatureSector, ...]
    fixed_hodge_signature: tuple[BranchingSignatureSector, ...]


@dataclass(frozen=True)
class GaugeEquivalenceCertificate:
    candidate_name: str
    standard_scope: str
    compatible_j_count: int
    global_pm_orbit_count: int
    intrinsic_branching_tables_match: bool
    fixed_sm_branching_tables_match_mod_global_pm: bool
    rule_generated_normalizer_orbit_certified: bool
    charge_conjugation_orbit_checked: bool
    relaxed_standard_supported: bool
    strict_standard_required: bool
    verdict: str
    patterns: tuple[GaugeEquivalentJPattern, ...]
    load_bearing_qca_bridge: bool = False


def _branching_signature() -> tuple[BranchingSignatureSector, ...]:
    return tuple(
        BranchingSignatureSector(
            n3=sector.n3,
            n2=sector.n2,
            multiplicity=sector.multiplicity,
            hypercharge=sector.hypercharge,
            label=sector.label,
        )
        for sector in branching_table()
    )


def _block_flips(
    candidate: FloquetAlphaNoncommutingCandidate,
    signs: tuple[int, ...],
) -> tuple[int, int]:
    alpha_flips = {
        signs[mode] * candidate.orientation_signs[mode] for mode in candidate.pattern.alpha_modes
    }
    eta_flips = {
        signs[mode] * candidate.orientation_signs[mode] for mode in candidate.pattern.eta_modes
    }
    if len(alpha_flips) != 1 or len(eta_flips) != 1:
        raise ValueError("compatible sign pattern is not constant on alpha/eta blocks")
    return next(iter(alpha_flips)), next(iter(eta_flips))


def _global_pm_class(alpha_flip: int, eta_flip: int) -> tuple[int, int]:
    if alpha_flip < 0:
        return (1, -eta_flip)
    return (1, eta_flip)


def _hodge_sector(sector: BranchingSector, *, alpha_flip: int, eta_flip: int) -> tuple[int, int]:
    n3 = 3 - sector.n3 if alpha_flip < 0 else sector.n3
    n2 = 2 - sector.n2 if eta_flip < 0 else sector.n2
    return n3, n2


def _preserves_even_chirality(*, alpha_flip: int, eta_flip: int) -> bool:
    flipped_dimension = (3 if alpha_flip < 0 else 0) + (2 if eta_flip < 0 else 0)
    return flipped_dimension % 2 == 0


def _fixed_hodge_signature(
    *,
    alpha_flip: int,
    eta_flip: int,
) -> tuple[BranchingSignatureSector, ...]:
    return tuple(
        sorted(
            BranchingSignatureSector(
                n3=n3,
                n2=n2,
                multiplicity=sector.multiplicity,
                hypercharge=sector.hypercharge,
                label=sector.label,
            )
            for sector in branching_table()
            for n3, n2 in (_hodge_sector(sector, alpha_flip=alpha_flip, eta_flip=eta_flip),)
        )
    )


def _fixed_hypercharge_preserved(*, alpha_flip: int, eta_flip: int) -> bool:
    if not _preserves_even_chirality(alpha_flip=alpha_flip, eta_flip=eta_flip):
        return False
    by_sector = {(sector.n3, sector.n2): sector for sector in _branching_signature()}
    for sector in branching_table():
        target = _hodge_sector(sector, alpha_flip=alpha_flip, eta_flip=eta_flip)
        reference = by_sector.get(target)
        if reference is None or reference.hypercharge != sector.hypercharge:
            return False
    return True


def _global_pm_fixed_hypercharge_preserved(*, alpha_flip: int, eta_flip: int) -> bool:
    return _fixed_hypercharge_preserved(
        alpha_flip=alpha_flip,
        eta_flip=eta_flip,
    ) or _fixed_hypercharge_preserved(
        alpha_flip=-alpha_flip,
        eta_flip=-eta_flip,
    )


def route1_gauge_equivalence_certificate(
    *,
    pattern_index: int = 0,
) -> GaugeEquivalenceCertificate:
    candidates = floquet_alpha_noncommuting_candidates(pattern_index=pattern_index)
    if not candidates:
        raise ValueError(f"unknown pattern index: {pattern_index}")
    candidate = candidates[0]
    intrinsic = _branching_signature()
    patterns = []
    for index in range(4):
        signs = floquet_alpha_noncommuting_completion_j_signs(candidate, j_index=index)
        alpha_flip, eta_flip = _block_flips(candidate, signs)
        patterns.append(
            GaugeEquivalentJPattern(
                index=index,
                pair_orientation_signs=signs,
                alpha_flip=alpha_flip,
                eta_flip=eta_flip,
                global_pm_class=_global_pm_class(alpha_flip, eta_flip),
                direct_hodge_preserves_even_chirality=_preserves_even_chirality(
                    alpha_flip=alpha_flip,
                    eta_flip=eta_flip,
                ),
                direct_fixed_hypercharge_preserved=_fixed_hypercharge_preserved(
                    alpha_flip=alpha_flip,
                    eta_flip=eta_flip,
                ),
                global_pm_fixed_hypercharge_preserved=_global_pm_fixed_hypercharge_preserved(
                    alpha_flip=alpha_flip,
                    eta_flip=eta_flip,
                ),
                intrinsic_branching_signature=intrinsic,
                fixed_hodge_signature=_fixed_hodge_signature(
                    alpha_flip=alpha_flip,
                    eta_flip=eta_flip,
                ),
            )
        )

    intrinsic_match = len({pattern.intrinsic_branching_signature for pattern in patterns}) == 1
    global_classes = {pattern.global_pm_class for pattern in patterns}
    fixed_match = all(pattern.global_pm_fixed_hypercharge_preserved for pattern in patterns)
    rule_orbit_certified = False
    relaxed_supported = (
        intrinsic_match and fixed_match and len(global_classes) == 1 and rule_orbit_certified
    )
    verdict = (
        "gauge_equivalent_bridge_standard_supported"
        if relaxed_supported
        else "strict_standard_required"
    )
    return GaugeEquivalenceCertificate(
        candidate_name=candidate.name,
        standard_scope="fixed_su3_su2_u1_hypercharge_no_charge_conjugation",
        compatible_j_count=len(patterns),
        global_pm_orbit_count=len(global_classes),
        intrinsic_branching_tables_match=intrinsic_match,
        fixed_sm_branching_tables_match_mod_global_pm=fixed_match,
        rule_generated_normalizer_orbit_certified=rule_orbit_certified,
        charge_conjugation_orbit_checked=False,
        relaxed_standard_supported=relaxed_supported,
        strict_standard_required=not relaxed_supported,
        verdict=verdict,
        patterns=tuple(patterns),
    )
