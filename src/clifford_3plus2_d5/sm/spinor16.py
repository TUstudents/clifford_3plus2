"""Guarded Spin(10) spinor reconstruction from the prior J-first data."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from clifford_3plus2_d5.algebra.projectors import projector_pair_check_passed
from clifford_3plus2_d5.algebra.real_carrier import (
    carrier_identities,
    phase_1_check_passed,
    standard_real_carrier,
)
from clifford_3plus2_d5.branching import (
    BranchingSector,
    SECTOR_LABELS,
    branching_check_passed,
    branching_table,
    total_multiplicity,
)
from clifford_3plus2_d5.exterior import ExteriorBasisElement, even_basis_3plus2
from clifford_3plus2_d5.search.addressability import structural_split_certificate
from clifford_3plus2_d5.sm.hypercharge import format_fraction, hypercharge, hypercharge_check_passed


Spinor16Verdict = Literal["derived_from_qca", "candidate_only", "falsified"]


@dataclass(frozen=True)
class ComplexMode:
    index: int
    real_basis: tuple[str, str]
    block: str


@dataclass(frozen=True)
class SpinorBasisState:
    subset: tuple[int, ...]
    degree: int
    n3: int
    n2: int
    sector: tuple[int, int]
    hypercharge: Fraction
    label: str


@dataclass(frozen=True)
class Spinor16Certificate:
    spinor16_dimension: int
    degree_dimensions: dict[int, int]
    sector_count: int
    hypercharge_check_passed: bool
    branching_table_check_passed: bool
    uses_phase_1_real_carrier: bool
    uses_phase_1_j: bool
    uses_phase_1_projectors: bool
    uses_phase_3_split_candidate: bool
    uses_existing_j_and_split: bool
    introduces_new_complex_structure: bool
    introduces_new_3plus2_split: bool
    qca_derives_spinor_inputs: bool
    spinor16_verdict: Spinor16Verdict
    load_bearing_qca_bridge: bool


def derived_complex_modes() -> tuple[ComplexMode, ...]:
    carrier = standard_real_carrier()
    return tuple(
        ComplexMode(
            index=index,
            real_basis=(carrier.basis[index - 1], carrier.basis[carrier.mode_dimension + index - 1]),
            block="C3" if index <= 3 else "C2",
        )
        for index in range(1, carrier.mode_dimension + 1)
    )


def even_spinor_basis() -> tuple[SpinorBasisState, ...]:
    return tuple(_basis_state(element) for element in even_basis_3plus2())


def _basis_state(element: ExteriorBasisElement) -> SpinorBasisState:
    return SpinorBasisState(
        subset=element.subset,
        degree=element.degree,
        n3=element.n3,
        n2=element.n2,
        sector=element.sector,
        hypercharge=hypercharge(element.n3, element.n2),
        label=SECTOR_LABELS[element.sector],
    )


def degree_dimensions() -> dict[int, int]:
    counts: dict[int, int] = {}
    for state in even_spinor_basis():
        counts[state.degree] = counts.get(state.degree, 0) + 1
    return counts


def hypercharge_operator() -> tuple[Fraction, ...]:
    return tuple(state.hypercharge for state in even_spinor_basis())


def spinor16_sectors() -> tuple[BranchingSector, ...]:
    return branching_table()


def spinor16_table_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "n3": sector.n3,
            "n2": sector.n2,
            "multiplicity": sector.multiplicity,
            "hypercharge": format_fraction(sector.hypercharge),
            "label": sector.label,
        }
        for sector in spinor16_sectors()
    )


def spinor16_certificate(*, qca_derives_spinor_inputs: bool = False) -> Spinor16Certificate:
    identities = carrier_identities()
    uses_phase_1_real_carrier = phase_1_check_passed()
    uses_phase_1_j = bool(
        identities["j_squared_minus_identity"]
        and identities["j_orthogonal"]
        and identities["j_determinant"] == 1
    )
    uses_phase_1_projectors = projector_pair_check_passed()
    split_certificate = structural_split_certificate()
    uses_phase_3_split_candidate = (
        split_certificate.projector_identities_passed
        and split_certificate.projectors_commute_with_j
        and split_certificate.addressability_algebra_safe
    )
    spinor_dimension = len(even_spinor_basis())
    hypercharge_passed = hypercharge_check_passed()
    branching_passed = branching_check_passed()
    uses_existing = (
        uses_phase_1_real_carrier
        and uses_phase_1_j
        and uses_phase_1_projectors
        and uses_phase_3_split_candidate
    )
    introduces_new_complex_structure = False
    introduces_new_3plus2_split = False
    falsified = not (
        spinor_dimension == 16
        and degree_dimensions() == {0: 1, 2: 10, 4: 5}
        and total_multiplicity() == 16
        and hypercharge_passed
        and branching_passed
        and uses_existing
        and not introduces_new_complex_structure
        and not introduces_new_3plus2_split
    )
    if falsified:
        verdict: Spinor16Verdict = "falsified"
    elif qca_derives_spinor_inputs:
        verdict = "derived_from_qca"
    else:
        verdict = "candidate_only"

    return Spinor16Certificate(
        spinor16_dimension=spinor_dimension,
        degree_dimensions=degree_dimensions(),
        sector_count=len(spinor16_sectors()),
        hypercharge_check_passed=hypercharge_passed,
        branching_table_check_passed=branching_passed,
        uses_phase_1_real_carrier=uses_phase_1_real_carrier,
        uses_phase_1_j=uses_phase_1_j,
        uses_phase_1_projectors=uses_phase_1_projectors,
        uses_phase_3_split_candidate=uses_phase_3_split_candidate,
        uses_existing_j_and_split=uses_existing,
        introduces_new_complex_structure=introduces_new_complex_structure,
        introduces_new_3plus2_split=introduces_new_3plus2_split,
        qca_derives_spinor_inputs=qca_derives_spinor_inputs,
        spinor16_verdict=verdict,
        load_bearing_qca_bridge=False,
    )


def certificate_to_dict(certificate: Spinor16Certificate) -> dict[str, object]:
    return {
        "spinor16_dimension": certificate.spinor16_dimension,
        "degree_dimensions": {
            str(degree): count for degree, count in sorted(certificate.degree_dimensions.items())
        },
        "sector_count": certificate.sector_count,
        "hypercharge_check_passed": certificate.hypercharge_check_passed,
        "branching_table_check_passed": certificate.branching_table_check_passed,
        "uses_phase_1_real_carrier": certificate.uses_phase_1_real_carrier,
        "uses_phase_1_j": certificate.uses_phase_1_j,
        "uses_phase_1_projectors": certificate.uses_phase_1_projectors,
        "uses_phase_3_split_candidate": certificate.uses_phase_3_split_candidate,
        "uses_existing_j_and_split": certificate.uses_existing_j_and_split,
        "introduces_new_complex_structure": certificate.introduces_new_complex_structure,
        "introduces_new_3plus2_split": certificate.introduces_new_3plus2_split,
        "qca_derives_spinor_inputs": certificate.qca_derives_spinor_inputs,
        "spinor16_table": list(spinor16_table_rows()),
        "spinor16_verdict": certificate.spinor16_verdict,
        "spinor16_check_passed": (
            certificate.spinor16_dimension == 16
            and certificate.hypercharge_check_passed
            and certificate.branching_table_check_passed
            and certificate.uses_existing_j_and_split
            and not certificate.introduces_new_complex_structure
            and not certificate.introduces_new_3plus2_split
        ),
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }
