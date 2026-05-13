"""Exact projector helpers for the structural 3+2 split."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import commutator, identity, is_zero_matrix
from clifford_3plus2_d5.algebra.real_carrier import RealCarrier, standard_real_carrier


@dataclass(frozen=True)
class ProjectorPairIdentities:
    dimension: int
    projector_3_idempotent: bool
    projector_2_idempotent: bool
    projector_sum_identity: bool
    projectors_orthogonal: bool
    projector_3_rank: int
    projector_2_rank: int
    projector_3_commutes_with_j: bool
    projector_2_commutes_with_j: bool


def projector_pair_identities(
    projector_3: sp.Matrix | None = None,
    projector_2: sp.Matrix | None = None,
    *,
    complex_structure: sp.Matrix | None = None,
    carrier: RealCarrier | None = None,
) -> ProjectorPairIdentities:
    carrier = carrier or standard_real_carrier()
    projector_3 = projector_3 if projector_3 is not None else carrier.projector_3
    projector_2 = projector_2 if projector_2 is not None else carrier.projector_2
    complex_structure = (
        complex_structure if complex_structure is not None else carrier.complex_structure
    )
    dimension = carrier.dimension

    return ProjectorPairIdentities(
        dimension=dimension,
        projector_3_idempotent=projector_3 * projector_3 == projector_3,
        projector_2_idempotent=projector_2 * projector_2 == projector_2,
        projector_sum_identity=projector_3 + projector_2 == identity(dimension),
        projectors_orthogonal=projector_3 * projector_2 == sp.zeros(dimension),
        projector_3_rank=projector_3.rank(),
        projector_2_rank=projector_2.rank(),
        projector_3_commutes_with_j=is_zero_matrix(commutator(complex_structure, projector_3)),
        projector_2_commutes_with_j=is_zero_matrix(commutator(complex_structure, projector_2)),
    )


def projector_pair_check_passed(identities: ProjectorPairIdentities | None = None) -> bool:
    identities = identities or projector_pair_identities()
    return (
        identities.dimension == 10
        and identities.projector_3_idempotent
        and identities.projector_2_idempotent
        and identities.projector_sum_identity
        and identities.projectors_orthogonal
        and identities.projector_3_rank == 6
        and identities.projector_2_rank == 4
        and identities.projector_3_commutes_with_j
        and identities.projector_2_commutes_with_j
    )


def projectors_commute_with_j(identities: ProjectorPairIdentities | None = None) -> bool:
    identities = identities or projector_pair_identities()
    return identities.projector_3_commutes_with_j and identities.projector_2_commutes_with_j


def mode_axis_projector(index: int, *, mode_dimension: int = 5) -> sp.Matrix:
    """Project onto one complex mode, represented as a real two-plane."""

    if not 0 <= index < mode_dimension:
        raise ValueError("mode axis index out of range")

    diagonal = [0] * mode_dimension
    diagonal[index] = 1
    return sp.kronecker_product(identity(2), sp.diag(*diagonal))


def color_axis_projectors() -> tuple[sp.Matrix, ...]:
    return tuple(mode_axis_projector(index) for index in range(3))


def weak_axis_projectors() -> tuple[sp.Matrix, ...]:
    return tuple(mode_axis_projector(index) for index in range(3, 5))
