"""Exact real carrier for the enhanced J-first attack."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import commutator, epsilon, identity, is_zero_matrix


@dataclass(frozen=True)
class RealCarrier:
    dimension: int
    mode_dimension: int
    basis: tuple[str, ...]
    metric: sp.Matrix
    complex_structure: sp.Matrix
    projector_3: sp.Matrix
    projector_2: sp.Matrix


def standard_basis(mode_dimension: int = 5) -> tuple[str, ...]:
    return tuple(
        [f"x_{index}" for index in range(1, mode_dimension + 1)]
        + [f"y_{index}" for index in range(1, mode_dimension + 1)]
    )


def clock_complex_structure(mode_dimension: int = 5) -> sp.Matrix:
    return sp.kronecker_product(epsilon(), identity(mode_dimension))


def split_projectors_3_2() -> tuple[sp.Matrix, sp.Matrix]:
    clock_identity = identity(2)
    pi3 = sp.diag(1, 1, 1, 0, 0)
    pi2 = sp.diag(0, 0, 0, 1, 1)
    return (
        sp.kronecker_product(clock_identity, pi3),
        sp.kronecker_product(clock_identity, pi2),
    )


def standard_real_carrier() -> RealCarrier:
    mode_dimension = 5
    dimension = 2 * mode_dimension
    projector_3, projector_2 = split_projectors_3_2()
    return RealCarrier(
        dimension=dimension,
        mode_dimension=mode_dimension,
        basis=standard_basis(mode_dimension),
        metric=identity(dimension),
        complex_structure=clock_complex_structure(mode_dimension),
        projector_3=projector_3,
        projector_2=projector_2,
    )


def carrier_identities(carrier: RealCarrier | None = None) -> dict[str, bool | int]:
    carrier = carrier or standard_real_carrier()
    dimension = carrier.dimension
    j = carrier.complex_structure
    p3 = carrier.projector_3
    p2 = carrier.projector_2
    metric = carrier.metric

    return {
        "dimension": dimension,
        "mode_dimension": carrier.mode_dimension,
        "metric_is_identity": metric == identity(dimension),
        "metric_is_real_symmetric": metric == metric.T
        and all(value.is_real for value in metric),
        "j_squared_minus_identity": j * j == -identity(dimension),
        "j_orthogonal": j.T * j == identity(dimension),
        "j_determinant": int(j.det()),
        "projector_sum_identity": p3 + p2 == identity(dimension),
        "projector_3_idempotent": p3 * p3 == p3,
        "projector_2_idempotent": p2 * p2 == p2,
        "projectors_orthogonal": p3 * p2 == sp.zeros(dimension),
        "projector_3_rank": p3.rank(),
        "projector_2_rank": p2.rank(),
        "projector_3_commutes_with_j": is_zero_matrix(commutator(j, p3)),
        "projector_2_commutes_with_j": is_zero_matrix(commutator(j, p2)),
    }


def phase_1_check_passed(carrier: RealCarrier | None = None) -> bool:
    identities = carrier_identities(carrier)
    return (
        identities["dimension"] == 10
        and identities["mode_dimension"] == 5
        and identities["metric_is_identity"] is True
        and identities["metric_is_real_symmetric"] is True
        and identities["j_squared_minus_identity"] is True
        and identities["j_orthogonal"] is True
        and identities["j_determinant"] == 1
        and identities["projector_sum_identity"] is True
        and identities["projector_3_idempotent"] is True
        and identities["projector_2_idempotent"] is True
        and identities["projectors_orthogonal"] is True
        and identities["projector_3_rank"] == 6
        and identities["projector_2_rank"] == 4
        and identities["projector_3_commutes_with_j"] is True
        and identities["projector_2_commutes_with_j"] is True
    )


def carrier_certificate() -> dict[str, object]:
    carrier = standard_real_carrier()
    identities = carrier_identities(carrier)
    return {
        "carrier": "R^2_clock tensor (R^3 plus R^2)",
        "basis": list(carrier.basis),
        **identities,
        "phase_1_real_carrier_check_passed": phase_1_check_passed(carrier),
        "qca_forces_j": False,
        "load_bearing_qca_bridge": False,
    }
