"""Carrier helpers for the leptonic bridge laboratory."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import clock_complex_structure, standard_basis


@dataclass(frozen=True)
class LeptonCarrier:
    dimension: int
    mode_dimension: int
    basis: tuple[str, ...]
    metric: sp.Matrix
    complex_structure: sp.Matrix
    target_projectors: tuple[sp.Matrix, ...] = ()


def lab_a_carrier() -> LeptonCarrier:
    mode_dimension = 2
    dimension = 2 * mode_dimension
    return LeptonCarrier(
        dimension=dimension,
        mode_dimension=mode_dimension,
        basis=standard_basis(mode_dimension),
        metric=identity(dimension),
        complex_structure=clock_complex_structure(mode_dimension),
    )


def lab_b_singlet_doublet_projectors() -> tuple[sp.Matrix, sp.Matrix]:
    """Return the R^6 singlet/doublet projectors with real ranks 2 and 4."""

    return lab_b_singlet_doublet_projectors_for_mode(0)


def lab_b_singlet_doublet_projectors_for_mode(singlet_mode: int) -> tuple[sp.Matrix, sp.Matrix]:
    """Return R^6 projectors for an arbitrary singlet mode index."""

    if not 0 <= singlet_mode < 3:
        raise ValueError("Lab B singlet mode must be in {0, 1, 2}")

    singlet_entries = [1 if mode == singlet_mode else 0 for mode in range(3)]
    doublet_entries = [1 - entry for entry in singlet_entries]
    singlet = sp.diag(*singlet_entries)
    doublet = sp.diag(*doublet_entries)
    clock_identity = identity(2)
    return (
        sp.kronecker_product(clock_identity, singlet),
        sp.kronecker_product(clock_identity, doublet),
    )


def lab_b_carrier() -> LeptonCarrier:
    mode_dimension = 3
    dimension = 2 * mode_dimension
    return LeptonCarrier(
        dimension=dimension,
        mode_dimension=mode_dimension,
        basis=standard_basis(mode_dimension),
        metric=identity(dimension),
        complex_structure=clock_complex_structure(mode_dimension),
        target_projectors=lab_b_singlet_doublet_projectors(),
    )


def lab_a_complex_structure() -> sp.Matrix:
    return lab_a_carrier().complex_structure


def lab_b_complex_structure() -> sp.Matrix:
    return lab_b_carrier().complex_structure


def lab_b_physical_wall_complex_structure() -> sp.Matrix:
    """Return the two-site R^12 clock complex structure for a physical wall."""

    j = lab_b_complex_structure()
    return sp.diag(j, j)


def lab_b_physical_wall_site_projectors() -> tuple[sp.Matrix, sp.Matrix]:
    """Return complementary left/right site projectors on R^6 ⊕ R^6."""

    zero = sp.zeros(6)
    one = identity(6)
    return (
        sp.Matrix.vstack(sp.Matrix.hstack(one, zero), sp.Matrix.hstack(zero, zero)),
        sp.Matrix.vstack(sp.Matrix.hstack(zero, zero), sp.Matrix.hstack(zero, one)),
    )


def lab_b_physical_wall_carrier() -> LeptonCarrier:
    mode_dimension = 6
    dimension = 2 * mode_dimension
    return LeptonCarrier(
        dimension=dimension,
        mode_dimension=mode_dimension,
        basis=tuple(f"L:{item}" for item in standard_basis(3))
        + tuple(f"R:{item}" for item in standard_basis(3)),
        metric=identity(dimension),
        complex_structure=lab_b_physical_wall_complex_structure(),
        target_projectors=lab_b_physical_wall_site_projectors(),
    )


def lab_a_carrier_identities() -> dict[str, bool | int]:
    carrier = lab_a_carrier()
    dimension = carrier.dimension
    one = identity(dimension)
    j = carrier.complex_structure
    return {
        "dimension": dimension,
        "mode_dimension": carrier.mode_dimension,
        "metric_is_identity": carrier.metric == one,
        "j_squared_minus_identity": j * j == -one,
        "j_orthogonal": j.T * j == one,
        "j_determinant": int(j.det()),
        "target_projector_count": len(carrier.target_projectors),
    }


def lab_a_carrier_check_passed() -> bool:
    identities = lab_a_carrier_identities()
    return (
        identities["dimension"] == 4
        and identities["mode_dimension"] == 2
        and identities["metric_is_identity"] is True
        and identities["j_squared_minus_identity"] is True
        and identities["j_orthogonal"] is True
        and identities["j_determinant"] == 1
        and identities["target_projector_count"] == 0
    )


def lab_b_carrier_identities() -> dict[str, bool | int | tuple[int, ...]]:
    carrier = lab_b_carrier()
    dimension = carrier.dimension
    one = identity(dimension)
    j = carrier.complex_structure
    projectors = carrier.target_projectors
    return {
        "dimension": dimension,
        "mode_dimension": carrier.mode_dimension,
        "metric_is_identity": carrier.metric == one,
        "j_squared_minus_identity": j * j == -one,
        "j_orthogonal": j.T * j == one,
        "j_determinant": int(j.det()),
        "target_projector_count": len(projectors),
        "target_projector_ranks": tuple(projector.rank() for projector in projectors),
        "target_projectors_sum_to_identity": sum(projectors, sp.zeros(dimension)) == one,
        "target_projectors_commute_with_j": all(projector * j == j * projector for projector in projectors),
    }


def lab_b_carrier_check_passed() -> bool:
    identities = lab_b_carrier_identities()
    return (
        identities["dimension"] == 6
        and identities["mode_dimension"] == 3
        and identities["metric_is_identity"] is True
        and identities["j_squared_minus_identity"] is True
        and identities["j_orthogonal"] is True
        and identities["j_determinant"] == 1
        and identities["target_projector_count"] == 2
        and identities["target_projector_ranks"] == (2, 4)
        and identities["target_projectors_sum_to_identity"] is True
        and identities["target_projectors_commute_with_j"] is True
    )


def lab_b_physical_wall_carrier_identities() -> dict[str, bool | int | tuple[int, ...]]:
    carrier = lab_b_physical_wall_carrier()
    dimension = carrier.dimension
    one = identity(dimension)
    j = carrier.complex_structure
    left, right = carrier.target_projectors
    return {
        "dimension": dimension,
        "mode_dimension": carrier.mode_dimension,
        "metric_is_identity": carrier.metric == one,
        "j_squared_minus_identity": j * j == -one,
        "j_orthogonal": j.T * j == one,
        "j_determinant": int(j.det()),
        "target_projector_count": len(carrier.target_projectors),
        "target_projector_ranks": tuple(projector.rank() for projector in carrier.target_projectors),
        "target_projectors_sum_to_identity": left + right == one,
        "target_projectors_are_orthogonal": left * right == sp.zeros(dimension),
        "target_projectors_commute_with_j": all(
            projector * j == j * projector for projector in carrier.target_projectors
        ),
    }


def lab_b_physical_wall_carrier_check_passed() -> bool:
    identities = lab_b_physical_wall_carrier_identities()
    return (
        identities["dimension"] == 12
        and identities["mode_dimension"] == 6
        and identities["metric_is_identity"] is True
        and identities["j_squared_minus_identity"] is True
        and identities["j_orthogonal"] is True
        and identities["j_determinant"] == 1
        and identities["target_projector_count"] == 2
        and identities["target_projector_ranks"] == (6, 6)
        and identities["target_projectors_sum_to_identity"] is True
        and identities["target_projectors_are_orthogonal"] is True
        and identities["target_projectors_commute_with_j"] is True
    )
