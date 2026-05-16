"""Checkerboard continuum tests for the Pati-Salam chiral-16 factorization."""

from __future__ import annotations

from typing import Literal

import sympy as sp

from clifford_3plus2_d5.lepton.checkerboard import (
    checkerboard_massless_floquet,
    edge_gauge_covariance_holds,
    expected_massless_generator,
    expected_massless_hamiltonian,
    floquet_eigenvalues_at,
    gauge_transform_link,
    pauli_z,
)
from clifford_3plus2_d5.lepton.clifford_patisalam import (
    patisalam_all_commute_with_chosen_j,
    patisalam_generator_is_valid,
    spin04_simple_bivector_j,
    su2_l_generators_from_spin04,
    su2_r_generators_from_spin04,
    su4_generators_from_spin06,
)
from clifford_3plus2_d5.lepton.continuum import (
    effective_generator_from_floquet,
    hamiltonian_from_real_skew_generator,
)

GaugeSector = Literal["su4", "su2_l", "su2_r"]


def _block_diag(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return sp.Matrix.vstack(
        sp.Matrix.hstack(left, sp.zeros(left.rows, right.cols)),
        sp.Matrix.hstack(sp.zeros(right.rows, left.cols), right),
    )


def patisalam_internal_identity() -> sp.Matrix:
    return sp.eye(32)


def patisalam_checkerboard_massless_floquet(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
) -> sp.Matrix:
    return checkerboard_massless_floquet(epsilon, k, internal_dim=32)


def patisalam_expected_massless_generator(k: sp.Symbol | sp.Expr) -> sp.Matrix:
    return expected_massless_generator(k, internal_dim=32)


def patisalam_expected_massless_hamiltonian(k: sp.Symbol | sp.Expr) -> sp.Matrix:
    return expected_massless_hamiltonian(k, internal_dim=32)


def patisalam_massless_effective_generator(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
) -> sp.Matrix:
    return effective_generator_from_floquet(
        patisalam_checkerboard_massless_floquet(epsilon, k),
        epsilon=epsilon,
        convention="real_skew",
    )


def patisalam_massless_effective_hamiltonian(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
) -> sp.Matrix:
    return hamiltonian_from_real_skew_generator(
        patisalam_massless_effective_generator(epsilon, k),
    )


def patisalam_background_generator(
    sector: GaugeSector = "su4",
    index: int = 0,
) -> sp.Matrix:
    if sector == "su4":
        basis = su4_generators_from_spin06()
    elif sector == "su2_l":
        basis = su2_l_generators_from_spin04()
    elif sector == "su2_r":
        basis = su2_r_generators_from_spin04()
    else:
        raise ValueError(f"unknown Pati-Salam gauge sector: {sector}")
    if not 0 <= index < len(basis):
        raise ValueError("Pati-Salam gauge generator index out of range")
    return basis[index]


def patisalam_background_generator_is_valid(
    sector: GaugeSector = "su4",
    index: int = 0,
) -> bool:
    generator = patisalam_background_generator(sector, index)
    return patisalam_generator_is_valid(generator) and patisalam_all_commute_with_chosen_j(
        (generator,),
    )


def patisalam_checkerboard_gauge_shift_bloch(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    """Massless shift with a first-order Pati-Salam background link."""

    if gauge_generator.shape != (32, 32):
        raise ValueError("gauge generator must be a 32x32 matrix")
    link = sp.eye(32) + epsilon * gauge_generator
    right = sp.exp(-sp.I * k * epsilon) * link
    left = sp.exp(sp.I * k * epsilon) * link
    return _block_diag(right, left).applyfunc(sp.simplify)


def patisalam_checkerboard_gauge_floquet(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return patisalam_checkerboard_gauge_shift_bloch(epsilon, k, gauge_generator)


def patisalam_expected_gauge_generator(
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return (
        sp.kronecker_product(sp.eye(2), gauge_generator)
        - sp.I * k * sp.kronecker_product(pauli_z(), sp.eye(32))
    ).applyfunc(sp.simplify)


def patisalam_expected_gauge_hamiltonian(
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return hamiltonian_from_real_skew_generator(
        patisalam_expected_gauge_generator(k, gauge_generator),
    )


def patisalam_gauge_effective_generator(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return effective_generator_from_floquet(
        patisalam_checkerboard_gauge_floquet(epsilon, k, gauge_generator),
        epsilon=epsilon,
        convention="real_skew",
    )


def patisalam_gauge_effective_hamiltonian(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return hamiltonian_from_real_skew_generator(
        patisalam_gauge_effective_generator(epsilon, k, gauge_generator),
    )


def patisalam_finite_spin04_transform() -> sp.Matrix:
    """An exact orthogonal finite Spin(4) element for covariance checks."""

    return spin04_simple_bivector_j()


def patisalam_gauge_transform_link(
    link: sp.Matrix,
    left_gauge: sp.Matrix,
    right_gauge: sp.Matrix,
) -> sp.Matrix:
    return gauge_transform_link(link, left_gauge, right_gauge)


def patisalam_edge_gauge_covariance_holds(
    link: sp.Matrix,
    left_gauge: sp.Matrix,
    right_gauge: sp.Matrix,
) -> bool:
    return edge_gauge_covariance_holds(link, left_gauge, right_gauge)


def patisalam_floquet_eigenvalues_at(
    epsilon: sp.Expr,
    k_value: sp.Expr,
) -> tuple[sp.Expr, ...]:
    return floquet_eigenvalues_at(epsilon, k_value, internal_dim=32)


def patisalam_has_gapless_eigenvalue_at(epsilon: sp.Expr, k_value: sp.Expr) -> bool:
    return any(
        sp.simplify(value - 1) == 0
        for value in patisalam_floquet_eigenvalues_at(epsilon, k_value)
    )


def patisalam_sample_gapless_momenta(epsilon: sp.Symbol) -> tuple[sp.Expr, ...]:
    samples = (0, sp.pi / (2 * epsilon), sp.pi / epsilon, -sp.pi / (2 * epsilon))
    return tuple(
        value for value in samples if patisalam_has_gapless_eigenvalue_at(epsilon, value)
    )


def patisalam_checkerboard_audit_payload() -> dict[str, object]:
    epsilon, k = sp.symbols("epsilon k")
    return {
        "internal_real_dimension": 32,
        "massless_floquet_shape": patisalam_checkerboard_massless_floquet(
            epsilon,
            k,
        ).shape,
        "massless_hamiltonian_eigenvalues": patisalam_expected_massless_hamiltonian(
            k,
        ).eigenvals(),
        "su4_background_valid": patisalam_background_generator_is_valid("su4"),
        "su2_l_background_valid": patisalam_background_generator_is_valid("su2_l"),
        "su2_r_background_valid": patisalam_background_generator_is_valid("su2_r"),
        "gapless_sample_momenta": patisalam_sample_gapless_momenta(epsilon),
        "note": "Pati-Salam backgrounds use the same checkerboard continuum form as Session 17.",
    }
