"""Checkerboard background tests for the extracted SM gauge algebra."""

from __future__ import annotations

from typing import Literal

import sympy as sp

from clifford_3plus2_d5.lepton.checkerboard_patisalam import (
    patisalam_checkerboard_gauge_floquet,
    patisalam_checkerboard_gauge_shift_bloch,
    patisalam_checkerboard_massless_floquet,
    patisalam_edge_gauge_covariance_holds,
    patisalam_expected_gauge_generator,
    patisalam_expected_gauge_hamiltonian,
    patisalam_expected_massless_generator,
    patisalam_expected_massless_hamiltonian,
    patisalam_finite_spin04_transform,
    patisalam_floquet_eigenvalues_at,
    patisalam_gauge_effective_generator,
    patisalam_gauge_effective_hamiltonian,
    patisalam_gauge_transform_link,
    patisalam_has_gapless_eigenvalue_at,
    patisalam_massless_effective_generator,
    patisalam_massless_effective_hamiltonian,
    patisalam_sample_gapless_momenta,
)
from clifford_3plus2_d5.lepton.clifford_patisalam import (
    patisalam_all_commute_with_chosen_j,
)
from clifford_3plus2_d5.lepton.patisalam_sm import (
    hypercharge_generator,
    sm_generator_is_valid,
    su3_c_generators_from_su4,
    su2_l_generators_from_spin04,
)

SmGaugeSector = Literal["su3_c", "su2_l", "u1_y"]


def sm_background_generator(sector: SmGaugeSector = "su3_c", index: int = 0) -> sp.Matrix:
    if sector == "su3_c":
        basis = su3_c_generators_from_su4()
        if not 0 <= index < len(basis):
            raise ValueError("SM SU(3)c generator index out of range")
        return basis[index]
    if sector == "su2_l":
        basis = su2_l_generators_from_spin04()
        if not 0 <= index < len(basis):
            raise ValueError("SM SU(2)L generator index out of range")
        return basis[index]
    if sector == "u1_y":
        if index != 0:
            raise ValueError("U(1)Y has a single generator")
        return hypercharge_generator()
    raise ValueError(f"unknown SM gauge sector: {sector}")


def sm_background_generator_is_valid(sector: SmGaugeSector = "su3_c", index: int = 0) -> bool:
    generator = sm_background_generator(sector, index)
    return sm_generator_is_valid(generator) and patisalam_all_commute_with_chosen_j(
        (generator,),
    )


def sm_checkerboard_gauge_shift_bloch(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return patisalam_checkerboard_gauge_shift_bloch(epsilon, k, gauge_generator)


def sm_checkerboard_gauge_floquet(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return patisalam_checkerboard_gauge_floquet(epsilon, k, gauge_generator)


def sm_expected_gauge_generator(
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return patisalam_expected_gauge_generator(k, gauge_generator)


def sm_expected_gauge_hamiltonian(
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return patisalam_expected_gauge_hamiltonian(k, gauge_generator)


def sm_gauge_effective_generator(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return patisalam_gauge_effective_generator(epsilon, k, gauge_generator)


def sm_gauge_effective_hamiltonian(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return patisalam_gauge_effective_hamiltonian(epsilon, k, gauge_generator)


def sm_checkerboard_audit_payload() -> dict[str, object]:
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
        "su3_c_background_valid": sm_background_generator_is_valid("su3_c"),
        "su2_l_background_valid": sm_background_generator_is_valid("su2_l"),
        "u1_y_background_valid": sm_background_generator_is_valid("u1_y"),
        "gapless_sample_momenta": patisalam_sample_gapless_momenta(epsilon),
        "note": "SM backgrounds reuse the Session 18 Pati-Salam checkerboard continuum form.",
    }


__all__ = [
    "SmGaugeSector",
    "patisalam_checkerboard_massless_floquet",
    "patisalam_edge_gauge_covariance_holds",
    "patisalam_expected_massless_generator",
    "patisalam_expected_massless_hamiltonian",
    "patisalam_finite_spin04_transform",
    "patisalam_floquet_eigenvalues_at",
    "patisalam_gauge_transform_link",
    "patisalam_has_gapless_eigenvalue_at",
    "patisalam_massless_effective_generator",
    "patisalam_massless_effective_hamiltonian",
    "patisalam_sample_gapless_momenta",
    "sm_background_generator",
    "sm_background_generator_is_valid",
    "sm_checkerboard_audit_payload",
    "sm_checkerboard_gauge_floquet",
    "sm_checkerboard_gauge_shift_bloch",
    "sm_expected_gauge_generator",
    "sm_expected_gauge_hamiltonian",
    "sm_gauge_effective_generator",
    "sm_gauge_effective_hamiltonian",
]
