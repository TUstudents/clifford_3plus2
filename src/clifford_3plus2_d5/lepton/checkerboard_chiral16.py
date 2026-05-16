"""1D checkerboard walk lifted to the Cl(0,10) chiral-16 carrier.

This is the Session 17 lift of the Session 16 Cl(8) checkerboard result.
The spatial walk is unchanged; the internal carrier is now the real
chiral-16 block of Cl(0,10), real dimension 32, with a chosen Cl(0,2)
commuting complex structure.
"""

from __future__ import annotations

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
from clifford_3plus2_d5.lepton.clifford_chiral16 import (
    all_commute_with_chosen_j,
    chiral16_su3_generator_is_valid,
    lifted_rigid_su3_transform_chiral16,
    lifted_su3_generators_chiral16,
)
from clifford_3plus2_d5.lepton.continuum import (
    effective_generator_from_floquet,
    hamiltonian_from_real_skew_generator,
)


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _block_diag(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return sp.Matrix.vstack(
        sp.Matrix.hstack(left, sp.zeros(left.rows, right.cols)),
        sp.Matrix.hstack(sp.zeros(right.rows, left.cols), right),
    )


def chiral16_internal_identity() -> sp.Matrix:
    return sp.eye(32)


def chiral16_checkerboard_massless_floquet(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
) -> sp.Matrix:
    return checkerboard_massless_floquet(epsilon, k, internal_dim=32)


def chiral16_expected_massless_generator(k: sp.Symbol | sp.Expr) -> sp.Matrix:
    return expected_massless_generator(k, internal_dim=32)


def chiral16_expected_massless_hamiltonian(k: sp.Symbol | sp.Expr) -> sp.Matrix:
    return expected_massless_hamiltonian(k, internal_dim=32)


def chiral16_massless_effective_generator(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
) -> sp.Matrix:
    return effective_generator_from_floquet(
        chiral16_checkerboard_massless_floquet(epsilon, k),
        epsilon=epsilon,
        convention="real_skew",
    )


def chiral16_massless_effective_hamiltonian(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
) -> sp.Matrix:
    return hamiltonian_from_real_skew_generator(
        chiral16_massless_effective_generator(epsilon, k),
    )


def chiral16_su3_background_generator(index: int = 0) -> sp.Matrix:
    basis = lifted_su3_generators_chiral16()
    if not 0 <= index < len(basis):
        raise ValueError("chiral16 SU(3) generator index out of range")
    return basis[index]


def chiral16_checkerboard_gauge_shift_bloch(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    """Massless chiral-16 shift with first-order background SU(3) link."""

    if gauge_generator.shape != (32, 32):
        raise ValueError("gauge generator must be a 32x32 matrix")
    link = sp.eye(32) + epsilon * gauge_generator
    right = sp.exp(-sp.I * k * epsilon) * link
    left = sp.exp(sp.I * k * epsilon) * link
    return _block_diag(right, left).applyfunc(sp.simplify)


def chiral16_checkerboard_gauge_floquet(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return chiral16_checkerboard_gauge_shift_bloch(epsilon, k, gauge_generator)


def chiral16_expected_gauge_generator(
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return (
        sp.kronecker_product(sp.eye(2), gauge_generator)
        - sp.I * k * sp.kronecker_product(pauli_z(), sp.eye(32))
    ).applyfunc(sp.simplify)


def chiral16_expected_gauge_hamiltonian(
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return hamiltonian_from_real_skew_generator(
        chiral16_expected_gauge_generator(k, gauge_generator),
    )


def chiral16_gauge_effective_generator(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return effective_generator_from_floquet(
        chiral16_checkerboard_gauge_floquet(epsilon, k, gauge_generator),
        epsilon=epsilon,
        convention="real_skew",
    )


def chiral16_gauge_effective_hamiltonian(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return hamiltonian_from_real_skew_generator(
        chiral16_gauge_effective_generator(epsilon, k, gauge_generator),
    )


def chiral16_rigid_su3_gauge_transform() -> sp.Matrix:
    return lifted_rigid_su3_transform_chiral16()


def chiral16_edge_gauge_covariance_holds(
    link: sp.Matrix,
    left_gauge: sp.Matrix,
    right_gauge: sp.Matrix,
) -> bool:
    return edge_gauge_covariance_holds(link, left_gauge, right_gauge)


def chiral16_gauge_transform_link(
    link: sp.Matrix,
    left_gauge: sp.Matrix,
    right_gauge: sp.Matrix,
) -> sp.Matrix:
    return gauge_transform_link(link, left_gauge, right_gauge)


def chiral16_floquet_eigenvalues_at(
    epsilon: sp.Expr,
    k_value: sp.Expr,
) -> tuple[sp.Expr, ...]:
    return floquet_eigenvalues_at(epsilon, k_value, internal_dim=32)


def chiral16_has_gapless_eigenvalue_at(epsilon: sp.Expr, k_value: sp.Expr) -> bool:
    return any(
        sp.simplify(value - 1) == 0
        for value in chiral16_floquet_eigenvalues_at(epsilon, k_value)
    )


def chiral16_sample_gapless_momenta(epsilon: sp.Symbol) -> tuple[sp.Expr, ...]:
    samples = (0, sp.pi / (2 * epsilon), sp.pi / epsilon, -sp.pi / (2 * epsilon))
    return tuple(
        value for value in samples if chiral16_has_gapless_eigenvalue_at(epsilon, value)
    )


def chiral16_background_su3_is_valid(index: int = 0) -> bool:
    generator = chiral16_su3_background_generator(index)
    return chiral16_su3_generator_is_valid(generator) and all_commute_with_chosen_j(
        (generator,),
    )


def chiral16_checkerboard_audit_payload() -> dict[str, object]:
    epsilon, k = sp.symbols("epsilon k")
    gauge_generator = chiral16_su3_background_generator()
    return {
        "internal_real_dimension": 32,
        "massless_floquet_shape": chiral16_checkerboard_massless_floquet(
            epsilon,
            k,
        ).shape,
        "massless_hamiltonian_eigenvalues": chiral16_expected_massless_hamiltonian(
            k,
        ).eigenvals(),
        "su3_background_valid": chiral16_background_su3_is_valid(),
        "su3_commutes_with_chosen_j": all_commute_with_chosen_j((gauge_generator,)),
        "gapless_sample_momenta": chiral16_sample_gapless_momenta(epsilon),
        "ew_note": "SU(3) lifts cleanly; SU(2)_L is not supplied by Spin(0,2).",
    }
