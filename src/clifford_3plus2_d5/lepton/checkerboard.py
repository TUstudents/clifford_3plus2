"""1D Cl(8)-internal checkerboard walk with background SU(3) links."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_dynamics import (
    StabilizerClass,
    chiral_block_matrix,
    clifford_dynamics_audit_entries,
    iter_clifford_dynamics_candidates,
)
from clifford_3plus2_d5.lepton.clifford_lie import (
    clifford_su3_basis,
    is_skew_symmetric,
    matrix_in_span,
)
from clifford_3plus2_d5.lepton.continuum import (
    effective_generator_from_floquet,
    hamiltonian_from_real_skew_generator,
)


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def pauli_z() -> sp.Matrix:
    return sp.diag(1, -1)


def internal_identity(internal_dim: int = 8) -> sp.Matrix:
    return sp.eye(internal_dim)


def checkerboard_shift_bloch(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    *,
    internal_dim: int = 8,
) -> sp.Matrix:
    """Return ``diag(exp(-i k eps), exp(+i k eps)) tensor I_internal``."""

    right_phase = sp.exp(-sp.I * k * epsilon)
    left_phase = sp.exp(sp.I * k * epsilon)
    return sp.diag(
        *([right_phase] * internal_dim),
        *([left_phase] * internal_dim),
    )


def checkerboard_massless_floquet(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    *,
    internal_dim: int = 8,
) -> sp.Matrix:
    return checkerboard_shift_bloch(epsilon, k, internal_dim=internal_dim)


def expected_massless_generator(k: sp.Symbol | sp.Expr, *, internal_dim: int = 8) -> sp.Matrix:
    return (-sp.I * k * sp.kronecker_product(pauli_z(), internal_identity(internal_dim))).applyfunc(sp.simplify)


def expected_massless_hamiltonian(k: sp.Symbol | sp.Expr, *, internal_dim: int = 8) -> sp.Matrix:
    return (k * sp.kronecker_product(pauli_z(), internal_identity(internal_dim))).applyfunc(sp.simplify)


def massless_effective_generator(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    *,
    internal_dim: int = 8,
) -> sp.Matrix:
    return effective_generator_from_floquet(
        checkerboard_massless_floquet(epsilon, k, internal_dim=internal_dim),
        epsilon=epsilon,
        convention="real_skew",
    )


def massless_effective_hamiltonian(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    *,
    internal_dim: int = 8,
) -> sp.Matrix:
    return hamiltonian_from_real_skew_generator(
        massless_effective_generator(epsilon, k, internal_dim=internal_dim),
    )


def su3_background_generator(index: int = 0) -> sp.Matrix:
    basis = clifford_su3_basis()
    if not 0 <= index < len(basis):
        raise ValueError("su3 background generator index out of range")
    return basis[index]


def checkerboard_gauge_shift_bloch(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    """Massless shift with a first-order background ``SU(3)`` link.

    The link is represented as ``I + epsilon A`` for real-skew
    ``A in su(3)``. This keeps the v1 continuum test exact and avoids
    unnecessary symbolic matrix exponentials.
    """

    if gauge_generator.shape != (8, 8):
        raise ValueError("gauge generator must be an 8x8 matrix")
    link = sp.eye(8) + epsilon * gauge_generator
    right = sp.exp(-sp.I * k * epsilon) * link
    left = sp.exp(sp.I * k * epsilon) * link
    return _block_diag(right, left).applyfunc(sp.simplify)


def _block_diag(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return sp.Matrix.vstack(
        sp.Matrix.hstack(left, sp.zeros(left.rows, right.cols)),
        sp.Matrix.hstack(sp.zeros(right.rows, left.cols), right),
    )


def checkerboard_gauge_floquet(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return checkerboard_gauge_shift_bloch(epsilon, k, gauge_generator)


def expected_gauge_generator(k: sp.Symbol | sp.Expr, gauge_generator: sp.Matrix) -> sp.Matrix:
    return (
        sp.kronecker_product(sp.eye(2), gauge_generator)
        - sp.I * k * sp.kronecker_product(pauli_z(), sp.eye(8))
    ).applyfunc(sp.simplify)


def expected_gauge_hamiltonian(k: sp.Symbol | sp.Expr, gauge_generator: sp.Matrix) -> sp.Matrix:
    return hamiltonian_from_real_skew_generator(expected_gauge_generator(k, gauge_generator))


def gauge_effective_generator(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return effective_generator_from_floquet(
        checkerboard_gauge_floquet(epsilon, k, gauge_generator),
        epsilon=epsilon,
        convention="real_skew",
    )


def gauge_effective_hamiltonian(
    epsilon: sp.Symbol,
    k: sp.Symbol | sp.Expr,
    gauge_generator: sp.Matrix,
) -> sp.Matrix:
    return hamiltonian_from_real_skew_generator(
        gauge_effective_generator(epsilon, k, gauge_generator),
    )


def rigid_su3_gauge_transform() -> sp.Matrix:
    """Return one exact nontrivial finite SU(3)-fixing automorphism."""

    candidates = iter_clifford_dynamics_candidates()
    entries = clifford_dynamics_audit_entries()
    identity = sp.eye(8)
    for candidate, entry in zip(candidates, entries, strict=True):
        if entry.stabilizer_class != StabilizerClass.SU3_FIXING_E7:
            continue
        block = chiral_block_matrix(candidate.matrix, "+")
        if block is not None and block != identity:
            return block
    raise RuntimeError("no nontrivial rigid SU(3)-fixing gauge transform found")


def gauge_transform_link(link: sp.Matrix, left_gauge: sp.Matrix, right_gauge: sp.Matrix) -> sp.Matrix:
    return (left_gauge * link * right_gauge.T).applyfunc(sp.simplify)


def edge_gauge_covariance_holds(link: sp.Matrix, left_gauge: sp.Matrix, right_gauge: sp.Matrix) -> bool:
    transformed = gauge_transform_link(link, left_gauge, right_gauge)
    return _same_matrix(transformed * right_gauge, left_gauge * link)


def su3_generator_is_valid(generator: sp.Matrix) -> bool:
    return generator.shape == (8, 8) and is_skew_symmetric(generator) and matrix_in_span(generator, clifford_su3_basis())


def floquet_eigenvalues_at(epsilon: sp.Expr, k_value: sp.Expr, *, internal_dim: int = 8) -> tuple[sp.Expr, ...]:
    right = sp.exp(-sp.I * k_value * epsilon)
    left = sp.exp(sp.I * k_value * epsilon)
    return tuple([sp.simplify(right)] * internal_dim + [sp.simplify(left)] * internal_dim)


def has_gapless_eigenvalue_at(epsilon: sp.Expr, k_value: sp.Expr) -> bool:
    return any(sp.simplify(value - 1) == 0 for value in floquet_eigenvalues_at(epsilon, k_value))


def sample_gapless_momenta(epsilon: sp.Symbol) -> tuple[sp.Expr, ...]:
    samples = (0, sp.pi / (2 * epsilon), sp.pi / epsilon, -sp.pi / (2 * epsilon))
    return tuple(value for value in samples if has_gapless_eigenvalue_at(epsilon, value))
