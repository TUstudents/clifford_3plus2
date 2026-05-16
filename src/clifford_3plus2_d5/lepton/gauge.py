"""Gauge-generator helpers for the R^6 lepton structural lab."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp


def complex_mode_matrix_realification(matrix: sp.Matrix) -> sp.Matrix:
    """Realify a complex-linear mode matrix in the ``x_0..x_n,y_0..y_n`` basis."""

    mode_dimension = matrix.rows
    if matrix.shape != (mode_dimension, mode_dimension):
        raise ValueError("mode matrix must be square")

    real_part = sp.zeros(mode_dimension)
    imag_part = sp.zeros(mode_dimension)
    for row in range(mode_dimension):
        for col in range(mode_dimension):
            real_value, imag_value = sp.expand(matrix[row, col]).as_real_imag()
            real_part[row, col] = sp.simplify(real_value)
            imag_part[row, col] = sp.simplify(imag_value)

    return sp.Matrix.vstack(
        sp.Matrix.hstack(real_part, -imag_part),
        sp.Matrix.hstack(imag_part, real_part),
    )


def _doublet_modes_for_singlet(singlet_mode: int) -> tuple[int, int]:
    if not 0 <= singlet_mode < 3:
        raise ValueError("Lab B singlet mode must be in {0, 1, 2}")
    modes = tuple(mode for mode in range(3) if mode != singlet_mode)
    return modes[0], modes[1]


def _doublet_mode_operator(block: sp.Matrix, *, singlet_mode: int = 0) -> sp.Matrix:
    if block.shape != (2, 2):
        raise ValueError("doublet block must be 2x2")
    matrix = sp.zeros(3)
    doublet_modes = _doublet_modes_for_singlet(singlet_mode)
    for row in range(2):
        for col in range(2):
            matrix[doublet_modes[row], doublet_modes[col]] = block[row, col]
    return matrix


def su2_l_generators_r6_for_singlet(singlet_mode: int) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return anti-Hermitian SU(2)_L generators for one singlet-mode frame."""

    sigma_1 = sp.Matrix([[0, 1], [1, 0]])
    sigma_2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sigma_3 = sp.Matrix([[1, 0], [0, -1]])
    return tuple(
        complex_mode_matrix_realification(_doublet_mode_operator(sp.I * sigma, singlet_mode=singlet_mode))
        for sigma in (sigma_1, sigma_2, sigma_3)
    )


def su2_l_generators_r6() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return anti-Hermitian SU(2)_L generators realified on R^6."""

    return su2_l_generators_r6_for_singlet(0)


def u1_y_generator_r6_for_singlet(
    singlet_mode: int,
    *,
    singlet_charge: int = -2,
    doublet_charge: int = -1,
) -> sp.Matrix:
    """Return the realified U(1)_Y phase generator on the R^6 carrier."""

    charges = sp.diag(
        *(singlet_charge if mode == singlet_mode else doublet_charge for mode in range(3))
    )
    return complex_mode_matrix_realification(sp.I * charges)


def u1_y_generator_r6(*, singlet_charge: int = -2, doublet_charge: int = -1) -> sp.Matrix:
    """Return the default-frame realified U(1)_Y phase generator."""

    return u1_y_generator_r6_for_singlet(
        0,
        singlet_charge=singlet_charge,
        doublet_charge=doublet_charge,
    )


def su2_l_u1_y_generators_r6_for_singlet(singlet_mode: int) -> tuple[sp.Matrix, ...]:
    return (*su2_l_generators_r6_for_singlet(singlet_mode), u1_y_generator_r6_for_singlet(singlet_mode))


def su2_l_u1_y_generators_r6() -> tuple[sp.Matrix, ...]:
    return su2_l_u1_y_generators_r6_for_singlet(0)


def transported_gauge_pairs(
    transition: sp.Matrix,
    gauge_generators: Sequence[sp.Matrix],
) -> tuple[tuple[sp.Matrix, sp.Matrix], ...]:
    """Pair each gauge generator with its orthogonal transition transport."""

    return tuple(
        (generator, (transition * generator * transition.T).applyfunc(sp.simplify))
        for generator in gauge_generators
    )
