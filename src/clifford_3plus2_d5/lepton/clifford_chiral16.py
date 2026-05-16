"""Exact Cl(0,10) chiral-16 carrier helpers.

This module lifts the Cl(0,8) octonion block to a real Cl(0,10) spinor
module. The full real module has dimension 64; the chiral projectors have
rank 32, corresponding to complex dimension 16 after choosing a commuting
complex structure supplied by the Cl(0,2) factor.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.lepton.checkerboard import rigid_su3_gauge_transform
from clifford_3plus2_d5.lepton.clifford_lie import (
    basis_span_dimension,
    clifford_su3_basis,
    is_skew_symmetric,
)
from clifford_3plus2_d5.lepton.clifford_octonion import (
    cl02_generators,
    cl08_gamma_matrices,
)

MatrixTuple = tuple[sp.Matrix, ...]


def _identity(dimension: int) -> sp.Matrix:
    return sp.eye(dimension)


def _zero(dimension: int) -> sp.Matrix:
    return sp.zeros(dimension)


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _basis_left_inverse(basis: sp.Matrix) -> sp.Matrix:
    return ((basis.T * basis).inv() * basis.T).applyfunc(sp.simplify)


def _cl02_grading_operator() -> sp.Matrix:
    """A square-``+I`` operator anticommuting with both Cl(0,2) generators."""

    x_matrix = sp.Matrix([[0, 1], [1, 0]])
    return sp.kronecker_product(x_matrix, sp.eye(2))


def _cl02_commuting_complex_structure() -> sp.Matrix:
    """A commuting square-root of ``-I`` from the Cl(0,2) module.

    It commutes with the two Cl(0,2) generators and with the grading operator.
    This is the chosen ``J`` for the chiral-16 block.
    """

    epsilon = sp.Matrix([[0, 1], [-1, 0]])
    return sp.kronecker_product(sp.eye(2), -epsilon)


@lru_cache(maxsize=1)
def cl010_gamma_matrices() -> MatrixTuple:
    gamma8 = cl08_gamma_matrices()
    eta1, eta2 = cl02_generators()
    rho = _cl02_grading_operator()
    identity_16 = _identity(16)
    return tuple(
        sp.kronecker_product(gamma, rho).applyfunc(sp.simplify)
        for gamma in gamma8
    ) + (
        sp.kronecker_product(identity_16, eta1).applyfunc(sp.simplify),
        sp.kronecker_product(identity_16, eta2).applyfunc(sp.simplify),
    )


def cl010_relations_pass(gammas: MatrixTuple | None = None) -> bool:
    gammas = cl010_gamma_matrices() if gammas is None else gammas
    identity = _identity(gammas[0].rows)
    zero = _zero(gammas[0].rows)
    for left_index, left in enumerate(gammas):
        for right_index, right in enumerate(gammas):
            anticommutator = (left * right + right * left).applyfunc(sp.simplify)
            expected = -2 * identity if left_index == right_index else zero
            if anticommutator != expected:
                return False
    return True


def cl010_volume_element(gammas: MatrixTuple | None = None) -> sp.Matrix:
    gammas = cl010_gamma_matrices() if gammas is None else gammas
    volume = _identity(gammas[0].rows)
    for gamma in gammas:
        volume = (volume * gamma).applyfunc(sp.simplify)
    return volume


def cl010_complex_structure_full() -> sp.Matrix:
    return sp.kronecker_product(_identity(16), _cl02_commuting_complex_structure()).applyfunc(sp.simplify)


def cl010_chirality_operator(gammas: MatrixTuple | None = None) -> sp.Matrix:
    """Return the real chirality involution.

    The raw 10-volume squares to ``-I`` in this real convention. Multiplying by
    the commuting Cl(0,2) complex structure gives an involution that still
    anticommutes with every odd gamma.
    """

    return (cl010_volume_element(gammas) * cl010_complex_structure_full()).applyfunc(sp.simplify)


def cl010_chirality_projectors(gammas: MatrixTuple | None = None) -> tuple[sp.Matrix, sp.Matrix]:
    chirality = cl010_chirality_operator(gammas)
    identity = _identity(chirality.rows)
    return (
        ((identity + chirality) / 2).applyfunc(sp.simplify),
        ((identity - chirality) / 2).applyfunc(sp.simplify),
    )


@lru_cache(maxsize=2)
def chiral16_basis_matrix(sign: str = "+") -> sp.Matrix:
    if sign not in {"+", "-"}:
        raise ValueError("sign must be '+' or '-'")
    projector = cl010_chirality_projectors()[0 if sign == "+" else 1]
    columns = projector.columnspace()
    if len(columns) != 32:
        raise RuntimeError("chiral16 projector did not produce rank 32")
    return sp.Matrix.hstack(*columns).applyfunc(sp.simplify)


@lru_cache(maxsize=2)
def _chiral16_left_inverse(sign: str = "+") -> sp.Matrix:
    return _basis_left_inverse(chiral16_basis_matrix(sign))


def chiral16_block_matrix(matrix: sp.Matrix, sign: str = "+") -> sp.Matrix | None:
    if matrix.shape != (64, 64):
        raise ValueError("chiral16 block extraction expects a 64x64 matrix")
    projector = cl010_chirality_projectors()[0 if sign == "+" else 1]
    basis = chiral16_basis_matrix(sign)
    image = (matrix * basis).applyfunc(sp.simplify)
    if not _same_matrix(projector * image, image):
        return None
    return (_chiral16_left_inverse(sign) * image).applyfunc(sp.simplify)


def cl02_factor_complex_structures_on_chiral16(sign: str = "+") -> MatrixTuple:
    eta1, eta2 = cl02_generators()
    candidates = (
        sp.kronecker_product(_identity(16), eta1).applyfunc(sp.simplify),
        sp.kronecker_product(_identity(16), eta2).applyfunc(sp.simplify),
        sp.kronecker_product(_identity(16), (eta1 * eta2).applyfunc(sp.simplify)).applyfunc(sp.simplify),
        cl010_complex_structure_full(),
    )
    blocks: list[sp.Matrix] = []
    for candidate in candidates:
        block = chiral16_block_matrix(candidate, sign)
        if block is not None and _same_matrix(block * block, -_identity(block.rows)):
            blocks.append(block)
    return tuple(blocks)


def chosen_chiral16_complex_structure(which: str = "commuting") -> sp.Matrix:
    if which != "commuting":
        raise ValueError("only the commuting Cl(0,2) complex structure is supported in v1")
    block = chiral16_block_matrix(cl010_complex_structure_full(), "+")
    if block is None:
        raise RuntimeError("chosen Cl(0,2) complex structure did not preserve chiral16")
    return block


def _cl08_chiral_basis(sign: str) -> sp.Matrix:
    from clifford_3plus2_d5.lepton.clifford_dynamics import chiral_basis_matrix

    return chiral_basis_matrix(sign)


def _cl08_block_operator(plus_block: sp.Matrix, minus_block: sp.Matrix | None = None) -> sp.Matrix:
    minus_block = plus_block if minus_block is None else minus_block
    plus_basis = _cl08_chiral_basis("+")
    minus_basis = _cl08_chiral_basis("-")
    plus_left = _basis_left_inverse(plus_basis)
    minus_left = _basis_left_inverse(minus_basis)
    return (
        plus_basis * plus_block * plus_left
        + minus_basis * minus_block * minus_left
    ).applyfunc(sp.simplify)


def lifted_su3_generators_chiral16() -> MatrixTuple:
    generators = []
    for generator in clifford_su3_basis():
        full_cl08 = _cl08_block_operator(generator)
        full_cl010 = sp.kronecker_product(full_cl08, _identity(4)).applyfunc(sp.simplify)
        block = chiral16_block_matrix(full_cl010, "+")
        if block is None:
            raise RuntimeError("lifted SU(3) generator failed to preserve chiral16")
        generators.append(block)
    return tuple(generators)


def lifted_su3_span_dimension() -> int:
    return basis_span_dimension(lifted_su3_generators_chiral16())


def lifted_rigid_su3_transform_chiral16() -> sp.Matrix:
    full_cl08 = _cl08_block_operator(rigid_su3_gauge_transform())
    full_cl010 = sp.kronecker_product(full_cl08, _identity(4)).applyfunc(sp.simplify)
    block = chiral16_block_matrix(full_cl010, "+")
    if block is None:
        raise RuntimeError("lifted rigid SU(3) transform failed to preserve chiral16")
    return block


def chiral16_su3_generator_is_valid(generator: sp.Matrix) -> bool:
    return generator.shape == (32, 32) and is_skew_symmetric(generator)


def cl02_electroweak_audit() -> dict[str, object]:
    j_candidates = cl02_factor_complex_structures_on_chiral16("+")
    chosen_j = chosen_chiral16_complex_structure()
    su3_commutes = all(
        _same_matrix(chosen_j * generator, generator * chosen_j)
        for generator in lifted_su3_generators_chiral16()
    )
    return {
        "cl02_j_candidate_count": len(j_candidates),
        "chirality_preserving_j_count": len(j_candidates),
        "chosen_j_commutes_with_su3": su3_commutes,
        "spin_02_dimension": 1,
        "spin_02_can_supply_su2": False,
        "note": "Spin(0,2) is U(1), not SU(2); electroweak SU(2)_L requires extra structure.",
    }


def all_commute_with_chosen_j(matrices: Sequence[sp.Matrix]) -> bool:
    chosen_j = chosen_chiral16_complex_structure()
    return all(_same_matrix(chosen_j * matrix, matrix * chosen_j) for matrix in matrices)
