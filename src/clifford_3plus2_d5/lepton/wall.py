"""Wall and intertwiner scaffolding for the leptonic domain-wall lab."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import permutations, product

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.qca.gates import is_real_matrix


class LocalityModel(StrEnum):
    MATRIX_BAND = "matrix_band"
    SITE_SUPPORT = "site_support"
    BLOCK_SUPPORT = "block_support"
    INTERNAL_SPARSITY = "internal_sparsity"


@dataclass(frozen=True)
class LocalityConstraint:
    model: LocalityModel
    sites: tuple[int, ...] = ()
    blocks: tuple[tuple[int, int], ...] = ()
    allowed_block_pairs: tuple[tuple[int, int], ...] = ()
    allowed_entries: tuple[tuple[int, int], ...] = ()
    radius: int = 0


@dataclass(frozen=True)
class WallContext:
    gauge_pairs: tuple[tuple[sp.Matrix, sp.Matrix], ...]
    left_complex_structure: sp.Matrix
    right_complex_structure: sp.Matrix
    transition: sp.Matrix
    locality_constraint: LocalityConstraint
    left_side_projector: sp.Matrix
    right_side_projector: sp.Matrix
    side_projectors_must_be_complementary: bool = True

    def consistency_certified(self) -> bool:
        n = self.transition.rows
        if self.transition.shape != (n, n):
            return False
        matrices = (
            self.left_complex_structure,
            self.right_complex_structure,
            self.left_side_projector,
            self.right_side_projector,
        )
        if any(matrix.shape != (n, n) for matrix in matrices):
            return False

        zero = sp.zeros(n)
        one = identity(n)
        transition = self.transition
        if not _same_matrix(transition.T * transition, one):
            return False

        if not _same_matrix(
            transition * self.left_complex_structure,
            self.right_complex_structure * transition,
        ):
            return False

        for left_generator, right_generator in self.gauge_pairs:
            if left_generator.shape != (n, n) or right_generator.shape != (n, n):
                return False
            if not _same_matrix(transition * left_generator, right_generator * transition):
                return False

        left_projector = self.left_side_projector
        right_projector = self.right_side_projector
        if not _same_matrix(left_projector * left_projector, left_projector):
            return False
        if not _same_matrix(right_projector * right_projector, right_projector):
            return False
        if self.side_projectors_must_be_complementary:
            if not _same_matrix(left_projector + right_projector, one):
                return False
            if not _same_matrix(left_projector * right_projector, zero):
                return False
        if not _same_matrix(left_projector * self.left_complex_structure, self.left_complex_structure * left_projector):
            return False
        if not _same_matrix(right_projector * self.right_complex_structure, self.right_complex_structure * right_projector):
            return False
        return True


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _is_real_orthogonal(matrix: sp.Matrix) -> bool:
    return (
        matrix.rows == matrix.cols
        and is_real_matrix(matrix)
        and _same_matrix(matrix.T * matrix, identity(matrix.rows))
    )


def locality_constraints(
    matrix: sp.Matrix,
    constraint: LocalityConstraint,
) -> list[sp.Expr]:
    n = matrix.rows
    equations: list[sp.Expr] = []
    if constraint.model == LocalityModel.MATRIX_BAND:
        for row in range(n):
            for col in range(n):
                if abs(row - col) > constraint.radius:
                    equations.append(matrix[row, col])
        return equations

    if constraint.model == LocalityModel.SITE_SUPPORT:
        allowed = set(constraint.sites)
        for row in range(n):
            for col in range(n):
                if row not in allowed or col not in allowed:
                    equations.append(matrix[row, col])
        return equations

    if constraint.model == LocalityModel.INTERNAL_SPARSITY:
        allowed_entries = set(constraint.allowed_entries)
        for row in range(n):
            for col in range(n):
                if (row, col) not in allowed_entries:
                    equations.append(matrix[row, col])
        return equations

    if constraint.model == LocalityModel.BLOCK_SUPPORT:
        block_index = [-1] * n
        for index, (start, end) in enumerate(constraint.blocks):
            if start < 0 or end < start or end > n:
                raise ValueError("invalid block range in locality constraint")
            for site in range(start, end):
                if block_index[site] != -1:
                    raise ValueError("overlapping block ranges in locality constraint")
                block_index[site] = index

        allowed_pairs = set(constraint.allowed_block_pairs)
        for row in range(n):
            for col in range(n):
                row_block = block_index[row]
                col_block = block_index[col]
                if row_block < 0 or col_block < 0:
                    equations.append(matrix[row, col])
                    continue
                if (row_block, col_block) not in allowed_pairs:
                    equations.append(matrix[row, col])
        return equations

    raise ValueError(f"unknown locality model: {constraint.model}")


def matrix_satisfies_locality(matrix: sp.Matrix, constraint: LocalityConstraint) -> bool:
    return all(sp.simplify(value) == 0 for value in locality_constraints(matrix, constraint))


def spectrum_matches(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Compare characteristic polynomials by simplified coefficients."""

    if left.shape != right.shape or left.rows != left.cols:
        return False
    left_coefficients = left.charpoly().as_poly().all_coeffs()
    right_coefficients = right.charpoly().as_poly().all_coeffs()
    return len(left_coefficients) == len(right_coefficients) and all(
        sp.simplify(a - b) == 0 for a, b in zip(left_coefficients, right_coefficients, strict=True)
    )


def mode_pair_permutations(mode_count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(permutations(range(mode_count)))


def sign_patterns(count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(product((-1, 1), repeat=count))


def build_mode_pair_permutation_matrix(
    permutation: Sequence[int],
    signs: Sequence[int],
    dimension: int,
) -> sp.Matrix:
    if dimension % 2:
        raise ValueError("mode-pair permutation requires even dimension")
    mode_count = dimension // 2
    if len(permutation) != mode_count or len(signs) != mode_count:
        raise ValueError("permutation and signs must match the mode count")
    if sorted(permutation) != list(range(mode_count)):
        raise ValueError("invalid mode-pair permutation")
    matrix = sp.zeros(dimension)
    for source_mode, target_mode in enumerate(permutation):
        sign = sp.Integer(signs[source_mode])
        if sign not in (-1, 1):
            raise ValueError("sign pattern entries must be ±1")
        matrix[target_mode, source_mode] = sign
        matrix[mode_count + target_mode, mode_count + source_mode] = sign
    return matrix


def _intertwines(transition: sp.Matrix, left: sp.Matrix, right: sp.Matrix) -> bool:
    return _same_matrix(transition * left, right * transition)


def _deduplicate(matrices: Sequence[sp.Matrix]) -> tuple[sp.Matrix, ...]:
    unique: list[sp.Matrix] = []
    for matrix in matrices:
        if not any(_same_matrix(matrix, existing) for existing in unique):
            unique.append(matrix)
    return tuple(unique)


def enumerate_split_reassignment_intertwiners(
    left: sp.Matrix,
    right: sp.Matrix,
    *,
    locality_constraint: LocalityConstraint,
) -> tuple[sp.Matrix, ...]:
    if left.shape != right.shape or left.rows % 2:
        return ()
    dimension = left.rows
    mode_count = dimension // 2
    candidates = []
    for permutation in mode_pair_permutations(mode_count):
        for signs in sign_patterns(mode_count):
            transition = build_mode_pair_permutation_matrix(permutation, signs, dimension)
            if not matrix_satisfies_locality(transition, locality_constraint):
                continue
            if _intertwines(transition, left, right):
                candidates.append(transition)
    return _deduplicate(candidates)


def _signed_permutation_matrix(
    permutation: Sequence[int],
    signs: Sequence[int],
) -> sp.Matrix:
    dimension = len(permutation)
    matrix = sp.zeros(dimension)
    for source, target in enumerate(permutation):
        sign = sp.Integer(signs[source])
        matrix[target, source] = sign
    return matrix


def enumerate_signed_permutation_intertwiners(
    left: sp.Matrix,
    right: sp.Matrix,
    *,
    locality_constraint: LocalityConstraint,
    max_dimension: int = 7,
) -> tuple[sp.Matrix, ...]:
    if left.shape != right.shape or left.rows > max_dimension:
        return ()
    dimension = left.rows
    candidates = []
    for permutation in permutations(range(dimension)):
        for signs in sign_patterns(dimension):
            transition = _signed_permutation_matrix(permutation, signs)
            if not matrix_satisfies_locality(transition, locality_constraint):
                continue
            if _intertwines(transition, left, right):
                candidates.append(transition)
    return _deduplicate(candidates)


def solve_quadratic_intertwiner_generic(
    left: sp.Matrix,
    right: sp.Matrix,
    *,
    locality_constraint: LocalityConstraint,
    max_intertwiner_basis: int = 16,
) -> tuple[tuple[sp.Matrix, ...], str]:
    dimension = left.rows
    variables = sp.symbols(f"t0:{dimension * dimension}")
    transition = sp.Matrix(dimension, dimension, variables)
    equations = list(transition * left - right * transition)
    equations.extend(locality_constraints(transition, locality_constraint))
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, variables)
    nullspace = coefficient_matrix.nullspace()
    if not nullspace:
        return (), "no_solutions"
    if len(nullspace) > max_intertwiner_basis:
        return (), "underdetermined"

    basis = [sp.Matrix(dimension, dimension, vector) for vector in nullspace]
    coefficients = sp.symbols(f"a0:{len(basis)}")
    parametrized = sp.zeros(dimension)
    for coefficient, basis_matrix in zip(coefficients, basis, strict=True):
        parametrized += coefficient * basis_matrix

    orthogonality = (parametrized.T * parametrized - identity(dimension)).applyfunc(sp.expand)
    orthogonality_equations = [value for value in orthogonality if value != 0]
    solutions = sp.solve(orthogonality_equations, coefficients, dict=True)
    candidates = []
    for solution in solutions:
        if not all(coefficient in solution for coefficient in coefficients):
            return (), "parametric_family"
        candidate = sp.zeros(dimension)
        for coefficient, basis_matrix in zip(coefficients, basis, strict=True):
            candidate += solution[coefficient] * basis_matrix
        candidate = candidate.applyfunc(sp.simplify)
        if is_real_matrix(candidate) and _is_real_orthogonal(candidate):
            candidates.append(candidate)
    return _deduplicate(candidates), "generic_quadratic" if candidates else "no_solutions"


def solve_t_intertwiner_orthogonal(
    left: sp.Matrix,
    right: sp.Matrix,
    *,
    locality_constraint: LocalityConstraint,
    max_intertwiner_basis: int = 16,
    include_generic: bool = True,
) -> tuple[tuple[sp.Matrix, ...], str]:
    """Return orthogonal intertwiners and the source tier that found them."""

    if left.shape != right.shape or left.rows != left.cols:
        return (), "no_solutions"
    if not spectrum_matches(left, right):
        return (), "no_solutions"

    split = enumerate_split_reassignment_intertwiners(
        left,
        right,
        locality_constraint=locality_constraint,
    )
    if split:
        return split, "split_reassignment"

    signed = enumerate_signed_permutation_intertwiners(
        left,
        right,
        locality_constraint=locality_constraint,
    )
    if signed:
        return signed, "signed_permutation"

    if not include_generic:
        return (), "no_solutions"
    generic, status = solve_quadratic_intertwiner_generic(
        left,
        right,
        locality_constraint=locality_constraint,
        max_intertwiner_basis=max_intertwiner_basis,
    )
    if status == "parametric_family":
        return (), "underdetermined"
    return generic, status


# Compatibility alias matching the plan notation.
solve_T_intertwiner_orthogonal = solve_t_intertwiner_orthogonal
