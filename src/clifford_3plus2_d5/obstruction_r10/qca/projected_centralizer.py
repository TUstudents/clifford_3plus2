"""Projected centralizer diagnostics for coarse QCA idempotents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import exact_matrix_span
from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.obstruction_r10.qca.gates import is_real_matrix
from clifford_3plus2_d5.obstruction_r10.qca.rule_verdict import CentralIdempotent, solve_central_idempotents


@dataclass(frozen=True)
class ProjectedCentralizerBlockDiagnostic:
    projector_rank: int
    projected_dimension: int
    multiplication_table_solved: bool
    commutative: bool
    idempotents_solved: bool
    idempotent_ranks: tuple[int, ...]
    primitive_component_dimensions: tuple[int, ...]
    primitive_component_types: tuple[str, ...]
    primitive_components_sum_to_projector: bool
    contains_complex_factor: bool
    classification: str


@dataclass(frozen=True)
class ProjectedCentralizerPairDiagnostic:
    pair_ranks: tuple[int, int]
    blocks: tuple[ProjectedCentralizerBlockDiagnostic, ...]


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _projected_basis(
    basis: Sequence[sp.Matrix],
    projector: sp.Matrix,
) -> tuple[sp.Matrix, ...]:
    span = exact_matrix_span(rows=projector.rows, cols=projector.cols, add_matrices=False)
    projected = []
    for matrix in (projector, *basis):
        candidate = (projector * matrix * projector).applyfunc(sp.simplify)
        if span.add(candidate):
            projected.append(candidate)
    return tuple(projected)


def _multiplication_table(
    basis: Sequence[sp.Matrix],
) -> tuple[bool, tuple[tuple[tuple[sp.Expr, ...], ...], ...]]:
    if not basis:
        return False, ()
    span = exact_matrix_span(basis)
    table = []
    for left in basis:
        row = []
        for right in basis:
            product = (left * right).applyfunc(sp.simplify)
            coordinates = span.coordinates(product)
            if coordinates is None:
                return False, ()
            row.append(tuple(sp.simplify(item) for item in coordinates))
        table.append(tuple(row))
    return True, tuple(table)


def _multiplication_table_commutative(
    table: tuple[tuple[tuple[sp.Expr, ...], ...], ...],
) -> bool:
    for left_index, row in enumerate(table):
        for right_index, coordinates in enumerate(row):
            opposite = table[right_index][left_index]
            if any(
                sp.simplify(left - right) != 0
                for left, right in zip(coordinates, opposite, strict=True)
            ):
                return False
    return True


def _nonzero_idempotents(
    idempotents: Sequence[CentralIdempotent],
    *,
    dimension: int,
) -> tuple[CentralIdempotent, ...]:
    zero = sp.zeros(dimension)
    return tuple(item for item in idempotents if not _same_matrix(item.matrix, zero))


def _idempotent_leq(left: sp.Matrix, right: sp.Matrix) -> bool:
    return _same_matrix(left * right, left) and _same_matrix(right * left, left)


def _primitive_idempotents(
    idempotents: Sequence[CentralIdempotent],
    *,
    dimension: int,
) -> tuple[CentralIdempotent, ...]:
    nonzero = _nonzero_idempotents(idempotents, dimension=dimension)
    primitive = []
    for candidate in nonzero:
        has_smaller = any(
            not _same_matrix(other.matrix, candidate.matrix)
            and _idempotent_leq(other.matrix, candidate.matrix)
            for other in nonzero
        )
        if not has_smaller:
            primitive.append(candidate)
    return tuple(primitive)


def _component_basis(
    basis: Sequence[sp.Matrix],
    idempotent: sp.Matrix,
) -> tuple[sp.Matrix, ...]:
    span = exact_matrix_span(rows=idempotent.rows, cols=idempotent.cols, add_matrices=False)
    component = []
    for matrix in (idempotent, *basis):
        candidate = (idempotent * matrix * idempotent).applyfunc(sp.simplify)
        if span.add(candidate):
            component.append(candidate)
    return tuple(component)


def _contains_square_root_of_minus_identity(
    basis: Sequence[sp.Matrix],
    idempotent: sp.Matrix,
) -> bool:
    if not basis:
        return False
    variables = sp.symbols(f"q0:{len(basis)}")
    candidate = sp.zeros(idempotent.rows)
    for variable, basis_matrix in zip(variables, basis, strict=True):
        candidate += variable * basis_matrix
    equations = tuple(
        sp.expand(value)
        for value in (candidate * candidate + idempotent)
        if sp.expand(value) != 0
    )
    if not equations:
        return True
    solutions = sp.solve(equations, tuple(variables), dict=True)
    for solution in solutions:
        if not all(variable in solution for variable in variables):
            continue
        matrix = sp.zeros(idempotent.rows)
        for variable, basis_matrix in zip(variables, basis, strict=True):
            matrix += sp.simplify(solution[variable]) * basis_matrix
        if is_real_matrix(matrix) and _same_matrix(matrix * matrix, -idempotent):
            return True
    return False


def _classify_component(
    basis: Sequence[sp.Matrix],
    idempotent: sp.Matrix,
) -> str:
    if len(basis) == 1:
        return "R"
    if _contains_square_root_of_minus_identity(basis, idempotent):
        return "C"
    return f"dim_{len(basis)}_no_complex_factor"


def _classify_block(
    basis: Sequence[sp.Matrix],
    projector: CentralIdempotent,
    *,
    dimension: int,
) -> ProjectedCentralizerBlockDiagnostic:
    projected_basis = _projected_basis(basis, projector.matrix)
    table_solved, table = _multiplication_table(projected_basis)
    commutative = table_solved and _multiplication_table_commutative(table)
    idempotents_solved = False
    idempotents: tuple[CentralIdempotent, ...] = ()
    if table_solved and commutative:
        idempotents_solved, idempotents = solve_central_idempotents(
            projected_basis,
            max_center_dimension=max(8, len(projected_basis)),
            dimension=dimension,
        )

    primitive_component_dimensions: tuple[int, ...] = ()
    primitive_component_types: tuple[str, ...] = ()
    primitive_sum = sp.zeros(dimension)
    primitive_sum_to_projector = False
    if idempotents_solved:
        primitive = _primitive_idempotents(idempotents, dimension=dimension)
        component_dimensions = []
        component_types = []
        for item in primitive:
            component_basis = _component_basis(projected_basis, item.matrix)
            component_dimensions.append(len(component_basis))
            component_types.append(_classify_component(component_basis, item.matrix))
            primitive_sum += item.matrix
        primitive_component_dimensions = tuple(component_dimensions)
        primitive_component_types = tuple(component_types)
        primitive_sum_to_projector = _same_matrix(primitive_sum, projector.matrix)

    contains_complex_factor = "C" in primitive_component_types
    if not table_solved:
        classification = "multiplication_table_not_closed"
    elif not commutative:
        classification = "noncommutative_projected_centralizer"
    elif not idempotents_solved:
        classification = "idempotents_not_solved"
    elif not primitive_sum_to_projector:
        classification = "primitive_decomposition_not_certified"
    elif contains_complex_factor:
        classification = "contains_complex_factor"
    elif primitive_component_types and all(item == "R" for item in primitive_component_types):
        classification = "split_real"
    else:
        classification = "no_complex_factor_unclassified"

    return ProjectedCentralizerBlockDiagnostic(
        projector_rank=projector.rank,
        projected_dimension=len(projected_basis),
        multiplication_table_solved=table_solved,
        commutative=commutative,
        idempotents_solved=idempotents_solved,
        idempotent_ranks=tuple(sorted(item.rank for item in idempotents)),
        primitive_component_dimensions=primitive_component_dimensions,
        primitive_component_types=primitive_component_types,
        primitive_components_sum_to_projector=primitive_sum_to_projector,
        contains_complex_factor=contains_complex_factor,
        classification=classification,
    )


def _complementary_rank_6_4_pairs(
    idempotents: Sequence[CentralIdempotent],
    *,
    dimension: int,
) -> tuple[tuple[CentralIdempotent, CentralIdempotent], ...]:
    one = identity(dimension)
    zero = sp.zeros(dimension)
    pairs = []
    for rank_6 in (item for item in idempotents if item.rank == 6):
        for rank_4 in (item for item in idempotents if item.rank == 4):
            if (
                _same_matrix(rank_6.matrix + rank_4.matrix, one)
                and _same_matrix(rank_6.matrix * rank_4.matrix, zero)
                and _same_matrix(rank_4.matrix * rank_6.matrix, zero)
            ):
                pairs.append((rank_6, rank_4))
    return tuple(pairs)


def projected_centralizer_pair_diagnostics(
    basis: Sequence[sp.Matrix],
    idempotents: Sequence[CentralIdempotent],
    *,
    dimension: int = 10,
) -> tuple[ProjectedCentralizerPairDiagnostic, ...]:
    """Classify compatible-centralizer restrictions to produced `6+4` blocks.

    The diagnostic is deliberately conservative. It proves a block is
    ``split_real`` only when the projected algebra has a solved primitive
    idempotent decomposition into one-dimensional real components. It reports
    ``contains_complex_factor`` only when a primitive component contains an
    exact real matrix square root of minus its identity.
    """

    diagnostics = []
    for rank_6, rank_4 in _complementary_rank_6_4_pairs(idempotents, dimension=dimension):
        blocks = (
            _classify_block(basis, rank_6, dimension=dimension),
            _classify_block(basis, rank_4, dimension=dimension),
        )
        diagnostics.append(
            ProjectedCentralizerPairDiagnostic(
                pair_ranks=(rank_6.rank, rank_4.rank),
                blocks=blocks,
            )
        )
    return tuple(diagnostics)
