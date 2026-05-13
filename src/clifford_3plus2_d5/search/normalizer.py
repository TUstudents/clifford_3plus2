"""Forcedness and normalizer checks for the candidate 3+2 bridge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import matrix_span_rank
from clifford_3plus2_d5.algebra.matrices import commutator, identity, is_zero_matrix, zero
from clifford_3plus2_d5.algebra.projectors import projector_pair_check_passed
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.search.addressability import (
    AddressableOperator,
    ClassifiedAddressableOperator,
    classify_addressable_operators,
    off_block_mixer_control,
    rank_one_color_projector_controls,
    rank_one_weak_projector_controls,
    standard_block_projectors,
)
from clifford_3plus2_d5.search.forced_j import is_complex_structure


ForcednessVerdict = Literal["forced", "weak_orbit", "candidate_only", "falsified"]


@dataclass(frozen=True)
class NormalizerCertificate:
    rule_data_source: str
    rule_data_operator_count: int
    addressable_operator_count: int
    centralizer_dimension: int
    orthogonal_normalizer_dimension: int
    addressability_algebra_dimension: int
    candidate_j_valid: bool
    candidate_split_valid: bool
    candidate_j_preserved_by_normalizer: bool
    normalizer_preserves_declared_split: bool
    continuous_j_alternatives_not_excluded: bool
    rank_three_projector_family_not_excluded: bool
    qca_rule_forces_j: bool
    qca_rule_forces_split: bool
    j_unique_or_forced: bool
    split_unique_or_forced: bool
    rank_one_color_projectors_addressable: bool
    rank_one_weak_projectors_addressable: bool
    off_block_controls_addressable: bool
    non_scalar_block_controls_addressable: bool
    addressability_algebra_safe: bool
    addressable_operators: tuple[ClassifiedAddressableOperator, ...]
    forcedness_verdict: ForcednessVerdict
    load_bearing_qca_bridge: bool


def _matrix_unit(row: int, column: int, dimension: int) -> sp.Matrix:
    matrix = sp.zeros(dimension)
    matrix[row, column] = 1
    return matrix


def _flat_basis_matrix(vector: sp.Matrix, dimension: int) -> sp.Matrix:
    return sp.Matrix(dimension, dimension, list(vector))


def _all_matrix_units(dimension: int) -> tuple[sp.Matrix, ...]:
    return tuple(
        _matrix_unit(row, column, dimension)
        for row in range(dimension)
        for column in range(dimension)
    )


def _skew_basis(dimension: int) -> tuple[sp.Matrix, ...]:
    basis = []
    for row in range(dimension):
        for column in range(row + 1, dimension):
            matrix = sp.zeros(dimension)
            matrix[row, column] = 1
            matrix[column, row] = -1
            basis.append(matrix)
    return tuple(basis)


def _validate_square_matrices(
    matrices: Sequence[sp.Matrix],
    *,
    dimension: int,
) -> None:
    for matrix in matrices:
        if matrix.shape != (dimension, dimension):
            msg = f"expected {dimension}x{dimension} matrices"
            raise ValueError(msg)


def centralizer_basis(
    matrices: Sequence[sp.Matrix],
    *,
    dimension: int = 10,
) -> tuple[sp.Matrix, ...]:
    """Return a basis for matrices X satisfying [X, A] = 0 for all A."""

    _validate_square_matrices(matrices, dimension=dimension)
    if not matrices:
        return _all_matrix_units(dimension)

    variables = sp.symbols(f"x0:{dimension * dimension}")
    candidate = sp.Matrix(dimension, dimension, variables)
    equations = [
        value
        for matrix in matrices
        for value in commutator(candidate, matrix)
    ]
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, variables)
    return tuple(
        _flat_basis_matrix(vector, dimension)
        for vector in coefficient_matrix.nullspace()
    )


def orthogonal_centralizer_basis(
    matrices: Sequence[sp.Matrix],
    *,
    dimension: int = 10,
) -> tuple[sp.Matrix, ...]:
    """Return the infinitesimal orthogonal normalizer preserving the data."""

    _validate_square_matrices(matrices, dimension=dimension)
    skew_basis = _skew_basis(dimension)
    if not matrices:
        return skew_basis

    variables = sp.symbols(f"a0:{len(skew_basis)}")
    candidate = zero(dimension)
    for variable, basis_matrix in zip(variables, skew_basis, strict=True):
        candidate += variable * basis_matrix

    equations = [
        value
        for matrix in matrices
        for value in commutator(candidate, matrix)
    ]
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, variables)
    normalizer_basis = []
    for vector in coefficient_matrix.nullspace():
        normalizer = zero(dimension)
        for coefficient, basis_matrix in zip(vector, skew_basis, strict=True):
            normalizer += coefficient * basis_matrix
        normalizer_basis.append(normalizer)
    return tuple(normalizer_basis)


def _append_if_independent(basis: list[sp.Matrix], candidate: sp.Matrix) -> bool:
    if not basis:
        basis.append(candidate)
        return True

    previous_rank = matrix_span_rank(basis)
    next_rank = matrix_span_rank([*basis, candidate])
    if next_rank > previous_rank:
        basis.append(candidate)
        return True
    return False


def addressability_algebra_basis(
    operators: Sequence[AddressableOperator],
    *,
    dimension: int = 10,
) -> tuple[sp.Matrix, ...]:
    """Return the finite-dimensional algebra generated by the controls."""

    matrices = [operator.matrix for operator in operators]
    _validate_square_matrices(matrices, dimension=dimension)

    basis: list[sp.Matrix] = []
    for matrix in (identity(dimension), *matrices):
        _append_if_independent(basis, matrix)

    changed = True
    while changed and len(basis) < dimension * dimension:
        changed = False
        current_basis = tuple(basis)
        for left in current_basis:
            for right in current_basis:
                changed = _append_if_independent(basis, left * right) or changed
                if len(basis) == dimension * dimension:
                    return tuple(basis)
    return tuple(basis)


def full_u5_like_addressability_controls() -> tuple[AddressableOperator, ...]:
    """Return a compact unsafe proxy for controls resolving all five modes."""

    return (
        *rank_one_color_projector_controls(),
        *rank_one_weak_projector_controls(),
        off_block_mixer_control(),
    )


def _classifications(
    classified: Sequence[ClassifiedAddressableOperator],
) -> tuple[str, ...]:
    return tuple(operator.classification for operator in classified)


def normalizer_certificate(
    *,
    rule_data_operators: tuple[AddressableOperator, ...] | None = None,
    addressable_operators: tuple[AddressableOperator, ...] | None = None,
    rule_data_source: str = "candidate_only",
    qca_rule_forces_j: bool = False,
    qca_rule_forces_split: bool = False,
) -> NormalizerCertificate:
    carrier = standard_real_carrier()
    addressable_operators = (
        addressable_operators
        if addressable_operators is not None
        else standard_block_projectors()
    )
    rule_data_operators = (
        rule_data_operators
        if rule_data_operators is not None
        else addressable_operators
    )
    rule_data_matrices = tuple(operator.matrix for operator in rule_data_operators)
    centralizer = centralizer_basis(rule_data_matrices, dimension=carrier.dimension)
    orthogonal_normalizer = orthogonal_centralizer_basis(
        rule_data_matrices,
        dimension=carrier.dimension,
    )
    classified = classify_addressable_operators(addressable_operators)
    classifications = _classifications(classified)
    rank_one_color = "rank_one_color_projector" in classifications
    rank_one_weak = "rank_one_weak_projector" in classifications
    off_block = "off_block_mixer" in classifications
    non_scalar_block = (
        "non_scalar_block_control" in classifications
        or "dimension_mismatch" in classifications
    )
    addressability_safe = not any(
        classification != "safe_block_projector" for classification in classifications
    )

    candidate_j_valid = is_complex_structure(carrier.complex_structure, carrier.metric)
    candidate_split_valid = projector_pair_check_passed()
    j_preserved = all(
        is_zero_matrix(commutator(generator, carrier.complex_structure))
        for generator in orthogonal_normalizer
    )
    split_preserved = all(
        is_zero_matrix(commutator(generator, carrier.projector_3))
        and is_zero_matrix(commutator(generator, carrier.projector_2))
        for generator in orthogonal_normalizer
    )
    source_backed = rule_data_source != "candidate_only"
    continuous_j_alternatives_not_excluded = (
        not source_backed
        or not qca_rule_forces_j
        or not j_preserved
    )
    rank_three_projector_family_not_excluded = (
        not source_backed
        or not qca_rule_forces_split
        or not split_preserved
        or not addressability_safe
    )
    j_unique_or_forced = candidate_j_valid and not continuous_j_alternatives_not_excluded
    split_unique_or_forced = (
        candidate_split_valid and not rank_three_projector_family_not_excluded
    )

    if not candidate_j_valid or not candidate_split_valid or not addressability_safe:
        verdict: ForcednessVerdict = "falsified"
    elif j_unique_or_forced and split_unique_or_forced:
        verdict = "forced"
    elif source_backed and split_preserved:
        verdict = "weak_orbit"
    else:
        verdict = "candidate_only"

    return NormalizerCertificate(
        rule_data_source=rule_data_source,
        rule_data_operator_count=len(rule_data_operators),
        addressable_operator_count=len(addressable_operators),
        centralizer_dimension=len(centralizer),
        orthogonal_normalizer_dimension=len(orthogonal_normalizer),
        addressability_algebra_dimension=len(
            addressability_algebra_basis(
                addressable_operators,
                dimension=carrier.dimension,
            )
        ),
        candidate_j_valid=candidate_j_valid,
        candidate_split_valid=candidate_split_valid,
        candidate_j_preserved_by_normalizer=j_preserved,
        normalizer_preserves_declared_split=split_preserved,
        continuous_j_alternatives_not_excluded=continuous_j_alternatives_not_excluded,
        rank_three_projector_family_not_excluded=(
            rank_three_projector_family_not_excluded
        ),
        qca_rule_forces_j=qca_rule_forces_j,
        qca_rule_forces_split=qca_rule_forces_split,
        j_unique_or_forced=j_unique_or_forced,
        split_unique_or_forced=split_unique_or_forced,
        rank_one_color_projectors_addressable=rank_one_color,
        rank_one_weak_projectors_addressable=rank_one_weak,
        off_block_controls_addressable=off_block,
        non_scalar_block_controls_addressable=non_scalar_block,
        addressability_algebra_safe=addressability_safe,
        addressable_operators=tuple(classified),
        forcedness_verdict=verdict,
        load_bearing_qca_bridge=False,
    )


def certificate_to_dict(certificate: NormalizerCertificate) -> dict[str, object]:
    return {
        "rule_data_source": certificate.rule_data_source,
        "rule_data_operator_count": certificate.rule_data_operator_count,
        "addressable_operator_count": certificate.addressable_operator_count,
        "centralizer_dimension": certificate.centralizer_dimension,
        "orthogonal_normalizer_dimension": (
            certificate.orthogonal_normalizer_dimension
        ),
        "addressability_algebra_dimension": (
            certificate.addressability_algebra_dimension
        ),
        "candidate_j_valid": certificate.candidate_j_valid,
        "candidate_split_valid": certificate.candidate_split_valid,
        "candidate_j_preserved_by_normalizer": (
            certificate.candidate_j_preserved_by_normalizer
        ),
        "normalizer_preserves_declared_split": (
            certificate.normalizer_preserves_declared_split
        ),
        "continuous_j_alternatives_not_excluded": (
            certificate.continuous_j_alternatives_not_excluded
        ),
        "rank_three_projector_family_not_excluded": (
            certificate.rank_three_projector_family_not_excluded
        ),
        "qca_rule_forces_j": certificate.qca_rule_forces_j,
        "qca_rule_forces_split": certificate.qca_rule_forces_split,
        "j_unique_or_forced": certificate.j_unique_or_forced,
        "split_unique_or_forced": certificate.split_unique_or_forced,
        "rank_one_color_projectors_addressable": (
            certificate.rank_one_color_projectors_addressable
        ),
        "rank_one_weak_projectors_addressable": (
            certificate.rank_one_weak_projectors_addressable
        ),
        "off_block_controls_addressable": certificate.off_block_controls_addressable,
        "non_scalar_block_controls_addressable": (
            certificate.non_scalar_block_controls_addressable
        ),
        "addressability_algebra_safe": certificate.addressability_algebra_safe,
        "addressable_operators": [
            {
                "name": operator.name,
                "classification": operator.classification,
            }
            for operator in certificate.addressable_operators
        ],
        "forcedness_verdict": certificate.forcedness_verdict,
        "normalizer_check_passed": (
            certificate.candidate_j_valid
            and certificate.candidate_split_valid
            and certificate.normalizer_preserves_declared_split
            and certificate.addressability_algebra_safe
        ),
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }
