"""Rigid Clifford primitive audit for the Cl(8) pivot.

Session 14 asks a narrower question than the earlier bridge verdicts: given
the pinned Cl(0,8) representation and the explicit octonion table from
Session 13, which rigid Clifford words preserve the selected octonionic
``G_2`` / ``SU(3)`` stabilizer structure on the positive chiral block?

This is an audit surface, not a full QCA verdict. The stabilizer embedding is
fixed as:

* identify the positive chiral spinor ``R^8_+`` with the chosen octonion
  table from :mod:`clifford_3plus2_d5.lepton.clifford_octonion`;
* take ``G_2`` as the automorphism group of that multiplication;
* take ``SU(3)`` as the stabilizer of the imaginary direction ``e_7``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from itertools import combinations

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_octonion import (
    chirality_projectors,
    cl08_gamma_matrices,
    octonion_derivation_basis,
    octonion_multiply,
    su3_stabilizer_basis,
)

MatrixTuple = tuple[sp.Matrix, ...]


class CliffordPrimitiveFamily(StrEnum):
    IDENTITY = "identity"
    REFLECTION = "reflection"
    BIVECTOR = "bivector"
    FOUR_VECTOR = "four_vector"


class StabilizerClass(StrEnum):
    SU3_FIXING_E7 = "su3_fixing_e7"
    SU3_FLIPPING_E7 = "su3_flipping_e7"
    G2_BEYOND_SU3 = "g2_beyond_su3"
    SPIN8_BEYOND_G2 = "spin8_beyond_g2"
    NOT_CHIRALITY_PRESERVING = "not_chirality_preserving"


@dataclass(frozen=True)
class CliffordDynamicsCandidate:
    name: str
    family: CliffordPrimitiveFamily
    word: tuple[int, ...]
    matrix: sp.Matrix


@dataclass(frozen=True)
class CliffordDynamicsAuditEntry:
    name: str
    family: CliffordPrimitiveFamily
    word: tuple[int, ...]
    chirality_preserving: bool
    stabilizer_class: StabilizerClass
    octonion_automorphism: bool
    normalizes_g2: bool
    normalizes_su3: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family.value,
            "word": tuple(index + 1 for index in self.word),
            "chirality_preserving": self.chirality_preserving,
            "stabilizer_class": self.stabilizer_class.value,
            "octonion_automorphism": self.octonion_automorphism,
            "normalizes_g2": self.normalizes_g2,
            "normalizes_su3": self.normalizes_su3,
        }


def _identity(dimension: int) -> sp.Matrix:
    return sp.eye(dimension)


def _zero(dimension: int) -> sp.Matrix:
    return sp.zeros(dimension)


def _basis_vector(index: int, dimension: int = 8) -> sp.Matrix:
    vector = sp.zeros(dimension, 1)
    vector[index] = 1
    return vector


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _is_orthogonal(matrix: sp.Matrix) -> bool:
    return matrix.rows == matrix.cols and _same_matrix(matrix.T * matrix, _identity(matrix.rows))


def _flatten(matrix: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(matrix.rows * matrix.cols, 1, list(matrix))


def _span_rank(matrices: Sequence[sp.Matrix]) -> int:
    if not matrices:
        return 0
    return sp.Matrix.hstack(*(_flatten(matrix) for matrix in matrices)).rank()


def _matrix_in_span(matrix: sp.Matrix, basis: Sequence[sp.Matrix]) -> bool:
    return _span_rank(tuple(basis)) == _span_rank((*basis, matrix))


@lru_cache(maxsize=2)
def chiral_basis_matrix(sign: str = "+") -> sp.Matrix:
    """Return the fixed 16x8 coordinate basis for one chiral block."""

    if sign not in {"+", "-"}:
        raise ValueError("sign must be '+' or '-'")
    projector = chirality_projectors()[0 if sign == "+" else 1]
    columns = projector.columnspace()
    if len(columns) != 8:
        raise RuntimeError("chirality projector did not produce an 8-dimensional block")
    return sp.Matrix.hstack(*columns).applyfunc(sp.simplify)


@lru_cache(maxsize=2)
def _chiral_left_inverse(sign: str = "+") -> sp.Matrix:
    basis = chiral_basis_matrix(sign)
    return ((basis.T * basis).inv() * basis.T).applyfunc(sp.simplify)


def chiral_block_matrix(matrix: sp.Matrix, sign: str = "+") -> sp.Matrix | None:
    """Return the selected 8x8 chiral block, or ``None`` if chirality swaps."""

    if matrix.shape != (16, 16):
        raise ValueError("chiral block extraction expects a 16x16 matrix")
    projector = chirality_projectors()[0 if sign == "+" else 1]
    basis = chiral_basis_matrix(sign)
    image = (matrix * basis).applyfunc(sp.simplify)
    if not _same_matrix(projector * image, image):
        return None
    return (_chiral_left_inverse(sign) * image).applyfunc(sp.simplify)


def _word_matrix(word: Sequence[int]) -> sp.Matrix:
    gammas = cl08_gamma_matrices()
    matrix = _identity(16)
    for index in word:
        matrix = (matrix * gammas[index]).applyfunc(sp.simplify)
    return matrix


@lru_cache(maxsize=1)
def iter_clifford_dynamics_candidates() -> tuple[CliffordDynamicsCandidate, ...]:
    """Enumerate the rigid v1 primitive family.

    The family is finite: identity, 8 odd reflections, 28 bivectors, and 70
    four-vectors. Continuous rotations are intentionally out of scope for the
    first audit.
    """

    candidates: list[CliffordDynamicsCandidate] = [
        CliffordDynamicsCandidate(
            name="identity",
            family=CliffordPrimitiveFamily.IDENTITY,
            word=(),
            matrix=_identity(16),
        )
    ]
    gammas = cl08_gamma_matrices()
    for index, gamma in enumerate(gammas):
        candidates.append(
            CliffordDynamicsCandidate(
                name=f"gamma_{index + 1}",
                family=CliffordPrimitiveFamily.REFLECTION,
                word=(index,),
                matrix=gamma,
            )
        )
    for first, second in combinations(range(8), 2):
        candidates.append(
            CliffordDynamicsCandidate(
                name=f"gamma_{first + 1}_{second + 1}",
                family=CliffordPrimitiveFamily.BIVECTOR,
                word=(first, second),
                matrix=_word_matrix((first, second)),
            )
        )
    for word in combinations(range(8), 4):
        one_based = "_".join(str(index + 1) for index in word)
        candidates.append(
            CliffordDynamicsCandidate(
                name=f"gamma_{one_based}",
                family=CliffordPrimitiveFamily.FOUR_VECTOR,
                word=tuple(word),
                matrix=_word_matrix(word),
            )
        )
    return tuple(candidates)


def is_octonion_automorphism(matrix: sp.Matrix) -> bool:
    """Group test: verify ``T(xy) = T(x)T(y)`` on octonion basis elements."""

    if matrix.shape != (8, 8) or not _is_orthogonal(matrix):
        return False
    basis = tuple(_basis_vector(index) for index in range(8))
    for left in basis:
        for right in basis:
            transported_product = (matrix * octonion_multiply(left, right)).applyfunc(sp.simplify)
            product_of_transports = octonion_multiply(matrix * left, matrix * right)
            if transported_product != product_of_transports:
                return False
    return True


def normalizes_lie_algebra(matrix: sp.Matrix, basis: Sequence[sp.Matrix]) -> bool:
    """Algebra test: verify ``T D T^{-1}`` remains in ``span(basis)``."""

    if matrix.shape != (8, 8) or not basis or matrix.det() == 0:
        return False
    inverse = matrix.T if _is_orthogonal(matrix) else matrix.inv()
    for generator in basis:
        transported = (matrix * generator * inverse).applyfunc(sp.simplify)
        if not _matrix_in_span(transported, basis):
            return False
    return True


def classify_chiral_block(block: sp.Matrix | None) -> StabilizerClass:
    if block is None:
        return StabilizerClass.NOT_CHIRALITY_PRESERVING
    if not is_octonion_automorphism(block):
        return StabilizerClass.SPIN8_BEYOND_G2

    e7 = _basis_vector(7)
    transported = (block * e7).applyfunc(sp.simplify)
    if transported == e7:
        return StabilizerClass.SU3_FIXING_E7
    if transported == -e7:
        return StabilizerClass.SU3_FLIPPING_E7
    return StabilizerClass.G2_BEYOND_SU3


def audit_clifford_dynamics_candidate(
    candidate: CliffordDynamicsCandidate,
) -> CliffordDynamicsAuditEntry:
    block = chiral_block_matrix(candidate.matrix, "+")
    g2_basis = octonion_derivation_basis()
    su3_basis = su3_stabilizer_basis(7)
    return CliffordDynamicsAuditEntry(
        name=candidate.name,
        family=candidate.family,
        word=candidate.word,
        chirality_preserving=block is not None,
        stabilizer_class=classify_chiral_block(block),
        octonion_automorphism=block is not None and is_octonion_automorphism(block),
        normalizes_g2=block is not None and normalizes_lie_algebra(block, g2_basis),
        normalizes_su3=block is not None and normalizes_lie_algebra(block, su3_basis),
    )


def known_octonion_automorphism() -> sp.Matrix:
    """Return a fixed nontrivial automorphism of the chosen octonion table."""

    permutation = (0, 1, 4, 5, 6, 7, 2, 3)
    matrix = sp.zeros(8)
    for source, target in enumerate(permutation):
        matrix[target, source] = 1
    return matrix


def generated_lie_algebra_dimension(
    generators: Sequence[sp.Matrix],
    *,
    ambient_basis: Sequence[sp.Matrix],
    max_dimension: int | None = None,
) -> int:
    """Return the commutator-closure dimension inside ``span(ambient_basis)``.

    The rigid audit candidates are finite group representatives, not
    infinitesimal generators. This function therefore seeds the closure only
    with matrices that already lie in the requested Lie-algebra span. That is
    the conservative test: if the finite representatives do not expose
    infinitesimal ``g_2`` / ``su(3)`` directions directly, the dimension is 0.
    """

    if not ambient_basis:
        return 0
    target_max = len(ambient_basis) if max_dimension is None else max_dimension
    basis: list[sp.Matrix] = []

    def add_if_independent(matrix: sp.Matrix) -> bool:
        simplified = matrix.applyfunc(sp.simplify)
        if simplified == _zero(simplified.rows):
            return False
        if not _matrix_in_span(simplified, ambient_basis):
            return False
        old_rank = _span_rank(basis)
        new_rank = _span_rank((*basis, simplified))
        if new_rank > old_rank:
            basis.append(simplified)
            return True
        return False

    for generator in generators:
        add_if_independent(generator)

    index = 0
    while index < len(basis):
        left = basis[index]
        for right in tuple(basis):
            commutator = (left * right - right * left).applyfunc(sp.simplify)
            add_if_independent(commutator)
            if len(basis) >= target_max:
                return len(basis)
        index += 1
    return len(basis)


@lru_cache(maxsize=1)
def clifford_dynamics_audit_entries() -> tuple[CliffordDynamicsAuditEntry, ...]:
    return tuple(audit_clifford_dynamics_candidate(candidate) for candidate in iter_clifford_dynamics_candidates())


def _count_by_family(entries: Sequence[CliffordDynamicsAuditEntry]) -> dict[str, int]:
    return {
        family.value: sum(1 for entry in entries if entry.family == family)
        for family in CliffordPrimitiveFamily
    }


def _count_by_stabilizer(entries: Sequence[CliffordDynamicsAuditEntry]) -> dict[str, int]:
    return {
        stabilizer_class.value: sum(1 for entry in entries if entry.stabilizer_class == stabilizer_class)
        for stabilizer_class in StabilizerClass
    }


def clifford_dynamics_audit_payload(*, include_entries: bool = False) -> dict[str, object]:
    candidates = iter_clifford_dynamics_candidates()
    entries = tuple(audit_clifford_dynamics_candidate(candidate) for candidate in candidates)
    g2_basis = octonion_derivation_basis()
    su3_basis = su3_stabilizer_basis(7)
    g2_blocks = tuple(
        block
        for candidate, entry in zip(candidates, entries, strict=True)
        if entry.octonion_automorphism
        if (block := chiral_block_matrix(candidate.matrix, "+")) is not None
    )
    su3_blocks = tuple(
        block
        for candidate, entry in zip(candidates, entries, strict=True)
        if entry.stabilizer_class
        in {StabilizerClass.SU3_FIXING_E7, StabilizerClass.SU3_FLIPPING_E7}
        if (block := chiral_block_matrix(candidate.matrix, "+")) is not None
    )

    payload: dict[str, object] = {
        "triality_embedding": "G2 stabilizer of the chosen octonion product on R8_+",
        "octonion_table": "session_13_fano_table",
        "su3_fixed_imaginary_direction": "e7",
        "candidate_count": len(entries),
        "family_counts": _count_by_family(entries),
        "chirality_preserving_count": sum(1 for entry in entries if entry.chirality_preserving),
        "stabilizer_class_counts": _count_by_stabilizer(entries),
        "octonion_automorphism_count": sum(1 for entry in entries if entry.octonion_automorphism),
        "g2_normalizer_count": sum(1 for entry in entries if entry.normalizes_g2),
        "su3_normalizer_count": sum(1 for entry in entries if entry.normalizes_su3),
        "g2_algebra_closure_dimension": generated_lie_algebra_dimension(
            g2_blocks,
            ambient_basis=g2_basis,
        ),
        "su3_algebra_closure_dimension": generated_lie_algebra_dimension(
            su3_blocks,
            ambient_basis=su3_basis,
        ),
        "expected_g2_dimension": len(g2_basis),
        "expected_su3_dimension": len(su3_basis),
        "qca_session_15_preview": (
            "Build a clifford_dynamics_profile using the Cl(2) J choice and "
            "the G2/SU3 stabilizer predicate as the commutant policy."
        ),
    }
    if include_entries:
        payload["entries"] = tuple(entry.to_dict() for entry in entries)
    return payload
