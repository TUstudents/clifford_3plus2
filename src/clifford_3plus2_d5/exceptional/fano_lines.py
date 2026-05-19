"""Phase 0b: three Fano-lines-through-e_7 triage.

A natural three-fold structure on the octonions: the three Fano lines
``{1, 6, 7}, {2, 5, 7}, {3, 4, 7}`` pass through e_7.  Each defines an
H-subalgebra ``{1, e_a, e_b, e_c}`` and naively an SU(2) acting on the
sub-quaternion space.

If three SU(2)s emerge this way and combine as "one SU(2)_L acting on
three doublets" (the SM pattern), this would be a candidate for three
generations.  The kill test:

1. Identify the three Fano lines through e_7.
2. Build the three "candidate SU(2)" generator sets from
   ``octonion_left_multiplication``.
3. **Crucial test**: check whether each candidate SU(2) is closed under
   commutator.  Due to octonion non-associativity,
   ``[L_a, L_b]`` is generically NOT a multiple of ``L_c`` (the
   associative SU(2) relation), so the candidates may fail to be
   Lie subalgebras at all.

Spoiler: octonion non-associativity prevents closure.  The three Fano
lines do not give three SU(2) Lie algebras, killing the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.exceptional.reuse import (
    octonion_fano_triples,
    octonion_left_multiplication,
)


def fano_lines_through(unit_index: int) -> tuple[tuple[int, int, int], ...]:
    """Return all Fano triples containing the given imaginary unit index."""

    return tuple(triple for triple in octonion_fano_triples() if unit_index in triple)


def candidate_su2_generators(line: tuple[int, int, int]) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return ``L_{e_a}, L_{e_b}, L_{e_c}`` for a Fano triple ``(a, b, c)``."""

    return tuple(octonion_left_multiplication(index) for index in line)  # type: ignore[return-value]


def _commutator(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return (left * right - right * left).applyfunc(sp.simplify)


def _flatten(matrix: sp.Matrix) -> list[sp.Expr]:
    return [matrix[row, col] for row in range(matrix.rows) for col in range(matrix.cols)]


def candidate_closes_as_lie_algebra(generators: tuple[sp.Matrix, sp.Matrix, sp.Matrix]) -> bool:
    """Return whether the 3 generators close under commutator inside their span.

    A genuine su(2) Lie algebra closes: ``[g_a, g_b] = scalar * g_c``.  For
    octonion ``L`` operators with non-associativity, this generically fails.
    """

    basis = sp.Matrix([_flatten(g) for g in generators]).T
    for i, gi in enumerate(generators):
        for j in range(i + 1, len(generators)):
            commutator = _commutator(gi, generators[j])
            augmented = basis.row_join(sp.Matrix(_flatten(commutator)))
            if augmented.rank() != basis.rank():
                return False
    return True


@lru_cache(maxsize=1)
def fano_e7_lines() -> tuple[tuple[int, int, int], ...]:
    return fano_lines_through(7)


def shared_generator_count() -> int:
    """Return how many imaginary units are shared between all three lines.

    For lines through ``e_7``, all three share at least ``e_7`` itself.
    """

    lines = fano_e7_lines()
    shared = set(lines[0])
    for line in lines[1:]:
        shared &= set(line)
    return len(shared)


def union_dimension() -> int:
    """Return the dimension of the linear span of all 9 generators."""

    lines = fano_e7_lines()
    generators = [
        octonion_left_multiplication(index)
        for line in lines
        for index in line
    ]
    return sp.Matrix([_flatten(generator) for generator in generators]).T.rank()


@dataclass(frozen=True)
class FanoLinesAuditPayload:
    line_count: int
    lines: tuple[tuple[int, int, int], ...]
    shared_generator_count: int
    union_span_dimension: int
    candidate_su2_count: int
    candidates_close_as_lie_algebras: tuple[bool, ...]
    any_candidate_is_lie_algebra: bool
    verdict: str
    interpretation: str


def fano_lines_audit_payload() -> FanoLinesAuditPayload:
    lines = fano_e7_lines()
    shared = shared_generator_count()
    union = union_dimension()
    closures = tuple(
        candidate_closes_as_lie_algebra(candidate_su2_generators(line))
        for line in lines
    )
    any_lie = any(closures)

    if not any_lie:
        verdict = "FANO KILL — non-associativity destroys SU(2) closure"
        interpretation = (
            "None of the three candidate Fano-line generator triples close "
            "as Lie algebras under commutator: octonion non-associativity "
            "produces ``[L_a, L_b]`` outside the linear span of "
            "``{L_a, L_b, L_c}`` for each Fano triple ``(a, b, c)``.  The "
            "three Fano lines through ``e_7`` do not give three independent "
            "su(2) Lie algebras at all.  The 'three SU(2)s = three "
            "generations' interpretation has no algebraic foundation."
        )
    elif union < 9 and shared >= 1:
        verdict = "FANO KILL — overlapping SU(2)s"
        interpretation = (
            f"All three candidate SU(2)s share {shared} generator(s) "
            f"({set.intersection(*(set(line) for line in lines))}), and "
            f"their union spans only {union} dimensions (not 9 = 3 × 3 for "
            "independent SU(2)s).  Not three independent SU(2)s; not the "
            "SM 'one SU(2)_L on three doublets' pattern."
        )
    else:
        verdict = "FANO UNEXPECTED"
        interpretation = (
            f"Closures: {closures}.  Shared: {shared}.  Union: {union}.  "
            f"Pattern doesn't match expected SM-shape; investigate."
        )

    return FanoLinesAuditPayload(
        line_count=len(lines),
        lines=lines,
        shared_generator_count=shared,
        union_span_dimension=union,
        candidate_su2_count=len(lines),
        candidates_close_as_lie_algebras=closures,
        any_candidate_is_lie_algebra=any_lie,
        verdict=verdict,
        interpretation=interpretation,
    )
