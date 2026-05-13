"""Bounded exact gate-word search for candidate complex structures."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import clock_complex_structure


@dataclass(frozen=True)
class PrimitiveGate:
    name: str
    matrix: sp.Matrix
    independently_addressable: bool = False


def matrix_product(gates: Sequence[sp.Matrix], *, dimension: int = 10) -> sp.Matrix:
    result = identity(dimension)
    for gate in gates:
        result = gate * result
    return result


def scan_gate_words(
    primitives: Sequence[PrimitiveGate], max_depth: int, *, dimension: int = 10
) -> Iterator[tuple[tuple[str, ...], sp.Matrix]]:
    if max_depth < 1:
        return

    for depth in range(1, max_depth + 1):
        for word in product(primitives, repeat=depth):
            yield (
                tuple(gate.name for gate in word),
                matrix_product([gate.matrix for gate in word], dimension=dimension),
            )


def standard_period_four_primitives() -> tuple[PrimitiveGate, ...]:
    return (
        PrimitiveGate(
            name="global_clock_tick",
            matrix=clock_complex_structure(),
            independently_addressable=False,
        ),
    )


def rank_one_pair_rotation(index: int, *, mode_dimension: int = 5) -> sp.Matrix:
    if not 0 <= index < mode_dimension:
        raise ValueError("rank-one pair index out of range")

    dimension = 2 * mode_dimension
    rotation = identity(dimension)
    x_index = index
    y_index = mode_dimension + index
    rotation[x_index, x_index] = 0
    rotation[x_index, y_index] = -1
    rotation[y_index, x_index] = 1
    rotation[y_index, y_index] = 0
    return rotation


def rank_one_pair_rotations() -> tuple[PrimitiveGate, ...]:
    return tuple(
        PrimitiveGate(
            name=f"rank_one_pair_rotation_{index + 1}",
            matrix=rank_one_pair_rotation(index),
            independently_addressable=True,
        )
        for index in range(5)
    )
