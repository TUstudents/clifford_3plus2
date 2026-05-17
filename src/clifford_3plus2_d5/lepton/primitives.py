"""Clock-plane primitive families for Lab A."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.obstruction_r10.qca.rule_verdict import RuleLayerInput


LAB_A_DIMENSION = 4
LAB_A_MODE_COUNT = 2


@dataclass(frozen=True)
class LabACandidate:
    name: str
    layers: tuple[RuleLayerInput, ...]


def _angle_for_order(order: int) -> sp.Expr:
    if order <= 0:
        raise ValueError("angle order must be positive")
    return 2 * sp.pi * sp.Rational(1, order)


def _clock_rotation_block(angle: sp.Expr) -> sp.Matrix:
    cosine = sp.simplify(sp.cos(angle))
    sine = sp.simplify(sp.sin(angle))
    return sp.Matrix([[cosine, -sine], [sine, cosine]])


def pair_rotation_matrix(orders: Sequence[int]) -> sp.Matrix:
    """Return a clock-plane rotation on each of the two R^2 mode pairs."""

    if len(orders) != LAB_A_MODE_COUNT:
        raise ValueError("Lab A pair rotation requires two mode orders")
    matrix = sp.zeros(LAB_A_DIMENSION)
    for mode, order in enumerate(orders):
        block = _clock_rotation_block(_angle_for_order(order))
        x_index = mode
        y_index = LAB_A_MODE_COUNT + mode
        matrix[x_index, x_index] = block[0, 0]
        matrix[x_index, y_index] = block[0, 1]
        matrix[y_index, x_index] = block[1, 0]
        matrix[y_index, y_index] = block[1, 1]
    return matrix.applyfunc(sp.simplify)


def signed_twist_matrix(signs: Sequence[int]) -> sp.Matrix:
    """Return a pair-preserving sign twist with the same sign on x_i and y_i."""

    if len(signs) != LAB_A_MODE_COUNT:
        raise ValueError("Lab A signed twist requires two signs")
    diagonal = []
    for sign in signs:
        if sign not in (-1, 1):
            raise ValueError("signed twist entries must be ±1")
        diagonal.append(sign)
    for sign in signs:
        diagonal.append(sign)
    return sp.diag(*diagonal)


def mode_swap_matrix() -> sp.Matrix:
    """Swap the two clock-plane mode pairs."""

    matrix = sp.zeros(LAB_A_DIMENSION)
    permutation = (1, 0, 3, 2)
    for source, target in enumerate(permutation):
        matrix[target, source] = 1
    return matrix


def _layer(name: str, matrix: sp.Matrix) -> RuleLayerInput:
    return RuleLayerInput(name=name, matrix=matrix, support=(0,), locality_radius=0)


def lab_a_primitive_layers(
    *,
    angle_orders: Sequence[int] = (2, 3, 4),
) -> tuple[RuleLayerInput, ...]:
    """Return the deterministic primitive layer catalogue for Lab A Session 3."""

    layers: list[RuleLayerInput] = []
    seen: set[tuple[sp.Expr, ...]] = set()

    def add(name: str, matrix: sp.Matrix) -> None:
        key = tuple(sp.simplify(value) for value in matrix)
        if key in seen:
            return
        seen.add(key)
        layers.append(_layer(name, matrix))

    for orders in product(angle_orders, repeat=LAB_A_MODE_COUNT):
        add(f"pair_rotation_o{orders[0]}_o{orders[1]}", pair_rotation_matrix(orders))

    for signs in product((-1, 1), repeat=LAB_A_MODE_COUNT):
        if all(sign == 1 for sign in signs):
            continue
        add(f"signed_twist_{signs[0]}_{signs[1]}", signed_twist_matrix(signs))

    add("mode_pair_swap", mode_swap_matrix())
    return tuple(layers)


def iter_lab_a_onsite_candidates(
    *,
    max_depth: int = 2,
    angle_orders: Sequence[int] = (2, 3, 4),
    max_candidates: int | None = None,
) -> Iterable[LabACandidate]:
    """Yield bounded ordered layer-tuples for the first Lab A on-site scan."""

    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    primitives = lab_a_primitive_layers(angle_orders=angle_orders)
    yielded = 0
    for depth in range(1, max_depth + 1):
        for layers in product(primitives, repeat=depth):
            name = "__".join(layer.name for layer in layers)
            yield LabACandidate(name=f"lab_a_onsite_{name}", layers=tuple(layers))
            yielded += 1
            if max_candidates is not None and yielded >= max_candidates:
                return


def lab_a_identity_layer() -> RuleLayerInput:
    return _layer("identity_r4", identity(LAB_A_DIMENSION))
