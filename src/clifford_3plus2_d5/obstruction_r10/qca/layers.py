"""Layer and update composition for QCA candidates."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.obstruction_r10.qca.gates import QCALocalGate, global_clock_tick_gate, is_real_orthogonal
from clifford_3plus2_d5.obstruction_r10.qca.locality import (
    layer_locality_check_passed,
    max_locality_radius,
    supports_are_disjoint,
)


@dataclass(frozen=True)
class QCALayer:
    name: str
    gates: tuple[QCALocalGate, ...]


@dataclass(frozen=True)
class QCAUpdate:
    name: str
    layers: tuple[QCALayer, ...]
    effective_hamiltonian_only: bool = False
    qca_rule_forces_update: bool = False
    microscopic_rule_source: str = "candidate_only"


def layer_operator(layer: QCALayer) -> sp.Matrix:
    result = identity(10)
    for gate in layer.gates:
        result = gate.matrix * result
    return result


def update_prefix_operator(update: QCAUpdate, layer_count: int) -> sp.Matrix:
    result = identity(10)
    for layer in update.layers[:layer_count]:
        result = layer_operator(layer) * result
    return result


def update_operator(update: QCAUpdate) -> sp.Matrix:
    return update_prefix_operator(update, len(update.layers))


def layer_certificate(layer: QCALayer) -> dict[str, object]:
    operator = layer_operator(layer)
    return {
        "name": layer.name,
        "gate_names": [gate.name for gate in layer.gates],
        "supports": [list(gate.support) for gate in layer.gates],
        "supports_are_disjoint": supports_are_disjoint(layer.gates),
        "locality_radius": max_locality_radius(layer.gates),
        "locality_check_passed": layer_locality_check_passed(layer.gates),
        "is_real_orthogonal": is_real_orthogonal(operator),
    }


def minimal_period_four_update() -> QCAUpdate:
    return QCAUpdate(
        name="minimal_period_four_clock_candidate",
        layers=tuple(
            QCALayer(name=f"clock_tick_{index}", gates=(global_clock_tick_gate(),))
            for index in range(1, 5)
        ),
    )
