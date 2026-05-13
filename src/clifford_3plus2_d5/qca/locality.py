"""Locality checks for finite-depth QCA layers."""

from __future__ import annotations

from clifford_3plus2_d5.qca.gates import QCALocalGate, gate_has_valid_locality


def supports_are_disjoint(gates: tuple[QCALocalGate, ...]) -> bool:
    seen: set[int] = set()
    for gate in gates:
        for site in gate.support:
            if site in seen:
                return False
            seen.add(site)
    return True


def max_locality_radius(gates: tuple[QCALocalGate, ...]) -> int:
    if not gates:
        return 0
    return max(gate.locality_radius for gate in gates)


def layer_locality_check_passed(gates: tuple[QCALocalGate, ...]) -> bool:
    return bool(gates) and supports_are_disjoint(gates) and all(
        gate_has_valid_locality(gate) for gate in gates
    )


def finite_depth_check_passed(layer_count: int, *, effective_hamiltonian_only: bool) -> bool:
    return layer_count > 0 and not effective_hamiltonian_only
