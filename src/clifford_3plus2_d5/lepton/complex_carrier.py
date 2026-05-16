"""Complex-linear carriers for the split-first leptonic lab."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ComplexCarrier:
    dimension: int
    basis: tuple[str, ...]
    target_projectors: tuple[sp.Matrix, ...] = ()


def _standard_basis(dimension: int) -> tuple[str, ...]:
    return tuple(f"e_{index}" for index in range(dimension))


def complex_projector(dimension: int, modes: tuple[int, ...]) -> sp.Matrix:
    if any(mode < 0 or mode >= dimension for mode in modes):
        raise ValueError("projector mode is out of range")
    return sp.diag(*(1 if mode in modes else 0 for mode in range(dimension)))


def complex_c2_doublet_carrier() -> ComplexCarrier:
    return ComplexCarrier(
        dimension=2,
        basis=_standard_basis(2),
    )


def complex_c3_lepton_family_projectors() -> tuple[sp.Matrix, sp.Matrix]:
    return (
        complex_projector(3, (0,)),
        complex_projector(3, (1, 2)),
    )


def complex_c3_lepton_family_carrier() -> ComplexCarrier:
    return ComplexCarrier(
        dimension=3,
        basis=_standard_basis(3),
        target_projectors=complex_c3_lepton_family_projectors(),
    )


def complex_c5_3plus2_projectors() -> tuple[sp.Matrix, sp.Matrix]:
    return (
        complex_projector(5, (0, 1, 2)),
        complex_projector(5, (3, 4)),
    )


def complex_c5_3plus2_carrier() -> ComplexCarrier:
    return ComplexCarrier(
        dimension=5,
        basis=_standard_basis(5),
        target_projectors=complex_c5_3plus2_projectors(),
    )


def complex_c3_carrier_check_passed() -> bool:
    carrier = complex_c3_lepton_family_carrier()
    projectors = carrier.target_projectors
    return (
        carrier.dimension == 3
        and tuple(projector.rank() for projector in projectors) == (1, 2)
        and sum(projectors, sp.zeros(3)) == sp.eye(3)
        and projectors[0] * projectors[1] == sp.zeros(3)
    )


def complex_c5_carrier_check_passed() -> bool:
    carrier = complex_c5_3plus2_carrier()
    projectors = carrier.target_projectors
    return (
        carrier.dimension == 5
        and tuple(projector.rank() for projector in projectors) == (3, 2)
        and sum(projectors, sp.zeros(5)) == sp.eye(5)
        and projectors[0] * projectors[1] == sp.zeros(5)
    )
