"""Primitive panels for the complex-linear split lab."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.lepton.complex_carrier import (
    complex_c3_lepton_family_projectors,
    complex_c5_3plus2_projectors,
)
from clifford_3plus2_d5.lepton.complex_verdict import ComplexLayerInput


@dataclass(frozen=True)
class ComplexCandidate:
    name: str
    layers: tuple[ComplexLayerInput, ...]
    metadata: tuple[tuple[str, str], ...] = ()


def _block_diag(first: sp.Matrix, second: sp.Matrix) -> sp.Matrix:
    return sp.diag(first, second)


def _layer(
    name: str,
    matrix: sp.Matrix,
    *,
    seeded: bool = False,
    metadata: tuple[tuple[str, str], ...] = (),
) -> ComplexLayerInput:
    return ComplexLayerInput(
        name=name,
        matrix=matrix.applyfunc(sp.simplify),
        is_seeded_split_layer=seeded,
        metadata=metadata,
    )


def c3_doublet_x() -> sp.Matrix:
    return sp.Matrix([[0, 1], [1, 0]])


def c3_doublet_z() -> sp.Matrix:
    return sp.Matrix([[1, 0], [0, -1]])


def c5_triplet_shift() -> sp.Matrix:
    return sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])


def c5_triplet_diagonal() -> sp.Matrix:
    return sp.diag(1, 2, 3)


def c5_doublet_x() -> sp.Matrix:
    return c3_doublet_x()


def c5_doublet_z() -> sp.Matrix:
    return c3_doublet_z()


def c3_seeded_split_control() -> ComplexCandidate:
    p1, p2 = complex_c3_lepton_family_projectors()
    return ComplexCandidate(
        name="c3_seeded_split_control",
        layers=(
            _layer("seed_P1", p1, seeded=True),
            _layer("seed_P2", p2, seeded=True),
            _layer("doublet_x", _block_diag(sp.Matrix([[1]]), c3_doublet_x())),
            _layer("doublet_z", _block_diag(sp.Matrix([[2]]), c3_doublet_z())),
        ),
        metadata=(("panel", "control"), ("expected", "seeded_split_control")),
    )


def c3_full_irreducible_control() -> ComplexCandidate:
    shift = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    diagonal = sp.diag(1, 2, 3)
    return ComplexCandidate(
        name="c3_full_irreducible_control",
        layers=(
            _layer("cyclic_shift", shift),
            _layer("distinct_diagonal", diagonal),
        ),
        metadata=(("panel", "control"), ("expected", "falsified_no_split")),
    )


def c3_rank_one_locking_control() -> ComplexCandidate:
    return ComplexCandidate(
        name="c3_rank_one_locking_control",
        layers=(_layer("three_distinct_phases", sp.diag(1, 2, 3)),),
        metadata=(("panel", "control"), ("expected", "falsified_rank_one_locking")),
    )


def c3_synthetic_split_candidate() -> ComplexCandidate:
    return ComplexCandidate(
        name="c3_synthetic_c_plus_m2_candidate",
        layers=(
            _layer("scalar_plus_doublet_x", _block_diag(sp.Matrix([[1]]), c3_doublet_x())),
            _layer("scalar_plus_doublet_z", _block_diag(sp.Matrix([[2]]), c3_doublet_z())),
        ),
        metadata=(("panel", "synthetic"), ("expected", "split_candidate")),
    )


def c3_phase_permutation_candidate(order: int) -> ComplexCandidate:
    if order <= 1:
        raise ValueError("phase order must be greater than one")
    omega = sp.exp(2 * sp.pi * sp.I / order)
    phase = sp.diag(1, omega, omega**2)
    swap = sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return ComplexCandidate(
        name=f"c3_phase_permutation_order_{order}",
        layers=(
            _layer(f"phase_order_{order}", phase),
            _layer("doublet_swap", swap),
        ),
        metadata=(("panel", "phase_permutation"), ("order", str(order))),
    )


def c5_seeded_split_control() -> ComplexCandidate:
    p3, p2 = complex_c5_3plus2_projectors()
    return ComplexCandidate(
        name="c5_seeded_split_control",
        layers=(
            _layer("seed_P3", p3, seeded=True),
            _layer("seed_P2", p2, seeded=True),
            _layer("triplet_shift", _block_diag(c5_triplet_shift(), c5_doublet_x())),
            _layer("triplet_diagonal", _block_diag(c5_triplet_diagonal(), c5_doublet_z())),
        ),
        metadata=(("panel", "control"), ("expected", "seeded_split_control")),
    )


def c5_full_irreducible_control() -> ComplexCandidate:
    shift = sp.Matrix(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ]
    )
    diagonal = sp.diag(1, 2, 3, 4, 5)
    return ComplexCandidate(
        name="c5_full_irreducible_control",
        layers=(
            _layer("cyclic_shift", shift),
            _layer("distinct_diagonal", diagonal),
        ),
        metadata=(("panel", "control"), ("expected", "falsified_no_split")),
    )


def c5_rank_one_locking_control() -> ComplexCandidate:
    return ComplexCandidate(
        name="c5_rank_one_locking_control",
        layers=(_layer("five_distinct_phases", sp.diag(1, 2, 3, 4, 5)),),
        metadata=(("panel", "control"), ("expected", "falsified_rank_one_locking")),
    )


def c5_synthetic_split_candidate() -> ComplexCandidate:
    return ComplexCandidate(
        name="c5_synthetic_m3_plus_m2_candidate",
        layers=(
            _layer(
                "triplet_shift_plus_doublet_x",
                _block_diag(c5_triplet_shift(), c5_doublet_x()),
            ),
            _layer(
                "triplet_diagonal_plus_doublet_z",
                _block_diag(c5_triplet_diagonal(), c5_doublet_z()),
            ),
        ),
        metadata=(("panel", "synthetic"), ("expected", "split_candidate")),
    )


def _mode_swap_matrix(dimension: int, left_mode: int, right_mode: int) -> sp.Matrix:
    if left_mode == right_mode:
        raise ValueError("mode swap requires distinct modes")
    matrix = sp.eye(dimension)
    matrix[left_mode, left_mode] = 0
    matrix[right_mode, right_mode] = 0
    matrix[left_mode, right_mode] = 1
    matrix[right_mode, left_mode] = 1
    return matrix


def _conjugate_candidate(candidate: ComplexCandidate, conjugator: sp.Matrix) -> ComplexCandidate:
    return ComplexCandidate(
        name=f"{candidate.name}_conjugated_swap_2_3",
        layers=tuple(
            _layer(
                f"{layer.name}_conjugated",
                (conjugator * layer.matrix * conjugator.T).applyfunc(sp.simplify),
                metadata=layer.metadata,
            )
            for layer in candidate.layers
        ),
        metadata=(
            ("panel", "conjugated_synthetic"),
            ("expected_fixed", "falsified_no_split"),
            ("expected_discovered", "split_candidate"),
            ("conjugation", "swap_2_3"),
        ),
    )


def c5_conjugated_synthetic_split_candidate() -> ComplexCandidate:
    return _conjugate_candidate(c5_synthetic_split_candidate(), _mode_swap_matrix(5, 2, 3))


def c5_phase_permutation_candidate(order: int) -> ComplexCandidate:
    if order <= 1:
        raise ValueError("phase order must be greater than one")
    omega = sp.exp(2 * sp.pi * sp.I / order)
    phase = sp.diag(1, omega, omega**2, omega**3, omega**4)
    swap = sp.Matrix(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    )
    return ComplexCandidate(
        name=f"c5_phase_permutation_order_{order}",
        layers=(
            _layer(f"phase_order_{order}", phase),
            _layer("doublet_swap", swap),
        ),
        metadata=(("panel", "phase_permutation"), ("order", str(order))),
    )


def iter_complex_c3_candidates(
    *,
    max_candidates: int | None = None,
    phase_orders: tuple[int, ...] = (3, 4),
) -> Iterable[ComplexCandidate]:
    yielded = 0
    candidates: list[ComplexCandidate] = [
        c3_seeded_split_control(),
        c3_full_irreducible_control(),
        c3_rank_one_locking_control(),
        c3_synthetic_split_candidate(),
    ]
    candidates.extend(c3_phase_permutation_candidate(order) for order in phase_orders)
    for candidate in candidates:
        yield candidate
        yielded += 1
        if max_candidates is not None and yielded >= max_candidates:
            return


def iter_complex_c5_candidates(
    *,
    max_candidates: int | None = None,
    phase_orders: tuple[int, ...] = (3, 4),
    include_conjugated: bool = False,
) -> Iterable[ComplexCandidate]:
    yielded = 0
    candidates: list[ComplexCandidate] = [
        c5_seeded_split_control(),
        c5_full_irreducible_control(),
        c5_rank_one_locking_control(),
        c5_synthetic_split_candidate(),
    ]
    if include_conjugated:
        candidates.append(c5_conjugated_synthetic_split_candidate())
    candidates.extend(c5_phase_permutation_candidate(order) for order in phase_orders)
    for candidate in candidates:
        yield candidate
        yielded += 1
        if max_candidates is not None and yielded >= max_candidates:
            return
