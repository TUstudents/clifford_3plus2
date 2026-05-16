"""Primitive panels for the complex-linear split lab."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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


def _permutation_matrix(permutation: Sequence[int]) -> sp.Matrix:
    dimension = len(permutation)
    if sorted(permutation) != list(range(dimension)):
        raise ValueError("permutation must contain each basis index exactly once")
    matrix = sp.zeros(dimension)
    for source, target in enumerate(permutation):
        matrix[target, source] = 1
    return matrix


def _phase_diagonal(order: int, exponents: Sequence[int]) -> sp.Matrix:
    if order <= 1:
        raise ValueError("phase order must be greater than one")
    omega = sp.exp(2 * sp.pi * sp.I / order)
    return sp.diag(*(omega**exponent for exponent in exponents))


def _dense_mixing_matrix() -> sp.Matrix:
    """Small exact invertible dense matrix with rational inverse."""
    return sp.Matrix(
        [
            [1, 1, 1, 1, 1],
            [1, -1, 1, -1, 2],
            [1, 1, -1, -2, 1],
            [1, -2, -1, 1, -1],
            [2, 1, -1, 1, -1],
        ]
    )


def _householder_matrix(vector: Sequence[int]) -> sp.Matrix:
    column = sp.Matrix(vector)
    denominator = (column.T * column)[0]
    if denominator == 0:
        raise ValueError("householder vector must be nonzero")
    dimension = len(vector)
    return (sp.eye(dimension) - sp.Rational(2, 1) * column * column.T / denominator).applyfunc(
        sp.simplify
    )


def _conjugate_matrix(matrix: sp.Matrix, conjugator: sp.Matrix) -> sp.Matrix:
    return (conjugator * matrix * conjugator.inv()).applyfunc(sp.simplify)


def _candidate_metadata(
    *,
    family: str,
    expected: str | None = None,
    synthetic: bool = False,
    control: bool = False,
    dense: bool = False,
    extra: tuple[tuple[str, str], ...] = (),
) -> tuple[tuple[str, str], ...]:
    base = (
        ("panel", family),
        ("family", family),
        ("synthetic", "true" if synthetic else "false"),
        ("control", "true" if control else "false"),
        ("dense", "true" if dense else "false"),
    )
    if expected is not None:
        base += (("expected", expected),)
    return base + extra


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


def c5_dense_conjugated_synthetic_split_candidate() -> ComplexCandidate:
    conjugator = _householder_matrix((1, 1, 1, 1, 1))
    candidate = c5_synthetic_split_candidate()
    return ComplexCandidate(
        name="c5_synthetic_m3_plus_m2_candidate_dense_householder_control",
        layers=tuple(
            _layer(
                f"{layer.name}_dense_conjugated",
                _conjugate_matrix(layer.matrix, conjugator),
                metadata=layer.metadata,
            )
            for layer in candidate.layers
        ),
        metadata=_candidate_metadata(
            family="dense-conjugated-control",
            synthetic=True,
            control=True,
            dense=True,
            expected="split_candidate",
            extra=(("conjugation", "householder_ones"),),
        ),
    )


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


def c5_phase_permutation_search_candidate(
    order: int,
    permutation: Sequence[int],
    *,
    label: str,
    exponents: Sequence[int] = (0, 1, 2, 3, 4),
) -> ComplexCandidate:
    phase = _phase_diagonal(order, exponents)
    permutation_matrix = _permutation_matrix(permutation)
    return ComplexCandidate(
        name=f"c5_phase_perm_{label}_order_{order}",
        layers=(
            _layer(f"phase_order_{order}", phase),
            _layer(f"perm_{label}", permutation_matrix),
        ),
        metadata=_candidate_metadata(
            family="phase-permutation",
            extra=(
                ("order", str(order)),
                ("permutation", ",".join(str(index) for index in permutation)),
            ),
        ),
    )


def c5_monomial_pair_candidate(
    order: int,
    first_permutation: Sequence[int],
    second_permutation: Sequence[int],
    *,
    label: str,
    first_exponents: Sequence[int] = (0, 1, 2, 0, 1),
    second_exponents: Sequence[int] = (0, 2, 1, 2, 0),
) -> ComplexCandidate:
    first = _phase_diagonal(order, first_exponents) * _permutation_matrix(first_permutation)
    second = _phase_diagonal(order, second_exponents) * _permutation_matrix(second_permutation)
    return ComplexCandidate(
        name=f"c5_monomial_{label}_order_{order}",
        layers=(
            _layer(f"monomial_a_{label}", first),
            _layer(f"monomial_b_{label}", second),
        ),
        metadata=_candidate_metadata(
            family="monomial",
            extra=(
                ("order", str(order)),
                ("first_permutation", ",".join(str(index) for index in first_permutation)),
                ("second_permutation", ",".join(str(index) for index in second_permutation)),
            ),
        ),
    )


def c5_finite_order_pair_candidate(
    first_permutation: Sequence[int],
    second_permutation: Sequence[int],
    *,
    label: str,
) -> ComplexCandidate:
    return ComplexCandidate(
        name=f"c5_finite_order_{label}",
        layers=(
            _layer(f"finite_perm_a_{label}", _permutation_matrix(first_permutation)),
            _layer(f"finite_perm_b_{label}", _permutation_matrix(second_permutation)),
        ),
        metadata=_candidate_metadata(
            family="finite-order",
            extra=(
                ("first_permutation", ",".join(str(index) for index in first_permutation)),
                ("second_permutation", ",".join(str(index) for index in second_permutation)),
            ),
        ),
    )


def c5_dense_hadamard_candidate(
    *,
    label: str,
    first_signs: Sequence[int],
    second_signs: Sequence[int],
) -> ComplexCandidate:
    conjugator = _dense_mixing_matrix()
    first = _conjugate_matrix(sp.diag(*first_signs), conjugator)
    second = _conjugate_matrix(sp.diag(*second_signs), conjugator)
    return ComplexCandidate(
        name=f"c5_dense_hadamard_{label}",
        layers=(
            _layer(f"dense_hadamard_a_{label}", first),
            _layer(f"dense_hadamard_b_{label}", second),
        ),
        metadata=_candidate_metadata(
            family="dense-hadamard",
            dense=True,
            extra=(
                ("first_signs", ",".join(str(sign) for sign in first_signs)),
                ("second_signs", ",".join(str(sign) for sign in second_signs)),
            ),
        ),
    )


def c5_dense_householder_candidate(
    *,
    label: str,
    first_vector: Sequence[int],
    second_vector: Sequence[int],
) -> ComplexCandidate:
    return ComplexCandidate(
        name=f"c5_dense_householder_{label}",
        layers=(
            _layer(f"householder_a_{label}", _householder_matrix(first_vector)),
            _layer(f"householder_b_{label}", _householder_matrix(second_vector)),
        ),
        metadata=_candidate_metadata(
            family="dense-householder",
            dense=True,
            extra=(
                ("first_vector", ",".join(str(item) for item in first_vector)),
                ("second_vector", ",".join(str(item) for item in second_vector)),
            ),
        ),
    )


def c5_dense_fourier_lite_candidate(
    first_permutation: Sequence[int],
    second_permutation: Sequence[int],
    *,
    label: str,
    signs: Sequence[int] = (1, -1, 1, -1, 1),
) -> ComplexCandidate:
    conjugator = _dense_mixing_matrix()
    first = _conjugate_matrix(_permutation_matrix(first_permutation), conjugator)
    second = _conjugate_matrix(sp.diag(*signs) * _permutation_matrix(second_permutation), conjugator)
    return ComplexCandidate(
        name=f"c5_dense_fourier_lite_{label}",
        layers=(
            _layer(f"dense_perm_a_{label}", first),
            _layer(f"dense_perm_b_{label}", second),
        ),
        metadata=_candidate_metadata(
            family="dense-fourier-lite",
            dense=True,
            extra=(
                ("first_permutation", ",".join(str(index) for index in first_permutation)),
                ("second_permutation", ",".join(str(index) for index in second_permutation)),
            ),
        ),
    )


def iter_complex_c5_phase_permutation_candidates(
    *,
    max_candidates: int | None = None,
    phase_orders: tuple[int, ...] = (2,),
) -> Iterable[ComplexCandidate]:
    yielded = 0
    permutations: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("doublet_swap", (0, 1, 2, 4, 3)),
        ("five_cycle", (1, 2, 3, 4, 0)),
        ("split_reassignment", (0, 1, 3, 2, 4)),
        ("two_transpositions", (1, 0, 2, 4, 3)),
    )
    for order in phase_orders:
        for label, permutation in permutations:
            yield c5_phase_permutation_search_candidate(order, permutation, label=label)
            yielded += 1
            if max_candidates is not None and yielded >= max_candidates:
                return


def iter_complex_c5_monomial_candidates(
    *,
    max_candidates: int | None = None,
    phase_orders: tuple[int, ...] = (2,),
) -> Iterable[ComplexCandidate]:
    yielded = 0
    pairs: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (
        ("cycle_plus_reflection", (1, 2, 3, 4, 0), (0, 4, 3, 2, 1)),
        ("split_mix", (0, 1, 3, 2, 4), (2, 1, 0, 4, 3)),
        ("transitive_pair", (1, 0, 2, 4, 3), (0, 2, 3, 4, 1)),
    )
    for order in phase_orders:
        for label, first, second in pairs:
            yield c5_monomial_pair_candidate(order, first, second, label=label)
            yielded += 1
            if max_candidates is not None and yielded >= max_candidates:
                return


def iter_complex_c5_finite_order_candidates(
    *,
    max_candidates: int | None = None,
) -> Iterable[ComplexCandidate]:
    yielded = 0
    pairs: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (
        ("cycle_reflection", (1, 2, 3, 4, 0), (0, 4, 3, 2, 1)),
        ("doublet_swap_with_cycle", (0, 1, 2, 4, 3), (1, 2, 3, 4, 0)),
        ("two_block_like_but_undeclared", (1, 2, 0, 4, 3), (2, 0, 1, 3, 4)),
        ("transitive_small", (1, 0, 3, 2, 4), (0, 2, 4, 1, 3)),
    )
    for label, first, second in pairs:
        yield c5_finite_order_pair_candidate(first, second, label=label)
        yielded += 1
        if max_candidates is not None and yielded >= max_candidates:
            return


def iter_complex_c5_dense_conjugated_control_candidates(
    *,
    max_candidates: int | None = None,
) -> Iterable[ComplexCandidate]:
    yield c5_dense_conjugated_synthetic_split_candidate()
    if max_candidates is not None and max_candidates <= 1:
        return


def iter_complex_c5_dense_hadamard_candidates(
    *,
    max_candidates: int | None = None,
) -> Iterable[ComplexCandidate]:
    yielded = 0
    sign_pairs: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (
        ("two_plus_three", (1, 1, 1, -1, -1), (1, -1, 1, -1, 1)),
        ("alternating", (1, -1, 1, -1, 1), (-1, 1, 1, -1, 1)),
        ("single_axis", (-1, 1, 1, 1, 1), (1, -1, 1, 1, -1)),
    )
    for label, first, second in sign_pairs:
        yield c5_dense_hadamard_candidate(label=label, first_signs=first, second_signs=second)
        yielded += 1
        if max_candidates is not None and yielded >= max_candidates:
            return


def iter_complex_c5_dense_householder_candidates(
    *,
    max_candidates: int | None = None,
) -> Iterable[ComplexCandidate]:
    yielded = 0
    vector_pairs: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (
        ("ones_vs_ramp", (1, 1, 1, 1, 1), (1, 2, 3, 4, 5)),
        ("sparse_dense", (1, 0, 1, 1, 0), (2, -1, 1, 0, 1)),
        ("mixed_sign", (1, -1, 2, -2, 1), (2, 1, -1, 1, -2)),
    )
    for label, first, second in vector_pairs:
        yield c5_dense_householder_candidate(
            label=label,
            first_vector=first,
            second_vector=second,
        )
        yielded += 1
        if max_candidates is not None and yielded >= max_candidates:
            return


def iter_complex_c5_dense_fourier_lite_candidates(
    *,
    max_candidates: int | None = None,
) -> Iterable[ComplexCandidate]:
    yielded = 0
    pairs: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (
        ("cycle_reflection", (1, 2, 3, 4, 0), (0, 4, 3, 2, 1)),
        ("split_mix", (0, 1, 3, 2, 4), (2, 1, 0, 4, 3)),
        ("transitive_pair", (1, 0, 2, 4, 3), (0, 2, 3, 4, 1)),
    )
    for label, first, second in pairs:
        yield c5_dense_fourier_lite_candidate(first, second, label=label)
        yielded += 1
        if max_candidates is not None and yielded >= max_candidates:
            return


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


def iter_complex_c5_discovered_candidates(
    *,
    family: str = "controls",
    max_candidates: int | None = None,
    phase_orders: tuple[int, ...] = (2,),
    include_conjugated: bool = True,
) -> Iterable[ComplexCandidate]:
    yielded = 0

    def emit(candidates: Iterable[ComplexCandidate]) -> Iterable[ComplexCandidate]:
        nonlocal yielded
        for candidate in candidates:
            yield candidate
            yielded += 1
            if max_candidates is not None and yielded >= max_candidates:
                return

    if family == "controls":
        yield from emit(
            iter_complex_c5_candidates(
                phase_orders=(),
                include_conjugated=include_conjugated,
            )
        )
        return

    if family == "phase-permutation":
        yield from emit(
            iter_complex_c5_phase_permutation_candidates(phase_orders=phase_orders)
        )
        return

    if family == "monomial":
        yield from emit(iter_complex_c5_monomial_candidates(phase_orders=phase_orders))
        return

    if family == "finite-order":
        yield from emit(iter_complex_c5_finite_order_candidates())
        return

    if family == "dense-conjugated-control":
        yield from emit(iter_complex_c5_dense_conjugated_control_candidates())
        return

    if family == "dense-hadamard":
        yield from emit(iter_complex_c5_dense_hadamard_candidates())
        return

    if family == "dense-householder":
        yield from emit(iter_complex_c5_dense_householder_candidates())
        return

    if family == "dense-fourier-lite":
        yield from emit(iter_complex_c5_dense_fourier_lite_candidates())
        return

    if family == "dense-all":
        streams = (
            iter_complex_c5_dense_hadamard_candidates(),
            iter_complex_c5_dense_householder_candidates(),
            iter_complex_c5_dense_fourier_lite_candidates(),
        )
        for stream in streams:
            yield from emit(stream)
            if max_candidates is not None and yielded >= max_candidates:
                return
        return

    if family == "all":
        streams = (
            iter_complex_c5_candidates(phase_orders=(), include_conjugated=include_conjugated),
            iter_complex_c5_phase_permutation_candidates(phase_orders=phase_orders),
            iter_complex_c5_monomial_candidates(phase_orders=phase_orders),
            iter_complex_c5_finite_order_candidates(),
            iter_complex_c5_dense_hadamard_candidates(),
            iter_complex_c5_dense_householder_candidates(),
            iter_complex_c5_dense_fourier_lite_candidates(),
        )
        for stream in streams:
            yield from emit(stream)
            if max_candidates is not None and yielded >= max_candidates:
                return
        return

    raise ValueError(f"unknown C5 discovered search family: {family}")
