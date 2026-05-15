"""Two-site Bloch-carrier diagnostics for Path A."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import matrix_span_rank
from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import split_projectors_3_2
from clifford_3plus2_d5.qca.bloch_rule import BlochSeedGuardrail
from clifford_3plus2_d5.qca.gates import is_real_matrix
from clifford_3plus2_d5.qca.rule_verdict import (
    RuleBlochTerm,
    RuleLayerInput,
    _bloch_layer_laurent_orthogonal,
    bloch_floquet_operators,
    center_basis_of_algebra,
    generated_algebra_closure,
    solve_central_idempotents,
)


@dataclass(frozen=True)
class EffectiveIdempotentProfile:
    rank: int
    sublattice_a_rank: int
    sublattice_b_rank: int
    off_diagonal_zero: bool
    balanced_effective_rank: int | None


@dataclass(frozen=True)
class TwoSiteBlochCertificate:
    rule_name: str
    variant: str
    period: int
    dimension: int
    term_count: int
    shifts: tuple[int, ...]
    locality_radius: int
    coefficient_matrices_real: bool
    laurent_orthogonal: bool
    seed_guardrail_passed: bool
    raw_seed_witnesses: tuple[str, ...]
    algebraic_seed_witnesses: tuple[str, ...]
    coefficient_algebra_dimension: int
    coefficient_algebra_closed: bool
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    center_dimension: int | None
    center_solved: bool
    central_idempotent_ranks: tuple[int, ...]
    effective_idempotent_profiles: tuple[EffectiveIdempotentProfile, ...]
    effective_rank_6_4_pairs: int
    route_label: str
    load_bearing_qca_bridge: bool = False


def _mode_edge_matrix(edges: tuple[tuple[int, int], ...], *, dimension: int = 10) -> sp.Matrix:
    mode_count = dimension // 2
    matrix = sp.zeros(dimension)
    for source, target in edges:
        matrix[target, source] = 1
        matrix[target + mode_count, source + mode_count] = 1
    return matrix


def _grouped_mode_edges(
    *,
    cycle: tuple[int, ...],
    source_shifts: tuple[int, ...],
    dimension: int = 10,
) -> tuple[tuple[int, sp.Matrix], ...]:
    if len(cycle) != dimension // 2 or sorted(cycle) != list(range(dimension // 2)):
        raise ValueError("cycle must be a permutation of mode indices")
    if len(source_shifts) != dimension // 2:
        raise ValueError("source_shifts must assign one shift per mode")
    terms = []
    for shift in sorted(set(source_shifts)):
        edges = tuple(
            (source, cycle[source])
            for source, source_shift in enumerate(source_shifts)
            if source_shift == shift
        )
        terms.append((shift, _mode_edge_matrix(edges, dimension=dimension)))
    return tuple(terms)


def _two_site_block(*, ab: sp.Matrix | None = None, ba: sp.Matrix | None = None) -> sp.Matrix:
    zero = sp.zeros(10)
    ab = ab if ab is not None else zero
    ba = ba if ba is not None else zero
    return zero.row_join(ab).col_join(ba.row_join(zero))


def two_site_bloch_forward_inverse_layer(
    *,
    cycle: tuple[int, ...] = (1, 2, 3, 4, 0),
    source_shifts: tuple[int, ...] = (4, 4, 4, 3, 3),
) -> RuleLayerInput:
    """Return the first two-site Path-A Bloch carrier.

    The layer acts on ``K_A + K_B``.  The forward half maps ``K_B`` to ``K_A``
    by a projector-free monomial hop with source windings ``(4,4,4,3,3)``; the
    inverse half maps ``K_A`` back to ``K_B`` with inverse shifts.  This keeps
    exact Laurent orthogonality while avoiding raw ``P_alpha/P_eta``
    coefficients.
    """

    forward_terms = _grouped_mode_edges(cycle=cycle, source_shifts=source_shifts)
    terms = [
        RuleBlochTerm(shift=shift, matrix=_two_site_block(ab=matrix))
        for shift, matrix in forward_terms
    ]
    terms.extend(
        RuleBlochTerm(shift=-shift, matrix=_two_site_block(ba=matrix.T))
        for shift, matrix in forward_terms
    )
    representative = sum((term.matrix for term in terms), sp.zeros(20))
    return RuleLayerInput(
        name=(
            "two_site_bloch_forward_inverse_"
            f"c{''.join(str(item) for item in cycle)}_"
            f"s{''.join(str(item) for item in source_shifts)}"
        ),
        matrix=representative,
        locality_radius=max(abs(shift) for shift in source_shifts),
        bloch_terms=tuple(terms),
    )


def _embedded_projectors() -> tuple[tuple[str, sp.Matrix], ...]:
    p_alpha, p_eta = split_projectors_3_2()
    zero = sp.zeros(10)
    return (
        ("P_alpha_A", p_alpha.row_join(zero).col_join(zero.row_join(zero))),
        ("P_eta_A", p_eta.row_join(zero).col_join(zero.row_join(zero))),
        ("P_alpha_B", zero.row_join(zero).col_join(zero.row_join(p_alpha))),
        ("P_eta_B", zero.row_join(zero).col_join(zero.row_join(p_eta))),
        ("P_alpha_A_plus_B", p_alpha.row_join(zero).col_join(zero.row_join(p_alpha))),
        ("P_eta_A_plus_B", p_eta.row_join(zero).col_join(zero.row_join(p_eta))),
    )


def _is_obvious_seed_projector(matrix: sp.Matrix) -> bool:
    if matrix == sp.zeros(matrix.rows) or matrix == identity(matrix.rows):
        return False
    if matrix * matrix != matrix:
        return False
    if matrix != matrix.T:
        return False
    rank = matrix.rank()
    return 0 < rank < matrix.rows


def _matrix_in_span(matrix: sp.Matrix, basis: tuple[sp.Matrix, ...]) -> bool:
    return matrix_span_rank((*basis, matrix)) == matrix_span_rank(basis)


def two_site_seed_guardrail(
    layer: RuleLayerInput,
    *,
    max_coefficient_algebra_dimension: int = 32,
) -> tuple[BlochSeedGuardrail, int, bool]:
    targets = _embedded_projectors()
    raw_witnesses: list[str] = []
    for term in layer.bloch_terms:
        for name, target in targets:
            if term.matrix == target:
                raw_witnesses.append(f"{layer.name}:shift_{term.shift}:{name}")
        if _is_obvious_seed_projector(term.matrix):
            raw_witnesses.append(
                f"{layer.name}:shift_{term.shift}:projector_rank_{term.matrix.rank()}"
            )

    coefficient_closure = generated_algebra_closure(
        tuple(term.matrix for term in layer.bloch_terms),
        dimension=20,
        max_dimension=max_coefficient_algebra_dimension,
    )
    algebraic_witnesses = []
    if coefficient_closure.closed:
        for name, target in targets:
            if _matrix_in_span(target, coefficient_closure.basis):
                algebraic_witnesses.append(f"coefficient_algebra:{name}")
    return (
        BlochSeedGuardrail(
            raw_seed_witnesses=tuple(raw_witnesses),
            algebraic_seed_witnesses=tuple(algebraic_witnesses),
        ),
        len(coefficient_closure.basis),
        coefficient_closure.closed,
    )


def _effective_profile(matrix: sp.Matrix) -> EffectiveIdempotentProfile:
    zero = sp.zeros(10)
    aa = matrix[:10, :10]
    ab = matrix[:10, 10:20]
    ba = matrix[10:20, :10]
    bb = matrix[10:20, 10:20]
    off_diagonal_zero = ab == zero and ba == zero
    rank_a = aa.rank()
    rank_b = bb.rank()
    return EffectiveIdempotentProfile(
        rank=matrix.rank(),
        sublattice_a_rank=rank_a,
        sublattice_b_rank=rank_b,
        off_diagonal_zero=off_diagonal_zero,
        balanced_effective_rank=rank_a if off_diagonal_zero and rank_a == rank_b else None,
    )


def _effective_rank_6_4_pairs(
    profiles: tuple[EffectiveIdempotentProfile, ...],
) -> int:
    return sum(1 for profile in profiles if profile.balanced_effective_rank == 6) * sum(
        1 for profile in profiles if profile.balanced_effective_rank == 4
    )


def _route_label(
    *,
    laurent_orthogonal: bool,
    seed_guardrail_passed: bool,
    generated_algebra_closed: bool,
    central_idempotent_ranks: tuple[int, ...],
    effective_rank_6_4_pairs: int,
) -> str:
    if not laurent_orthogonal:
        return "two_site_invalid_laurent"
    if not seed_guardrail_passed:
        return "two_site_seed_guardrail_rejected"
    if not generated_algebra_closed:
        return "two_site_cap_boundary"
    if effective_rank_6_4_pairs:
        return "two_site_effective_6_4_unresolved"
    if central_idempotent_ranks == (0, 20):
        return "two_site_trivial_center_no_effective_split"
    return "two_site_no_effective_6_4"


def two_site_bloch_certificate(
    *,
    variant: str = "winding-4-3",
    max_generated_algebra_dimension: int = 16,
    max_center_dimension: int = 8,
    max_coefficient_algebra_dimension: int = 32,
) -> TwoSiteBlochCertificate:
    if variant == "winding-4-3":
        layer = two_site_bloch_forward_inverse_layer(source_shifts=(4, 4, 4, 3, 3))
    elif variant == "uniform":
        layer = two_site_bloch_forward_inverse_layer(source_shifts=(1, 1, 1, 1, 1))
    else:
        raise ValueError(f"unknown two-site Bloch variant: {variant}")
    coefficient_real = all(is_real_matrix(term.matrix) for term in layer.bloch_terms)
    laurent = _bloch_layer_laurent_orthogonal(layer, dimension=20)
    guardrail, coefficient_dimension, coefficient_closed = two_site_seed_guardrail(
        layer,
        max_coefficient_algebra_dimension=max_coefficient_algebra_dimension,
    )
    samples = bloch_floquet_operators((layer,), bloch_period=12, dimension=20)
    closure = generated_algebra_closure(
        samples,
        dimension=20,
        max_dimension=max_generated_algebra_dimension,
    )

    center_dimension = None
    center_solved = False
    idempotent_ranks: tuple[int, ...] = ()
    profiles: tuple[EffectiveIdempotentProfile, ...] = ()
    if closure.closed:
        center = center_basis_of_algebra(closure.basis, dimension=20)
        center_dimension = len(center)
        center_solved, idempotents = solve_central_idempotents(
            center,
            max_center_dimension=max_center_dimension,
            dimension=20,
        )
        idempotent_ranks = tuple(item.rank for item in idempotents)
        profiles = tuple(_effective_profile(item.matrix) for item in idempotents)
    effective_pairs = _effective_rank_6_4_pairs(profiles)

    return TwoSiteBlochCertificate(
        rule_name=layer.name,
        variant=variant,
        period=12,
        dimension=20,
        term_count=len(layer.bloch_terms),
        shifts=tuple(term.shift for term in layer.bloch_terms),
        locality_radius=layer.locality_radius,
        coefficient_matrices_real=coefficient_real,
        laurent_orthogonal=laurent,
        seed_guardrail_passed=guardrail.passed,
        raw_seed_witnesses=guardrail.raw_seed_witnesses,
        algebraic_seed_witnesses=guardrail.algebraic_seed_witnesses,
        coefficient_algebra_dimension=coefficient_dimension,
        coefficient_algebra_closed=coefficient_closed,
        generated_algebra_dimension=len(closure.basis),
        generated_algebra_closed=closure.closed,
        center_dimension=center_dimension,
        center_solved=center_solved,
        central_idempotent_ranks=idempotent_ranks,
        effective_idempotent_profiles=profiles,
        effective_rank_6_4_pairs=effective_pairs,
        route_label=_route_label(
            laurent_orthogonal=laurent,
            seed_guardrail_passed=guardrail.passed,
            generated_algebra_closed=closure.closed,
            central_idempotent_ranks=idempotent_ranks,
            effective_rank_6_4_pairs=effective_pairs,
        ),
    )
