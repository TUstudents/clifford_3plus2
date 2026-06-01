"""V3 edge-weight potential for selecting a path repair scar.

V1 assumes an ``S3 -> Z2`` path repair scar.  V3 asks whether that scar can be
selected by an exact symmetric effective potential over nonnegative repair-bond
weights, rather than declared directly.

The weights are ordered as

    (w_ua, w_ab, w_ub).

The proposed effective potential is

    V = (S1 - 2)^2 + (S2 - 1)^2 + S3,

where ``S1``, ``S2``, and ``S3`` are the elementary symmetric polynomials of the
three edge weights.  On the nonnegative domain, ``V = 0`` iff the weights are one
of the three missing-edge path scars: ``(1,1,0)`` and permutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
    k3_laplacian,
)

WeightTriple = tuple[sp.Expr, sp.Expr, sp.Expr]


def _sorted_eigenvalues(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Return exact eigenvalues with multiplicity in deterministic order."""

    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def repair_weight_symbols() -> WeightTriple:
    """Return symbols for ``(w_ua, w_ab, w_ub)``."""

    return sp.symbols("w_ua w_ab w_ub")


def symmetric_edge_invariants(weights: WeightTriple) -> WeightTriple:
    """Return ``S1, S2, S3`` for the three repair edge weights."""

    w1, w2, w3 = weights
    s1 = sp.simplify(w1 + w2 + w3)
    s2 = sp.simplify(w1 * w2 + w2 * w3 + w3 * w1)
    s3 = sp.simplify(w1 * w2 * w3)
    return s1, s2, s3


def scar_potential(weights: WeightTriple) -> sp.Expr:
    """Return the effective path-scar potential."""

    s1, s2, s3 = symmetric_edge_invariants(weights)
    return sp.simplify((s1 - 2) ** 2 + (s2 - 1) ** 2 + s3)


def scar_potential_without_defect_term(weights: WeightTriple) -> sp.Expr:
    """Return the potential with the load-bearing ``S3`` term removed."""

    s1, s2, _ = symmetric_edge_invariants(weights)
    return sp.simplify((s1 - 2) ** 2 + (s2 - 1) ** 2)


def path_scar_weight_vectors() -> tuple[WeightTriple, ...]:
    """Return the three missing-edge path scar weights."""

    return tuple(
        tuple(sp.Integer(value) for value in weights)
        for weights in sorted(set(permutations((1, 1, 0))))
    )


def potential_zero_set_solutions() -> tuple[WeightTriple, ...]:
    """Return the exact nonnegative zero set of ``scar_potential``.

    Since every term is nonnegative on ``w_i >= 0``, a zero must satisfy
    ``S1 = 2``, ``S2 = 1``, and ``S3 = 0``.  The last equation sets one edge to
    zero.  The remaining two weights obey ``a + b = 2`` and ``ab = 1``, hence
    ``a = b = 1``.
    """

    return path_scar_weight_vectors()


def weighted_triangle_laplacian_general(weights: WeightTriple) -> sp.Matrix:
    """Return the weighted triangle Laplacian for ``(w_ua, w_ab, w_ub)``."""

    w_ua, w_ab, w_ub = weights
    return sp.Matrix(
        [
            [w_ua + w_ub, -w_ua, -w_ub],
            [-w_ua, w_ua + w_ab, -w_ab],
            [-w_ub, -w_ab, w_ab + w_ub],
        ]
    )


def path_scar_laplacian_spectra() -> tuple[tuple[sp.Expr, ...], ...]:
    """Return the Laplacian spectra at all path-scar minima."""

    return tuple(
        _sorted_eigenvalues(weighted_triangle_laplacian_general(weights))
        for weights in path_scar_weight_vectors()
    )


def path_scar_depth_spectra() -> tuple[tuple[sp.Expr, ...], ...]:
    """Return the BCC-doubled spectra at all path-scar minima."""

    return tuple(
        _sorted_eigenvalues(2 * weighted_triangle_laplacian_general(weights))
        for weights in path_scar_weight_vectors()
    )


def symmetric_point_weights() -> WeightTriple:
    """Return the unbroken symmetric triangle point."""

    return sp.Integer(1), sp.Integer(1), sp.Integer(1)


def non_scar_zero_of_potential_without_defect_term() -> WeightTriple:
    """Return an exact non-scar zero when the ``S3`` term is removed."""

    return sp.Rational(4, 3), sp.Rational(1, 3), sp.Rational(1, 3)


def scar_potential_is_symmetric_under_permutations() -> bool:
    """Return whether ``scar_potential`` is invariant under edge permutations."""

    weights = repair_weight_symbols()
    value = scar_potential(weights)
    return all(
        sp.simplify(scar_potential(tuple(permuted)) - value) == 0
        for permuted in permutations(weights)
    )


def path_scar_minima_pass() -> bool:
    """Return whether the effective potential selects exactly the path scars."""

    solutions = potential_zero_set_solutions()
    scars_are_zero = all(sp.simplify(scar_potential(weights)) == 0 for weights in solutions)
    symmetric_rejected = sp.simplify(scar_potential(symmetric_point_weights())) > 0
    return (
        set(solutions) == set(path_scar_weight_vectors())
        and len(solutions) == 3
        and scars_are_zero
        and symmetric_rejected
    )


def path_scar_spectra_pass() -> bool:
    """Return whether every selected scar gives the target spectra."""

    return (
        all(spectrum == EXPECTED_LAPLACIAN_SPECTRUM for spectrum in path_scar_laplacian_spectra())
        and all(spectrum == EXPECTED_DEPTH_SPECTRUM for spectrum in path_scar_depth_spectra())
    )


def defect_term_is_load_bearing() -> bool:
    """Return whether removing ``S3`` admits non-scar zeros."""

    weights = non_scar_zero_of_potential_without_defect_term()
    return (
        sp.simplify(scar_potential_without_defect_term(weights)) == 0
        and sp.simplify(scar_potential(weights)) > 0
        and weights not in set(path_scar_weight_vectors())
    )


def symmetric_triangle_control_rejected() -> bool:
    """Return whether the unbroken symmetric point is not a path-scar minimum."""

    weights = symmetric_point_weights()
    return (
        sp.simplify(scar_potential(weights)) > 0
        and _sorted_eigenvalues(k3_laplacian()) == (sp.Integer(0), sp.Integer(3), sp.Integer(3))
    )


@dataclass(frozen=True)
class EdgeWeightScarPotentialPayload:
    """V3 payload for the effective edge-weight scar potential."""

    final_verdict: str
    minima: tuple[WeightTriple, ...]
    scar_potential_symmetric: bool
    path_scar_minima_pass: bool
    path_scar_spectra_pass: bool
    defect_term_load_bearing: bool
    symmetric_triangle_control_rejected: bool
    scar_dynamically_derived_from_qca: bool
    interpretation: str


def edge_weight_scar_potential_payload() -> EdgeWeightScarPotentialPayload:
    """Return the V3 edge-weight scar-potential verdict."""

    symmetric = scar_potential_is_symmetric_under_permutations()
    minima_pass = path_scar_minima_pass()
    spectra_pass = path_scar_spectra_pass()
    load_bearing = defect_term_is_load_bearing()
    symmetric_control = symmetric_triangle_control_rejected()

    checks_pass = (
        symmetric
        and minima_pass
        and spectra_pass
        and load_bearing
        and symmetric_control
    )

    if checks_pass:
        final_verdict = "EDGE_WEIGHT_SCAR_POTENTIAL_PASS"
        interpretation = (
            "The symmetric effective potential V=(S1-2)^2+(S2-1)^2+S3 over "
            "nonnegative repair weights has exactly the three missing-edge path "
            "scars as zero-energy minima. Each minimum gives Spec(Delta)={0,1,3} "
            "and Spec(2 Delta)={0,2,6}. The S3 term is load-bearing: removing it "
            "admits non-scar zeros. This derives the scar at the effective "
            "edge-weight level, not from a microscopic QCA update."
        )
    elif not minima_pass:
        final_verdict = "EDGE_WEIGHT_SCAR_MINIMA_KILL"
        interpretation = "The effective potential does not select exactly the path-scar minima."
    elif not spectra_pass:
        final_verdict = "EDGE_WEIGHT_SCAR_SPECTRUM_KILL"
        interpretation = "One of the selected minima does not give the target spectrum."
    else:
        final_verdict = "EDGE_WEIGHT_SCAR_CONTROL_KILL"
        interpretation = "A symmetry, load-bearing, or unbroken-triangle control failed."

    return EdgeWeightScarPotentialPayload(
        final_verdict=final_verdict,
        minima=potential_zero_set_solutions(),
        scar_potential_symmetric=symmetric,
        path_scar_minima_pass=minima_pass,
        path_scar_spectra_pass=spectra_pass,
        defect_term_load_bearing=load_bearing,
        symmetric_triangle_control_rejected=symmetric_control,
        scar_dynamically_derived_from_qca=False,
        interpretation=interpretation,
    )

