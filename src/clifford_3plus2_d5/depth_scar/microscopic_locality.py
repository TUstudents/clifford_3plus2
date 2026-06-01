"""V9 microscopic locality/minimality theorem for the repair scar.

V7/V8 prove that minimal rank-2 length-3 nilpotent repair selects the path
flag.  V9 separates the remaining issue into two parts:

* support minimality: one-tick locality in a three-level defect filtration
  forbids the shortcut ``b -> u``;
* normalization minimality: equal unit weights require partial-isometry
  saturation of the active repair block.

The nilpotent object here is not the full unitary QCA update.  It is the
strictly height-lowering retarded repair subblock extracted from the boundary
update.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from clifford_3plus2_d5.depth_scar.local_flag_unitarity import (
    partial_isometry_forces_unit_magnitudes,
    phased_unit_flag_gauge_equivalent_to_canonical,
)
from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_laplacian_from_nilpotent,
    nilpotent_flag_operator,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
    path_laplacian,
)
from clifford_3plus2_d5.depth_scar.support_classification import (
    is_length_three_nilpotent,
    support_key,
)

PORTS = ("u", "a", "b")
HEIGHTS = (0, 1, 2)
BIPARTITE_PARITIES = (0, 1, 0)

# Matrix convention: N[target, source] is the directed edge source -> target.
PATH_REPAIR_EDGES = ((0, 1), (1, 2))
SHORTCUT_EDGE = (0, 2)


def _sorted_eigenvalues(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Return exact eigenvalues with multiplicity in deterministic order."""

    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def defect_height(port: int) -> int:
    """Return the residual defect height of a port."""

    return HEIGHTS[port]


def boundary_distance(left: int, right: int) -> int:
    """Return path distance in the residual boundary geometry ``u-a-b``."""

    return abs(left - right)


def bipartite_parity(port: int) -> int:
    """Return the residual BCC bipartite parity label."""

    return BIPARTITE_PARITIES[port]


def is_strictly_height_lowering(edge: tuple[int, int]) -> bool:
    """Return whether ``edge=(target, source)`` strictly lowers defect height."""

    target, source = edge
    return defect_height(target) < defect_height(source)


def is_one_tick_local(edge: tuple[int, int]) -> bool:
    """Return whether a repair edge is one-tick adjacent in boundary geometry."""

    target, source = edge
    return boundary_distance(target, source) == 1


def is_bipartite_allowed(edge: tuple[int, int]) -> bool:
    """Return whether an edge changes BCC bipartite parity."""

    target, source = edge
    return bipartite_parity(target) != bipartite_parity(source)


def strictly_lowering_edges() -> tuple[tuple[int, int], ...]:
    """Return all possible strictly height-lowering directed repair edges."""

    candidate_order = (*PATH_REPAIR_EDGES, SHORTCUT_EDGE)
    return tuple(edge for edge in candidate_order if is_strictly_height_lowering(edge))


def one_tick_local_lowering_edges() -> tuple[tuple[int, int], ...]:
    """Return strictly lowering edges allowed by one-tick locality."""

    return tuple(edge for edge in strictly_lowering_edges() if is_one_tick_local(edge))


def support_from_edges(edges: tuple[tuple[int, int], ...]) -> sp.Matrix:
    """Return a binary support matrix for directed repair edges."""

    matrix = sp.zeros(3, 3)
    for target, source in edges:
        matrix[target, source] = 1
    return matrix


def local_support_operator() -> sp.Matrix:
    """Return the unique one-tick local rank-complete support operator."""

    return support_from_edges(PATH_REPAIR_EDGES)


def shortcut_support_operator() -> sp.Matrix:
    """Return the monotone shortcut support allowed when locality is relaxed."""

    return support_from_edges((*PATH_REPAIR_EDGES, SHORTCUT_EDGE))


def local_rank_complete_supports() -> tuple[sp.Matrix, ...]:
    """Return rank-2 supports using only local height-lowering edges."""

    edges = one_tick_local_lowering_edges()
    supports: list[sp.Matrix] = []
    for count in range(1, len(edges) + 1):
        for subset in combinations(edges, count):
            matrix = support_from_edges(tuple(subset))
            if matrix.rank() == 2 and is_length_three_nilpotent(matrix):
                supports.append(matrix)
    return tuple(supports)


def relaxed_monotone_rank_complete_supports() -> tuple[sp.Matrix, ...]:
    """Return rank-2 length-3 supports when one-tick locality is removed."""

    edges = strictly_lowering_edges()
    supports: list[sp.Matrix] = []
    for count in range(1, len(edges) + 1):
        for subset in combinations(edges, count):
            matrix = support_from_edges(tuple(subset))
            if matrix.rank() == 2 and is_length_three_nilpotent(matrix):
                supports.append(matrix)
    return tuple(supports)


def shortcut_forbidden_by_one_tick_locality() -> bool:
    """Return whether the direct ``b -> u`` shortcut fails one-tick locality."""

    return (
        SHORTCUT_EDGE in strictly_lowering_edges()
        and SHORTCUT_EDGE not in one_tick_local_lowering_edges()
        and boundary_distance(*SHORTCUT_EDGE) == 2
        and not is_bipartite_allowed(SHORTCUT_EDGE)
    )


def microscopic_support_minimality_pass() -> bool:
    """Return whether V9a support minimality follows from local repair axioms."""

    supports = local_rank_complete_supports()
    return (
        strictly_lowering_edges() == ((*PATH_REPAIR_EDGES, SHORTCUT_EDGE))
        and one_tick_local_lowering_edges() == PATH_REPAIR_EDGES
        and shortcut_forbidden_by_one_tick_locality()
        and len(supports) == 1
        and support_key(supports[0]) == support_key(nilpotent_flag_operator())
    )


def shortcut_admitted_when_locality_is_relaxed() -> bool:
    """Return whether the shortcut support enters when one-tick locality is removed."""

    relaxed = relaxed_monotone_rank_complete_supports()
    relaxed_keys = {support_key(matrix) for matrix in relaxed}
    return (
        len(relaxed) == 2
        and support_key(local_support_operator()) in relaxed_keys
        and support_key(shortcut_support_operator()) in relaxed_keys
    )


def generic_local_repair_operator(alpha: sp.Expr, beta: sp.Expr) -> sp.Matrix:
    """Return ``alpha |u><a| + beta |a><b|``."""

    return sp.Matrix(
        [
            [0, alpha, 0],
            [0, 0, beta],
            [0, 0, 0],
        ]
    )


def generic_local_repair_nilpotent_pass() -> bool:
    """Return whether nonzero local repair is a length-3 nilpotent flag."""

    alpha, beta = sp.symbols("alpha beta", nonzero=True)
    operator = generic_local_repair_operator(alpha, beta)
    return (
        sp.simplify(operator**2 - sp.Matrix([[0, 0, alpha * beta], [0, 0, 0], [0, 0, 0]]))
        == sp.zeros(3, 3)
        and sp.simplify(operator**3) == sp.zeros(3, 3)
    )


def local_support_induces_path_laplacian() -> bool:
    """Return whether the local support induces ``Delta(P3)``."""

    return (
        flag_laplacian_from_nilpotent(local_support_operator()) == path_laplacian()
        and _sorted_eigenvalues(path_laplacian()) == EXPECTED_LAPLACIAN_SPECTRUM
        and _sorted_eigenvalues(2 * path_laplacian()) == EXPECTED_DEPTH_SPECTRUM
    )


def weighted_path_laplacian(w1: sp.Expr, w2: sp.Expr) -> sp.Matrix:
    """Return the weighted two-edge path Laplacian."""

    return sp.Matrix(
        [
            [w1, -w1, 0],
            [-w1, w1 + w2, -w2],
            [0, -w2, w2],
        ]
    )


def weighted_path_spectrum_formula(w1: sp.Expr, w2: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the closed-form weighted path spectrum."""

    radical = sp.sqrt(w1**2 - w1 * w2 + w2**2)
    return (
        sp.Integer(0),
        sp.simplify(w1 + w2 - radical),
        sp.simplify(w1 + w2 + radical),
    )


def weighted_path_spectrum_formula_pass() -> bool:
    """Return whether the weighted spectrum formula matches the exact spectrum."""

    w1, w2 = sp.symbols("w1 w2", positive=True)
    formula = weighted_path_spectrum_formula(w1, w2)
    exact = _sorted_eigenvalues(weighted_path_laplacian(w1, w2))
    return all(sp.simplify(left - right) == 0 for left, right in zip(exact, formula, strict=True))


def target_spectrum_forces_unit_weights() -> bool:
    """Return whether ``{0,1,3}`` forces equal unit weights on the weighted path."""

    w1, w2 = sp.symbols("w1 w2", real=True)
    # The nonzero eigenvalues have sum 2(w1+w2) and product 3w1w2.
    solutions = sp.solve((sp.Eq(w1 + w2, 2), sp.Eq(w1 * w2, 1)), (w1, w2), dict=True)
    return solutions == [{w1: sp.Integer(1), w2: sp.Integer(1)}]


def partial_isometry_saturation_conditional_pass() -> bool:
    """Return whether V6 supplies the conditional unit-weight normalization."""

    return (
        partial_isometry_forces_unit_magnitudes()
        and phased_unit_flag_gauge_equivalent_to_canonical()
        and target_spectrum_forces_unit_weights()
    )


def microscopic_locality_minimality_pass() -> bool:
    """Return whether V9a/V9b combined theorem gates pass."""

    return (
        microscopic_support_minimality_pass()
        and shortcut_admitted_when_locality_is_relaxed()
        and generic_local_repair_nilpotent_pass()
        and local_support_induces_path_laplacian()
        and weighted_path_spectrum_formula_pass()
        and partial_isometry_saturation_conditional_pass()
    )


@dataclass(frozen=True)
class MicroscopicLocalityPayload:
    """V9 payload for microscopic locality/minimality."""

    final_verdict: str
    support_verdict: str
    normalization_verdict: str
    monotone_edges: tuple[tuple[int, int], ...]
    one_tick_edges: tuple[tuple[int, int], ...]
    shortcut_forbidden_by_locality: bool
    local_rank_complete_support_count: int
    relaxed_rank_complete_support_count: int
    shortcut_admitted_when_locality_relaxed: bool
    local_support_induces_path_laplacian: bool
    weighted_path_formula_pass: bool
    target_spectrum_forces_unit_weights: bool
    partial_isometry_saturation_conditional_pass: bool
    height_filtration_microscopically_derived: bool
    one_tick_boundary_geometry_microscopically_derived: bool
    interpretation: str


def microscopic_locality_payload() -> MicroscopicLocalityPayload:
    """Return the V9 locality/minimality verdict."""

    support_pass = microscopic_support_minimality_pass()
    relaxed_shortcut = shortcut_admitted_when_locality_is_relaxed()
    laplacian = local_support_induces_path_laplacian()
    weighted_formula = weighted_path_spectrum_formula_pass()
    unit_weights = target_spectrum_forces_unit_weights()
    saturation = partial_isometry_saturation_conditional_pass()

    checks_pass = (
        support_pass
        and relaxed_shortcut
        and laplacian
        and weighted_formula
        and unit_weights
        and saturation
    )

    if checks_pass:
        final_verdict = "MICROSCOPIC_LOCALITY_MINIMALITY_CONDITIONAL_PASS"
        support_verdict = "V9A_MICROSCOPIC_SUPPORT_MINIMALITY_PASS"
        normalization_verdict = "V9B_MICROSCOPIC_NORMALIZATION_MINIMALITY_CONDITIONAL_PASS"
        interpretation = (
            "Given a three-level defect filtration and one-tick residual "
            "boundary geometry u-a-b, monotone rank-complete repair has exactly "
            "the path-flag support. The shortcut b->u is allowed by monotonicity "
            "but forbidden by one-tick locality. Equal unit weights still require "
            "partial-isometry saturation; without it, V9 gives a weighted path."
        )
    elif not support_pass or not relaxed_shortcut:
        final_verdict = "MICROSCOPIC_SUPPORT_LOCALITY_KILL"
        support_verdict = "V9A_MICROSCOPIC_SUPPORT_MINIMALITY_KILL"
        normalization_verdict = "V9B_NOT_REACHED"
        interpretation = "One-tick locality did not uniquely force the path-flag support."
    elif not saturation:
        final_verdict = "MICROSCOPIC_NORMALIZATION_SATURATION_KILL"
        support_verdict = "V9A_MICROSCOPIC_SUPPORT_MINIMALITY_PASS"
        normalization_verdict = "V9B_MICROSCOPIC_NORMALIZATION_MINIMALITY_KILL"
        interpretation = "Support locality passed, but partial-isometry saturation did not fix unit weights."
    else:
        final_verdict = "MICROSCOPIC_LOCALITY_CONTROL_KILL"
        support_verdict = "V9A_CONTROL_KILL"
        normalization_verdict = "V9B_CONTROL_KILL"
        interpretation = "A path-Laplacian, weighted-spectrum, or target-weight control failed."

    return MicroscopicLocalityPayload(
        final_verdict=final_verdict,
        support_verdict=support_verdict,
        normalization_verdict=normalization_verdict,
        monotone_edges=strictly_lowering_edges(),
        one_tick_edges=one_tick_local_lowering_edges(),
        shortcut_forbidden_by_locality=shortcut_forbidden_by_one_tick_locality(),
        local_rank_complete_support_count=len(local_rank_complete_supports()),
        relaxed_rank_complete_support_count=len(relaxed_monotone_rank_complete_supports()),
        shortcut_admitted_when_locality_relaxed=relaxed_shortcut,
        local_support_induces_path_laplacian=laplacian,
        weighted_path_formula_pass=weighted_formula,
        target_spectrum_forces_unit_weights=unit_weights,
        partial_isometry_saturation_conditional_pass=saturation,
        height_filtration_microscopically_derived=False,
        one_tick_boundary_geometry_microscopically_derived=False,
        interpretation=interpretation,
    )
