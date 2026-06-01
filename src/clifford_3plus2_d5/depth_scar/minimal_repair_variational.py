"""V8 minimal causal-repair variational principle.

V7 proves that two-edge, rank-2, length-3 nilpotent supports form one
``S3`` orbit: the path flag.  V8 recasts the two-edge condition as the
solution of a finite constrained optimization:

    minimize edge_count(N)

subject to:

    N**3 = 0,
    N**2 != 0,
    rank(N) = 2,
    all three ports participate.

The result is finite and exact.  The minimum cost is ``2`` and its minimizers
are exactly the V7 path-flag orbit.  The remaining open point is physical, not
combinatorial: this sidecar does not derive that the microscopic QCA must use
edge count as its causal repair cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
)
from clifford_3plus2_d5.depth_scar.support_classification import (
    OFF_DIAGONAL_POSITIONS,
    accepted_minimal_nilpotent_supports,
    accepted_minimal_orbit_keys,
    all_no_self_loop_binary_supports,
    canonical_support_key,
    directed_cycle_controls_rejected,
    directed_cycle_supports,
    flag_laplacian_from_nilpotent,
    is_length_three_nilpotent,
    is_nilpotent,
    nonminimal_length_three_supports,
    support_edge_count,
    support_key,
)


def _sorted_eigenvalues(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Return exact eigenvalues with multiplicity in deterministic order."""

    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def _support_key_set(supports: tuple[sp.Matrix, ...]) -> set[tuple[int, ...]]:
    """Return stable support keys for matrix-set comparisons."""

    return {support_key(support) for support in supports}


def causal_repair_cost(matrix: sp.Matrix) -> int:
    """Return the V8 causal repair cost: the number of directed repair edges."""

    return support_edge_count(matrix)


def active_ports(matrix: sp.Matrix) -> tuple[int, ...]:
    """Return ports that appear in at least one directed repair edge.

    The convention is ``matrix[target, source] = 1``.  A port participates if
    it appears as either a source or target of a nonzero off-diagonal entry.
    """

    ports: set[int] = set()
    for row, col in OFF_DIAGONAL_POSITIONS:
        if matrix[row, col] != 0:
            ports.add(row)
            ports.add(col)
    return tuple(sorted(ports))


def all_ports_participate(matrix: sp.Matrix) -> bool:
    """Return whether all three residual ports are active."""

    return active_ports(matrix) == (0, 1, 2)


def variational_feasible_supports() -> tuple[sp.Matrix, ...]:
    """Return supports satisfying the V8 finite variational constraints."""

    return tuple(
        matrix
        for matrix in all_no_self_loop_binary_supports()
        if is_length_three_nilpotent(matrix)
        and matrix.rank() == 2
        and all_ports_participate(matrix)
    )


def minimal_causal_repair_cost() -> int:
    """Return the minimum causal repair cost over feasible supports."""

    feasible = variational_feasible_supports()
    if not feasible:
        raise ValueError("no feasible repair supports")
    return min(causal_repair_cost(matrix) for matrix in feasible)


def minimal_causal_repair_minimizers() -> tuple[sp.Matrix, ...]:
    """Return the feasible supports with minimum causal repair cost."""

    minimum = minimal_causal_repair_cost()
    return tuple(
        matrix
        for matrix in variational_feasible_supports()
        if causal_repair_cost(matrix) == minimum
    )


def minimizer_orbit_keys() -> tuple[tuple[int, ...], ...]:
    """Return distinct ``S3`` orbit keys for V8 minimizers."""

    return tuple(sorted({canonical_support_key(matrix) for matrix in minimal_causal_repair_minimizers()}))


def minimizers_equivalent_to_flag() -> bool:
    """Return whether V8 minimizers are exactly V7's accepted flag orbit."""

    return _support_key_set(minimal_causal_repair_minimizers()) == _support_key_set(
        accepted_minimal_nilpotent_supports()
    )


def minimizer_spectra_pass() -> bool:
    """Return whether every V8 minimizer induces the target path spectra."""

    return all(
        _sorted_eigenvalues(flag_laplacian_from_nilpotent(matrix))
        == EXPECTED_LAPLACIAN_SPECTRUM
        and _sorted_eigenvalues(2 * flag_laplacian_from_nilpotent(matrix))
        == EXPECTED_DEPTH_SPECTRUM
        for matrix in minimal_causal_repair_minimizers()
    )


def nonminimal_shortcuts_excluded_by_cost() -> bool:
    """Return whether feasible shortcut supports are excluded by edge count."""

    shortcuts = nonminimal_length_three_supports()
    minimizer_keys = _support_key_set(minimal_causal_repair_minimizers())
    return (
        len(shortcuts) == 6
        and all(matrix in variational_feasible_supports() for matrix in shortcuts)
        and all(causal_repair_cost(matrix) == 3 for matrix in shortcuts)
        and all(support_key(matrix) not in minimizer_keys for matrix in shortcuts)
    )


def unconstrained_nonzero_nilpotent_minimizers() -> tuple[sp.Matrix, ...]:
    """Return minimum-cost nonzero nilpotents after dropping V8 constraints."""

    candidates = tuple(
        matrix
        for matrix in all_no_self_loop_binary_supports()
        if causal_repair_cost(matrix) > 0 and is_nilpotent(matrix)
    )
    minimum = min(causal_repair_cost(matrix) for matrix in candidates)
    return tuple(matrix for matrix in candidates if causal_repair_cost(matrix) == minimum)


def rank_one_relaxed_control_pass() -> bool:
    """Return whether relaxed minimization collapses to rank-one repairs."""

    minimizers = unconstrained_nonzero_nilpotent_minimizers()
    return (
        len(minimizers) == 6
        and all(causal_repair_cost(matrix) == 1 for matrix in minimizers)
        and all(matrix.rank() == 1 for matrix in minimizers)
        and all(sp.simplify(matrix**2) == sp.zeros(3, 3) for matrix in minimizers)
    )


def cycle_relaxed_control_pass() -> bool:
    """Return whether nilpotency is load-bearing against cycle closure."""

    cycles = directed_cycle_supports()
    return (
        len(cycles) == 2
        and all(causal_repair_cost(matrix) == 3 for matrix in cycles)
        and directed_cycle_controls_rejected()
    )


def constant_cost_minimizers() -> tuple[sp.Matrix, ...]:
    """Return minimizers if every feasible support is assigned constant cost."""

    return variational_feasible_supports()


def constant_cost_control_pass() -> bool:
    """Return whether constant cost loses the unique path-flag orbit."""

    minimizers = constant_cost_minimizers()
    orbit_keys = {canonical_support_key(matrix) for matrix in minimizers}
    return (
        len(minimizers) == 12
        and len(orbit_keys) > len(accepted_minimal_orbit_keys())
        and _support_key_set(nonminimal_length_three_supports()).issubset(_support_key_set(minimizers))
    )


def minimal_causal_repair_variational_pass() -> bool:
    """Return whether the V8 variational theorem and controls pass."""

    return (
        len(variational_feasible_supports()) == 12
        and minimal_causal_repair_cost() == 2
        and len(minimal_causal_repair_minimizers()) == 6
        and len(minimizer_orbit_keys()) == 1
        and minimizers_equivalent_to_flag()
        and minimizer_spectra_pass()
        and nonminimal_shortcuts_excluded_by_cost()
        and rank_one_relaxed_control_pass()
        and cycle_relaxed_control_pass()
        and constant_cost_control_pass()
    )


@dataclass(frozen=True)
class MinimalCausalRepairVariationalPayload:
    """V8 payload for the minimal causal-repair variational principle."""

    final_verdict: str
    feasible_support_count: int
    minimal_cost: int
    minimizer_count: int
    minimizer_orbit_count: int
    minimizers_equivalent_to_flag: bool
    minimizer_spectra_pass: bool
    shortcuts_excluded_by_cost: bool
    rank_one_relaxed_control_pass: bool
    cycle_relaxed_control_pass: bool
    constant_cost_control_pass: bool
    microscopic_cost_principle_derived: bool
    interpretation: str


def minimal_causal_repair_variational_payload() -> MinimalCausalRepairVariationalPayload:
    """Return the V8 minimal causal-repair variational verdict."""

    feasible = variational_feasible_supports()
    minimizers = minimal_causal_repair_minimizers()
    orbit_count = len(minimizer_orbit_keys())
    equivalent = minimizers_equivalent_to_flag()
    spectra = minimizer_spectra_pass()
    shortcuts = nonminimal_shortcuts_excluded_by_cost()
    rank_one_control = rank_one_relaxed_control_pass()
    cycle_control = cycle_relaxed_control_pass()
    constant_cost_control = constant_cost_control_pass()

    checks_pass = (
        len(feasible) == 12
        and minimal_causal_repair_cost() == 2
        and len(minimizers) == 6
        and orbit_count == 1
        and equivalent
        and spectra
        and shortcuts
        and rank_one_control
        and cycle_control
        and constant_cost_control
    )

    if checks_pass:
        final_verdict = "MINIMAL_CAUSAL_REPAIR_VARIATIONAL_PASS"
        interpretation = (
            "The path flag is the unique S3 orbit minimizing directed repair "
            "edge count under the length-3 nilpotent, rank-2, all-port-active "
            "constraints. The six shortcut supports remain feasible but have "
            "cost 3, so shortest causal repair excludes them. The microscopic "
            "origin of the edge-count cost principle is still not derived."
        )
    elif not equivalent or orbit_count != 1:
        final_verdict = "VARIATIONAL_MINIMIZER_NOT_PATH_KILL"
        interpretation = "The constrained edge-count minimizers are not uniquely the path-flag orbit."
    elif not rank_one_control or not cycle_control:
        final_verdict = "VARIATIONAL_CONSTRAINT_CONTROL_KILL"
        interpretation = "A relaxed-constraint control failed, so the variational constraints are unclear."
    elif not constant_cost_control or not shortcuts:
        final_verdict = "VARIATIONAL_COST_CONTROL_KILL"
        interpretation = "The edge-count cost did not separate path flags from shortcut supports."
    else:
        final_verdict = "VARIATIONAL_CONTROL_KILL"
        interpretation = "A variational count or target-spectrum control failed."

    return MinimalCausalRepairVariationalPayload(
        final_verdict=final_verdict,
        feasible_support_count=len(feasible),
        minimal_cost=minimal_causal_repair_cost(),
        minimizer_count=len(minimizers),
        minimizer_orbit_count=orbit_count,
        minimizers_equivalent_to_flag=equivalent,
        minimizer_spectra_pass=spectra,
        shortcuts_excluded_by_cost=shortcuts,
        rank_one_relaxed_control_pass=rank_one_control,
        cycle_relaxed_control_pass=cycle_control,
        constant_cost_control_pass=constant_cost_control,
        microscopic_cost_principle_derived=False,
        interpretation=interpretation,
    )
