"""V7 finite classification of minimal nilpotent repair supports.

V6 leaves one support-level input: the microscopic boundary must realize the
length-3 repair support ``b -> a -> u``.  V7 classifies all binary directed
supports on three ports, with no self-loops, and asks what is forced by minimal
nilpotent repair.

The support convention matches the rest of ``depth_scar``: a matrix entry
``N[target, source] = 1`` represents the directed repair edge
``source -> target``.

The finite result is:

    nilpotent + N^2 != 0 + rank 2 + exactly two edges
        => one S3 orbit, represented by |u><a| + |a><b|.

If the two-edge minimality condition is removed, additional acyclic shortcut
supports survive.  Thus V7 closes the support input only modulo the named
minimality assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import sympy as sp

from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_laplacian_from_nilpotent,
    nilpotent_flag_operator,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
    k3_laplacian,
)

OFF_DIAGONAL_POSITIONS = tuple((row, col) for row in range(3) for col in range(3) if row != col)


def _sorted_eigenvalues(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Return exact eigenvalues with multiplicity in deterministic order."""

    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def support_matrix_from_bits(bits: tuple[int, ...]) -> sp.Matrix:
    """Return a no-self-loop binary support matrix from six off-diagonal bits."""

    if len(bits) != len(OFF_DIAGONAL_POSITIONS):
        raise ValueError("expected six off-diagonal bits")
    matrix = sp.zeros(3, 3)
    for bit, (row, col) in zip(bits, OFF_DIAGONAL_POSITIONS, strict=True):
        matrix[row, col] = sp.Integer(bit)
    return matrix


def all_no_self_loop_binary_supports() -> tuple[sp.Matrix, ...]:
    """Return all ``2^6`` no-self-loop binary directed supports."""

    return tuple(support_matrix_from_bits(tuple(bits)) for bits in product((0, 1), repeat=6))


def support_edge_count(matrix: sp.Matrix) -> int:
    """Return the number of directed edges in a binary support matrix."""

    return int(sum(int(matrix[row, col]) for row, col in OFF_DIAGONAL_POSITIONS))


def is_nilpotent(matrix: sp.Matrix) -> bool:
    """Return whether a three-port support is nilpotent."""

    return sp.simplify(matrix**3) == sp.zeros(3, 3)


def is_length_three_nilpotent(matrix: sp.Matrix) -> bool:
    """Return whether the support has nilpotent order exactly three."""

    return is_nilpotent(matrix) and sp.simplify(matrix**2) != sp.zeros(3, 3)


def permutation_matrix(permutation: tuple[int, int, int]) -> sp.Matrix:
    """Return the basis permutation matrix for a port relabeling."""

    matrix = sp.zeros(3, 3)
    for new_index, old_index in enumerate(permutation):
        matrix[new_index, old_index] = 1
    return matrix


def permuted_supports(matrix: sp.Matrix) -> tuple[sp.Matrix, ...]:
    """Return all port relabelings of a support."""

    return tuple(
        sp.simplify(perm * matrix * perm.T)
        for perm in (permutation_matrix(p) for p in permutations((0, 1, 2)))
    )


def support_key(matrix: sp.Matrix) -> tuple[int, ...]:
    """Return a stable tuple key for support comparisons."""

    return tuple(int(matrix[row, col]) for row in range(3) for col in range(3))


def canonical_support_key(matrix: sp.Matrix) -> tuple[int, ...]:
    """Return the lexicographically minimal key in the ``S3`` orbit."""

    return min(support_key(permuted) for permuted in permuted_supports(matrix))


def permutation_equivalent(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return whether two supports are equivalent under port relabeling."""

    return canonical_support_key(left) == canonical_support_key(right)


def accepted_minimal_nilpotent_supports() -> tuple[sp.Matrix, ...]:
    """Return minimal length-3 nilpotent rank-2 supports."""

    return tuple(
        matrix
        for matrix in all_no_self_loop_binary_supports()
        if support_edge_count(matrix) == 2
        and matrix.rank() == 2
        and is_length_three_nilpotent(matrix)
    )


def accepted_minimal_orbit_keys() -> tuple[tuple[int, ...], ...]:
    """Return the distinct ``S3`` orbit keys of accepted minimal supports."""

    return tuple(sorted({canonical_support_key(matrix) for matrix in accepted_minimal_nilpotent_supports()}))


def all_minimal_supports_equivalent_to_flag() -> bool:
    """Return whether every accepted minimal support is a relabeling of V5's flag."""

    flag = nilpotent_flag_operator()
    supports = accepted_minimal_nilpotent_supports()
    return bool(supports) and all(permutation_equivalent(matrix, flag) for matrix in supports)


def minimal_support_spectra_pass() -> bool:
    """Return whether every accepted minimal support induces the path spectra."""

    return all(
        _sorted_eigenvalues(flag_laplacian_from_nilpotent(matrix))
        == EXPECTED_LAPLACIAN_SPECTRUM
        and _sorted_eigenvalues(2 * flag_laplacian_from_nilpotent(matrix))
        == EXPECTED_DEPTH_SPECTRUM
        for matrix in accepted_minimal_nilpotent_supports()
    )


def rank_one_nilpotent_supports() -> tuple[sp.Matrix, ...]:
    """Return nonzero nilpotent supports with ``N^2 = 0``."""

    return tuple(
        matrix
        for matrix in all_no_self_loop_binary_supports()
        if support_edge_count(matrix) > 0
        and is_nilpotent(matrix)
        and sp.simplify(matrix**2) == sp.zeros(3, 3)
    )


def directed_cycle_supports() -> tuple[sp.Matrix, ...]:
    """Return directed three-cycles."""

    return tuple(
        matrix
        for matrix in all_no_self_loop_binary_supports()
        if support_edge_count(matrix) == 3 and sp.simplify(matrix**3 - sp.eye(3)) == sp.zeros(3, 3)
    )


def directed_cycle_controls_rejected() -> bool:
    """Return whether cycles fail nilpotency and return the ``K3`` sector."""

    cycles = directed_cycle_supports()
    return (
        len(cycles) == 2
        and all(not is_nilpotent(matrix) for matrix in cycles)
        and all(flag_laplacian_from_nilpotent(matrix) == k3_laplacian() for matrix in cycles)
    )


def nonminimal_length_three_supports() -> tuple[sp.Matrix, ...]:
    """Return length-3 rank-2 nilpotent supports with more than two edges."""

    return tuple(
        matrix
        for matrix in all_no_self_loop_binary_supports()
        if support_edge_count(matrix) > 2
        and matrix.rank() == 2
        and is_length_three_nilpotent(matrix)
    )


def broad_length_three_orbit_keys() -> tuple[tuple[int, ...], ...]:
    """Return orbit keys without imposing the two-edge minimality condition."""

    return tuple(
        sorted(
            {
                canonical_support_key(matrix)
                for matrix in all_no_self_loop_binary_supports()
                if matrix.rank() == 2 and is_length_three_nilpotent(matrix)
            }
        )
    )


def minimality_assumption_is_load_bearing() -> bool:
    """Return whether dropping minimal edge count admits additional support orbits."""

    return (
        len(nonminimal_length_three_supports()) > 0
        and len(broad_length_three_orbit_keys()) > len(accepted_minimal_orbit_keys())
    )


def support_classification_controls_pass() -> bool:
    """Return whether the finite-census controls distinguish the theorem."""

    return (
        len(all_no_self_loop_binary_supports()) == 64
        and len(rank_one_nilpotent_supports()) > 0
        and directed_cycle_controls_rejected()
        and minimality_assumption_is_load_bearing()
    )


@dataclass(frozen=True)
class MinimalNilpotentSupportPayload:
    """V7 payload for minimal nilpotent support classification."""

    final_verdict: str
    total_support_count: int
    accepted_minimal_count: int
    accepted_minimal_orbit_count: int
    all_minimal_equivalent_to_flag: bool
    minimal_support_spectra_pass: bool
    rank_one_controls_exist: bool
    cycle_controls_rejected: bool
    nonminimal_shortcut_count: int
    minimality_load_bearing: bool
    microscopic_minimality_derived: bool
    interpretation: str


def minimal_nilpotent_support_payload() -> MinimalNilpotentSupportPayload:
    """Return the V7 minimal support-classification verdict."""

    total = len(all_no_self_loop_binary_supports())
    accepted = accepted_minimal_nilpotent_supports()
    orbit_count = len(accepted_minimal_orbit_keys())
    equivalent = all_minimal_supports_equivalent_to_flag()
    spectra = minimal_support_spectra_pass()
    rank_one_exists = len(rank_one_nilpotent_supports()) > 0
    cycles_rejected = directed_cycle_controls_rejected()
    shortcuts = nonminimal_length_three_supports()
    minimality_load_bearing = minimality_assumption_is_load_bearing()

    checks_pass = (
        total == 64
        and len(accepted) > 0
        and orbit_count == 1
        and equivalent
        and spectra
        and rank_one_exists
        and cycles_rejected
        and minimality_load_bearing
    )

    if checks_pass:
        final_verdict = "MINIMAL_NILPOTENT_SUPPORT_CLASSIFICATION_PASS"
        interpretation = (
            "Among all 64 no-self-loop binary supports on three ports, the "
            "rank-2 length-3 nilpotent supports with exactly two edges form one "
            "S3 orbit: the path flag of V5. Dropping the two-edge minimality "
            "condition admits shortcut acyclic supports, so minimality remains "
            "a named support assumption."
        )
    elif orbit_count != 1 or not equivalent:
        final_verdict = "NILPOTENT_SUPPORT_NOT_UNIQUE_KILL"
        interpretation = "Minimal length-3 nilpotent supports are not unique up to port relabeling."
    elif not minimality_load_bearing:
        final_verdict = "MINIMALITY_ASSUMPTION_LOAD_BEARING_KILL"
        interpretation = "The finite census did not show minimality as a load-bearing assumption."
    else:
        final_verdict = "SUPPORT_CLASSIFICATION_CONTROL_KILL"
        interpretation = "A census-count, spectrum, rank-one, or cycle control failed."

    return MinimalNilpotentSupportPayload(
        final_verdict=final_verdict,
        total_support_count=total,
        accepted_minimal_count=len(accepted),
        accepted_minimal_orbit_count=orbit_count,
        all_minimal_equivalent_to_flag=equivalent,
        minimal_support_spectra_pass=spectra,
        rank_one_controls_exist=rank_one_exists,
        cycle_controls_rejected=cycles_rejected,
        nonminimal_shortcut_count=len(shortcuts),
        minimality_load_bearing=minimality_load_bearing,
        microscopic_minimality_derived=False,
        interpretation=interpretation,
    )
