"""Exact prediction ledger for the depth-scar sidecar.

V1 proves the built-in operator output ``Spec(2 Delta(P3)) = {0,2,6}``.
This module records the stronger consequences that follow after accepting that
operator: fixed projectors, a fixed port-space transfer kernel, endpoint-parity
selection, the Wolfenstein exponent pattern, no intrinsic CP on a tree, and the
mass-exponent semigroup for two-sided transfer factors.
"""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    defect_modes,
    transfer_operator_port_basis,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon

LAMBDA_SYMBOL = "lambda = epsilon^2"


def family_projectors() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return ``P0, P2, P6`` ordered by depth."""

    return tuple(sp.simplify(mode * mode.T) for mode in defect_modes())


def projectors_resolve_identity() -> bool:
    """Return whether the depth projectors are rank-one orthogonal and complete."""

    projectors = family_projectors()
    idempotent = all(sp.simplify(projector * projector - projector) == sp.zeros(3, 3)
                     for projector in projectors)
    rank_one = all(projector.rank() == 1 for projector in projectors)
    orthogonal = all(
        sp.simplify(left * right) == sp.zeros(3, 3)
        for i, left in enumerate(projectors)
        for j, right in enumerate(projectors)
        if i != j
    )
    complete = sp.simplify(sum(projectors, sp.zeros(3, 3)) - sp.eye(3)) == sp.zeros(3, 3)
    return idempotent and rank_one and orthogonal and complete


def transfer_kernel_from_projectors() -> sp.Matrix:
    """Return ``T = P0 + epsilon^2 P2 + epsilon^6 P6`` in the port basis."""

    eps = epsilon()
    p0, p2, p6 = family_projectors()
    return sp.simplify(p0 + eps**2 * p2 + eps**6 * p6)


def transfer_kernel_matches_v1() -> bool:
    """Return whether the projector ledger reproduces the V1 transfer operator."""

    return sp.simplify(
        transfer_kernel_from_projectors() - transfer_operator_port_basis()
    ) == sp.zeros(3, 3)


def port_transfer_relations_hold() -> bool:
    """Return the exact port-space relations forced by the path scar."""

    transfer = transfer_kernel_from_projectors()
    return (
        sp.simplify(transfer[0, 0] - transfer[2, 2]) == 0
        and sp.simplify(transfer[0, 1] - transfer[1, 2]) == 0
        and sp.simplify(transfer[0, 2] - transfer[0, 1]) != 0
    )


def endpoint_reflection_matrix() -> sp.Matrix:
    """Return the surviving endpoint reflection ``u <-> b``."""

    return sp.Matrix(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )


def mode_endpoint_parities() -> tuple[int, int, int]:
    """Return endpoint-reflection parities for the modes ordered by depth."""

    reflection = endpoint_reflection_matrix()
    parities: list[int] = []
    for mode in defect_modes():
        if sp.simplify(reflection * mode - mode) == sp.zeros(3, 1):
            parities.append(1)
        elif sp.simplify(reflection * mode + mode) == sp.zeros(3, 1):
            parities.append(-1)
        else:
            raise ValueError("mode is not an endpoint-parity eigenvector")
    return tuple(parities)


def generic_endpoint_symmetric_operator() -> sp.Matrix:
    """Return the generic operator commuting with endpoint reflection."""

    a, b, c, d, e = sp.symbols("a b c d e")
    return sp.Matrix(
        [
            [a, b, c],
            [d, e, d],
            [c, b, a],
        ]
    )


def endpoint_parity_blocks_even_odd() -> bool:
    """Return whether reflection symmetry forbids even/odd matrix elements."""

    operator = generic_endpoint_symmetric_operator()
    mode0, mode2, mode6 = defect_modes()
    return (
        sp.simplify((mode0.T * operator * mode2)[0]) == 0
        and sp.simplify((mode6.T * operator * mode2)[0]) == 0
        and sp.simplify(endpoint_reflection_matrix() * operator - operator * endpoint_reflection_matrix())
        == sp.zeros(3, 3)
    )


def leading_transfer_kernel() -> sp.Matrix:
    """Return the leading ``epsilon -> 0`` transfer kernel."""

    return family_projectors()[0]


def leading_kernel_is_democratic_rank_one() -> bool:
    """Return whether the leading kernel is democratic and rank one."""

    kernel = leading_transfer_kernel()
    return kernel == sp.ones(3, 3) / 3 and kernel.rank() == 1


def graph_cycle_rank(edges: tuple[tuple[int, int], ...], *, vertices: int = 3) -> int:
    """Return the cycle-space dimension of an undirected graph."""

    if not edges:
        return 0

    parent = list(range(vertices))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(left, right)
    components = len({find(node) for node in range(vertices)})
    return len(edges) - vertices + components


def pure_path_has_no_intrinsic_cp_holonomy() -> bool:
    """Return whether the path graph has no cycle for gauge-invariant phases."""

    return graph_cycle_rank(((0, 1), (1, 2))) == 0


def restored_triangle_has_one_loop() -> bool:
    """Return whether adding the missing edge creates one possible holonomy."""

    return graph_cycle_rank(((0, 1), (1, 2), (0, 2))) == 1


def ckm_depth_exponents() -> dict[tuple[int, int], int]:
    """Return CKM transfer-depth exponents from depth differences."""

    d0, d2, d6 = EXPECTED_DEPTH_SPECTRUM
    return {
        (1, 2): int(d2 - d0),
        (2, 3): int(d6 - d2),
        (1, 3): int(d6 - d0),
    }


def ckm_lambda_exponents() -> dict[tuple[int, int], int]:
    """Return exponents in ``lambda = epsilon^2`` units."""

    return {pair: depth // 2 for pair, depth in ckm_depth_exponents().items()}


def one_sided_mass_depth_exponents() -> tuple[int, int, int]:
    """Return one-sided depth exponents before choosing any mass model."""

    return tuple(int(depth) for depth in EXPECTED_DEPTH_SPECTRUM)


def two_sided_mass_depth_semigroup() -> tuple[int, ...]:
    """Return possible two-sided mass exponents from sums of ``{0,2,6}``."""

    depths = one_sided_mass_depth_exponents()
    return tuple(sorted({left + right for left in depths for right in depths}))


def two_sided_lambda_power_semigroup() -> tuple[int, ...]:
    """Return the same semigroup in powers of ``lambda = epsilon^2``."""

    return tuple(exponent // 2 for exponent in two_sided_mass_depth_semigroup())

