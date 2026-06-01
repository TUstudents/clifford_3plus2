"""Exact repair-graph operators for the boundary depth-scar audit.

The sidecar's core question is whether the quark family depth spectrum
``{0, 2, 6}`` can be represented as a positive graph Laplacian rather than as a
hand-written diagonal spurion.

The candidate defect deletes one residual repair closure from the unbroken
``K3`` port graph, leaving the path

    u -- a -- b.

The resulting operator is ``D_scar = 2 Delta(P3)``.  The factor of two is the
BCC bipartite transfer-depth unit already used by the boundary-response track.
"""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.transfer import epsilon

VERTEX_LABELS = ("u", "a", "b")
EXPECTED_LAPLACIAN_SPECTRUM = (sp.Integer(0), sp.Integer(1), sp.Integer(3))
EXPECTED_DEPTH_SPECTRUM = (sp.Integer(0), sp.Integer(2), sp.Integer(6))


def _sorted_eigenvalues(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Return exact eigenvalues with multiplicity in deterministic order."""

    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def spectrum_dict(matrix: sp.Matrix) -> dict[sp.Expr, int]:
    """Return exact eigenvalues with multiplicities."""

    return {sp.simplify(k): v for k, v in matrix.eigenvals().items()}


def path_incidence_matrix() -> sp.Matrix:
    """Return the oriented incidence matrix for the repair path ``u-a-b``."""

    return sp.Matrix(
        [
            [1, -1, 0],
            [0, 1, -1],
        ]
    )


def path_laplacian() -> sp.Matrix:
    """Return ``Delta(P3) = partial.T * partial`` in the ``(u,a,b)`` basis."""

    incidence = path_incidence_matrix()
    return incidence.T * incidence


def depth_scar_operator() -> sp.Matrix:
    """Return ``D_scar = 2 Delta(P3)``."""

    return 2 * path_laplacian()


def k3_laplacian() -> sp.Matrix:
    """Return the unbroken residual ``K3`` Laplacian."""

    return 3 * sp.eye(3) - sp.ones(3, 3)


def doubled_k3_laplacian() -> sp.Matrix:
    """Return the BCC-doubled unbroken ``K3`` depth operator."""

    return 2 * k3_laplacian()


def hand_written_diagonal_depth_operator() -> sp.Matrix:
    """Return the non-graph-native diagonal depth control."""

    return sp.diag(*EXPECTED_DEPTH_SPECTRUM)


def defect_modes() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return normalized eigenmodes of ``Delta(P3)`` ordered by depth."""

    uniform = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    endpoint_antisymmetric = sp.Matrix([1, 0, -1]) / sp.sqrt(2)
    middle_compression = sp.Matrix([1, -2, 1]) / sp.sqrt(6)
    return uniform, endpoint_antisymmetric, middle_compression


def defect_eigenbasis_matrix() -> sp.Matrix:
    """Return the orthonormal eigenbasis matrix with defect modes as columns."""

    return sp.Matrix.hstack(*defect_modes())


def defect_mode_depths() -> tuple[sp.Expr, ...]:
    """Return the exact depth associated with each defect eigenmode."""

    operator = depth_scar_operator()
    return tuple(sp.simplify((mode.T * operator * mode)[0]) for mode in defect_modes())


def transfer_operator_eigenbasis() -> sp.Matrix:
    """Return ``epsilon**D_scar`` in the defect eigenbasis."""

    eps = epsilon()
    return sp.diag(sp.Integer(1), eps**2, eps**6)


def transfer_operator_port_basis() -> sp.Matrix:
    """Return the exact transfer operator in the ``(u,a,b)`` port basis."""

    basis = defect_eigenbasis_matrix()
    return sp.simplify(basis * transfer_operator_eigenbasis() * basis.T)


def transfer_operator_eigenvalues() -> tuple[sp.Expr, ...]:
    """Return the transfer suppression factors ordered by depth."""

    eps = epsilon()
    return (sp.Integer(1), sp.simplify(eps**2), sp.simplify(eps**6))


def transition_depth_differences() -> dict[tuple[int, int], int]:
    """Return pairwise depth differences for the ordered defect modes."""

    depths = tuple(int(depth) for depth in EXPECTED_DEPTH_SPECTRUM)
    return {
        (1, 2): depths[1] - depths[0],
        (2, 3): depths[2] - depths[1],
        (1, 3): depths[2] - depths[0],
    }


def laplacian_from_edges(edges: tuple[tuple[int, int], ...], *, vertices: int = 3) -> sp.Matrix:
    """Return the unweighted graph Laplacian for an edge list."""

    laplacian = sp.zeros(vertices, vertices)
    for left, right in edges:
        laplacian[left, left] += 1
        laplacian[right, right] += 1
        laplacian[left, right] -= 1
        laplacian[right, left] -= 1
    return laplacian


def permuted_path_laplacians() -> tuple[sp.Matrix, ...]:
    """Return the three equivalent ``K3`` scars with one missing edge."""

    return (
        laplacian_from_edges(((0, 1), (1, 2))),
        laplacian_from_edges(((0, 1), (0, 2))),
        laplacian_from_edges(((0, 2), (1, 2))),
    )


def weighted_triangle_laplacian(x: sp.Expr, y: sp.Expr) -> sp.Matrix:
    """Return the weighted ``K3`` scar with weights ``x,x,y``.

    The labels are ``w_ua = x``, ``w_ab = x``, and ``w_ub = y``.
    """

    return sp.Matrix(
        [
            [x + y, -x, -y],
            [-x, 2 * x, -x],
            [-y, -x, x + y],
        ]
    )


def weighted_triangle_eigenvalue_formula() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the exact spectrum formula for the weighted triangle scar."""

    x, y = sp.symbols("x y")
    matrix = weighted_triangle_laplacian(x, y)
    return _sorted_eigenvalues(matrix)


def path_spectrum_passes() -> bool:
    """Return whether ``Delta(P3)`` and ``D_scar`` have the target spectra."""

    return (
        _sorted_eigenvalues(path_laplacian()) == EXPECTED_LAPLACIAN_SPECTRUM
        and _sorted_eigenvalues(depth_scar_operator()) == EXPECTED_DEPTH_SPECTRUM
    )


def unbroken_k3_control_fails_hierarchy() -> bool:
    """Return whether doubled ``K3`` stays degenerate and misses ``{0,2,6}``."""

    return _sorted_eigenvalues(doubled_k3_laplacian()) == (
        sp.Integer(0),
        sp.Integer(6),
        sp.Integer(6),
    )


def hand_written_diagonal_is_not_graph_native() -> bool:
    """Return true when the diagonal control matches spectrum but differs from the scar."""

    diagonal = hand_written_diagonal_depth_operator()
    return (
        _sorted_eigenvalues(diagonal) == EXPECTED_DEPTH_SPECTRUM
        and sp.simplify(diagonal - depth_scar_operator()) != sp.zeros(3, 3)
    )


def weighted_scar_controls_pass() -> bool:
    """Return whether the path limit and ``1:1:4`` weighted scar hit the target."""

    path_limit = weighted_triangle_laplacian(sp.Integer(1), sp.Integer(0))
    strong_bond = weighted_triangle_laplacian(sp.Rational(1, 3), sp.Rational(4, 3))
    symmetric = weighted_triangle_laplacian(sp.Integer(1), sp.Integer(1))
    return (
        _sorted_eigenvalues(path_limit) == EXPECTED_LAPLACIAN_SPECTRUM
        and _sorted_eigenvalues(strong_bond) == EXPECTED_LAPLACIAN_SPECTRUM
        and _sorted_eigenvalues(symmetric) != EXPECTED_LAPLACIAN_SPECTRUM
    )

