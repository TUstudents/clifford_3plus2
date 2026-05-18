"""BT-1 kill test: Yukawa overlap matrix from triality projection.

Construction (from ``PLAN.md``):

    Y_ij = <Pi_SM (tau^i v_*), Pi_SM (tau^j v_*)>

where ``v_*`` is a fixed SM Cartan basis vector, ``Pi_SM`` is orthogonal
projection onto the 3-dim SM-inside-Spin(8) Cartan subspace, and the inner
product is the Euclidean inner product on R^4.

The default ``v_*`` is the restricted hypercharge ``Y'``.  This matches the
physical picture where the Higgs is aligned with the U(1)_Y direction
(color singlet, SU(2)_L doublet with hypercharge 1/2).

Pass condition: non-zero off-diagonal entries AND non-degenerate
eigenvalues (3 distinct eigenvalues).

Fail condition: Y trivially diagonal (no flavor mixing) or fewer than 3
distinct eigenvalues.

Documented feature: with ``v_* = Y'``, the resulting Y has rank 2 (one
eigenvalue is exactly zero) because of a residual ``H_1 <-> H_2`` swap
symmetry in the SM Cartan that survives the projection.  This is a
known artifact of the choice of v_*; it carries through to BT-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.broken_triality.reuse import (
    apply_triality_to_cartan_vector,
    g_sm_8_cartan_basis_coords,
    restricted_hypercharge_cartan_coords,
)


def sm_cartan_span_matrix() -> sp.Matrix:
    """Return a 4x3 matrix whose columns are the SM Cartan basis vectors."""

    return sp.Matrix.hstack(*g_sm_8_cartan_basis_coords())


def project_onto_sm_cartan(vector: sp.Matrix) -> sp.Matrix:
    """Orthogonal projection of a 4x1 vector onto the SM Cartan span.

    Uses Euclidean inner product on R^4.
    """

    span = sm_cartan_span_matrix()
    coordinates = span.solve_least_squares(vector)
    return (span * coordinates).applyfunc(sp.simplify)


def triality_orbit(v_star: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return ``(v, tau v, tau^2 v)`` as 4x1 vectors."""

    v_0 = v_star.applyfunc(sp.simplify)
    v_1 = apply_triality_to_cartan_vector(v_0)
    v_2 = apply_triality_to_cartan_vector(v_1)
    return v_0, v_1, v_2


def projected_triality_orbit(
    v_star: sp.Matrix,
) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return ``(Pi_SM v, Pi_SM tau v, Pi_SM tau^2 v)`` as 4x1 vectors."""

    return tuple(project_onto_sm_cartan(v) for v in triality_orbit(v_star))


def yukawa_overlap_matrix(v_star: sp.Matrix | None = None) -> sp.Matrix:
    """Return the 3x3 BT-1 Yukawa overlap matrix.

    Default ``v_*`` is ``Y'`` (restricted hypercharge), matching the
    physical picture of a Higgs aligned with U(1)_Y.
    """

    if v_star is None:
        v_star = restricted_hypercharge_cartan_coords()
    u_0, u_1, u_2 = projected_triality_orbit(v_star)
    u = (u_0, u_1, u_2)
    matrix = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            entry = sum(u[i][k, 0] * u[j][k, 0] for k in range(4))
            matrix[i, j] = sp.simplify(entry)
    return matrix


@lru_cache(maxsize=1)
def default_yukawa() -> sp.Matrix:
    return yukawa_overlap_matrix()


def yukawa_eigenvalues() -> dict[sp.Expr, int]:
    """Return the eigenvalue spectrum of the default Yukawa as ``{value: mult}``."""

    return {sp.simplify(value): mult for value, mult in default_yukawa().eigenvals().items()}


def yukawa_nonzero_eigenvalues() -> tuple[sp.Expr, ...]:
    """Return non-zero eigenvalues of the default Yukawa, sorted descending."""

    spectrum = yukawa_eigenvalues()
    values: list[sp.Expr] = []
    for value, multiplicity in spectrum.items():
        if sp.simplify(value) != 0:
            values.extend([value] * multiplicity)
    return tuple(sorted(values, key=lambda v: -float(v)))


def yukawa_rank() -> int:
    return default_yukawa().rank()


def yukawa_off_diagonal_entries() -> tuple[sp.Expr, ...]:
    y = default_yukawa()
    return tuple(y[i, j] for i in range(3) for j in range(3) if i != j)


def all_off_diagonals_nonzero() -> bool:
    return all(entry != 0 for entry in yukawa_off_diagonal_entries())


def eigenvalues_non_degenerate() -> bool:
    return len(yukawa_eigenvalues()) == 3


@dataclass(frozen=True)
class BT1Audit:
    v_star_description: str
    yukawa_matrix_entries: tuple[tuple[sp.Expr, ...], ...]
    yukawa_rank: int
    off_diagonal_entries_all_nonzero: bool
    eigenvalue_spectrum: dict[sp.Expr, int]
    distinct_eigenvalue_count: int
    nonzero_eigenvalues: tuple[sp.Expr, ...]
    passes: bool
    verdict: str
    interpretation: str


def bt1_audit_payload() -> BT1Audit:
    y = default_yukawa()
    entries = tuple(tuple(y[i, j] for j in range(3)) for i in range(3))
    rank = y.rank()
    spectrum = yukawa_eigenvalues()
    distinct = len(spectrum)
    nonzero = yukawa_nonzero_eigenvalues()
    off_diag_nonzero = all_off_diagonals_nonzero()
    non_degenerate = eigenvalues_non_degenerate()
    passes = off_diag_nonzero and non_degenerate

    if passes:
        if rank == 3:
            interpretation = (
                "Yukawa has full rank with 3 distinct eigenvalues and non-zero "
                "off-diagonal mixing.  BT-1 passes cleanly."
            )
        else:
            interpretation = (
                f"Yukawa has 3 distinct eigenvalues with off-diagonal mixing, "
                f"but rank {rank} < 3 (one eigenvalue is zero).  BT-1 passes "
                f"the literal kill condition; the rank deficit is forced by a "
                f"residual H_1 <-> H_2 symmetry in the SM Cartan that survives "
                f"the projection.  BT-2 will need to handle the zero eigenvalue."
            )
        verdict = "BT-1 PASS"
    else:
        reasons: list[str] = []
        if not off_diag_nonzero:
            reasons.append("at least one off-diagonal entry is zero")
        if not non_degenerate:
            reasons.append("eigenvalues degenerate (fewer than 3 distinct)")
        verdict = "BT-1 FAIL"
        interpretation = (
            "BT-1 fails: " + "; ".join(reasons) + ".  No flavor structure "
            "from triality projection; the broken-triality program is closed."
        )

    return BT1Audit(
        v_star_description="restricted hypercharge Y' (default)",
        yukawa_matrix_entries=entries,
        yukawa_rank=rank,
        off_diagonal_entries_all_nonzero=off_diag_nonzero,
        eigenvalue_spectrum=spectrum,
        distinct_eigenvalue_count=distinct,
        nonzero_eigenvalues=nonzero,
        passes=passes,
        verdict=verdict,
        interpretation=interpretation,
    )
