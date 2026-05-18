"""Beta audit: J-misalignment in the Higgs-like map space.

The Cl(0,4)-commutant complex-structure candidates from
``lepton.clifford_patisalam.cl04_commutant_complex_structures`` form a small
set.  Each lifts to a 32x32 complex structure on the chiral-16; only one
of them actually preserves the chiral-16 subspace under the construction
in ``patisalam_chiral16_block_matrix``.

For the Higgs-like internal map ``M`` from
``spacetime_qca.yukawa.higgs_like_charge_shift_candidate``, the natural
CP measure is whether ``M`` commutes with the lifted ``J``:

- ``[M, J] = 0``: ``M`` is "real-linear" under the J complex structure;
  the corresponding Yukawa is CP-preserving.
- ``[M, J] != 0``: ``M`` has anti-J-linear content; the corresponding
  Yukawa is CP-violating.

The J-decomposition:

```text
M = M_c + M_a
M_c = (M - J M J) / 2     commutes with J  (CP-preserving)
M_a = (M + J M J) / 2     anticommutes with J  (CP-violating)
```

The CP-violating fraction ``||M_a||^2 / ||M||^2`` (Frobenius) quantifies
how much of the Higgs map is genuinely CP-violating.

Beta passes if there exists at least one viable J such that the
CP-violating fraction is non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.cp.reuse import (
    cl04_commutant_complex_structures,
    higgs_like_charge_shift_candidate,
    patisalam_chiral16_block_matrix,
    patisalam_chosen_complex_structure,
)


def _frobenius_squared(matrix: sp.Matrix) -> sp.Expr:
    return sp.simplify(
        sum(matrix[r, c] ** 2 for r in range(matrix.rows) for c in range(matrix.cols)),
    )


def lift_cl04_j_to_chiral16(j_cl04: sp.Matrix) -> sp.Matrix | None:
    """Lift an 8x8 Cl(0,4) J candidate to the 32x32 chiral-16 carrier.

    Returns ``None`` if the lift does not preserve the chiral-16 subspace.
    """

    j_64 = sp.kronecker_product(sp.eye(8), j_cl04).applyfunc(sp.simplify)
    return patisalam_chiral16_block_matrix(j_64)


def viable_j_candidates() -> tuple[sp.Matrix, ...]:
    """Return all 32x32 J candidates that preserve the chiral-16."""

    lifted: list[sp.Matrix] = []
    for j_cl04 in cl04_commutant_complex_structures():
        lifted_j = lift_cl04_j_to_chiral16(j_cl04)
        if lifted_j is not None:
            lifted.append(lifted_j)
    return tuple(lifted)


def j_decomposition(matrix: sp.Matrix, j_full: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    """Return ``(M_c, M_a)`` with ``M = M_c + M_a``, ``M_c`` commuting
    with ``J`` and ``M_a`` anticommuting with ``J``.

    Uses ``J^2 = -I`` (so ``J^{-1} = -J``).
    """

    j_m_j = (j_full * matrix * j_full).applyfunc(sp.simplify)
    m_c = ((matrix - j_m_j) * sp.Rational(1, 2)).applyfunc(sp.simplify)
    m_a = ((matrix + j_m_j) * sp.Rational(1, 2)).applyfunc(sp.simplify)
    return m_c, m_a


def cp_violating_fraction(matrix: sp.Matrix, j_full: sp.Matrix) -> sp.Expr:
    """Return ``||M_a||^2 / ||M||^2`` (Frobenius)."""

    _, m_a = j_decomposition(matrix, j_full)
    total = _frobenius_squared(matrix)
    if total == 0:
        return sp.Integer(0)
    anti = _frobenius_squared(m_a)
    return sp.simplify(anti / total)


@lru_cache(maxsize=1)
def _cached_higgs_map() -> sp.Matrix:
    return higgs_like_charge_shift_candidate()


@dataclass(frozen=True)
class BetaAuditPayload:
    j_candidate_count_in_cl04: int
    j_candidate_count_viable: int
    higgs_frobenius_norm_squared: sp.Expr
    chosen_j_commuting_norm_squared: sp.Expr
    chosen_j_anticommuting_norm_squared: sp.Expr
    chosen_j_cp_violating_fraction: sp.Expr
    chosen_j_cp_violating_fraction_float: float
    higgs_commutes_with_chosen_j: bool
    passes: bool
    verdict: str
    interpretation: str


def beta_audit_payload() -> BetaAuditPayload:
    """Run the beta audit on the Higgs-like map under the chosen J."""

    cl04_count = len(cl04_commutant_complex_structures())
    viable = viable_j_candidates()

    higgs_map = _cached_higgs_map()
    j_chosen = patisalam_chosen_complex_structure()

    m_c, m_a = j_decomposition(higgs_map, j_chosen)
    fro_M = _frobenius_squared(higgs_map)
    fro_c = _frobenius_squared(m_c)
    fro_a = _frobenius_squared(m_a)
    fraction = sp.simplify(fro_a / fro_M) if fro_M != 0 else sp.Integer(0)
    fraction_float = float(fraction)
    commutes = (m_a.applyfunc(sp.simplify) == sp.zeros(higgs_map.rows, higgs_map.cols))

    passes = (fraction != 0) and (not commutes)

    if passes and sp.simplify(fraction - sp.Rational(1, 2)) == 0:
        interpretation = (
            "BETA PASS (maximal mixing): the Higgs-like internal map M "
            "splits exactly 50/50 between J-commuting (CP-preserving) and "
            "J-anticommuting (CP-violating) parts.  This is the strongest "
            "possible CP-content signal from the algebraic structure: M "
            "carries equal CP-even and CP-odd content under the chosen J."
        )
    elif passes:
        interpretation = (
            f"BETA PASS: the Higgs-like internal map M has nonzero "
            f"CP-violating content under the chosen J, with fraction "
            f"{fraction} ({fraction_float:.4f})."
        )
    else:
        interpretation = (
            "BETA FAIL: the Higgs-like internal map M commutes with the "
            "chosen J.  The map is purely real-linear under J; no "
            "CP-violating slot from this audit."
        )

    return BetaAuditPayload(
        j_candidate_count_in_cl04=cl04_count,
        j_candidate_count_viable=len(viable),
        higgs_frobenius_norm_squared=fro_M,
        chosen_j_commuting_norm_squared=fro_c,
        chosen_j_anticommuting_norm_squared=fro_a,
        chosen_j_cp_violating_fraction=fraction,
        chosen_j_cp_violating_fraction_float=fraction_float,
        higgs_commutes_with_chosen_j=commutes,
        passes=passes,
        verdict=("BETA PASS" if passes else "BETA FAIL"),
        interpretation=interpretation,
    )
