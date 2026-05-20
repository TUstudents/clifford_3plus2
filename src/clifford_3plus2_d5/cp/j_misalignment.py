"""Beta audit: J-misalignment in the Higgs-like map space.

The Cl(0,4)-commutant complex-structure candidates from
``lepton.clifford_patisalam.cl04_commutant_complex_structures`` form a small
set.  Each lifts to a 32x32 complex structure on the chiral-16; only one
of them actually preserves the chiral-16 subspace under the construction
in ``patisalam_chiral16_block_matrix``.

For the Higgs-like internal map ``M`` from
``spacetime_qca.yukawa.higgs_like_charge_shift_candidate``, the natural
question is whether ``M`` commutes with the lifted ``J``:

- ``[M, J] = 0``: ``M`` is real-linear under the J complex structure
  (J-commuting).
- ``[M, J] != 0``: ``M`` has anti-J-linear content (J-anticommuting).

The J-decomposition (using ``J² = -I``):

```text
M = M_c + M_a
M_c = (M - J M J) / 2     commutes with J  (J-commuting / real-linear)
M_a = (M + J M J) / 2     anticommutes with J  (J-anticommuting / anti-J-linear)
```

**Scope note** (was renamed away from "CP" in the Tier-A audit, 2026-05-20):
This decomposition measures algebraic anti-J-linear content of the
Higgs-like map under a chosen complex structure J.  It is NOT a
physical CP measurement — CP would require an explicit transformation
of the dynamical fields and the resulting symmetry condition on the
action, not just an anti-linear projection on a static algebraic map.
The original code identified anti-J-linear content with CP violation;
that identification has been retired.  The J-decomposition remains
useful as an algebraic structural property of the Higgs-like map
space; it is not load-bearing for any physical CP claim.

The J-anticommuting fraction ``||M_a||^2 / ||M||^2`` (Frobenius) quantifies
how much of the Higgs map is anti-J-linear under the chosen J.

Beta passes if there exists at least one viable J such that the
J-anticommuting fraction is non-zero.
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
from clifford_3plus2_d5.spacetime_qca.yukawa import (
    color_singlet_charge_shift_basis,
    conjugate_charge_shift_component,
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


def j_anticommuting_fraction(matrix: sp.Matrix, j_full: sp.Matrix) -> sp.Expr:
    """Return the J-anticommuting Frobenius fraction ``||M_a||^2 / ||M||^2``.

    Measures the fraction of the Frobenius norm of ``M`` carried by its
    J-anticommuting component (anti-J-linear content under the chosen
    complex structure J).
    """

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
    chosen_j_anticommuting_fraction: sp.Expr
    chosen_j_anticommuting_fraction_float: float
    higgs_commutes_with_chosen_j: bool
    passes: bool
    verdict: str
    interpretation: str


def beta_audit_payload() -> BetaAuditPayload:
    """Run the beta audit on the Higgs-like map under the chosen J.

    Measures the J-anticommuting fraction of the Higgs-like internal map
    under the chosen complex structure J.  This is an algebraic
    structural property of the Higgs map space; it does NOT carry a
    direct physical-CP interpretation.
    """

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
            "splits exactly 50/50 between J-commuting (real-linear) and "
            "J-anticommuting (anti-J-linear) parts.  This is a clean "
            "algebraic structural feature of the dim-4 Higgs-like map "
            "space: M carries equal J-commuting and J-anticommuting "
            "content under the chosen J.  Note: this is NOT a physical "
            "CP measurement — it is an algebraic property of the static "
            "Higgs map relative to the chosen complex structure."
        )
    elif passes:
        interpretation = (
            f"BETA PASS: the Higgs-like internal map M has nonzero "
            f"J-anticommuting content under the chosen J, with fraction "
            f"{fraction} ({fraction_float:.4f})."
        )
    else:
        interpretation = (
            "BETA FAIL: the Higgs-like internal map M commutes with the "
            "chosen J.  The map is purely real-linear under J; no "
            "J-anticommuting content from this audit."
        )

    return BetaAuditPayload(
        j_candidate_count_in_cl04=cl04_count,
        j_candidate_count_viable=len(viable),
        higgs_frobenius_norm_squared=fro_M,
        chosen_j_commuting_norm_squared=fro_c,
        chosen_j_anticommuting_norm_squared=fro_a,
        chosen_j_anticommuting_fraction=fraction,
        chosen_j_anticommuting_fraction_float=fraction_float,
        higgs_commutes_with_chosen_j=commutes,
        passes=passes,
        verdict=("BETA PASS" if passes else "BETA FAIL"),
        interpretation=interpretation,
    )


# ---- Multi-element beta audit (covers all 4 basis + 4 transposes) ----


@dataclass(frozen=True)
class MultiElementBetaAuditPayload:
    basis_dimension: int
    per_basis_fractions: tuple[sp.Expr, ...]
    per_transpose_fractions: tuple[sp.Expr, ...]
    all_equal_one_half: bool
    any_zero: bool
    passes: bool
    verdict: str
    interpretation: str


def multi_element_beta_audit_payload() -> MultiElementBetaAuditPayload:
    """Iterate the full dim-4 Higgs-like basis and the 4 transposes.

    Computes the J-anticommuting Frobenius fraction for each.  Verdict
    distinguishes:

    - all 8 equal exactly 1/2 (ROBUST PASS),
    - all 8 non-zero but not all 1/2 (PASS),
    - some zero (MIXED),
    - all zero (FAIL).
    """

    j_chosen = patisalam_chosen_complex_structure()
    basis = color_singlet_charge_shift_basis()
    per_basis: list[sp.Expr] = []
    per_transpose: list[sp.Expr] = []
    for matrix in basis:
        per_basis.append(j_anticommuting_fraction(matrix, j_chosen))
        per_transpose.append(
            j_anticommuting_fraction(conjugate_charge_shift_component(matrix), j_chosen),
        )

    all_fractions = tuple(per_basis) + tuple(per_transpose)
    half = sp.Rational(1, 2)
    all_equal_half = all(sp.simplify(f - half) == 0 for f in all_fractions)
    any_zero = any(sp.simplify(f) == 0 for f in all_fractions)
    all_nonzero = all(sp.simplify(f) != 0 for f in all_fractions)

    passes = all_nonzero

    if all_equal_half:
        verdict = "BETA-MULTI ROBUST PASS"
        interpretation = (
            "All 4 basis elements and all 4 transpose components have "
            "J-anticommuting fraction exactly 1/2 under the chosen J.  "
            "The 50/50 mixing is a universal algebraic feature of the "
            "dim-4 Higgs-like map space, not an artifact of basis[0].  "
            "This is a structural property of the algebra, not a "
            "physical-CP measurement."
        )
    elif passes:
        verdict = "BETA-MULTI PASS"
        interpretation = (
            "All 8 elements have non-zero J-anticommuting fraction, but "
            "magnitudes differ.  Anti-J-linear content is generic but the "
            "exact 1/2 value was basis[0]-specific."
        )
    elif any_zero:
        zero_count = sum(1 for f in all_fractions if sp.simplify(f) == 0)
        verdict = "BETA-MULTI MIXED"
        interpretation = (
            f"{zero_count} of {len(all_fractions)} elements have "
            f"J-anticommuting fraction zero.  Anti-J-linear content is "
            f"basis-dependent within the dim-4 space.  Report the "
            f"distribution; the 50/50 result is not universal."
        )
    else:
        verdict = "BETA-MULTI FAIL"
        interpretation = (
            "All elements have J-anticommuting fraction zero.  This would "
            "contradict the basis[0] result and indicates a regression."
        )

    return MultiElementBetaAuditPayload(
        basis_dimension=len(basis),
        per_basis_fractions=tuple(per_basis),
        per_transpose_fractions=tuple(per_transpose),
        all_equal_one_half=all_equal_half,
        any_zero=any_zero,
        passes=passes,
        verdict=verdict,
        interpretation=interpretation,
    )
