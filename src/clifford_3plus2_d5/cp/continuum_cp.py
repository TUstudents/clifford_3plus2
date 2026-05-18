"""O(ε²) continuum audit: extract H^(1) of the BCC Dirac walk and decompose.

The bare massless BCC Dirac walk has leading-order effective Hamiltonian
``H^(0)(k) = α · k``.  The first lattice correction is ``H^(1)(k)``,
extracted from the second-order Taylor coefficient ``B_2(k)`` of the Bloch
operator via Baker-Campbell-Hausdorff:

```text
H^(1)(k) = i · (B_2(k) - B_2(k)^†) / 2.
```

This is the Hermitian extraction; ``B_2`` itself is not Hermitian.

The audit decomposes ``H^(1)(k)`` along two orthogonal axes:

1. **CP**: CP-even vs CP-odd under ``cp_operator()`` from
   ``cp/discrete_symmetries.py``.  CP is antiunitary with no momentum flip,
   so its action on an operator ``M(k)`` is ``CP_mat · M(k)* · CP_mat^{-1}``.
2. **Cubic-harmonic**: ``A_{1g} ⊕ E_g ⊕ T_{2g}`` irreducible representations
   of the cubic point group ``O_h`` on the 6-dim degree-2 polynomial space.

Reports Frobenius-norm-squared values in the 2 × 3 cell grid.  A CP-odd
cell with nonzero magnitude indicates the lattice CP-violation slot at
O(ε).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import sympy as sp

from clifford_3plus2_d5.cp.cubic_harmonics import (
    IRREP_NAMES,
    decompose_matrix_of_polynomials,
)
from clifford_3plus2_d5.cp.discrete_symmetries import cp_operator
from clifford_3plus2_d5.cp.reuse import bcc_dirac_symbol
from clifford_3plus2_d5.spacetime_qca.continuum import nth_order_in_epsilon

if TYPE_CHECKING:
    from clifford_3plus2_d5.cp.cubic_harmonics import IrrepName


def symbolic_momentum() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]:
    """Return ``(epsilon, kx, ky, kz)`` as positive/real SymPy symbols."""

    epsilon = sp.symbols("epsilon", positive=True)
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    return epsilon, kx, ky, kz


@lru_cache(maxsize=1)
def _cached_bloch_order_two() -> sp.Matrix:
    eps, kx, ky, kz = symbolic_momentum()
    bloch = bcc_dirac_symbol(eps, kx, ky, kz)
    return nth_order_in_epsilon(bloch, eps, 2)


def bloch_order_two() -> sp.Matrix:
    """Return ``B_2(k)`` = ε²-coefficient of the BCC Dirac Bloch operator."""

    return _cached_bloch_order_two()


@lru_cache(maxsize=1)
def effective_hamiltonian_first_correction() -> sp.Matrix:
    """Return ``H^(1)(k) = i (B_2 - B_2^†) / 2``.

    By construction this is Hermitian whenever ``B_2`` is the ε²-coefficient
    of a unitary walk.
    """

    b2 = bloch_order_two()
    correction = (sp.I * (b2 - b2.H) / 2).applyfunc(sp.simplify)
    return correction


def cp_action_on_operator(matrix: sp.Matrix) -> sp.Matrix:
    """Return ``CP_mat · M^* · CP_mat^{-1}``.

    CP is antiunitary with no momentum flip, so the spinor-level action on
    an operator with polynomial-in-k entries is ``CP_mat · entry-conjugate
    · CP_mat^{-1}`` (k coordinates are real and unchanged).
    """

    op = cp_operator()
    cp_mat = op.spinor_matrix
    cp_inv = cp_mat.inv()
    conjugated = matrix.applyfunc(sp.conjugate)
    return (cp_mat * conjugated * cp_inv).applyfunc(sp.simplify)


def cp_even_part(matrix: sp.Matrix) -> sp.Matrix:
    """Return ``(M + CP M^* CP^{-1}) / 2``."""

    return ((matrix + cp_action_on_operator(matrix)) / 2).applyfunc(sp.simplify)


def cp_odd_part(matrix: sp.Matrix) -> sp.Matrix:
    """Return ``(M - CP M^* CP^{-1}) / 2``."""

    return ((matrix - cp_action_on_operator(matrix)) / 2).applyfunc(sp.simplify)


def polynomial_matrix_norm_squared(
    matrix: sp.Matrix,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> sp.Expr:
    """Return ``Σ_ij Σ_α |c_α(M_ij)|^2`` over the degree-2 monomial basis.

    Treats monomials as orthonormal.  Sums squared moduli of complex
    coefficients.
    """

    from clifford_3plus2_d5.cp.cubic_harmonics import (
        polynomial_to_coefficient_vector,
    )

    total = sp.Integer(0)
    for row in range(matrix.rows):
        for col in range(matrix.cols):
            entry = matrix[row, col]
            if entry == 0:
                continue
            vector = polynomial_to_coefficient_vector(entry, k_symbols)
            for index in range(6):
                coefficient = vector[index, 0]
                magnitude_squared = (
                    coefficient * sp.conjugate(coefficient)
                )
                total = total + magnitude_squared
    return sp.simplify(total)


def cp_irrep_decomposition() -> dict[tuple[str, "IrrepName"], sp.Matrix]:
    """Return a 2×3 dict keyed by (CP-parity, irrep) of polynomial matrices."""

    _, kx, ky, kz = symbolic_momentum()
    k_symbols = (kx, ky, kz)
    h1 = effective_hamiltonian_first_correction()
    even = cp_even_part(h1)
    odd = cp_odd_part(h1)
    even_decomp = decompose_matrix_of_polynomials(even, k_symbols)
    odd_decomp = decompose_matrix_of_polynomials(odd, k_symbols)
    result: dict[tuple[str, IrrepName], sp.Matrix] = {}
    for irrep in IRREP_NAMES:
        result[("CP-even", irrep)] = even_decomp[irrep]
        result[("CP-odd", irrep)] = odd_decomp[irrep]
    return result


def cp_irrep_norm_table() -> dict[tuple[str, "IrrepName"], sp.Expr]:
    """Return the ``||cell||²`` polynomial-coefficient norm for each cell."""

    _, kx, ky, kz = symbolic_momentum()
    k_symbols = (kx, ky, kz)
    decomposition = cp_irrep_decomposition()
    return {
        key: polynomial_matrix_norm_squared(value, k_symbols)
        for key, value in decomposition.items()
    }


def h1_total_norm_squared() -> sp.Expr:
    _, kx, ky, kz = symbolic_momentum()
    return polynomial_matrix_norm_squared(
        effective_hamiltonian_first_correction(),
        (kx, ky, kz),
    )


def h1_is_hermitian() -> bool:
    h1 = effective_hamiltonian_first_correction()
    return (h1 - h1.H).applyfunc(sp.simplify) == sp.zeros(h1.rows, h1.cols)


@dataclass(frozen=True)
class ContinuumCPAuditPayload:
    """Result of the O(ε²) continuum decomposition audit."""

    h1_is_hermitian: bool
    h1_total_norm_squared: sp.Expr
    cell_norms: dict[tuple[str, str], sp.Expr]
    cp_even_total_norm_squared: sp.Expr
    cp_odd_total_norm_squared: sp.Expr
    cp_odd_carrying_irreps: tuple[str, ...]
    cp_violating_fraction_at_order_epsilon: sp.Expr
    verdict: str
    interpretation: str


def continuum_cp_audit_payload() -> ContinuumCPAuditPayload:
    """Run the full O(ε²) continuum CP audit."""

    hermitian = h1_is_hermitian()
    total = h1_total_norm_squared()
    table = cp_irrep_norm_table()

    cell_norms: dict[tuple[str, str], sp.Expr] = {
        (parity, irrep): value for (parity, irrep), value in table.items()
    }

    cp_even_total = sp.simplify(
        sum(value for key, value in table.items() if key[0] == "CP-even"),
    )
    cp_odd_total = sp.simplify(
        sum(value for key, value in table.items() if key[0] == "CP-odd"),
    )

    cp_odd_irreps = tuple(
        irrep
        for irrep in IRREP_NAMES
        if sp.simplify(table[("CP-odd", irrep)]) != 0
    )

    if total == 0:
        fraction: sp.Expr = sp.Integer(0)
    else:
        fraction = sp.simplify(cp_odd_total / total)

    if not hermitian:
        verdict = "ALPHA-CONT ANOMALY"
        interpretation = (
            "H^(1) extracted via BCH is not Hermitian, indicating a "
            "convention error or implementation bug in the BCH extraction."
        )
    elif sp.simplify(total) == 0:
        verdict = "ALPHA-CONT FAIL"
        interpretation = (
            "H^(1) ≡ 0.  The O(ε²) coefficient of the BCC Bloch operator "
            "is purely Hermitian; lattice CP violation must appear at "
            "higher order in ε.  Soft fail; (optional) recurse to O(ε³)."
        )
    elif sp.simplify(cp_odd_total) == 0:
        verdict = "ALPHA-CONT SOFT PASS"
        interpretation = (
            "H^(1) is nonzero but its CP-odd part vanishes.  Lattice CP "
            "violation is suppressed beyond O(ε); recurse to O(ε²) or "
            "higher if pursued.  Lattice CP still vanishes in continuum, "
            "but the leading correction is CP-even."
        )
    else:
        verdict = "ALPHA-CONT PASS"
        interpretation = (
            f"H^(1) is nonzero and contains CP-odd content carried by "
            f"irrep(s) {cp_odd_irreps}.  The lattice CP-violation slot "
            f"lives at O(ε), in specific cubic-harmonic component(s).  "
            f"CP-violating fraction at this order: {fraction}."
        )

    return ContinuumCPAuditPayload(
        h1_is_hermitian=hermitian,
        h1_total_norm_squared=total,
        cell_norms=cell_norms,
        cp_even_total_norm_squared=cp_even_total,
        cp_odd_total_norm_squared=cp_odd_total,
        cp_odd_carrying_irreps=cp_odd_irreps,
        cp_violating_fraction_at_order_epsilon=fraction,
        verdict=verdict,
        interpretation=interpretation,
    )
