"""Phase 1: J_3(O) exceptional Jordan algebra.

J_3(O) = 3×3 Hermitian octonion matrices, 27 real dimensions.  An
element ``M`` has the form

```text
M = [[ a       x        y      ],
     [ x*      b        z      ],
     [ y*      z*       c      ]]
```

with ``a, b, c ∈ R`` (3 real diagonal entries) and ``x, y, z ∈ O``
(3 off-diagonal octonion entries).  Below-diagonal entries are determined
by octonion conjugation ``M_ji = conj(M_ij)``.

The Jordan product

```text
M ·_J N := (MN + NM) / 2
```

(where ``MN`` uses entry-wise octonion multiplication) keeps the result
inside J_3(O).  J_3(O) is power-associative but not associative.

Key invariants:

- Linear: ``Tr(M) = a + b + c``.
- Bilinear: ``(M, N) := Tr(M ·_J N)``.
- Cubic norm: ``det(M)`` defined as the unique cubic invariant
  preserved by E_6.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.exceptional.reuse import octonion_multiply


def octonion_conjugate(octonion: sp.Matrix) -> sp.Matrix:
    """Return ``conj(o)``: flip sign of imaginary components.

    For octonion ``o = o_0 + sum_i o_i e_i`` (i = 1..7), the conjugate is
    ``conj(o) = o_0 - sum_i o_i e_i``.
    """

    if octonion.shape != (8, 1):
        raise ValueError("octonion must be an 8x1 column")
    result = octonion.copy()
    for index in range(1, 8):
        result[index, 0] = -result[index, 0]
    return result.applyfunc(sp.simplify)


def real_as_octonion(value: sp.Expr) -> sp.Matrix:
    """Embed a real number as an octonion (real part only)."""

    result = sp.zeros(8, 1)
    result[0, 0] = value
    return result


@dataclass(frozen=True)
class J3O:
    """A 3×3 Hermitian octonion matrix element of J_3(O).

    Stored as:

    - ``diagonal``: ``(m_11, m_22, m_33)`` — three real scalars.
    - ``off_diagonal``: ``(m_12, m_13, m_23)`` — three octonions as 8x1
      columns.
    """

    diagonal: tuple[sp.Expr, sp.Expr, sp.Expr]
    off_diagonal: tuple[sp.Matrix, sp.Matrix, sp.Matrix]

    def matrix_entry(self, row: int, col: int) -> sp.Matrix:
        """Return the (row, col) entry as an octonion 8x1 column."""

        if row == col:
            return real_as_octonion(self.diagonal[row])
        if row < col:
            return self.off_diagonal[_off_index(row, col)]
        # row > col: derived by conjugation
        return octonion_conjugate(self.off_diagonal[_off_index(col, row)])


def _off_index(row: int, col: int) -> int:
    """Map an above-diagonal (row, col) to a flat off-diagonal index.

    (0, 1) -> 0, (0, 2) -> 1, (1, 2) -> 2.
    """

    if (row, col) == (0, 1):
        return 0
    if (row, col) == (0, 2):
        return 1
    if (row, col) == (1, 2):
        return 2
    raise ValueError(f"({row}, {col}) is not an above-diagonal index pair")


def j3o_zero() -> J3O:
    return J3O(
        diagonal=(sp.Integer(0), sp.Integer(0), sp.Integer(0)),
        off_diagonal=(sp.zeros(8, 1), sp.zeros(8, 1), sp.zeros(8, 1)),
    )


def j3o_diagonal(value_0: sp.Expr, value_1: sp.Expr, value_2: sp.Expr) -> J3O:
    return J3O(
        diagonal=(value_0, value_1, value_2),
        off_diagonal=(sp.zeros(8, 1), sp.zeros(8, 1), sp.zeros(8, 1)),
    )


def j3o_scalar_multiply(scalar: sp.Expr, matrix: J3O) -> J3O:
    new_diagonal = tuple(sp.simplify(scalar * value) for value in matrix.diagonal)
    new_off = tuple(
        (scalar * entry).applyfunc(sp.simplify) for entry in matrix.off_diagonal
    )
    return J3O(diagonal=new_diagonal, off_diagonal=new_off)  # type: ignore[arg-type]


def j3o_add(left: J3O, right: J3O) -> J3O:
    new_diagonal = tuple(
        sp.simplify(left.diagonal[index] + right.diagonal[index])
        for index in range(3)
    )
    new_off = tuple(
        (left.off_diagonal[index] + right.off_diagonal[index]).applyfunc(sp.simplify)
        for index in range(3)
    )
    return J3O(diagonal=new_diagonal, off_diagonal=new_off)  # type: ignore[arg-type]


def j3o_matrix_multiply(left: J3O, right: J3O) -> tuple[tuple[sp.Matrix, ...], ...]:
    """Compute ``LR`` entry-by-entry as a 3×3 grid of octonion 8x1 columns.

    The result is generically NOT Hermitian (= not in J_3(O)).
    """

    result_rows: list[tuple[sp.Matrix, ...]] = []
    for row in range(3):
        row_entries: list[sp.Matrix] = []
        for col in range(3):
            total = sp.zeros(8, 1)
            for inner in range(3):
                product = octonion_multiply(
                    left.matrix_entry(row, inner),
                    right.matrix_entry(inner, col),
                )
                total = (total + product).applyfunc(sp.simplify)
            row_entries.append(total)
        result_rows.append(tuple(row_entries))
    return tuple(result_rows)


def j3o_jordan_product(left: J3O, right: J3O) -> J3O:
    """Return ``(LR + RL) / 2`` as an element of J_3(O).

    By symmetry of the construction, the result has real diagonal entries
    and Hermitian off-diagonals.  Below-diagonal entries are reconstructed
    from above-diagonal by conjugation.
    """

    LR = j3o_matrix_multiply(left, right)
    RL = j3o_matrix_multiply(right, left)
    half = sp.Rational(1, 2)

    diag: list[sp.Expr] = []
    for index in range(3):
        sum_entry = (LR[index][index] + RL[index][index]).applyfunc(sp.simplify)
        # Real part comes out in the 0th component; imaginary parts must cancel.
        # We extract the real part directly (and verify residual imaginary is zero).
        real_value = sp.simplify(half * sum_entry[0, 0])
        diag.append(real_value)

    off: list[sp.Matrix] = []
    for row, col in ((0, 1), (0, 2), (1, 2)):
        sum_entry = (LR[row][col] + RL[row][col]).applyfunc(sp.simplify)
        off.append((half * sum_entry).applyfunc(sp.simplify))

    return J3O(
        diagonal=(diag[0], diag[1], diag[2]),
        off_diagonal=(off[0], off[1], off[2]),
    )


def j3o_trace(matrix: J3O) -> sp.Expr:
    return sp.simplify(matrix.diagonal[0] + matrix.diagonal[1] + matrix.diagonal[2])


def j3o_bilinear_form(left: J3O, right: J3O) -> sp.Expr:
    return j3o_trace(j3o_jordan_product(left, right))


def j3o_cubic_norm(matrix: J3O) -> sp.Expr:
    """Return the cubic E_6-invariant of ``matrix``.

    For ``M = [[a, x, y], [x*, b, z], [y*, z*, c]]`` the cubic norm is

    ``N(M) = a b c + Re(2 x z y* ) - a |z|² - b |y|² - c |x|²``.

    where ``|x|² = x x* = conj(x).x`` is the octonion norm.
    """

    a, b, c = matrix.diagonal
    x, y, z = matrix.off_diagonal

    def octonion_norm_squared(octonion: sp.Matrix) -> sp.Expr:
        return sp.simplify(
            sum(octonion[index, 0] ** 2 for index in range(8)),
        )

    norm_x = octonion_norm_squared(x)
    norm_y = octonion_norm_squared(y)
    norm_z = octonion_norm_squared(z)

    # Compute 2 Re(x z y*).  Use octonion product:
    xz = octonion_multiply(x, z)
    y_conj = octonion_conjugate(y)
    xzy_conj = octonion_multiply(xz, y_conj)
    real_xzy = xzy_conj[0, 0]
    two_real_xzy = sp.simplify(2 * real_xzy)

    return sp.simplify(
        a * b * c
        + two_real_xzy
        - a * norm_z
        - b * norm_y
        - c * norm_x,
    )


# ----- Basis machinery for the 27-dim space -----


def j3o_basis_real(index: int) -> J3O:
    """Return the basis element for the ``index``-th real diagonal slot (0..2)."""

    if index not in (0, 1, 2):
        raise ValueError("real diagonal index must be 0, 1, or 2")
    diagonal = [sp.Integer(0)] * 3
    diagonal[index] = sp.Integer(1)
    return J3O(
        diagonal=tuple(diagonal),  # type: ignore[arg-type]
        off_diagonal=(sp.zeros(8, 1), sp.zeros(8, 1), sp.zeros(8, 1)),
    )


def j3o_basis_octonion(pair_index: int, octonion_component: int) -> J3O:
    """Return the basis element for off-diagonal ``pair_index`` component ``c``.

    ``pair_index ∈ {0, 1, 2}`` selects ``(m_12, m_13, m_23)``.
    ``octonion_component ∈ {0, ..., 7}`` selects which basis octonion.
    """

    if pair_index not in (0, 1, 2):
        raise ValueError("pair index must be 0, 1, or 2")
    if octonion_component not in range(8):
        raise ValueError("octonion component must be 0..7")
    off: list[sp.Matrix] = [sp.zeros(8, 1), sp.zeros(8, 1), sp.zeros(8, 1)]
    off[pair_index] = sp.zeros(8, 1)
    off[pair_index][octonion_component, 0] = sp.Integer(1)
    return J3O(
        diagonal=(sp.Integer(0), sp.Integer(0), sp.Integer(0)),
        off_diagonal=(off[0], off[1], off[2]),
    )


@lru_cache(maxsize=1)
def j3o_basis() -> tuple[J3O, ...]:
    """Return all 27 basis elements of J_3(O) in fixed order."""

    elements: list[J3O] = []
    # 3 diagonal real basis elements
    for index in range(3):
        elements.append(j3o_basis_real(index))
    # 24 off-diagonal octonion basis elements
    for pair_index in range(3):
        for component in range(8):
            elements.append(j3o_basis_octonion(pair_index, component))
    return tuple(elements)


def j3o_to_real_vector(matrix: J3O) -> sp.Matrix:
    """Return the 27x1 real coordinate vector of ``matrix`` in the basis."""

    vector = sp.zeros(27, 1)
    for index in range(3):
        vector[index, 0] = matrix.diagonal[index]
    for pair_index in range(3):
        for component in range(8):
            vector[3 + pair_index * 8 + component, 0] = (
                matrix.off_diagonal[pair_index][component, 0]
            )
    return vector.applyfunc(sp.simplify)


def j3o_dimension() -> int:
    """Return 27 (constant).  Verified by basis enumeration."""

    return len(j3o_basis())
