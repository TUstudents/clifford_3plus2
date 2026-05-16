"""Standard Model subalgebra extracted from the Pati-Salam factorization.

Session 19a starts from the Session 18 algebra

``su4 + su2_l + su2_r``

on the real chiral-16 carrier and extracts

``su3_c + su2_l + u1_y``.

The ``B-L`` convention is the sum of the three commuting Cartan bivectors in
the ``Spin(0,6) ~= SU(4)`` factor.  Its centralizer in ``su4`` has dimension
9, and the trace-orthogonal part of that centralizer gives the 8-dimensional
``su3_c`` subalgebra.  Hypercharge is represented by the unnormalized
Pati-Salam combination ``Y_raw = T3_R + (B-L)/2``; charge normalization and
the field table are left to Session 19b.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_lie import (
    basis_span_dimension,
    is_skew_symmetric,
    matrix_in_span,
)
from clifford_3plus2_d5.lepton.clifford_patisalam import (
    patisalam_all_commute_with_chosen_j,
    su2_l_generators_from_spin04,
    su2_r_generators_from_spin04,
    su4_generators_from_spin06,
)

MatrixTuple = tuple[sp.Matrix, ...]


def _zero(dimension: int) -> sp.Matrix:
    return sp.zeros(dimension)


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _flatten(matrix: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(matrix.rows * matrix.cols, 1, list(matrix))


def _commutator(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return (left * right - right * left).applyfunc(sp.simplify)


def _linear_combination(basis: Sequence[sp.Matrix], coefficients: sp.Matrix) -> sp.Matrix:
    return sum(
        (coefficients[index] * basis[index] for index in range(len(basis))),
        _zero(basis[0].rows),
    ).applyfunc(sp.simplify)


def _commuting_constraint_matrix(basis: Sequence[sp.Matrix], generator: sp.Matrix) -> sp.Matrix:
    return sp.Matrix.hstack(
        *(_flatten(_commutator(item, generator)) for item in basis),
    )


def _trace_orthogonality_constraint(basis: Sequence[sp.Matrix], generator: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(
        [[sp.trace((item.T * generator).applyfunc(sp.simplify)) for item in basis]],
    )


def _subspace_from_constraints(
    basis: Sequence[sp.Matrix],
    constraints: Sequence[sp.Matrix],
) -> MatrixTuple:
    constraint_matrix = sp.Matrix.vstack(*constraints)
    return tuple(
        _linear_combination(basis, vector)
        for vector in constraint_matrix.nullspace()
    )


def _all_commute(left_basis: Sequence[sp.Matrix], right_basis: Sequence[sp.Matrix]) -> bool:
    return all(
        _same_matrix(_commutator(left, right), _zero(left.rows))
        for left in left_basis
        for right in right_basis
    )


def _closes_inside(basis: Sequence[sp.Matrix], ambient: Sequence[sp.Matrix]) -> bool:
    return all(
        matrix_in_span(_commutator(left, right), ambient)
        for left_index, left in enumerate(basis)
        for right in basis[:left_index]
    )


def _independent_from(generator: sp.Matrix, basis: Sequence[sp.Matrix]) -> bool:
    return not matrix_in_span(generator, basis)


def b_minus_l_generator_from_su4() -> sp.Matrix:
    """Return the exact Pati-Salam ``B-L`` breaking direction.

    The basis order in ``su4_generators_from_spin06`` follows
    ``combinations(range(6), 2)``.  The disjoint Cartan bivectors are
    therefore ``(0,1)``, ``(2,3)``, and ``(4,5)``, at indices 0, 9, and 14.
    """

    su4 = su4_generators_from_spin06()
    return (su4[0] + su4[9] + su4[14]).applyfunc(sp.simplify)


@lru_cache(maxsize=1)
def b_minus_l_centralizer_in_su4() -> MatrixTuple:
    su4 = su4_generators_from_spin06()
    b_minus_l = b_minus_l_generator_from_su4()
    return _subspace_from_constraints(
        su4,
        (_commuting_constraint_matrix(su4, b_minus_l),),
    )


@lru_cache(maxsize=1)
def su3_c_generators_from_su4() -> MatrixTuple:
    """Return the 8-dimensional color subalgebra inside ``su4``."""

    su4 = su4_generators_from_spin06()
    b_minus_l = b_minus_l_generator_from_su4()
    return _subspace_from_constraints(
        su4,
        (
            _commuting_constraint_matrix(su4, b_minus_l),
            _trace_orthogonality_constraint(su4, b_minus_l),
        ),
    )


def t3_r_generator_from_su2_r() -> sp.Matrix:
    """Return a fixed Cartan convention for ``SU(2)_R``."""

    return su2_r_generators_from_spin04()[0]


def hypercharge_generator() -> sp.Matrix:
    """Return unnormalized ``Y = T3_R + (B-L)/2``."""

    return (
        t3_r_generator_from_su2_r()
        + sp.Rational(1, 2) * b_minus_l_generator_from_su4()
    ).applyfunc(sp.simplify)


def sm_gauge_generators() -> MatrixTuple:
    return (
        *su3_c_generators_from_su4(),
        *su2_l_generators_from_spin04(),
        hypercharge_generator(),
    )


def sm_generator_is_valid(generator: sp.Matrix) -> bool:
    return generator.shape == (32, 32) and is_skew_symmetric(generator)


def su3_c_closes() -> bool:
    su3 = su3_c_generators_from_su4()
    return _closes_inside(su3, su3)


def sm_commutators_pass() -> dict[str, bool]:
    su3 = su3_c_generators_from_su4()
    su2_l = su2_l_generators_from_spin04()
    hypercharge = hypercharge_generator()
    return {
        "su3_commutes_with_su2_l": _all_commute(su3, su2_l),
        "su3_commutes_with_y": _all_commute(su3, (hypercharge,)),
        "su2_l_commutes_with_y": _all_commute(su2_l, (hypercharge,)),
    }


def sm_algebra_audit_payload() -> dict[str, object]:
    su4 = su4_generators_from_spin06()
    su3 = su3_c_generators_from_su4()
    su2_l = su2_l_generators_from_spin04()
    hypercharge = hypercharge_generator()
    commutators = sm_commutators_pass()
    return {
        "source_algebra": "su4 + su2_l + su2_r from Cl(0,6) tensor Cl(0,4)",
        "su4_dimension": basis_span_dimension(su4),
        "b_minus_l_centralizer_dimension": basis_span_dimension(
            b_minus_l_centralizer_in_su4(),
        ),
        "su3_c_dimension": basis_span_dimension(su3),
        "su3_c_closes": su3_c_closes(),
        "su2_l_dimension": basis_span_dimension(su2_l),
        "u1_y_dimension": 1,
        "sm_total_dimension": basis_span_dimension((*su3, *su2_l, hypercharge)),
        "su3_commutes_with_su2_l": commutators["su3_commutes_with_su2_l"],
        "su3_commutes_with_y": commutators["su3_commutes_with_y"],
        "su2_l_commutes_with_y": commutators["su2_l_commutes_with_y"],
        "chosen_j_commutes_with_sm": patisalam_all_commute_with_chosen_j(
            sm_gauge_generators(),
        ),
        "hypercharge_nonzero": not _same_matrix(hypercharge, _zero(32)),
        "hypercharge_skew_symmetric": is_skew_symmetric(hypercharge),
        "hypercharge_independent_from_su3_su2": _independent_from(
            hypercharge,
            (*su3, *su2_l),
        ),
        "normalization_note": "Y is the raw Pati-Salam generator; textbook charges are Session 19b.",
    }
