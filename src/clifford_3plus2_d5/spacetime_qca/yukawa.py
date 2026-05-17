"""Representation-level Higgs/Yukawa audit helpers.

Session 23 does not implement a dynamical Higgs field.  It asks a narrower
question: whether the internal chiral-16 carrier contains exact maps with the
charge profile of a Higgs/Yukawa insertion.

For an internal operator ``M`` and charge observable ``Q``, a charge-shifting
map satisfies

``Q M - M Q = delta_q M``.

The Higgs-like control used here is color-singlet and shifts
``(Y, T3_L)`` by ``(+1/2, +1/2)``.  Its real transpose is the conjugate
charge-shift component with ``(-1/2, -1/2)``.  These are representation-level
static maps, not a Hermitian Yukawa Hamiltonian or a dynamical Higgs field.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_patisalam import su2_l_generators_from_spin04
from clifford_3plus2_d5.lepton.patisalam_sm import su3_c_generators_from_su4
from clifford_3plus2_d5.lepton.sm_hypercharge import (
    EXPECTED_JOINT_Y_T3L_TABLE,
    hypercharge_observable,
    joint_y_t3l_table,
    normalized_t3_l_observable,
    physical_hypercharge_generator,
)
from clifford_3plus2_d5.spacetime_qca.dirac import gamma5
from clifford_3plus2_d5.spacetime_qca.mass import (
    beta_matrix,
    commutator,
    commutes_with_all,
    projector_control_mass,
    scalar_internal_mass,
)


def _zero(rows: int, cols: int | None = None) -> sp.Matrix:
    return sp.zeros(rows, rows if cols is None else cols)


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == _zero(left.rows, left.cols)


def _flatten(matrix: sp.Matrix) -> list[sp.Expr]:
    return [matrix[row, col] for row in range(matrix.rows) for col in range(matrix.cols)]


def left_right_projectors() -> tuple[sp.Matrix, sp.Matrix]:
    """Return spacetime ``(P_R, P_L)`` in the package chiral basis."""

    chirality = gamma5()
    identity = sp.eye(chirality.rows)
    return (
        ((identity + chirality) / 2).applyfunc(sp.simplify),
        ((identity - chirality) / 2).applyfunc(sp.simplify),
    )


def yukawa_spacetime_coupler() -> sp.Matrix:
    """Return the off-diagonal spacetime coupler ``beta = gamma^0``."""

    return beta_matrix()


def beta_is_off_diagonal_between_chiralities() -> bool:
    p_right, p_left = left_right_projectors()
    beta = yukawa_spacetime_coupler()
    return (
        _same_matrix(p_right * beta * p_right, _zero(4))
        and _same_matrix(p_left * beta * p_left, _zero(4))
        and not _same_matrix(p_right * beta * p_left, _zero(4))
        and not _same_matrix(p_left * beta * p_right, _zero(4))
    )


def internal_commutator_profile(
    matrix: sp.Matrix,
    generators: Sequence[sp.Matrix],
) -> tuple[bool, ...]:
    """Return one boolean per generator indicating commutation with ``matrix``."""

    return tuple(_same_matrix(commutator(matrix, generator), _zero(matrix.rows)) for generator in generators)


def gauge_breaking_summary(
    internal_matrix: sp.Matrix,
    *,
    su3: Sequence[sp.Matrix],
    su2_l: Sequence[sp.Matrix],
    hypercharge: sp.Matrix,
) -> dict[str, bool]:
    """Summarize which SM sectors a static internal operator preserves."""

    preserves_color = commutes_with_all(internal_matrix, su3)
    preserves_su2_l = commutes_with_all(internal_matrix, su2_l)
    preserves_hypercharge = commutes_with_all(internal_matrix, (hypercharge,))
    return {
        "preserves_color": preserves_color,
        "preserves_su2_l": preserves_su2_l,
        "preserves_hypercharge": preserves_hypercharge,
        "commutes_with_sm": preserves_color and preserves_su2_l and preserves_hypercharge,
    }


def charge_shift_residual(
    matrix: sp.Matrix,
    observable: sp.Matrix,
    shift: sp.Expr,
) -> sp.Matrix:
    """Return ``Q M - M Q - shift M``."""

    return (observable * matrix - matrix * observable - shift * matrix).applyfunc(sp.simplify)


def has_charge_shift(matrix: sp.Matrix, observable: sp.Matrix, shift: sp.Expr) -> bool:
    return _same_matrix(charge_shift_residual(matrix, observable, shift), _zero(matrix.rows, matrix.cols))


def _matrix_from_vector(vector: sp.Matrix, dimension: int) -> sp.Matrix:
    return sp.Matrix(dimension, dimension, list(vector)).applyfunc(sp.simplify)


@lru_cache(maxsize=None)
def color_singlet_charge_shift_basis(
    hypercharge_shift: sp.Expr = sp.Rational(1, 2),
    t3_l_shift: sp.Expr = sp.Rational(1, 2),
) -> tuple[sp.Matrix, ...]:
    """Return exact internal maps with color-singlet Higgs-like charge shifts."""

    dimension = 32
    variables = sp.symbols(f"y0:{dimension * dimension}")
    matrix = sp.Matrix(dimension, dimension, variables)
    equations: list[sp.Expr] = []
    for generator in su3_c_generators_from_su4():
        equations.extend(_flatten(commutator(matrix, generator)))
    equations.extend(
        _flatten(
            charge_shift_residual(
                matrix,
                hypercharge_observable(),
                hypercharge_shift,
            ),
        ),
    )
    equations.extend(
        _flatten(
            charge_shift_residual(
                matrix,
                normalized_t3_l_observable(),
                t3_l_shift,
            ),
        ),
    )
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, variables)
    return tuple(_matrix_from_vector(vector, dimension) for vector in coefficient_matrix.nullspace())


def higgs_like_charge_shift_candidate(
    hypercharge_shift: sp.Expr = sp.Rational(1, 2),
    t3_l_shift: sp.Expr = sp.Rational(1, 2),
) -> sp.Matrix:
    """Return one exact color-singlet charge-shift representative."""

    basis = color_singlet_charge_shift_basis(hypercharge_shift, t3_l_shift)
    if not basis:
        raise RuntimeError("no Higgs-like charge-shift candidate found")
    return basis[0]


def conjugate_charge_shift_component(matrix: sp.Matrix) -> sp.Matrix:
    """Return the transpose component with opposite charge shifts.

    The charge observables are real symmetric.  If ``M`` satisfies
    ``Q M - M Q = delta M``, then ``M.T`` satisfies the same relation with
    ``-delta``.  This gives the conjugate Higgs-like component without solving
    a second nullspace problem.
    """

    return matrix.T.applyfunc(sp.simplify)


def higgs_like_charge_shift_pair(
    hypercharge_shift: sp.Expr = sp.Rational(1, 2),
    t3_l_shift: sp.Expr = sp.Rational(1, 2),
) -> tuple[sp.Matrix, sp.Matrix]:
    """Return ``(+shift, -shift)`` Higgs-like charge components."""

    positive = higgs_like_charge_shift_candidate(hypercharge_shift, t3_l_shift)
    return positive, conjugate_charge_shift_component(positive)


def is_higgs_like_charge_shift(
    matrix: sp.Matrix,
    *,
    hypercharge_shift: sp.Expr = sp.Rational(1, 2),
    t3_l_shift: sp.Expr = sp.Rational(1, 2),
) -> bool:
    return (
        not _same_matrix(matrix, _zero(matrix.rows, matrix.cols))
        and commutes_with_all(matrix, su3_c_generators_from_su4())
        and has_charge_shift(matrix, hypercharge_observable(), hypercharge_shift)
        and has_charge_shift(matrix, normalized_t3_l_observable(), t3_l_shift)
    )


@dataclass(frozen=True)
class YukawaCandidateAudit:
    name: str
    preserves_color: bool
    preserves_su2_l: bool
    preserves_hypercharge: bool
    commutes_with_sm: bool
    higgs_like_charge_shift: bool
    hypercharge_shift: sp.Expr | None
    t3_l_shift: sp.Expr | None
    interpretation: str


def audit_yukawa_candidate(
    name: str,
    internal_matrix: sp.Matrix,
    *,
    hypercharge_shift: sp.Expr | None = None,
    t3_l_shift: sp.Expr | None = None,
) -> YukawaCandidateAudit:
    summary = gauge_breaking_summary(
        internal_matrix,
        su3=su3_c_generators_from_su4(),
        su2_l=su2_l_generators_from_spin04(),
        hypercharge=physical_hypercharge_generator(),
    )
    higgs_like = False
    if hypercharge_shift is not None and t3_l_shift is not None:
        higgs_like = is_higgs_like_charge_shift(
            internal_matrix,
            hypercharge_shift=hypercharge_shift,
            t3_l_shift=t3_l_shift,
        )
    if higgs_like:
        interpretation = (
            "Representation-level Higgs-like color-singlet charge-shift map; "
            "not yet a dynamical Higgs/Yukawa field."
        )
    elif summary["commutes_with_sm"]:
        interpretation = "Gauge-preserving scalar/control operator; not Higgs-like."
    else:
        interpretation = "Symmetry-breaking control operator; not Higgs-like."
    return YukawaCandidateAudit(
        name=name,
        preserves_color=summary["preserves_color"],
        preserves_su2_l=summary["preserves_su2_l"],
        preserves_hypercharge=summary["preserves_hypercharge"],
        commutes_with_sm=summary["commutes_with_sm"],
        higgs_like_charge_shift=higgs_like,
        hypercharge_shift=hypercharge_shift,
        t3_l_shift=t3_l_shift,
        interpretation=interpretation,
    )


def universal_scalar_yukawa_audit() -> YukawaCandidateAudit:
    return audit_yukawa_candidate("universal_scalar", scalar_internal_mass(32, sp.Integer(1)))


def projector_control_yukawa_audit() -> YukawaCandidateAudit:
    return audit_yukawa_candidate("projector_control", projector_control_mass(32, 16))


def higgs_like_yukawa_audit(
    hypercharge_shift: sp.Expr = sp.Rational(1, 2),
    t3_l_shift: sp.Expr = sp.Rational(1, 2),
) -> YukawaCandidateAudit:
    return audit_yukawa_candidate(
        "higgs_like_charge_shift",
        higgs_like_charge_shift_candidate(hypercharge_shift, t3_l_shift),
        hypercharge_shift=hypercharge_shift,
        t3_l_shift=t3_l_shift,
    )


def yukawa_representation_audit_payload() -> dict[str, object]:
    """Return the Session 23 representation-level audit payload."""

    shift_basis = color_singlet_charge_shift_basis()
    positive_component, negative_component = higgs_like_charge_shift_pair()
    return {
        "joint_y_t3l_table_matches_sm": joint_y_t3l_table() == EXPECTED_JOINT_Y_T3L_TABLE,
        "beta_off_diagonal_between_chiralities": beta_is_off_diagonal_between_chiralities(),
        "universal_scalar": universal_scalar_yukawa_audit(),
        "projector_control": projector_control_yukawa_audit(),
        "higgs_like": higgs_like_yukawa_audit(),
        "higgs_like_shift_basis_dimension": len(shift_basis),
        "positive_component_is_higgs_like": is_higgs_like_charge_shift(positive_component),
        "negative_component_is_conjugate_higgs_like": is_higgs_like_charge_shift(
            negative_component,
            hypercharge_shift=sp.Rational(-1, 2),
            t3_l_shift=sp.Rational(-1, 2),
        ),
        "conjugate_component_construction": "transpose of the positive charge-shift map",
        "interpretation": (
            "This is a representation audit. It finds static internal maps "
            "with both Higgs-like charge-shift components but does not "
            "introduce a dynamical Higgs field, Yukawa couplings, or a finite "
            "real-space update."
        ),
    }
