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
from typing import Sequence, TypeAlias

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_lie import basis_span_dimension
from clifford_3plus2_d5.lepton.clifford_patisalam import (
    patisalam_chosen_complex_structure,
    su2_l_generators_from_spin04,
)
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
    mass_hamiltonian,
    projector_control_mass,
    scalar_internal_mass,
)

ComplexPair: TypeAlias = tuple[sp.Expr, sp.Expr]


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


def _matrix_span_basis(matrices: Sequence[sp.Matrix]) -> tuple[sp.Matrix, ...]:
    if not matrices:
        return ()
    rows = matrices[0].rows
    cols = matrices[0].cols
    columns = [matrix.reshape(rows * cols, 1) for matrix in matrices]
    _, pivots = sp.Matrix.hstack(*columns).rref()
    return tuple(matrices[index] for index in pivots)


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


@lru_cache(maxsize=1)
def su2_l_lowered_higgs_like_basis() -> tuple[sp.Matrix, ...]:
    """Return the ``Delta T3_L = -1/2`` space generated by SU(2)_L action.

    Starting from the ``(+1/2,+1/2)`` color-singlet charge-shift space, the
    second non-Cartan ``SU(2)_L`` generator lowers the ``T3_L`` charge shift.
    This avoids solving a second large nullspace while testing the doublet
    relation directly.
    """

    lowering_generator = su2_l_generators_from_spin04()[1]
    lowered = tuple(
        commutator(lowering_generator, matrix).applyfunc(sp.simplify)
        for matrix in color_singlet_charge_shift_basis(sp.Rational(1, 2), sp.Rational(1, 2))
    )
    return _matrix_span_basis(lowered)


def higgs_like_doublet_map_basis() -> tuple[tuple[sp.Matrix, ...], tuple[sp.Matrix, ...]]:
    """Return ``(upper, lower)`` Higgs-like map spaces.

    The upper space has charge shifts ``(Delta Y, Delta T3_L) = (+1/2,+1/2)``.
    The lower space is generated from it by the ``SU(2)_L`` action and has
    charge shifts ``(+1/2,-1/2)``.  These are Yukawa-map spaces carrying
    Higgs-doublet charge structure, not a standalone dynamical Higgs field.
    """

    return color_singlet_charge_shift_basis(sp.Rational(1, 2), sp.Rational(1, 2)), su2_l_lowered_higgs_like_basis()


def all_higgs_like_charge_shift(
    matrices: Sequence[sp.Matrix],
    *,
    hypercharge_shift: sp.Expr,
    t3_l_shift: sp.Expr,
) -> bool:
    return all(
        is_higgs_like_charge_shift(
            matrix,
            hypercharge_shift=hypercharge_shift,
            t3_l_shift=t3_l_shift,
        )
        for matrix in matrices
    )


def commutes_with_internal_j(matrix: sp.Matrix) -> bool:
    return _same_matrix(commutator(matrix, patisalam_chosen_complex_structure()), _zero(matrix.rows))


def all_commute_with_internal_j(matrices: Sequence[sp.Matrix]) -> bool:
    return all(commutes_with_internal_j(matrix) for matrix in matrices)


def static_yukawa_internal_control(coefficients: Sequence[sp.Expr] | None = None) -> sp.Matrix:
    """Return a real-symmetric static Yukawa control from upper components.

    For real coefficients ``c_i`` and upper charge-shift maps ``M_i``, the
    control is ``sum_i c_i (M_i + M_i.T)``.  It is Hermitian as a real matrix,
    but it is a fixed charge-breaking background rather than a dynamical Higgs
    doublet.
    """

    upper, _ = higgs_like_doublet_map_basis()
    if coefficients is None:
        coefficients = tuple(sp.Integer(1) for _ in upper)
    if len(coefficients) != len(upper):
        raise ValueError(f"expected {len(upper)} coefficients")
    control = sp.zeros(32)
    for coefficient, matrix in zip(coefficients, upper, strict=True):
        control += coefficient * (matrix + matrix.T)
    return control.applyfunc(sp.simplify)


def static_yukawa_hamiltonian(coefficients: Sequence[sp.Expr] | None = None) -> sp.Matrix:
    """Return ``beta x Y_static`` for the static Hermitian Yukawa control."""

    return mass_hamiltonian(static_yukawa_internal_control(coefficients))


def matrix_is_real_symmetric(matrix: sp.Matrix) -> bool:
    return _same_matrix(matrix, matrix.T)


def electromagnetic_charge_observable() -> sp.Matrix:
    """Return the SM electromagnetic charge observable ``Q = Y + T3_L``."""

    return (hypercharge_observable() + normalized_t3_l_observable()).applyfunc(sp.simplify)


def static_higgs_doublet_internal_control(
    *,
    upper_coefficients: Sequence[sp.Expr] | None = None,
    lower_coefficients: Sequence[sp.Expr] | None = None,
) -> sp.Matrix:
    """Return a real-symmetric static control from both Higgs-doublet components.

    The upper maps have ``(Delta Y, Delta T3_L) = (+1/2,+1/2)`` and the
    lower maps have ``(+1/2,-1/2)``.  Real coefficients build the Hermitian
    real-form combination ``M + M.T`` for each selected map.  This remains a
    fixed background control, not a position-dependent dynamical Higgs field.
    """

    upper, lower = higgs_like_doublet_map_basis()
    if upper_coefficients is None:
        upper_coefficients = tuple(sp.Integer(1) for _ in upper)
    if lower_coefficients is None:
        lower_coefficients = tuple(sp.Integer(1) for _ in lower)
    if len(upper_coefficients) != len(upper):
        raise ValueError(f"expected {len(upper)} upper coefficients")
    if len(lower_coefficients) != len(lower):
        raise ValueError(f"expected {len(lower)} lower coefficients")

    control = sp.zeros(32)
    for coefficient, matrix in zip(upper_coefficients, upper, strict=True):
        control += coefficient * (matrix + matrix.T)
    for coefficient, matrix in zip(lower_coefficients, lower, strict=True):
        control += coefficient * (matrix + matrix.T)
    return control.applyfunc(sp.simplify)


def static_higgs_doublet_hamiltonian(
    *,
    upper_coefficients: Sequence[sp.Expr] | None = None,
    lower_coefficients: Sequence[sp.Expr] | None = None,
) -> sp.Matrix:
    """Return ``beta x Y_static`` for the full static Higgs-doublet control."""

    return mass_hamiltonian(
        static_higgs_doublet_internal_control(
            upper_coefficients=upper_coefficients,
            lower_coefficients=lower_coefficients,
        ),
    )


def static_neutral_higgs_vev_control(coefficients: Sequence[sp.Expr] | None = None) -> sp.Matrix:
    """Return the neutral lower-component static Higgs VEV control.

    The lower component has ``Delta Q_em = Delta(Y + T3_L) = 0``.  Its
    Hermitian real-form combination therefore preserves electromagnetism while
    breaking ``Y`` and ``T3_L`` separately, matching the static SM Higgs-VEV
    charge pattern.
    """

    _, lower = higgs_like_doublet_map_basis()
    if coefficients is None:
        coefficients = tuple(sp.Integer(1) for _ in lower)
    if len(coefficients) != len(lower):
        raise ValueError(f"expected {len(lower)} coefficients")
    control = sp.zeros(32)
    for coefficient, matrix in zip(coefficients, lower, strict=True):
        control += coefficient * (matrix + matrix.T)
    return control.applyfunc(sp.simplify)


def static_neutral_higgs_vev_hamiltonian(
    coefficients: Sequence[sp.Expr] | None = None,
) -> sp.Matrix:
    """Return ``beta x Y_static`` for the neutral static Higgs-VEV control."""

    return mass_hamiltonian(static_neutral_higgs_vev_control(coefficients))


def selected_higgs_phi_basis() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return the canonical two-complex ``Phi`` map slice.

    The full upper and lower Higgs-like map spaces are each four-dimensional
    over the real carrier.  Session 38 exposes a physics-facing static Higgs
    doublet API by selecting the first two deterministic basis maps from each
    component:

    ``(U0, U1, L0, L1) = (upper[0], upper[1], lower[0], lower[1])``.

    A complex coefficient is represented on the exact real carrier by its
    explicit real and imaginary coordinates against those two selected maps.
    This is a fixed two-complex slice of the known module, not a claim that the
    whole eight-real-dimensional map module has been complex-basis reduced.
    """

    upper, lower = higgs_like_doublet_map_basis()
    if len(upper) < 2 or len(lower) < 2:
        raise RuntimeError("Higgs-like doublet map spaces must each have at least two basis maps")
    return upper[0], upper[1], lower[0], lower[1]


def _validate_complex_pair(name: str, value: ComplexPair) -> ComplexPair:
    """Return exact real/imaginary coordinates for a SymPy complex scalar."""

    if len(value) != 2:
        raise ValueError(f"{name} must be an explicit (real, imag) pair")
    return sp.sympify(value[0]), sp.sympify(value[1])


def higgs_phi_raising_map(
    phi_plus: ComplexPair = (sp.Integer(0), sp.Integer(0)),
    phi_zero: ComplexPair = (sp.Integer(1), sp.Integer(0)),
) -> sp.Matrix:
    """Return the non-Hermitian charge-raising map ``A(Phi)``.

    ``phi_plus`` and ``phi_zero`` are explicit real-form complex pairs
    ``(real_part, imag_part)``.  The default is the neutral VEV direction
    ``Phi = (0, 1)``.
    """

    plus_re, plus_im = _validate_complex_pair("phi_plus", phi_plus)
    zero_re, zero_im = _validate_complex_pair("phi_zero", phi_zero)
    upper_re, upper_im, lower_re, lower_im = selected_higgs_phi_basis()
    return (
        plus_re * upper_re
        + plus_im * upper_im
        + zero_re * lower_re
        + zero_im * lower_im
    ).applyfunc(sp.simplify)


def hermitian_yukawa_internal_control(
    phi_plus: ComplexPair = (sp.Integer(0), sp.Integer(0)),
    phi_zero: ComplexPair = (sp.Integer(1), sp.Integer(0)),
) -> sp.Matrix:
    """Return the real-form Hermitian internal Yukawa control ``Y(Phi)``.

    For ``A(Phi)`` in the selected Higgs-doublet map slice, the internal
    control is ``A(Phi) + A(Phi).T``.  The transpose is the real-form
    opposite-charge counterpart used by Sessions 23 and 25.  This is the
    Hermitian static-control construction; it is not yet a full complex
    dynamical Higgs-field conjugation law.
    """

    raising = higgs_phi_raising_map(phi_plus=phi_plus, phi_zero=phi_zero)
    return (raising + raising.T).applyfunc(sp.simplify)


def hermitian_yukawa_hamiltonian(
    phi_plus: ComplexPair = (sp.Integer(0), sp.Integer(0)),
    phi_zero: ComplexPair = (sp.Integer(1), sp.Integer(0)),
) -> sp.Matrix:
    """Return ``beta x Y(Phi)`` for the static Hermitian Yukawa control."""

    return mass_hamiltonian(hermitian_yukawa_internal_control(phi_plus=phi_plus, phi_zero=phi_zero))


def neutral_yukawa_internal_control(vev: sp.Expr = sp.Integer(1)) -> sp.Matrix:
    """Return ``Y(Phi)`` for the neutral Higgs direction ``Phi = (0, vev)``."""

    return hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(0), sp.Integer(0)),
        phi_zero=(sp.sympify(vev), sp.Integer(0)),
    )


def neutral_yukawa_hamiltonian(vev: sp.Expr = sp.Integer(1)) -> sp.Matrix:
    """Return ``beta x Y(Phi)`` for the neutral Higgs direction."""

    return mass_hamiltonian(neutral_yukawa_internal_control(vev))


def preserves_electromagnetism(matrix: sp.Matrix) -> bool:
    """Return whether ``matrix`` commutes with ``Q_em = Y + T3_L``."""

    return _same_matrix(commutator(matrix, electromagnetic_charge_observable()), _zero(matrix.rows))


def matrix_rank_nullity(matrix: sp.Matrix) -> tuple[int, int]:
    rank = matrix.rank()
    return rank, matrix.cols - rank


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


@dataclass(frozen=True)
class HiggsDoubletMapAudit:
    upper_dimension: int
    lower_dimension: int
    combined_dimension: int
    upper_charge_shifts_valid: bool
    lower_charge_shifts_valid: bool
    lowering_generators_span_same_space: bool
    upper_commutes_with_j: bool
    lower_commutes_with_j: bool
    static_control_symmetric: bool
    static_hamiltonian_hermitian: bool
    full_doublet_control_rank: int
    full_doublet_control_nullity: int
    neutral_vev_preserves_color: bool
    neutral_vev_preserves_electromagnetism: bool
    neutral_vev_breaks_hypercharge: bool
    neutral_vev_breaks_t3_l: bool
    neutral_vev_rank: int
    neutral_vev_nullity: int
    interpretation: str


@dataclass(frozen=True)
class HermitianYukawaPhiAudit:
    phi_api: str
    selected_upper_dimension: int
    selected_lower_dimension: int
    zero_phi_is_zero: bool
    linearity_passed: bool
    internal_control_symmetric: bool
    hamiltonian_hermitian: bool
    neutral_preserves_color: bool
    neutral_preserves_electromagnetism: bool
    neutral_breaks_hypercharge: bool
    neutral_breaks_t3_l: bool
    charged_component_breaks_electromagnetism: bool
    neutral_rank: int
    neutral_nullity: int
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


def higgs_doublet_map_audit_payload() -> HiggsDoubletMapAudit:
    """Return the Session 25 Higgs-like map-space audit payload."""

    upper, lower = higgs_like_doublet_map_basis()
    second_lower = _matrix_span_basis(
        tuple(
            commutator(su2_l_generators_from_spin04()[2], matrix).applyfunc(sp.simplify)
            for matrix in upper
        ),
    )
    control = static_higgs_doublet_internal_control()
    hamiltonian = static_higgs_doublet_hamiltonian()
    full_rank, full_nullity = matrix_rank_nullity(control)
    neutral_control = static_neutral_higgs_vev_control()
    neutral_rank, neutral_nullity = matrix_rank_nullity(neutral_control)
    return HiggsDoubletMapAudit(
        upper_dimension=basis_span_dimension(upper),
        lower_dimension=basis_span_dimension(lower),
        combined_dimension=basis_span_dimension((*upper, *lower)),
        upper_charge_shifts_valid=all_higgs_like_charge_shift(
            upper,
            hypercharge_shift=sp.Rational(1, 2),
            t3_l_shift=sp.Rational(1, 2),
        ),
        lower_charge_shifts_valid=all_higgs_like_charge_shift(
            lower,
            hypercharge_shift=sp.Rational(1, 2),
            t3_l_shift=sp.Rational(-1, 2),
        ),
        lowering_generators_span_same_space=(
            basis_span_dimension(lower)
            == basis_span_dimension(second_lower)
            == basis_span_dimension((*lower, *second_lower))
        ),
        upper_commutes_with_j=all_commute_with_internal_j(upper),
        lower_commutes_with_j=all_commute_with_internal_j(lower),
        static_control_symmetric=matrix_is_real_symmetric(control),
        static_hamiltonian_hermitian=_same_matrix(hamiltonian, hamiltonian.H),
        full_doublet_control_rank=full_rank,
        full_doublet_control_nullity=full_nullity,
        neutral_vev_preserves_color=commutes_with_all(neutral_control, su3_c_generators_from_su4()),
        neutral_vev_preserves_electromagnetism=preserves_electromagnetism(neutral_control),
        neutral_vev_breaks_hypercharge=not commutes_with_all(
            neutral_control,
            (hypercharge_observable(),),
        ),
        neutral_vev_breaks_t3_l=not commutes_with_all(
            neutral_control,
            (normalized_t3_l_observable(),),
        ),
        neutral_vev_rank=neutral_rank,
        neutral_vev_nullity=neutral_nullity,
        interpretation=(
            "The color-singlet Yukawa-map module carries Higgs-doublet charge "
            "structure under SU(2)_L, but the maps are not internal-J-linear by "
            "themselves.  The lower-component static VEV preserves Q_em while "
            "breaking Y and T3_L separately.  These controls are fixed "
            "backgrounds, not dynamical Higgs fields."
        ),
    )


def audit_hermitian_yukawa_phi() -> HermitianYukawaPhiAudit:
    """Return the Session 38 Hermitian ``Y(Phi)`` audit payload."""

    zero = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(0), sp.Integer(0)),
        phi_zero=(sp.Integer(0), sp.Integer(0)),
    )
    sample = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(2), sp.Integer(-3)),
        phi_zero=(sp.Rational(5, 2), sp.Rational(7, 3)),
    )
    sample_hamiltonian = hermitian_yukawa_hamiltonian(
        phi_plus=(sp.Integer(2), sp.Integer(-3)),
        phi_zero=(sp.Rational(5, 2), sp.Rational(7, 3)),
    )
    left = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(1), sp.Integer(2)),
        phi_zero=(sp.Integer(3), sp.Integer(4)),
    )
    right = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(5), sp.Integer(6)),
        phi_zero=(sp.Integer(7), sp.Integer(8)),
    )
    combined = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(6), sp.Integer(8)),
        phi_zero=(sp.Integer(10), sp.Integer(12)),
    )
    neutral = neutral_yukawa_internal_control(sp.Integer(1))
    charged = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(1), sp.Integer(0)),
        phi_zero=(sp.Integer(0), sp.Integer(0)),
    )
    neutral_rank, neutral_nullity = matrix_rank_nullity(neutral)
    return HermitianYukawaPhiAudit(
        phi_api="two_complex_explicit_re_im",
        selected_upper_dimension=2,
        selected_lower_dimension=2,
        zero_phi_is_zero=_same_matrix(zero, _zero(32)),
        linearity_passed=_same_matrix(combined, left + right),
        internal_control_symmetric=matrix_is_real_symmetric(sample),
        hamiltonian_hermitian=_same_matrix(sample_hamiltonian, sample_hamiltonian.H),
        neutral_preserves_color=commutes_with_all(neutral, su3_c_generators_from_su4()),
        neutral_preserves_electromagnetism=preserves_electromagnetism(neutral),
        neutral_breaks_hypercharge=not commutes_with_all(neutral, (hypercharge_observable(),)),
        neutral_breaks_t3_l=not commutes_with_all(neutral, (normalized_t3_l_observable(),)),
        charged_component_breaks_electromagnetism=not preserves_electromagnetism(charged),
        neutral_rank=neutral_rank,
        neutral_nullity=neutral_nullity,
        interpretation=(
            "Session 38 promotes the static Higgs-like map module to a "
            "Hermitian two-complex Phi control on a deterministic real-form "
            "slice.  It is still a static background Yukawa layer, not a "
            "site-local dynamical Higgs field or Yukawa hierarchy."
        ),
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
