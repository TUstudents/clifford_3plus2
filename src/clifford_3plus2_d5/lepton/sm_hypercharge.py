"""Hypercharge spectrum for the extracted Standard Model algebra.

Session 19b converts real-skew gauge generators into real charge
observables using the chosen Pati-Salam complex structure ``J``:

``Q_A = -J A``.

The raw Session 19a generator ``Y_raw = T3_R + (B-L)/2`` is algebraically
valid but not charge-normalized: the Clifford basis gives ``T3_R`` raw
eigenvalues ``+-1`` and ``B-L`` raw eigenvalues ``+-1/2, +-3/2``.  The
physical Pati-Salam normalization used here is therefore

``T3_R_phys = T3_R_raw / 2``
``(B-L)_phys = 2 (B-L)_raw / 3``
``Y = T3_R_phys + (B-L)_phys / 2 = T3_R_raw/2 + (B-L)_raw/3``.
"""

from __future__ import annotations

from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_patisalam import (
    patisalam_chosen_complex_structure,
    su2_l_generators_from_spin04,
)
from clifford_3plus2_d5.lepton.patisalam_sm import (
    b_minus_l_generator_from_su4,
    hypercharge_generator,
    t3_r_generator_from_su2_r,
)

Spectrum = dict[sp.Expr, int]
JointSpectrum = dict[tuple[sp.Expr, sp.Expr], int]


EXPECTED_HYPERCHARGE_SPECTRUM: Spectrum = {
    sp.Rational(1, 6): 6,
    sp.Rational(-2, 3): 3,
    sp.Rational(1, 3): 3,
    sp.Rational(-1, 2): 2,
    sp.Integer(1): 1,
    sp.Integer(0): 1,
}

EXPECTED_JOINT_Y_T3L_TABLE: JointSpectrum = {
    (sp.Rational(1, 6), sp.Rational(1, 2)): 3,
    (sp.Rational(1, 6), sp.Rational(-1, 2)): 3,
    (sp.Rational(-2, 3), sp.Integer(0)): 3,
    (sp.Rational(1, 3), sp.Integer(0)): 3,
    (sp.Rational(-1, 2), sp.Rational(1, 2)): 1,
    (sp.Rational(-1, 2), sp.Rational(-1, 2)): 1,
    (sp.Integer(1), sp.Integer(0)): 1,
    (sp.Integer(0), sp.Integer(0)): 1,
}


def _identity(dimension: int = 32) -> sp.Matrix:
    return sp.eye(dimension)


def _zero(dimension: int = 32) -> sp.Matrix:
    return sp.zeros(dimension)


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _commutator(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return (left * right - right * left).applyfunc(sp.simplify)


def _normalize_spectrum(spectrum: Spectrum) -> Spectrum:
    return {sp.simplify(key): value for key, value in spectrum.items()}


def generator_commutes_with_j(generator: sp.Matrix) -> bool:
    j = patisalam_chosen_complex_structure()
    return _same_matrix(_commutator(generator, j), _zero(generator.rows))


def charge_observable(generator: sp.Matrix) -> sp.Matrix:
    if not generator_commutes_with_j(generator):
        raise ValueError("charge observable requires the generator to commute with J")
    return (-patisalam_chosen_complex_structure() * generator).applyfunc(sp.simplify)


def observable_is_symmetric(observable: sp.Matrix) -> bool:
    return _same_matrix(observable, observable.T)


def b_minus_l_observable() -> sp.Matrix:
    return charge_observable(b_minus_l_generator_from_su4())


def normalized_b_minus_l_generator() -> sp.Matrix:
    return (sp.Rational(2, 3) * b_minus_l_generator_from_su4()).applyfunc(sp.simplify)


def normalized_b_minus_l_observable() -> sp.Matrix:
    return charge_observable(normalized_b_minus_l_generator())


def t3_r_observable() -> sp.Matrix:
    return charge_observable(t3_r_generator_from_su2_r())


def normalized_t3_r_generator() -> sp.Matrix:
    return (sp.Rational(1, 2) * t3_r_generator_from_su2_r()).applyfunc(sp.simplify)


def normalized_t3_r_observable() -> sp.Matrix:
    return charge_observable(normalized_t3_r_generator())


def t3_l_generator() -> sp.Matrix:
    return su2_l_generators_from_spin04()[0]


def t3_l_observable() -> sp.Matrix:
    return charge_observable(t3_l_generator())


def normalized_t3_l_generator() -> sp.Matrix:
    return (sp.Rational(1, 2) * t3_l_generator()).applyfunc(sp.simplify)


def normalized_t3_l_observable() -> sp.Matrix:
    return charge_observable(normalized_t3_l_generator())


def raw_hypercharge_observable() -> sp.Matrix:
    return charge_observable(hypercharge_generator())


def physical_hypercharge_generator() -> sp.Matrix:
    return (
        normalized_t3_r_generator()
        + sp.Rational(1, 2) * normalized_b_minus_l_generator()
    ).applyfunc(sp.simplify)


def hypercharge_observable() -> sp.Matrix:
    return charge_observable(physical_hypercharge_generator())


def charge_observables_commute() -> bool:
    observables = (
        hypercharge_observable(),
        normalized_t3_l_observable(),
        normalized_t3_r_observable(),
        normalized_b_minus_l_observable(),
    )
    return all(
        _same_matrix(_commutator(left, right), _zero())
        for left_index, left in enumerate(observables)
        for right in observables[:left_index]
    )


def real_charge_spectrum(observable: sp.Matrix) -> Spectrum:
    if not observable_is_symmetric(observable):
        raise ValueError("charge spectrum requires a symmetric observable")
    return _normalize_spectrum(observable.eigenvals())


def complex_charge_spectrum(observable: sp.Matrix) -> Spectrum:
    spectrum = real_charge_spectrum(observable)
    if any(multiplicity % 2 != 0 for multiplicity in spectrum.values()):
        raise ValueError("real charge multiplicities must be even")
    return {charge: multiplicity // 2 for charge, multiplicity in spectrum.items()}


@lru_cache(maxsize=1)
def raw_hypercharge_spectrum() -> Spectrum:
    return complex_charge_spectrum(raw_hypercharge_observable())


@lru_cache(maxsize=1)
def hypercharge_spectrum() -> Spectrum:
    return complex_charge_spectrum(hypercharge_observable())


def _scaled_spectrum(spectrum: Spectrum, scale: sp.Expr) -> Spectrum:
    return {
        sp.simplify(scale * charge): multiplicity
        for charge, multiplicity in spectrum.items()
    }


def match_normalization(
    raw_spectrum: Spectrum,
    expected_spectrum: Spectrum = EXPECTED_HYPERCHARGE_SPECTRUM,
) -> tuple[bool, sp.Expr | None]:
    """Return whether a single scale maps ``raw_spectrum`` to ``expected``."""

    raw = _normalize_spectrum(raw_spectrum)
    expected = _normalize_spectrum(expected_spectrum)
    if raw == expected:
        return True, sp.Integer(1)
    raw_nonzero = [charge for charge in raw if charge != 0]
    expected_nonzero = [charge for charge in expected if charge != 0]
    for raw_charge in raw_nonzero:
        for expected_charge in expected_nonzero:
            scale = sp.simplify(expected_charge / raw_charge)
            if _scaled_spectrum(raw, scale) == expected:
                return True, scale
    return False, None


def normalized_hypercharge_spectrum() -> Spectrum:
    matched, scale = match_normalization(hypercharge_spectrum())
    if not matched or scale is None:
        raise RuntimeError("physical hypercharge spectrum did not ratio-match SM")
    return _scaled_spectrum(hypercharge_spectrum(), scale)


def joint_charge_decomposition(
    first_observable: sp.Matrix,
    second_observable: sp.Matrix,
    first_values: tuple[sp.Expr, ...],
    second_values: tuple[sp.Expr, ...],
) -> JointSpectrum:
    dimension = first_observable.rows
    identity = _identity(dimension)
    table: JointSpectrum = {}
    for first_value in first_values:
        for second_value in second_values:
            constraints = sp.Matrix.vstack(
                first_observable - first_value * identity,
                second_observable - second_value * identity,
            )
            real_multiplicity = len(constraints.nullspace())
            if real_multiplicity:
                if real_multiplicity % 2 != 0:
                    raise ValueError("joint real multiplicity must be even")
                table[(sp.simplify(first_value), sp.simplify(second_value))] = (
                    real_multiplicity // 2
                )
    return table


@lru_cache(maxsize=1)
def joint_y_t3l_table() -> JointSpectrum:
    return joint_charge_decomposition(
        hypercharge_observable(),
        normalized_t3_l_observable(),
        tuple(EXPECTED_HYPERCHARGE_SPECTRUM),
        (
            sp.Rational(1, 2),
            sp.Rational(-1, 2),
            sp.Integer(0),
        ),
    )


def sm_field_multiplicity_table() -> dict[str, dict[str, object]]:
    table = joint_y_t3l_table()
    return {
        "Q": {"Y": sp.Rational(1, 6), "complex_multiplicity": 6},
        "u^c": {"Y": sp.Rational(-2, 3), "complex_multiplicity": 3},
        "d^c": {"Y": sp.Rational(1, 3), "complex_multiplicity": 3},
        "L": {"Y": sp.Rational(-1, 2), "complex_multiplicity": 2},
        "e^c": {"Y": sp.Integer(1), "complex_multiplicity": 1},
        "nu^c": {"Y": sp.Integer(0), "complex_multiplicity": 1},
        "joint_y_t3l_table": dict(table),
    }


def hypercharge_audit_payload() -> dict[str, object]:
    raw_matched, raw_scale = match_normalization(raw_hypercharge_spectrum())
    matched, scale = match_normalization(hypercharge_spectrum())
    normalized = normalized_hypercharge_spectrum() if matched else {}
    joint = joint_y_t3l_table()
    observables = (
        raw_hypercharge_observable(),
        hypercharge_observable(),
        normalized_t3_l_observable(),
        normalized_t3_r_observable(),
        normalized_b_minus_l_observable(),
    )
    return {
        "real_dimension": 32,
        "complex_dimension": 16,
        "y_raw_commutes_with_j": generator_commutes_with_j(hypercharge_generator()),
        "y_physical_commutes_with_j": generator_commutes_with_j(physical_hypercharge_generator()),
        "observables_symmetric": all(observable_is_symmetric(item) for item in observables),
        "charge_observables_commute": charge_observables_commute(),
        "raw_hypercharge_spectrum": raw_hypercharge_spectrum(),
        "raw_common_scale_matches_sm": raw_matched,
        "raw_common_scale": raw_scale,
        "component_normalization_required": not raw_matched,
        "b_minus_l_normalization_factor": sp.Rational(2, 3),
        "t3_r_normalization_factor": sp.Rational(1, 2),
        "normalization_factor": scale,
        "normalized_hypercharge_spectrum": normalized,
        "matches_sm_hypercharge_spectrum": normalized == EXPECTED_HYPERCHARGE_SPECTRUM,
        "joint_y_t3l_table": joint,
        "matches_sm_joint_table": joint == EXPECTED_JOINT_Y_T3L_TABLE,
        "failure_mode_note": "Raw Y needs component normalization; after PS normalization the SM table matches.",
    }
