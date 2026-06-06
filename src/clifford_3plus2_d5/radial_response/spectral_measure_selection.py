"""R13 finite spectral-measure selection study.

R12 proves that the current finite S3/silver-transfer data do not force radial
mass poles or residues.  This study makes the next distinction precise: any
positive finite spectral measure can be encoded as a Jacobi bath, but that is
not a QCA derivation unless a simple forward bath selects the measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.radial_response.pole_residue_rigidity import (
    bath_self_energy,
    one_level_triality_bath,
    two_level_tail_triality_bath,
)
from clifford_3plus2_d5.radial_response.silver_transfer_inheritance import (
    inherited_eta,
)
from clifford_3plus2_d5.radial_response.unitary_defect import self_energy_at_z_two


@dataclass(frozen=True)
class DiscreteSpectralMeasure:
    """Finite positive spectral measure seen by a boundary vector."""

    label: str
    poles: tuple[sp.Expr, ...]
    weights: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class SpectralMeasureSelectionPayload:
    """Payload for the R13 spectral-measure selection study."""

    final_verdict: str
    inherited_eta: sp.Expr
    up_measure: DiscreteSpectralMeasure
    down_baseline_measure: DiscreteSpectralMeasure
    down_candidate_measure: DiscreteSpectralMeasure
    up_jacobi: sp.Matrix
    down_baseline_jacobi: sp.Matrix
    down_candidate_jacobi: sp.Matrix
    positive_measures: bool
    jacobi_round_trips: bool
    r12_baths_match_targets: bool
    p3_repair_matches_targets: bool
    silver_tail_matches_targets: bool
    minimal_unitary_matches_up_at_z2: bool
    simple_forward_bath_selected: bool
    interpretation: str


def _z_symbol(z: sp.Symbol | None = None) -> sp.Symbol:
    return sp.Symbol("z") if z is None else z


def _target_eta() -> sp.Expr:
    """Return the inherited radial amplitude factor eta = epsilon^2."""

    return inherited_eta()


def up_target_measure() -> DiscreteSpectralMeasure:
    """Return the normalized up-sector pole/residue target."""

    eta = _target_eta()
    poles = (
        sp.simplify(sp.Rational(1, 4) * eta**6),
        sp.simplify(eta**3 / sp.sqrt(2)),
        sp.Integer(1),
    )
    weights = (sp.Rational(1, 25), sp.Rational(8, 25), sp.Rational(16, 25))
    return DiscreteSpectralMeasure("up_scalar_target", poles, weights)


def down_baseline_target_measure() -> DiscreteSpectralMeasure:
    """Return the S3-baseline down-sector pole/residue target."""

    eta = _target_eta()
    poles = (
        sp.simplify(sp.sqrt(sp.Rational(3, 2)) * eta**4),
        sp.simplify(eta**2 / sp.sqrt(2)),
        sp.Integer(1),
    )
    weights = (sp.Rational(1, 2), sp.Rational(1, 6), sp.Rational(1, 3))
    return DiscreteSpectralMeasure("down_s3_baseline_target", poles, weights)


def down_candidate_target_measure() -> DiscreteSpectralMeasure:
    """Return the data-improved down-sector pole/residue target."""

    eta = _target_eta()
    poles = (
        sp.simplify(sp.sqrt(sp.Rational(6, 5)) * eta**4),
        sp.simplify(sp.sqrt(sp.Rational(2, 5)) * eta**2),
        sp.Integer(1),
    )
    weights = (sp.Rational(6, 13), sp.Rational(2, 13), sp.Rational(5, 13))
    return DiscreteSpectralMeasure("down_odd_shell_candidate_target", poles, weights)


def weights_are_normalized(measure: DiscreteSpectralMeasure) -> bool:
    """Return whether the measure weights sum to one."""

    return sp.simplify(sum(measure.weights, sp.Integer(0)) - 1) == 0


def measure_is_positive(measure: DiscreteSpectralMeasure) -> bool:
    """Return whether all poles and weights are numerically positive."""

    return weights_are_normalized(measure) and all(
        float(sp.N(value)) > 0 for value in (*measure.poles, *measure.weights)
    )


def stieltjes_self_energy(
    measure: DiscreteSpectralMeasure,
    z: sp.Symbol | None = None,
) -> sp.Expr:
    """Return ``sum_i w_i / (z - lambda_i)`` for a finite measure."""

    z_symbol = _z_symbol(z)
    return sp.factor(
        sum(
            weight / (z_symbol - pole)
            for pole, weight in zip(measure.poles, measure.weights, strict=True)
        )
    )


def jacobi_from_measure(measure: DiscreteSpectralMeasure) -> sp.Matrix:
    """Return the Jacobi bath whose boundary measure is ``measure``."""

    if not weights_are_normalized(measure):
        raise ValueError("measure weights must sum to one")
    if len(measure.poles) not in {1, 2, 3}:
        raise ValueError("only one-, two-, and three-point measures are supported")

    if len(measure.poles) == 1:
        return sp.Matrix([[measure.poles[0]]])

    moments = [measure_moment(measure, power) for power in range(5)]
    alpha0 = moments[1]
    beta1_squared = sp.factor(moments[2] - alpha0**2)
    beta1 = sp.sqrt(beta1_squared)

    if len(measure.poles) == 2:
        alpha1 = sp.simplify(sum(measure.poles, sp.Integer(0)) - alpha0)
        return sp.Matrix([[alpha0, beta1], [beta1, alpha1]]).applyfunc(sp.factor)

    alpha1 = sp.factor(
        (moments[3] - 2 * alpha0 * moments[2] + alpha0**2 * moments[1])
        / beta1_squared
    )
    x2_q1_squared = sp.factor(
        (moments[4] - 2 * alpha0 * moments[3] + alpha0**2 * moments[2])
        / beta1_squared
    )
    beta2_squared = sp.factor(x2_q1_squared - alpha1**2 - beta1_squared)
    beta2 = sp.sqrt(beta2_squared)
    alpha2 = sp.factor(sum(measure.poles, sp.Integer(0)) - alpha0 - alpha1)
    return sp.Matrix(
        [
            [alpha0, beta1, 0],
            [beta1, alpha1, beta2],
            [0, beta2, alpha2],
        ]
    ).applyfunc(sp.factor)


def jacobi_self_energy(jacobi: sp.Matrix, z: sp.Symbol | None = None) -> sp.Expr:
    """Return the boundary resolvent element of a Jacobi bath."""

    z_symbol = _z_symbol(z)
    e0 = sp.zeros(jacobi.rows, 1)
    e0[0, 0] = 1
    return sp.factor((e0.T * (z_symbol * sp.eye(jacobi.rows) - jacobi).inv() * e0)[0, 0])


def measure_moment(measure: DiscreteSpectralMeasure, power: int) -> sp.Expr:
    """Return the finite spectral moment ``sum_i w_i lambda_i^power``."""

    return sp.simplify(
        sum(
            weight * pole**power
            for pole, weight in zip(measure.poles, measure.weights, strict=True)
        )
    )


def jacobi_moment(jacobi: sp.Matrix, power: int) -> sp.Expr:
    """Return the boundary moment ``<e0|J^power|e0>``."""

    e0 = sp.zeros(jacobi.rows, 1)
    e0[0, 0] = 1
    if power == 0:
        return sp.Integer(1)
    return (e0.T * (jacobi**power) * e0)[0, 0]


def measure_from_jacobi(
    jacobi: sp.Matrix,
    z: sp.Symbol | None = None,
) -> DiscreteSpectralMeasure:
    """Return the finite spectral measure seen by the first Jacobi site."""

    z_symbol = _z_symbol(z)
    sigma = jacobi_self_energy(jacobi, z_symbol)
    denominator = sp.denom(sp.together(sigma))
    poles = tuple(
        sorted(
            (sp.simplify(pole) for pole in sp.solve(sp.Eq(denominator, 0), z_symbol)),
            key=lambda pole: float(sp.N(pole)),
        )
    )
    weights = tuple(sp.simplify(sp.residue(sigma, z_symbol, pole)) for pole in poles)
    return DiscreteSpectralMeasure("jacobi_boundary_measure", poles, weights)


def measure_round_trips(measure: DiscreteSpectralMeasure) -> bool:
    """Return whether inverse Jacobi reconstruction preserves sampled moments."""

    jacobi = jacobi_from_measure(measure)
    return all(
        abs(float(sp.N(measure_moment(measure, power) - jacobi_moment(jacobi, power), 50)))
        < 1e-40
        for power in range(2 * len(measure.poles))
    )


def p3_repair_jacobi() -> sp.Matrix:
    """Return the canonical three-site path-repair Jacobi control."""

    return sp.Matrix(
        [
            [1, -1, 0],
            [-1, 2, -1],
            [0, -1, 1],
        ]
    )


def silver_tail_jacobi() -> sp.Matrix:
    """Return a three-site constant silver-tail control."""

    eta = _target_eta()
    return sp.Matrix(
        [
            [0, eta, 0],
            [eta, 0, eta],
            [0, eta, 0],
        ]
    )


def self_energies_equal(left: sp.Expr, right: sp.Expr, z: sp.Symbol | None = None) -> bool:
    """Return exact equality of two scalar self-energies."""

    return sp.simplify(sp.together(left - right)) == 0


def existing_r12_baths_match_target(measure: DiscreteSpectralMeasure) -> bool:
    """Return whether either R12 bath already selects the target measure."""

    z = sp.Symbol("z")
    target = stieltjes_self_energy(measure, z)
    sample_points = (sp.Integer(2), sp.Integer(3), sp.Integer(5))
    return all(
        sp.simplify(
            target.subs(z, point)
            - bath_self_energy(one_level_triality_bath(), z).subs(z, point)
        )
        == 0
        for point in sample_points
    ) or all(
        sp.simplify(
            target.subs(z, point)
            - bath_self_energy(two_level_tail_triality_bath(), z).subs(z, point)
        )
        == 0
        for point in sample_points
    )


def jacobi_control_matches_target(control: sp.Matrix, measure: DiscreteSpectralMeasure) -> bool:
    """Return whether a Jacobi control has the target self-energy."""

    return all(
        abs(float(sp.N(jacobi_moment(control, power) - measure_moment(measure, power), 50)))
        < 1e-40
        for power in range(2 * len(measure.poles))
    )


def minimal_unitary_matches_up_target_at_z2() -> bool:
    """Return whether the minimal unitary toy matches the up target at z=2."""

    z_value = sp.Integer(2)
    return sp.simplify(
        self_energy_at_z_two()[0, 0]
        - stieltjes_self_energy(up_target_measure(), sp.Symbol("z")).subs(sp.Symbol("z"), z_value)
    ) == 0


def spectral_measure_selection_payload() -> SpectralMeasureSelectionPayload:
    """Return the R13 spectral-measure selection verdict."""

    up = up_target_measure()
    down_baseline = down_baseline_target_measure()
    down_candidate = down_candidate_target_measure()
    measures = (up, down_baseline, down_candidate)
    up_jacobi = jacobi_from_measure(up)
    down_baseline_jacobi = jacobi_from_measure(down_baseline)
    down_candidate_jacobi = jacobi_from_measure(down_candidate)

    positive = all(measure_is_positive(measure) for measure in measures)
    round_trips = all(measure_round_trips(measure) for measure in measures)
    r12_matches = any(existing_r12_baths_match_target(measure) for measure in measures)
    p3_matches = any(jacobi_control_matches_target(p3_repair_jacobi(), measure) for measure in measures)
    silver_matches = any(
        jacobi_control_matches_target(silver_tail_jacobi(), measure) for measure in measures
    )
    unitary_matches = minimal_unitary_matches_up_target_at_z2()
    simple_forward = r12_matches or p3_matches or silver_matches or unitary_matches

    if positive and round_trips and not simple_forward:
        final_verdict = "RADIAL_SPECTRAL_MEASURE_RECONSTRUCTION_ONLY"
        interpretation = (
            "The up, down-baseline, and down-candidate mass textures define "
            "valid positive finite spectral measures, and each reconstructs "
            "a unique Jacobi bath by the finite inverse spectral problem. "
            "However, the existing R12 baths, the P3 repair Jacobi bath, the "
            "constant silver-tail control, and the minimal unitary S3 toy do "
            "not select those measures. The mass poles/residues are therefore "
            "encoded by inverse reconstruction, not derived by a simple forward "
            "QCA bath already present in the theory."
        )
    elif positive and round_trips and simple_forward:
        final_verdict = "RADIAL_SPECTRAL_MEASURE_SELECTION_PASS"
        interpretation = "A simple forward bath selects one of the target spectral measures."
    else:
        final_verdict = "RADIAL_SPECTRAL_MEASURE_SELECTION_KILL"
        interpretation = "At least one target texture failed positive-measure or Jacobi round-trip tests."

    return SpectralMeasureSelectionPayload(
        final_verdict=final_verdict,
        inherited_eta=_target_eta(),
        up_measure=up,
        down_baseline_measure=down_baseline,
        down_candidate_measure=down_candidate,
        up_jacobi=up_jacobi,
        down_baseline_jacobi=down_baseline_jacobi,
        down_candidate_jacobi=down_candidate_jacobi,
        positive_measures=positive,
        jacobi_round_trips=round_trips,
        r12_baths_match_targets=r12_matches,
        p3_repair_matches_targets=p3_matches,
        silver_tail_matches_targets=silver_matches,
        minimal_unitary_matches_up_at_z2=unitary_matches,
        simple_forward_bath_selected=simple_forward,
        interpretation=interpretation,
    )
