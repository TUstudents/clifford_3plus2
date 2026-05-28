"""V22 label-conserving dynamics no-go for the max-entropy prior.

V21 derives the six entropy atoms from conserved-label distinguishability.
V22 asks whether dynamics that preserves those labels can also force the
uniform Jaynes state ``I_6 / 6``.

The answer is no.  A label-conserving boundary scattering operator has the
form

    U(phi) = sum_i exp(i phi_i) P_i,

and a diagonal population state has the form

    rho(p) = sum_i p_i P_i.

Every such ``rho(p)`` commutes with every such ``U(phi)``.  Label dephasing
removes coherences but preserves all populations.  Therefore the uniform state
is stationary, but it is not unique; the stationary simplex has dimension five.
Dynamic convergence to uniform requires a label-mixing mechanism, which
violates the V21 conserved-label condition.  The max-entropy prior remains an
inference principle, not a consequence of label-conserving dynamics alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from clifford_3plus2_d5.boundary_response.conserved_label_partition import (
    REMAINING_DECLARED_INPUTS,
    conserved_label_projectors,
    label_mixing_control_operator,
    scattering_preserves_conserved_labels,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
)


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two exact matrices agree after simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def _validate_length(values: Sequence[sp.Expr], *, name: str) -> tuple[sp.Expr, ...]:
    """Return sympified values after checking shell length."""

    if len(values) != SHELL_DIMENSION:
        raise ValueError(f"{name} must contain {SHELL_DIMENSION} entries")
    return tuple(sp.sympify(value) for value in values)


def label_conserving_scattering(phases: Sequence[sp.Expr]) -> sp.Matrix:
    """Return ``U(phi) = sum_i exp(i phi_i) P_i``."""

    selected = _validate_length(phases, name="phases")
    return sum(
        (sp.exp(sp.I * phase) * projector for phase, projector in zip(
            selected,
            conserved_label_projectors(),
            strict=True,
        )),
        sp.zeros(SHELL_DIMENSION, SHELL_DIMENSION),
    )


def label_population_density(probabilities: Sequence[sp.Expr]) -> sp.Matrix:
    """Return ``rho(p) = sum_i p_i P_i``."""

    selected = _validate_length(probabilities, name="probabilities")
    return sum(
        (
            probability * projector
            for probability, projector in zip(
                selected,
                conserved_label_projectors(),
                strict=True,
            )
        ),
        sp.zeros(SHELL_DIMENSION, SHELL_DIMENSION),
    )


def label_dephasing_channel(matrix: sp.Matrix) -> sp.Matrix:
    """Return ``D(rho) = sum_i P_i rho P_i`` for conserved-label projectors."""

    return sp.simplify(
        sum(
            (projector * matrix * projector for projector in conserved_label_projectors()),
            sp.zeros(SHELL_DIMENSION, SHELL_DIMENSION),
        )
    )


def label_population_vector(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Return populations ``Tr(P_i matrix)``."""

    return tuple(
        sp.simplify(sp.trace(projector * matrix))
        for projector in conserved_label_projectors()
    )


def stationary_simplex_dimension() -> int:
    """Return the dimension of trace-one diagonal stationary states."""

    return SHELL_DIMENSION - 1


def all_diagonal_populations_stationary() -> bool:
    """Return true when generic diagonal populations commute with generic phases."""

    probabilities = sp.symbols(f"p0:{SHELL_DIMENSION}")
    phases = sp.symbols(f"phi0:{SHELL_DIMENSION}", real=True)
    density = label_population_density(probabilities)
    scattering = label_conserving_scattering(phases)
    return _matrix_equal(scattering * density - density * scattering, sp.zeros(SHELL_DIMENSION))


def dephasing_preserves_populations() -> bool:
    """Return true when label dephasing preserves generic primitive populations."""

    entries = sp.symbols(f"m0:{SHELL_DIMENSION * SHELL_DIMENSION}")
    matrix = sp.Matrix(SHELL_DIMENSION, SHELL_DIMENSION, entries)
    return label_population_vector(label_dephasing_channel(matrix)) == label_population_vector(matrix)


def dephasing_removes_coherence() -> bool:
    """Return true when label dephasing removes a representative off-diagonal term."""

    matrix = sp.zeros(SHELL_DIMENSION, SHELL_DIMENSION)
    matrix[1, 3] = 1
    matrix[3, 1] = 1
    return _matrix_equal(label_dephasing_channel(matrix), sp.zeros(SHELL_DIMENSION))


def uniform_population_density() -> sp.Matrix:
    """Return the uniform six-channel density."""

    return sp.eye(SHELL_DIMENSION) / SHELL_DIMENSION


def uniform_stationary() -> bool:
    """Return true when the uniform density is stationary under label phases."""

    phases = sp.symbols(f"phi0:{SHELL_DIMENSION}", real=True)
    scattering = label_conserving_scattering(phases)
    uniform = uniform_population_density()
    return _matrix_equal(scattering * uniform - uniform * scattering, sp.zeros(SHELL_DIMENSION))


def uniform_unique_stationary() -> bool:
    """Return false: nonuniform diagonal states are also stationary."""

    nonuniform = label_population_density(
        (
            sp.Rational(1, 2),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
            sp.Rational(1, 10),
        )
    )
    phases = sp.symbols(f"phi0:{SHELL_DIMENSION}", real=True)
    scattering = label_conserving_scattering(phases)
    nonuniform_stationary = _matrix_equal(
        scattering * nonuniform - nonuniform * scattering,
        sp.zeros(SHELL_DIMENSION),
    )
    return not nonuniform_stationary


@dataclass(frozen=True)
class LabelConservingDynamicsAuditPayload:
    """Verdict payload for the V22 label-conserving dynamics no-go."""

    final_verdict: str
    stationary_simplex_dimension: int
    all_diagonal_populations_stationary: bool
    dephasing_preserves_populations: bool
    uniform_stationary: bool
    uniform_unique_stationary: bool
    label_mixing_required_for_dynamic_uniform: bool
    label_mixing_violates_conservation: bool
    max_entropy_prior_remains_declared: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def label_conserving_dynamics_audit_payload() -> LabelConservingDynamicsAuditPayload:
    """Return the V22 label-conserving dynamics no-go verdict."""

    diagonal_stationary = all_diagonal_populations_stationary()
    populations_preserved = dephasing_preserves_populations()
    coherence_removed = dephasing_removes_coherence()
    uniform_is_stationary = uniform_stationary()
    uniform_is_unique = uniform_unique_stationary()
    label_mixing_violates = not scattering_preserves_conserved_labels(label_mixing_control_operator())
    label_mixing_required = (
        diagonal_stationary
        and populations_preserved
        and uniform_is_stationary
        and not uniform_is_unique
    )

    checks_pass = (
        stationary_simplex_dimension() == 5
        and diagonal_stationary
        and populations_preserved
        and coherence_removed
        and uniform_is_stationary
        and not uniform_is_unique
        and label_mixing_required
        and label_mixing_violates
    )

    if checks_pass:
        final_verdict = "LABEL_CONSERVING_DYNAMICS_MAX_ENTROPY_NO_GO_PASS"
        interpretation = (
            "Label-conserving scattering preserves every primitive population. "
            "Dephasing removes coherences but leaves those populations fixed, "
            "so the uniform Jaynes density is stationary but not unique. "
            "Dynamic convergence to uniform requires label mixing, which "
            "violates the conserved-label partition. The max-entropy prior "
            "therefore remains an inference principle."
        )
    else:
        final_verdict = "LABEL_CONSERVING_DYNAMICS_MAX_ENTROPY_NO_GO_KILL"
        interpretation = (
            "The stationarity, dephasing, non-uniqueness, or label-mixing "
            "control failed."
        )

    return LabelConservingDynamicsAuditPayload(
        final_verdict=final_verdict,
        stationary_simplex_dimension=stationary_simplex_dimension(),
        all_diagonal_populations_stationary=diagonal_stationary,
        dephasing_preserves_populations=populations_preserved,
        uniform_stationary=uniform_is_stationary,
        uniform_unique_stationary=uniform_is_unique,
        label_mixing_required_for_dynamic_uniform=label_mixing_required,
        label_mixing_violates_conservation=label_mixing_violates,
        max_entropy_prior_remains_declared=True,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS,
        interpretation=interpretation,
    )
