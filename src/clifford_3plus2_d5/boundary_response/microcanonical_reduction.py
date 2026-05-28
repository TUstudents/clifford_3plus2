"""V23 equal-degeneracy microcanonical reduction theorem.

V22 proves that label-conserving dynamics cannot dynamically select the
uniform primitive-label state.  V23 proves a different statement: if the
unresolved boundary shell is microcanonical and each conserved primitive
label has the same unresolved bath degeneracy, tracing out the bath gives
the V20 Jaynes state ``I_6 / 6``.

For an unresolved space

    H_Q = direct_sum_i ( |i>_label tensor B_i ),

with bath dimensions ``d_i``, the full microcanonical state reduces to

    rho_label = sum_i (d_i / D) P_i,    D = sum_i d_i.

Equal degeneracy therefore gives the uniform primitive density.  Unequal
degeneracy produces a nonuniform Jaynes density, and compressed macrochannel
counting reproduces the wrong V17 branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from clifford_3plus2_d5.boundary_response.conserved_label_partition import (
    conserved_label_projectors,
)
from clifford_3plus2_d5.boundary_response.jaynes_primitive_ergodicity import (
    phase_from_alpha,
    primitive_ratio_from_alpha,
)
from clifford_3plus2_d5.boundary_response.label_conserving_dynamics import (
    label_conserving_dynamics_audit_payload,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)

REMAINING_DECLARED_INPUTS_AFTER_REDUCTION = (
    "vacuum_framing",
    "transfer_probe",
    "equal_boundary_degeneracy_or_max_entropy_prior",
)


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two exact matrices agree after simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def _validate_degeneracies(degeneracies: Sequence[sp.Expr]) -> tuple[sp.Expr, ...]:
    """Return sympified degeneracies after checking shell length."""

    if len(degeneracies) != SHELL_DIMENSION:
        raise ValueError(f"degeneracies must contain {SHELL_DIMENSION} entries")
    return tuple(sp.sympify(value) for value in degeneracies)


def microcanonical_label_weights(degeneracies: Sequence[sp.Expr]) -> tuple[sp.Expr, ...]:
    """Return reduced label weights ``d_i / sum_j d_j``."""

    selected = _validate_degeneracies(degeneracies)
    total = sp.simplify(sum(selected))
    return tuple(sp.simplify(value / total) for value in selected)


def microcanonical_reduced_density(degeneracies: Sequence[sp.Expr]) -> sp.Matrix:
    """Return ``rho_label = sum_i (d_i / D) P_i``."""

    weights = microcanonical_label_weights(degeneracies)
    return sum(
        (
            weight * projector
            for weight, projector in zip(
                weights,
                conserved_label_projectors(),
                strict=True,
            )
        ),
        sp.zeros(SHELL_DIMENSION, SHELL_DIMENSION),
    )


def equal_degeneracy_reduced_density() -> sp.Matrix:
    """Return the reduced density for equal primitive degeneracy."""

    return microcanonical_reduced_density(tuple(sp.Integer(1) for _ in range(SHELL_DIMENSION)))


def alpha_from_label_weights(weights: Sequence[sp.Expr]) -> sp.Expr:
    """Return the even-channel alpha parameter from reduced label weights."""

    if len(weights) != SHELL_DIMENSION:
        raise ValueError(f"weights must contain {SHELL_DIMENSION} entries")
    return sp.sympify(weights[0])


def ratio_from_degeneracies(degeneracies: Sequence[sp.Expr]) -> sp.Expr:
    """Return the V20 primitive ratio induced by reduced degeneracies."""

    return primitive_ratio_from_alpha(alpha_from_label_weights(microcanonical_label_weights(degeneracies)))


def phase_from_degeneracies(degeneracies: Sequence[sp.Expr]) -> sp.Expr:
    """Return the V20 phase induced by reduced degeneracies."""

    return phase_from_alpha(alpha_from_label_weights(microcanonical_label_weights(degeneracies)))


def compressed_macro_degeneracy_control_weights() -> tuple[sp.Expr, sp.Expr]:
    """Return the wrong compressed ``{even, odd_total}`` microcanonical weights."""

    return (sp.Rational(1, 2), sp.Rational(1, 2))


def compressed_macro_degeneracy_control_ratio() -> sp.Expr:
    """Return the V15 ratio from treating odd_total as one macrostate."""

    alpha = compressed_macro_degeneracy_control_weights()[0]
    return primitive_ratio_from_alpha(alpha)


def compressed_macro_degeneracy_control_phase() -> sp.Expr:
    """Return the V15 phase from the compressed macrostate control."""

    alpha = compressed_macro_degeneracy_control_weights()[0]
    return phase_from_alpha(alpha)


@dataclass(frozen=True)
class MicrocanonicalReductionAuditPayload:
    """Verdict payload for the V23 microcanonical reduction theorem."""

    final_verdict: str
    equal_degeneracy_density_uniform: bool
    equal_degeneracy_ratio: sp.Expr
    equal_degeneracy_phase: sp.Expr
    unequal_degeneracy_control_rejected: bool
    compressed_degeneracy_control_rejected: bool
    consistent_with_v22_no_go: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def microcanonical_reduction_audit_payload() -> MicrocanonicalReductionAuditPayload:
    """Return the V23 microcanonical reduction verdict."""

    symbolic_degeneracies = sp.symbols(f"d0:{SHELL_DIMENSION}", positive=True)
    weights = microcanonical_label_weights(symbolic_degeneracies)
    equal_density = equal_degeneracy_reduced_density()
    uniform_density = sp.eye(SHELL_DIMENSION) / SHELL_DIMENSION
    equal_density_uniform = _matrix_equal(equal_density, uniform_density)
    equal_degeneracies = tuple(sp.Integer(1) for _ in range(SHELL_DIMENSION))
    equal_ratio = ratio_from_degeneracies(equal_degeneracies)
    equal_phase = phase_from_degeneracies(equal_degeneracies)

    unequal_degeneracies = (
        sp.Integer(2),
        sp.Integer(1),
        sp.Integer(1),
        sp.Integer(1),
        sp.Integer(1),
        sp.Integer(1),
    )
    unequal_density = microcanonical_reduced_density(unequal_degeneracies)
    unequal_rejected = (
        not _matrix_equal(unequal_density, uniform_density)
        and sp.simplify(phase_from_degeneracies(unequal_degeneracies) - quark_boundary_phase_angle()) != 0
    )

    compressed_phase = compressed_macro_degeneracy_control_phase()
    compressed_rejected = (
        sp.simplify(compressed_macro_degeneracy_control_ratio() - 1 / sp.sqrt(5)) == 0
        and sp.simplify(compressed_phase - sp.pi / 4) == 0
        and sp.simplify(compressed_phase - quark_boundary_phase_angle()) != 0
    )
    v22 = label_conserving_dynamics_audit_payload()
    consistent_with_v22 = v22.final_verdict == "LABEL_CONSERVING_DYNAMICS_MAX_ENTROPY_NO_GO_PASS"

    checks_pass = (
        sp.simplify(sum(weights) - 1) == 0
        and equal_density_uniform
        and sp.simplify(equal_ratio - 1) == 0
        and sp.simplify(equal_phase - quark_boundary_phase_angle()) == 0
        and unequal_rejected
        and compressed_rejected
        and consistent_with_v22
    )

    if checks_pass:
        final_verdict = "EQUAL_DEGENERACY_MICROCANONICAL_REDUCTION_PASS"
        interpretation = (
            "A full microcanonical state on unresolved sectors with equal "
            "bath degeneracy per conserved primitive label reduces to I6/6, "
            "recovering r=1 and atan(sqrt(5)). Unequal degeneracy and "
            "compressed macrochannel controls fail. This is compatible with "
            "V22 because it is a reduced-state inference theorem, not a "
            "label-conserving thermalization theorem."
        )
    else:
        final_verdict = "EQUAL_DEGENERACY_MICROCANONICAL_REDUCTION_KILL"
        interpretation = (
            "The reduced weights, equal-degeneracy uniform density, CKM phase, "
            "negative controls, or V22 compatibility failed."
        )

    return MicrocanonicalReductionAuditPayload(
        final_verdict=final_verdict,
        equal_degeneracy_density_uniform=equal_density_uniform,
        equal_degeneracy_ratio=equal_ratio,
        equal_degeneracy_phase=equal_phase,
        unequal_degeneracy_control_rejected=unequal_rejected,
        compressed_degeneracy_control_rejected=compressed_rejected,
        consistent_with_v22_no_go=consistent_with_v22,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_REDUCTION,
        interpretation=interpretation,
    )
