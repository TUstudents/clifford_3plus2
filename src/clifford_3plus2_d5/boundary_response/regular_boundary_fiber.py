"""V24 regular boundary-fiber equal-degeneracy audit.

V23 proves that equal unresolved bath degeneracy per conserved primitive
label reduces the microcanonical boundary state to ``I_6 / 6``.  V24 sharpens
that input.  Conserved-label dynamics alone leaves six independent primitive
populations and six independent bath dimensions.  A regular boundary fiber,

    H_Q = direct_sum_i ( |i>_label tensor B ),

uses the same unresolved bath template ``B`` for every conserved primitive
label.  That regularity forces equal degeneracy, while arbitrary
label-preserving degeneracies remain free.

This is deliberately conservative: it does not claim BCC geometry alone
forces equal degeneracy.  It names the extra physical principle needed to
replace the vaguer V23 input.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.label_conserving_dynamics import (
    label_conserving_dynamics_audit_payload,
)
from clifford_3plus2_d5.boundary_response.microcanonical_reduction import (
    compressed_macro_degeneracy_control_phase,
    compressed_macro_degeneracy_control_ratio,
    microcanonical_label_weights,
    microcanonical_reduced_density,
    phase_from_degeneracies,
    ratio_from_degeneracies,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)

REMAINING_DECLARED_INPUTS_AFTER_REGULAR_FIBER = (
    "vacuum_framing",
    "transfer_probe",
    "regular_boundary_fiber_or_max_entropy_prior",
)


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two exact matrices agree after simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def arbitrary_label_preserving_degeneracies() -> tuple[sp.Expr, ...]:
    """Return symbolic bath dimensions allowed by conserved-label structure."""

    return sp.symbols(f"d0:{SHELL_DIMENSION}", positive=True)


def regular_fiber_degeneracies(fiber_dim: sp.Expr) -> tuple[sp.Expr, ...]:
    """Return equal bath dimensions for a regular primitive boundary fiber."""

    selected = sp.sympify(fiber_dim)
    return tuple(selected for _ in range(SHELL_DIMENSION))


def regular_fiber_degeneracies_are_equal(fiber_dim: sp.Expr) -> bool:
    """Return true when all regular-fiber degeneracies are identical."""

    degeneracies = regular_fiber_degeneracies(fiber_dim)
    reference = degeneracies[0]
    return all(sp.simplify(value - reference) == 0 for value in degeneracies)


def arbitrary_degeneracies_remain_free() -> bool:
    """Return true when conservation alone leaves degeneracy ratios unfixed."""

    weights = microcanonical_label_weights(arbitrary_label_preserving_degeneracies())
    return sp.simplify(weights[0] - weights[1]) != 0


def regular_fiber_reduced_density(fiber_dim: sp.Expr) -> sp.Matrix:
    """Return the reduced primitive density for a regular boundary fiber."""

    return microcanonical_reduced_density(regular_fiber_degeneracies(fiber_dim))


def regular_fiber_density_is_uniform(fiber_dim: sp.Expr) -> bool:
    """Return true when a regular fiber reduces to ``I_6 / 6``."""

    return _matrix_equal(
        regular_fiber_reduced_density(fiber_dim),
        sp.eye(SHELL_DIMENSION) / SHELL_DIMENSION,
    )


def regular_fiber_ratio_and_phase(fiber_dim: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Return the V20 primitive ratio and phase induced by regularity."""

    degeneracies = regular_fiber_degeneracies(fiber_dim)
    return (
        ratio_from_degeneracies(degeneracies),
        phase_from_degeneracies(degeneracies),
    )


def unequal_degeneracy_control_rejected() -> bool:
    """Return true when a label-preserving unequal bath fails CKM flatness."""

    degeneracies = (
        sp.Integer(2),
        sp.Integer(1),
        sp.Integer(1),
        sp.Integer(1),
        sp.Integer(1),
        sp.Integer(1),
    )
    density = microcanonical_reduced_density(degeneracies)
    return (
        not _matrix_equal(density, sp.eye(SHELL_DIMENSION) / SHELL_DIMENSION)
        and sp.simplify(phase_from_degeneracies(degeneracies) - quark_boundary_phase_angle())
        != 0
    )


def compressed_macro_degeneracy_control_rejected() -> bool:
    """Return true when compressed macrochannel counting gives the wrong branch."""

    compressed_phase = compressed_macro_degeneracy_control_phase()
    return (
        sp.simplify(compressed_macro_degeneracy_control_ratio() - 1 / sp.sqrt(5)) == 0
        and sp.simplify(compressed_phase - sp.pi / 4) == 0
        and sp.simplify(compressed_phase - quark_boundary_phase_angle()) != 0
    )


@dataclass(frozen=True)
class RegularBoundaryFiberAuditPayload:
    """Verdict payload for the V24 regular boundary-fiber theorem."""

    final_verdict: str
    regular_degeneracies_equal: bool
    arbitrary_degeneracies_remain_free: bool
    regular_density_uniform: bool
    regular_ratio: sp.Expr
    regular_phase: sp.Expr
    unequal_degeneracy_control_rejected: bool
    compressed_macro_control_rejected: bool
    consistent_with_v22_no_go: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def regular_boundary_fiber_audit_payload() -> RegularBoundaryFiberAuditPayload:
    """Return the V24 regular boundary-fiber verdict."""

    fiber_dim = sp.symbols("D", positive=True)
    regular_equal = regular_fiber_degeneracies_are_equal(fiber_dim)
    arbitrary_free = arbitrary_degeneracies_remain_free()
    regular_uniform = regular_fiber_density_is_uniform(fiber_dim)
    regular_ratio, regular_phase = regular_fiber_ratio_and_phase(fiber_dim)
    unequal_rejected = unequal_degeneracy_control_rejected()
    compressed_rejected = compressed_macro_degeneracy_control_rejected()
    v22 = label_conserving_dynamics_audit_payload()
    consistent_with_v22 = (
        v22.final_verdict == "LABEL_CONSERVING_DYNAMICS_MAX_ENTROPY_NO_GO_PASS"
    )

    checks_pass = (
        regular_equal
        and arbitrary_free
        and regular_uniform
        and sp.simplify(regular_ratio - 1) == 0
        and sp.simplify(regular_phase - quark_boundary_phase_angle()) == 0
        and unequal_rejected
        and compressed_rejected
        and consistent_with_v22
    )

    if checks_pass:
        final_verdict = "REGULAR_BOUNDARY_FIBER_EQUAL_DEGENERACY_PASS"
        interpretation = (
            "Conserved-label dynamics leaves arbitrary primitive bath "
            "degeneracies free. A regular unresolved boundary fiber uses the "
            "same bath template for each conserved primitive label, forcing "
            "equal degeneracy and reducing the microcanonical state to I6/6. "
            "This recovers r=1 and atan(sqrt(5)) while rejecting unequal and "
            "compressed controls. The result is compatible with V22 because "
            "it is a structural degeneracy theorem, not thermalization."
        )
    else:
        final_verdict = "REGULAR_BOUNDARY_FIBER_EQUAL_DEGENERACY_KILL"
        interpretation = (
            "Regular degeneracy, arbitrary-degeneracy freedom, uniform "
            "reduction, CKM phase recovery, negative controls, or V22 "
            "compatibility failed."
        )

    return RegularBoundaryFiberAuditPayload(
        final_verdict=final_verdict,
        regular_degeneracies_equal=regular_equal,
        arbitrary_degeneracies_remain_free=arbitrary_free,
        regular_density_uniform=regular_uniform,
        regular_ratio=regular_ratio,
        regular_phase=regular_phase,
        unequal_degeneracy_control_rejected=unequal_rejected,
        compressed_macro_control_rejected=compressed_rejected,
        consistent_with_v22_no_go=consistent_with_v22,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_REGULAR_FIBER,
        interpretation=interpretation,
    )
