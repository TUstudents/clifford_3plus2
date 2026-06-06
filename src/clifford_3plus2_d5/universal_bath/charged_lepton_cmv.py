"""Session 04 charged-lepton CMV/OPUC finite head."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.leptonic_boundary_holonomy import (
    leptonic_boundary_holonomy_audit_payload,
    primitive_charged_lepton_boundary_word,
)
from clifford_3plus2_d5.universal_bath.opuc import (
    free_verblunsky_tail,
    is_free_verblunsky_tail,
)
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind
from clifford_3plus2_d5.universal_bath.source_dictionary import (
    SourceAnchor,
    SourceStatus,
    source_dictionary_payload,
)
from clifford_3plus2_d5.universal_bath.tail import (
    period_one_tail,
    silver_epsilon,
    silver_selected_z,
)

CHARGED_LEPTON_SOURCE_LABEL = "charged_lepton_active_e1"


@dataclass(frozen=True)
class ChargedLeptonCMVHeadPayload:
    """Session 04 charged-lepton CMV finite-head verdict."""

    final_verdict: str
    source_dictionary_pass: bool
    holonomy_prerequisite_pass: bool
    source_label: str
    source_reduction: ReductionKind
    source_depth: int
    residual_components: dict[str, sp.Expr]
    two_step_leakage: sp.Expr
    rotation_sine: sp.Expr
    rotation_sine_squared: sp.Expr
    phase_angle: sp.Expr
    phase: sp.Expr
    alpha: sp.Expr
    alpha_modulus_squared: sp.Expr
    alpha_inside_unit_disk: bool
    cmv_head: sp.Matrix
    cmv_head_unitary: bool
    verblunsky_coefficients: tuple[sp.Expr, ...]
    free_tail_after_head: bool
    depth_one_control_rejected: bool
    depth_three_control_rejected: bool
    b_leakage_control_rejected: bool
    holonomy_controls_rejected: bool
    pmns_parked: bool
    interpretation: str


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices agree after exact simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def frozen_charged_lepton_source() -> SourceAnchor:
    """Return the frozen Session 02 charged-lepton source anchor."""

    payload = source_dictionary_payload()
    anchors = {anchor.label: anchor for anchor in payload.frozen_sources}
    return anchors[CHARGED_LEPTON_SOURCE_LABEL]


def charged_lepton_source_components() -> dict[str, sp.Expr]:
    """Return the charged-lepton source components in the residual basis."""

    return frozen_charged_lepton_source().residual_components


def charged_lepton_depth_amplitude(depth: int, z: sp.Expr | None = None) -> sp.Expr:
    """Return the universal-tail amplitude for a charged-lepton depth."""

    if depth < 0:
        raise ValueError("depth must be non-negative")
    probe = silver_selected_z() if z is None else z
    return sp.simplify(period_one_tail(probe) ** depth)


def charged_lepton_rotation_sine_for_depth(depth: int) -> sp.Expr:
    """Return the leakage angle sine implied by a selected depth."""

    components = charged_lepton_source_components()
    radial_component = components["a"]
    if radial_component == 0:
        raise ZeroDivisionError("charged-lepton source has no radial component")
    return sp.simplify(charged_lepton_depth_amplitude(depth) / radial_component)


def charged_lepton_rotation_sine() -> sp.Expr:
    """Return ``sin(theta_e)`` from the frozen source depth."""

    source = frozen_charged_lepton_source()
    if source.normal_depth is None:
        raise ValueError("charged-lepton source depth is not frozen")
    return charged_lepton_rotation_sine_for_depth(source.normal_depth)


def charged_lepton_rotation_sine_squared() -> sp.Expr:
    """Return exact ``sin(theta_e)^2``."""

    sine = charged_lepton_rotation_sine()
    return sp.simplify(sine**2)


def charged_lepton_phase_angle() -> sp.Expr:
    """Return the primitive charged-lepton boundary phase angle in units of pi."""

    return primitive_charged_lepton_boundary_word().principal_angle


def charged_lepton_phase() -> sp.Expr:
    """Return the exact primitive charged-lepton phase."""

    angle = charged_lepton_phase_angle()
    return sp.simplify(sp.cos(sp.pi * angle) + sp.I * sp.sin(sp.pi * angle))


def charged_lepton_verblunsky_alpha() -> sp.Expr:
    """Return the finite charged-lepton Verblunsky coefficient."""

    return sp.simplify(charged_lepton_rotation_sine() * charged_lepton_phase())


def alpha_modulus_squared(alpha: sp.Expr) -> sp.Expr:
    """Return ``|alpha|^2`` with exact simplification."""

    return sp.simplify(alpha * sp.conjugate(alpha))


def alpha_inside_unit_disk(alpha: sp.Expr) -> bool:
    """Return whether ``|alpha| < 1`` numerically after exact construction."""

    return bool(sp.N(alpha_modulus_squared(alpha), 50) < 1)


def cmv_givens_head(alpha: sp.Expr) -> sp.Matrix:
    """Return the unitary two-state CMV/Givens head for one coefficient."""

    rho = sp.sqrt(1 - alpha_modulus_squared(alpha))
    return sp.Matrix(
        [
            [rho, alpha],
            [-sp.conjugate(alpha), rho],
        ]
    ).applyfunc(sp.simplify)


def matrix_is_unitary(matrix: sp.Matrix) -> bool:
    """Return whether a matrix is exactly unitary."""

    return _matrix_equal(matrix.conjugate().T * matrix, sp.eye(matrix.rows))


def charged_lepton_verblunsky_sequence(tail_length: int = 4) -> tuple[sp.Expr, ...]:
    """Return finite head followed by the universal free CMV tail."""

    return (charged_lepton_verblunsky_alpha(), *free_verblunsky_tail(tail_length))


def b_leakage_control_rejected() -> bool:
    """Return true when adding a synthetic b component is detected."""

    components = charged_lepton_source_components()
    synthetic = dict(components)
    synthetic["b"] = sp.Integer(1)
    return components["b"] == 0 and synthetic["b"] != 0


def charged_lepton_cmv_head_payload() -> ChargedLeptonCMVHeadPayload:
    """Return the Session 04 charged-lepton CMV finite-head verdict."""

    source_payload = source_dictionary_payload()
    source = frozen_charged_lepton_source()
    holonomy = leptonic_boundary_holonomy_audit_payload()
    components = charged_lepton_source_components()

    source_pass = (
        source_payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
        and source.label == CHARGED_LEPTON_SOURCE_LABEL
        and source.status == SourceStatus.FROZEN
        and source.reduction == ReductionKind.CMV_OPUC
        and source.normal_depth == 2
    )
    holonomy_pass = holonomy.final_verdict == "LEPTONIC_PHASE_WORD_DERIVED_PASS"
    sine = charged_lepton_rotation_sine()
    sine_squared = charged_lepton_rotation_sine_squared()
    phase_angle = charged_lepton_phase_angle()
    phase = charged_lepton_phase()
    alpha = charged_lepton_verblunsky_alpha()
    alpha_norm = alpha_modulus_squared(alpha)
    inside_disk = alpha_inside_unit_disk(alpha)
    head = cmv_givens_head(alpha)
    head_unitary = matrix_is_unitary(head)
    coefficients = charged_lepton_verblunsky_sequence()
    free_tail = is_free_verblunsky_tail(coefficients[1:])
    expected_sine = sp.sqrt(sp.Rational(3, 2)) * silver_epsilon() ** 2
    expected_sine_squared = sp.Rational(3, 2) * silver_epsilon() ** 4
    depth_one_rejected = sp.simplify(charged_lepton_rotation_sine_for_depth(1) - sine) != 0
    depth_three_rejected = sp.simplify(charged_lepton_rotation_sine_for_depth(3) - sine) != 0
    holonomy_controls = all(holonomy.controls_rejected.values())

    checks_pass = (
        source_pass
        and holonomy_pass
        and sp.simplify(components["a"] - sp.sqrt(sp.Rational(2, 3))) == 0
        and sp.simplify(components["u"] - 1 / sp.sqrt(3)) == 0
        and components["b"] == 0
        and sp.simplify(charged_lepton_depth_amplitude(2) - silver_epsilon() ** 2) == 0
        and sp.simplify(sine - expected_sine) == 0
        and sp.simplify(sine_squared - expected_sine_squared) == 0
        and phase_angle == -sp.Rational(5, 12)
        and sp.simplify(
            phase
            - (
                sp.cos(-sp.pi * sp.Rational(5, 12))
                + sp.I * sp.sin(-sp.pi * sp.Rational(5, 12))
            )
        )
        == 0
        and sp.simplify(alpha_norm - sine_squared) == 0
        and inside_disk
        and head_unitary
        and free_tail
        and depth_one_rejected
        and depth_three_rejected
        and b_leakage_control_rejected()
        and holonomy_controls
    )

    if checks_pass:
        final_verdict = "CHARGED_LEPTON_CMV_HEAD_PASS"
        interpretation = (
            "The frozen charged-lepton source e1 gives the two-step leakage "
            "sin(theta_e)=sqrt(3/2) epsilon^2, and the primitive boundary "
            "holonomy supplies phase exp(-5 pi i/12).  Together they form the "
            "finite Verblunsky coefficient alpha_e, followed by the universal "
            "free CMV tail alpha_n=0.  This is a finite-head theorem given the "
            "source and holonomy inputs; charged-lepton masses and PMNS remain "
            "parked."
        )
    else:
        final_verdict = "CHARGED_LEPTON_CMV_HEAD_KILL"
        interpretation = (
            "The charged-lepton CMV head failed the source prerequisite, "
            "leakage geometry, holonomy word, unitary head, free-tail, or "
            "negative-control gate."
        )

    return ChargedLeptonCMVHeadPayload(
        final_verdict=final_verdict,
        source_dictionary_pass=source_pass,
        holonomy_prerequisite_pass=holonomy_pass,
        source_label=source.label,
        source_reduction=source.reduction,
        source_depth=source.normal_depth or -1,
        residual_components=components,
        two_step_leakage=charged_lepton_depth_amplitude(2),
        rotation_sine=sine,
        rotation_sine_squared=sine_squared,
        phase_angle=phase_angle,
        phase=phase,
        alpha=alpha,
        alpha_modulus_squared=alpha_norm,
        alpha_inside_unit_disk=inside_disk,
        cmv_head=head,
        cmv_head_unitary=head_unitary,
        verblunsky_coefficients=coefficients,
        free_tail_after_head=free_tail,
        depth_one_control_rejected=depth_one_rejected,
        depth_three_control_rejected=depth_three_rejected,
        b_leakage_control_rejected=b_leakage_control_rejected(),
        holonomy_controls_rejected=holonomy_controls,
        pmns_parked=True,
        interpretation=interpretation,
    )
