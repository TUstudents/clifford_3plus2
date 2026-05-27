"""V7 charged-lepton two-step leakage gate.

V6 proves the neutrino-core response

    K_nu = epsilon^2 P_u + P_b

from the semi-infinite product sterile tail.  V7 audits only the next
conditional PMNS ingredient, Assumption L1: the selected charged-lepton/Higgs
frame has a two-step residual leakage relative to the neutrino transfer basis.

The selected active port is the first residual basis vector

    e1 = (1, 0, 0),

which decomposes as

    e1 = sqrt(2/3) a + 1/sqrt(3) u.

If the leading leakage is two Weyl-transfer steps, then

    sqrt(2/3) sin(theta_e) = epsilon^2,

so

    sin(theta_e) = sqrt(3/2) epsilon^2.

This module does not assemble PMNS and does not derive the leptonic phase word.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import transfer_probe
from clifford_3plus2_d5.boundary_response.residual_basis import (
    residual_basis_matrix,
    standard_basis,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon_fourth
from clifford_3plus2_d5.boundary_response.weyl_sterile import semi_infinite_weyl_function


def selected_charged_lepton_port() -> sp.Matrix:
    """Return the selected charged-lepton/Higgs residual port ``e1``."""

    return standard_basis()[0]


def selected_port_residual_components() -> dict[str, sp.Expr]:
    """Return components of ``e1`` in the residual ``(a,u,b)`` basis."""

    port = selected_charged_lepton_port()
    basis = residual_basis_matrix(("a", "u", "b"))
    components = (basis.T * port).applyfunc(sp.simplify)
    return {
        "a": components[0, 0],
        "u": components[1, 0],
        "b": components[2, 0],
    }


def charged_lepton_leakage_depth_amplitude(
    depth: int,
    z: sp.Expr | None = None,
) -> sp.Expr:
    """Return the exact Weyl-transfer leakage amplitude at integer depth."""

    if depth < 0:
        raise ValueError("depth must be non-negative")
    probe = transfer_probe() if z is None else z
    return sp.simplify(semi_infinite_weyl_function(probe) ** depth)


def charged_lepton_rotation_sine(z: sp.Expr | None = None) -> sp.Expr:
    """Return the L1 charged-lepton rotation sine from two-step leakage."""

    components = selected_port_residual_components()
    a_projection = components["a"]
    if a_projection == 0:
        raise ZeroDivisionError("selected port has no radial component")
    return sp.simplify(charged_lepton_leakage_depth_amplitude(2, z=z) / a_projection)


def charged_lepton_rotation_sine_squared(z: sp.Expr | None = None) -> sp.Expr:
    """Return the exact squared charged-lepton rotation sine."""

    sine = charged_lepton_rotation_sine(z=z)
    return sp.simplify(sine**2)


def charged_lepton_rotation_sine_for_depth(depth: int, z: sp.Expr | None = None) -> sp.Expr:
    """Return the rotation sine implied by a non-default leakage depth."""

    components = selected_port_residual_components()
    return sp.simplify(charged_lepton_leakage_depth_amplitude(depth, z=z) / components["a"])


def synthetic_b_leakage_response(
    *,
    b_weight: sp.Expr = sp.Integer(1),
) -> sp.Matrix:
    """Return a synthetic leakage vector with forbidden ``b`` contamination."""

    basis = residual_basis_matrix(("a", "u", "b"))
    components = sp.Matrix(
        [
            selected_port_residual_components()["a"],
            selected_port_residual_components()["u"],
            b_weight,
        ]
    )
    return (basis * components).applyfunc(sp.simplify)


@dataclass(frozen=True)
class ChargedLeptonLeakageAuditPayload:
    """Verdict payload for the V7 charged-lepton leakage gate."""

    final_verdict: str
    selected_port: sp.Matrix
    component_a: sp.Expr
    component_u: sp.Expr
    component_b: sp.Expr
    two_step_leakage: sp.Expr
    rotation_sine: sp.Expr
    rotation_sine_squared: sp.Expr
    depth_one_control_matches: bool
    depth_three_control_matches: bool
    b_leakage_control_detected: bool
    pmns_ckm_parked: bool
    interpretation: str


def charged_lepton_leakage_audit_payload() -> ChargedLeptonLeakageAuditPayload:
    """Return the V7 charged-lepton leakage verdict."""

    components = selected_port_residual_components()
    expected_sine = sp.sqrt(sp.Rational(3, 2)) * charged_lepton_leakage_depth_amplitude(2)
    expected_sine_squared = sp.Rational(3, 2) * epsilon_fourth()
    rotation_sine = charged_lepton_rotation_sine()
    rotation_sine_squared = charged_lepton_rotation_sine_squared()

    depth_one_matches = sp.simplify(charged_lepton_rotation_sine_for_depth(1) - expected_sine) == 0
    depth_three_matches = sp.simplify(charged_lepton_rotation_sine_for_depth(3) - expected_sine) == 0

    synthetic = synthetic_b_leakage_response()
    basis = residual_basis_matrix(("a", "u", "b"))
    synthetic_components = (basis.T * synthetic).applyfunc(sp.simplify)
    b_leakage_detected = sp.simplify(synthetic_components[2, 0]) != 0

    checks_pass = (
        sp.simplify(components["a"] - sp.sqrt(sp.Rational(2, 3))) == 0
        and sp.simplify(components["u"] - 1 / sp.sqrt(3)) == 0
        and components["b"] == 0
        and sp.simplify(rotation_sine - expected_sine) == 0
        and sp.simplify(rotation_sine_squared - expected_sine_squared) == 0
        and not depth_one_matches
        and not depth_three_matches
        and b_leakage_detected
    )

    if checks_pass:
        final_verdict = "CHARGED_LEPTON_LEAKAGE_PASS"
        interpretation = (
            "The selected charged-lepton/Higgs port decomposes as "
            "sqrt(2/3) a + 1/sqrt(3) u with no b component, and two exact "
            "Weyl-transfer steps give sin(theta_e) = sqrt(3/2) epsilon^2. "
            "This derives only Assumption L1; the leptonic phase word and "
            "PMNS assembly remain parked."
        )
    else:
        final_verdict = "CHARGED_LEPTON_LEAKAGE_KILL"
        interpretation = (
            "The selected-port decomposition, two-step Weyl leakage, depth "
            "controls, or b-leakage control failed. PMNS and CKM remain "
            "parked."
        )

    return ChargedLeptonLeakageAuditPayload(
        final_verdict=final_verdict,
        selected_port=selected_charged_lepton_port(),
        component_a=components["a"],
        component_u=components["u"],
        component_b=components["b"],
        two_step_leakage=charged_lepton_leakage_depth_amplitude(2),
        rotation_sine=rotation_sine,
        rotation_sine_squared=rotation_sine_squared,
        depth_one_control_matches=depth_one_matches,
        depth_three_control_matches=depth_three_matches,
        b_leakage_control_detected=b_leakage_detected,
        pmns_ckm_parked=True,
        interpretation=interpretation,
    )
