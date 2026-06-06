"""R11 silver-transfer inheritance gate.

The silver transfer root is already derived in ``boundary_response`` and reused
by ``flavor_a_track``.  This gate prevents ``radial_response`` from becoming a
second transfer calculator: it imports the established root and records how the
radial mass sidecar uses its powers.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import transfer_probe
from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    residual_graph_decaying_factor,
)
from clifford_3plus2_d5.boundary_response.transfer import (
    epsilon,
    epsilon_fourth,
    epsilon_squared,
)
from clifford_3plus2_d5.boundary_response.weyl_sterile import (
    semi_infinite_weyl_function,
)
from clifford_3plus2_d5.flavor_a_track.a32_quark_transfer_schur import (
    quark_transfer_schur_audit_payload,
)
from clifford_3plus2_d5.flavor_a_track.u1_shared_transfer import (
    shared_transfer_audit_payload,
)
from clifford_3plus2_d5.radial_response.unitary_defect import (
    minimal_unitary_defect_payload,
)


INDEPENDENT_ETA_CONTROL = sp.Rational(172089, 1000000)


@dataclass(frozen=True)
class SilverTransferInheritancePayload:
    """Payload for the R11 radial silver-transfer inheritance gate."""

    final_verdict: str
    inherited_root: sp.Expr
    eta: sp.Expr
    intensity_ratio: sp.Expr
    eta_matches_root_square: bool
    intensity_matches_root_fourth: bool
    residual_graph_source_matches: bool
    weyl_chain_source_matches: bool
    flavor_shared_transfer_passes: bool
    quark_transfer_schur_passes: bool
    independent_eta_control: sp.Expr
    independent_eta_control_rejected: bool
    k2_root_control_rejected: bool
    k4_root_control_rejected: bool
    duplicate_local_derivation_used: bool
    minimal_unitary_value_forcing_rejected: bool
    source_modules: tuple[str, ...]
    interpretation: str


def inherited_transfer_root() -> sp.Expr:
    """Return the established silver transfer root from ``boundary_response``."""

    return sp.simplify(epsilon())


def inherited_eta() -> sp.Expr:
    """Return the inherited amplitude-level two-step factor."""

    return sp.simplify(epsilon_squared())


def inherited_intensity_ratio() -> sp.Expr:
    """Return the inherited probability/intensity-level factor."""

    return sp.simplify(epsilon_fourth())


def residual_graph_source_matches() -> bool:
    """Return whether the residual K3 graph source matches the inherited root."""

    return sp.simplify(residual_graph_decaying_factor(3) - inherited_transfer_root()) == 0


def weyl_chain_source_matches() -> bool:
    """Return whether the sterile-chain Weyl source matches the inherited root."""

    return sp.simplify(semi_infinite_weyl_function(transfer_probe()) - inherited_transfer_root()) == 0


def candidate_eta_is_inherited(candidate: sp.Expr) -> bool:
    """Return whether a candidate ``eta`` is exactly the inherited ``epsilon^2``."""

    return sp.simplify(sp.sympify(candidate) - inherited_eta()) == 0


def independent_eta_control_rejected(
    candidate: sp.Expr = INDEPENDENT_ETA_CONTROL,
) -> bool:
    """Return whether a fitted independent eta value is rejected."""

    return not candidate_eta_is_inherited(candidate)


def candidate_transfer_root_is_inherited(candidate: sp.Expr) -> bool:
    """Return whether a candidate transfer root equals the inherited root."""

    return sp.simplify(sp.sympify(candidate) - inherited_transfer_root()) == 0


def second_transfer_root_controls_rejected() -> bool:
    """Return whether K2 and K4 graph roots are rejected as alternate roots."""

    return (
        not candidate_transfer_root_is_inherited(residual_graph_decaying_factor(2))
        and not candidate_transfer_root_is_inherited(residual_graph_decaying_factor(4))
    )


def minimal_unitary_value_forcing_rejected() -> bool:
    """Return whether the minimal ``U=S C`` toy is rejected as value-forcing."""

    payload = minimal_unitary_defect_payload()
    return (
        payload.final_verdict == "MINIMAL_UNITARY_S3_DEFECT_FORM_PASS"
        and payload.coin_angle_changes_self_energy
        and payload.defect_vector_changes_self_energy
        and not payload.phase_and_radial_values_forced_by_form
    )


def radial_silver_transfer_inheritance_pass() -> bool:
    """Return whether radial_response inherits the already-derived transfer root."""

    root = inherited_transfer_root()
    eta = inherited_eta()
    intensity = inherited_intensity_ratio()
    shared = shared_transfer_audit_payload()
    quark = quark_transfer_schur_audit_payload()
    return (
        sp.simplify(eta - root**2) == 0
        and sp.simplify(intensity - root**4) == 0
        and residual_graph_source_matches()
        and weyl_chain_source_matches()
        and shared.final_verdict == "SHARED_TRANSFER_INVARIANT"
        and quark.final_verdict == "QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR"
        and independent_eta_control_rejected()
        and second_transfer_root_controls_rejected()
        and minimal_unitary_value_forcing_rejected()
    )


def silver_transfer_inheritance_payload() -> SilverTransferInheritancePayload:
    """Return the R11 silver-transfer inheritance verdict."""

    root = inherited_transfer_root()
    eta = inherited_eta()
    intensity = inherited_intensity_ratio()
    residual_matches = residual_graph_source_matches()
    weyl_matches = weyl_chain_source_matches()
    shared = shared_transfer_audit_payload()
    quark = quark_transfer_schur_audit_payload()
    independent_rejected = independent_eta_control_rejected()
    k2_rejected = not candidate_transfer_root_is_inherited(residual_graph_decaying_factor(2))
    k4_rejected = not candidate_transfer_root_is_inherited(residual_graph_decaying_factor(4))
    unitary_rejected = minimal_unitary_value_forcing_rejected()
    checks_pass = radial_silver_transfer_inheritance_pass()

    if checks_pass:
        final_verdict = "RADIAL_SILVER_TRANSFER_INHERITANCE_PASS"
        interpretation = (
            "radial_response inherits the silver transfer root from the existing "
            "boundary_response/flavor_a_track transfer stack instead of "
            "recomputing or fitting it locally. The inherited root epsilon = "
            "sqrt(2)-1 gives eta = epsilon^2 for radial mass amplitudes and "
            "r = epsilon^4 for intensity/mixing relations. Independent fitted "
            "eta, K2/K4 alternate roots, and the claim that the minimal U=S C "
            "toy alone forces radial values are all rejected. The open radial "
            "burden is pole/residue rigidity, not silver-root derivation."
        )
    else:
        final_verdict = "RADIAL_SILVER_TRANSFER_INHERITANCE_KILL"
        interpretation = (
            "radial_response failed to inherit the established silver transfer "
            "root, or a negative control did not reject an independent radial "
            "transfer value."
        )

    return SilverTransferInheritancePayload(
        final_verdict=final_verdict,
        inherited_root=root,
        eta=eta,
        intensity_ratio=intensity,
        eta_matches_root_square=sp.simplify(eta - root**2) == 0,
        intensity_matches_root_fourth=sp.simplify(intensity - root**4) == 0,
        residual_graph_source_matches=residual_matches,
        weyl_chain_source_matches=weyl_matches,
        flavor_shared_transfer_passes=shared.final_verdict == "SHARED_TRANSFER_INVARIANT",
        quark_transfer_schur_passes=quark.final_verdict
        == "QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR",
        independent_eta_control=INDEPENDENT_ETA_CONTROL,
        independent_eta_control_rejected=independent_rejected,
        k2_root_control_rejected=k2_rejected,
        k4_root_control_rejected=k4_rejected,
        duplicate_local_derivation_used=False,
        minimal_unitary_value_forcing_rejected=unitary_rejected,
        source_modules=(
            "boundary_response.transfer",
            "boundary_response.residual_graph_transfer",
            "boundary_response.weyl_sterile",
            "flavor_a_track.u1_shared_transfer",
            "flavor_a_track.a32_quark_transfer_schur",
        ),
        interpretation=interpretation,
    )
