"""Single import surface for the texture-provenance (A3b) sidecar.

Collects the CKM/PMNS texture factor functions from ``boundary_response`` and the
ergodicity-prior remaining-input marker. No new physics is defined here.
"""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    charged_lepton_rotation_sine,
    selected_port_residual_components,
)
from clifford_3plus2_d5.boundary_response.leptonic_boundary_holonomy import (
    leptonic_boundary_holonomy_audit_payload,
)
from clifford_3plus2_d5.boundary_response.local_boundary_fiber import (
    REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
    quark_gamma_sum,
    quark_shell_dimension_breakdown,
)
from clifford_3plus2_d5.boundary_response.quark_clebsch_factors import (
    bcc_antisymmetric_projection_factor,
    bcc_symmetric_two_path_factor,
    color_return_contraction,
    color_return_factor,
)
from clifford_3plus2_d5.boundary_response.quark_transfer_hierarchy import (
    quark_family_depths,
    quark_transition_depths,
)

__all__ = [
    "REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER",
    "bcc_antisymmetric_projection_factor",
    "bcc_symmetric_two_path_factor",
    "charged_lepton_rotation_sine",
    "color_return_contraction",
    "color_return_factor",
    "leptonic_boundary_holonomy_audit_payload",
    "quark_boundary_phase_angle",
    "quark_family_depths",
    "quark_gamma_sum",
    "quark_shell_dimension_breakdown",
    "quark_transition_depths",
    "selected_port_residual_components",
]
