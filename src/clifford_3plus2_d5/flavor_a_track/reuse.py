"""Single external import surface for the flavor A-track.

Collects everything the A-track gates borrow from ``boundary_response`` (the
transfer invariant, the sterile-chain ``H_Q`` and its Schur self-energy, the
sector boundary shells, the quark Clebsch/coin data, the V10 leptonic holonomy,
the ergodicity-prior marker) and from ``lepton`` (the chiral-16 SM quantum-number
content). No new boundary physics is defined here.

This surface re-exports only *external* (boundary_response / lepton) symbols.
Intra-A-track references (one gate using another, e.g. A3-3 reading U3's
color split) use ordinary sibling imports, not this module — that keeps ``reuse``
free of any ``flavor_a_track`` import and avoids an import cycle.
"""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    charged_lepton_leakage_depth_amplitude,
    selected_port_residual_components,
)
from clifford_3plus2_d5.boundary_response.conserved_label_partition import (
    conserved_label_partition_is_complete,
    primitive_conserved_labels,
)
from clifford_3plus2_d5.boundary_response.explicit_hq import (
    green_transfer_amplitude,
    transfer_probe,
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
    CKM_TRANSITIONS,
    quark_family_depths,
    quark_transition_amplitude,
    quark_transition_depths,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    k_nu_operator,
    residual_vectors,
)
from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    residual_graph_decaying_factor,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon, epsilon_fourth
from clifford_3plus2_d5.boundary_response.weyl_sterile import (
    semi_infinite_weyl_function,
    weyl_product_sterile_normalized_response,
)
from clifford_3plus2_d5.lepton.sm_hypercharge import sm_field_multiplicity_table

__all__ = [
    "CKM_TRANSITIONS",
    "REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER",
    "bcc_antisymmetric_projection_factor",
    "bcc_symmetric_two_path_factor",
    "charged_lepton_leakage_depth_amplitude",
    "color_return_contraction",
    "color_return_factor",
    "conserved_label_partition_is_complete",
    "epsilon",
    "epsilon_fourth",
    "green_transfer_amplitude",
    "k_nu_operator",
    "leptonic_boundary_holonomy_audit_payload",
    "primitive_conserved_labels",
    "quark_boundary_phase_angle",
    "quark_family_depths",
    "quark_gamma_sum",
    "quark_shell_dimension_breakdown",
    "quark_transition_amplitude",
    "quark_transition_depths",
    "residual_graph_decaying_factor",
    "residual_vectors",
    "selected_port_residual_components",
    "semi_infinite_weyl_function",
    "sm_field_multiplicity_table",
    "transfer_probe",
    "weyl_product_sterile_normalized_response",
]
