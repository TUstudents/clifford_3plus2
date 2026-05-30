"""Single import surface for the universality sidecar.

Bridges ``boundary_response`` (the per-sector boundary shells and the transfer
invariant) and ``lepton`` (the chiral-16 SM quantum-number content).  No new
Clifford / boundary code is defined here.
"""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    charged_lepton_leakage_depth_amplitude,
)
from clifford_3plus2_d5.boundary_response.conserved_label_partition import (
    conserved_label_partition_is_complete,
    primitive_conserved_labels,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    BCC,
    COLOR,
    DIRECT,
    EVEN,
    ODD,
    quark_primitive_channels,
    quark_shell_dimension_breakdown,
)
from clifford_3plus2_d5.boundary_response.quark_transfer_hierarchy import (
    quark_family_depths,
    quark_transition_amplitude,
)
from clifford_3plus2_d5.boundary_response.residual_basis import residual_vectors
from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    residual_graph_decaying_factor,
    residual_graph_degree,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon, epsilon_fourth
from clifford_3plus2_d5.lepton.sm_hypercharge import sm_field_multiplicity_table

__all__ = [
    "BCC",
    "COLOR",
    "DIRECT",
    "EVEN",
    "ODD",
    "charged_lepton_leakage_depth_amplitude",
    "conserved_label_partition_is_complete",
    "epsilon",
    "epsilon_fourth",
    "primitive_conserved_labels",
    "quark_family_depths",
    "quark_primitive_channels",
    "quark_shell_dimension_breakdown",
    "quark_transition_amplitude",
    "residual_graph_decaying_factor",
    "residual_graph_degree",
    "residual_vectors",
    "sm_field_multiplicity_table",
]
