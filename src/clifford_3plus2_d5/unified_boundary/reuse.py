"""Single import surface for the unified-boundary (A3a) sidecar.

Bridges ``boundary_response`` (the common sterile-chain ``H_Q``, the lepton Schur
self-energy, the quark transfer/Clebsch/coin data), ``universality`` (the V_f
quantum-number catalog), and ``lepton`` (chiral-16 color structure). No new
boundary physics is defined here.
"""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.explicit_hq import (
    green_transfer_amplitude,
    transfer_probe,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
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
from clifford_3plus2_d5.boundary_response.residual_basis import k_nu_operator
from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    residual_graph_decaying_factor,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon
from clifford_3plus2_d5.boundary_response.weyl_sterile import (
    semi_infinite_weyl_function,
    weyl_product_sterile_normalized_response,
)
from clifford_3plus2_d5.universality.u3_coupling_catalog import (
    color_split_is_quark_triplet_lepton_singlet,
)

__all__ = [
    "CKM_TRANSITIONS",
    "bcc_antisymmetric_projection_factor",
    "bcc_symmetric_two_path_factor",
    "color_return_contraction",
    "color_return_factor",
    "color_split_is_quark_triplet_lepton_singlet",
    "epsilon",
    "green_transfer_amplitude",
    "k_nu_operator",
    "quark_boundary_phase_angle",
    "quark_family_depths",
    "quark_transition_amplitude",
    "quark_transition_depths",
    "residual_graph_decaying_factor",
    "semi_infinite_weyl_function",
    "transfer_probe",
    "weyl_product_sterile_normalized_response",
]
