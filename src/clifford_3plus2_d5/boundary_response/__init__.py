"""Boundary-response sidecar for the residual BCC flavor core audit.

This package is intentionally narrow.  It audits the operator path

    H_Q, V_a -> Sigma(z) -> K_nu = epsilon^2 P_u + P_b

and the explicitly gated conditional PMNS and CKM assemblies.  Those texture
assemblies remain below theorem status unless their boundary-shell assumptions
are derived.
"""

from clifford_3plus2_d5.boundary_response.audit import (
    BoundaryCoreAuditPayload,
    boundary_core_audit_payload,
)
from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    ChargedLeptonLeakageAuditPayload,
    charged_lepton_leakage_audit_payload,
)
from clifford_3plus2_d5.boundary_response.chiral_boundary_normalization import (
    ChiralBoundaryNormalizationAuditPayload,
    chiral_boundary_normalization_audit_payload,
    normalized_odd_collective_vector,
    orthogonal_chiral_swap,
)
from clifford_3plus2_d5.boundary_response.ckm_conditional import (
    CKMConditionalAuditPayload,
    CKMSines,
    ckm_conditional_audit_payload,
    ckm_jarlskog,
    ckm_magnitude_matrix,
    ckm_sines,
    conditional_ckm_matrix,
)
from clifford_3plus2_d5.boundary_response.conserved_label_partition import (
    ConservedLabelPartitionAuditPayload,
    compressed_partition_merges_conserved_labels,
    conserved_label_partition_audit_payload,
    conserved_label_partition_is_complete,
    conserved_label_projectors,
)
from clifford_3plus2_d5.boundary_response.framed_sterile import (
    FramedSterileAuditPayload,
    framed_sterile_audit_payload,
)
from clifford_3plus2_d5.boundary_response.impedance import (
    ImpedanceAuditPayload,
    impedance_audit_payload,
)
from clifford_3plus2_d5.boundary_response.intertwiner_normalization import (
    AlgebraicIntertwinerAuditPayload,
    algebraic_intertwiner_audit_payload,
    even_odd_intertwiner,
    spectral_lift_operator,
)
from clifford_3plus2_d5.boundary_response.jaynes_primitive_ergodicity import (
    JaynesPrimitiveErgodicityAuditPayload,
    jaynes_primitive_density,
    jaynes_primitive_entropy,
    jaynes_primitive_ergodicity_audit_payload,
    primitive_ratio_from_alpha,
)
from clifford_3plus2_d5.boundary_response.label_conserving_dynamics import (
    LabelConservingDynamicsAuditPayload,
    label_conserving_dynamics_audit_payload,
    label_conserving_scattering,
    label_dephasing_channel,
    label_population_density,
)
from clifford_3plus2_d5.boundary_response.microcanonical_reduction import (
    MicrocanonicalReductionAuditPayload,
    equal_degeneracy_reduced_density,
    microcanonical_label_weights,
    microcanonical_reduced_density,
    microcanonical_reduction_audit_payload,
)
from clifford_3plus2_d5.boundary_response.leptonic_boundary_holonomy import (
    LeptonicBoundaryHolonomyAuditPayload,
    leptonic_boundary_holonomy_audit_payload,
)
from clifford_3plus2_d5.boundary_response.leptonic_phase_word import (
    LeptonicPhaseWordAuditPayload,
    leptonic_phase_word_audit_payload,
)
from clifford_3plus2_d5.boundary_response.pmns_conditional import (
    PMNSConditionalAuditPayload,
    PMNSMixingObservables,
    conditional_pmns_matrix,
    pmns_conditional_audit_payload,
    pmns_mixing_observables,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    PrimitiveErgodicityAuditPayload,
    primitive_ergodicity_audit_payload,
)
from clifford_3plus2_d5.boundary_response.primitive_entropy_ergodicity import (
    PrimitiveEntropyErgodicityAuditPayload,
    compressed_macro_entropy,
    primitive_entropy_ergodicity_audit_payload,
    primitive_entropy_probabilities,
    primitive_shannon_entropy,
)
from clifford_3plus2_d5.boundary_response.product_sterile import (
    ProductSterileAuditPayload,
    product_sterile_audit_payload,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    QuarkBoundaryShellAuditPayload,
    quark_boundary_shell_audit_payload,
)
from clifford_3plus2_d5.boundary_response.quark_clebsch_factors import (
    QuarkClebschAuditPayload,
    QuarkClebschSummary,
    quark_clebsch_audit_payload,
)
from clifford_3plus2_d5.boundary_response.quark_coin_rigidity import (
    QuarkCoinRigidityAuditPayload,
    isotropic_quark_coin,
    isotropic_quark_phase_angle,
    quark_coin_rigidity_audit_payload,
)
from clifford_3plus2_d5.boundary_response.quark_transfer_hierarchy import (
    QuarkTransferHierarchyAuditPayload,
    quark_transfer_hierarchy_audit_payload,
)
from clifford_3plus2_d5.boundary_response.regular_boundary_fiber import (
    RegularBoundaryFiberAuditPayload,
    arbitrary_label_preserving_degeneracies,
    regular_boundary_fiber_audit_payload,
    regular_fiber_degeneracies,
    regular_fiber_reduced_density,
)
from clifford_3plus2_d5.boundary_response.weyl_sterile import (
    WeylSterileAuditPayload,
    weyl_sterile_audit_payload,
)
from clifford_3plus2_d5.boundary_response.explicit_hq import (
    ExplicitHqAuditPayload,
    explicit_hq_audit_payload,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    k_nu_operator,
    residual_projectors,
)
from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    ResidualGraphTransferAuditPayload,
    residual_complete_graph_adjacency,
    residual_graph_decaying_factor,
    residual_graph_transfer_audit_payload,
)
from clifford_3plus2_d5.boundary_response.transfer import (
    epsilon,
    epsilon_fourth,
)
from clifford_3plus2_d5.boundary_response.transfer_probe_theorem import (
    TransferProbeTheoremAuditPayload,
    probe_from_transfer_factor,
    transfer_probe_theorem_audit_payload,
)
from clifford_3plus2_d5.boundary_response.unit_continuation import (
    UnitContinuationAuditPayload,
    residual_basis_continuation_couplings,
    unit_continuation_audit_payload,
    unit_outward_matching,
)
from clifford_3plus2_d5.boundary_response.vacuum_framing import (
    VacuumFramingAuditPayload,
    bcc_unoriented_exit_representatives,
    framed_residual_exits,
    vacuum_framing_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector import (
    VacuumSelectorAuditPayload,
    ground_exit_indices,
    selector_energies,
    selector_gap,
    selector_order_parameter,
    vacuum_selector_audit_payload,
)

__all__ = [
    "AlgebraicIntertwinerAuditPayload",
    "BoundaryCoreAuditPayload",
    "CKMConditionalAuditPayload",
    "CKMSines",
    "ChargedLeptonLeakageAuditPayload",
    "ChiralBoundaryNormalizationAuditPayload",
    "ConservedLabelPartitionAuditPayload",
    "ExplicitHqAuditPayload",
    "FramedSterileAuditPayload",
    "ImpedanceAuditPayload",
    "JaynesPrimitiveErgodicityAuditPayload",
    "LabelConservingDynamicsAuditPayload",
    "LeptonicBoundaryHolonomyAuditPayload",
    "LeptonicPhaseWordAuditPayload",
    "MicrocanonicalReductionAuditPayload",
    "PMNSConditionalAuditPayload",
    "PMNSMixingObservables",
    "PrimitiveEntropyErgodicityAuditPayload",
    "PrimitiveErgodicityAuditPayload",
    "ProductSterileAuditPayload",
    "QuarkBoundaryShellAuditPayload",
    "QuarkClebschAuditPayload",
    "QuarkClebschSummary",
    "QuarkCoinRigidityAuditPayload",
    "QuarkTransferHierarchyAuditPayload",
    "RegularBoundaryFiberAuditPayload",
    "ResidualGraphTransferAuditPayload",
    "TransferProbeTheoremAuditPayload",
    "UnitContinuationAuditPayload",
    "VacuumFramingAuditPayload",
    "VacuumSelectorAuditPayload",
    "WeylSterileAuditPayload",
    "algebraic_intertwiner_audit_payload",
    "arbitrary_label_preserving_degeneracies",
    "boundary_core_audit_payload",
    "bcc_unoriented_exit_representatives",
    "charged_lepton_leakage_audit_payload",
    "chiral_boundary_normalization_audit_payload",
    "ckm_conditional_audit_payload",
    "ckm_jarlskog",
    "ckm_magnitude_matrix",
    "ckm_sines",
    "compressed_macro_entropy",
    "compressed_partition_merges_conserved_labels",
    "conserved_label_partition_audit_payload",
    "conserved_label_partition_is_complete",
    "conserved_label_projectors",
    "conditional_ckm_matrix",
    "conditional_pmns_matrix",
    "equal_degeneracy_reduced_density",
    "epsilon",
    "epsilon_fourth",
    "even_odd_intertwiner",
    "explicit_hq_audit_payload",
    "framed_sterile_audit_payload",
    "framed_residual_exits",
    "ground_exit_indices",
    "impedance_audit_payload",
    "isotropic_quark_coin",
    "isotropic_quark_phase_angle",
    "jaynes_primitive_density",
    "jaynes_primitive_entropy",
    "jaynes_primitive_ergodicity_audit_payload",
    "k_nu_operator",
    "label_conserving_dynamics_audit_payload",
    "label_conserving_scattering",
    "label_dephasing_channel",
    "label_population_density",
    "leptonic_boundary_holonomy_audit_payload",
    "leptonic_phase_word_audit_payload",
    "microcanonical_label_weights",
    "microcanonical_reduced_density",
    "microcanonical_reduction_audit_payload",
    "normalized_odd_collective_vector",
    "orthogonal_chiral_swap",
    "pmns_conditional_audit_payload",
    "pmns_mixing_observables",
    "probe_from_transfer_factor",
    "primitive_entropy_ergodicity_audit_payload",
    "primitive_entropy_probabilities",
    "primitive_ergodicity_audit_payload",
    "primitive_ratio_from_alpha",
    "primitive_shannon_entropy",
    "product_sterile_audit_payload",
    "quark_boundary_shell_audit_payload",
    "quark_clebsch_audit_payload",
    "quark_coin_rigidity_audit_payload",
    "quark_transfer_hierarchy_audit_payload",
    "regular_boundary_fiber_audit_payload",
    "regular_fiber_degeneracies",
    "regular_fiber_reduced_density",
    "residual_complete_graph_adjacency",
    "residual_basis_continuation_couplings",
    "residual_graph_decaying_factor",
    "residual_graph_transfer_audit_payload",
    "residual_projectors",
    "selector_energies",
    "selector_gap",
    "selector_order_parameter",
    "spectral_lift_operator",
    "transfer_probe_theorem_audit_payload",
    "unit_continuation_audit_payload",
    "unit_outward_matching",
    "vacuum_framing_audit_payload",
    "vacuum_selector_audit_payload",
    "weyl_sterile_audit_payload",
]
