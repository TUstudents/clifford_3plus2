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
from clifford_3plus2_d5.boundary_response.local_boundary_fiber import (
    LocalBoundaryFiberAuditPayload,
    local_boundary_fiber_audit_payload,
    local_fiber_degeneracies,
    local_fiber_reduced_density,
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
from clifford_3plus2_d5.boundary_response.vacuum_selector_bb_induced_breaking import (
    BBInducedRadialBreakingAuditPayload,
    bb_induced_radial_breaking_audit_payload,
    bb_induced_radial_energy_samples,
    bb_induced_radial_minimum_candidates,
    quartic_backreaction_potential,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    ChiralBBSelectorSignAuditPayload,
    bb_scalar_h2_quadratic_coefficients,
    bb_scalar_h2_xyz_coefficient,
    bb_trace_b3_xyz_coefficient,
    chiral_bb_selector_sign_audit_payload,
    filled_band_parity_odd_energy,
    filled_band_selector_ratio,
    filled_band_selector_sign_passes,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_condensation import (
    VacuumSelectorCondensationAuditPayload,
    selector_locking_potential,
    tetrahedral_cubic_polynomial,
    vacuum_selector_condensation_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_filled_band_potential import (
    FilledBandSelectorPotentialAuditPayload,
    filled_band_selector_branch_energies,
    filled_band_selector_branch_gap,
    filled_band_selector_candidate_energy,
    filled_band_selector_potential_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_higgs_stabilizer import (
    HiggsBackreactionRadialStabilizerAuditPayload,
    higgs_backreaction_radial_stabilizer_audit_payload,
    mexican_hat_radial_potential,
    stabilized_radial_energy_samples,
    stabilized_radial_minimum_candidates,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_higgs_origin import (
    HiggsRadialLandauOriginAuditPayload,
    completed_square_vev_squared,
    higgs_radial_landau_origin_audit_payload,
    spacetime_higgs_potential_pullback,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_radial_theorem import (
    AnalyticRadialBreakingTheoremAuditPayload,
    analytic_radial_breaking_theorem_audit_payload,
    analytic_radial_energy,
    massless_quartic_stationary_radius,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_closure import (
    VacuumSelectorClosureAuditPayload,
    vacuum_selector_closure_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_landau import (
    TetrahedralLandauAuditPayload,
    invariant_space_dimension,
    lowest_selector_anisotropy_degree,
    tetrahedral_landau_audit_payload,
    tetrahedral_landau_locking_potential,
    tetrahedral_rotation_group,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_microscopic_potential import (
    MicroscopicFilledBandPotentialAuditPayload,
    filled_band_effective_energy,
    filled_band_even_radial_energy,
    filled_band_odd_selector_energy,
    microscopic_selector_potential_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_radial_no_go import (
    FreeBBRadialNoGoAuditPayload,
    free_bb_radial_stabilization_audit_payload,
    radial_energy_finite_differences,
    radial_selector_energy_samples,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_schur_landau import (
    SchurShellLandauAuditPayload,
    schur_shell_cubic_coefficient,
    schur_shell_landau_audit_payload,
    schur_shell_landau_series,
    tetrahedral_shell_power_sum,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    VacuumSelectorPotentialAuditPayload,
    selector_potential_energy,
    tetrahedral_selector_candidates,
    vacuum_selector_potential_audit_payload,
)

__all__ = [
    "AlgebraicIntertwinerAuditPayload",
    "AnalyticRadialBreakingTheoremAuditPayload",
    "BBInducedRadialBreakingAuditPayload",
    "BoundaryCoreAuditPayload",
    "CKMConditionalAuditPayload",
    "CKMSines",
    "ChargedLeptonLeakageAuditPayload",
    "ChiralBBSelectorSignAuditPayload",
    "ChiralBoundaryNormalizationAuditPayload",
    "ConservedLabelPartitionAuditPayload",
    "ExplicitHqAuditPayload",
    "FilledBandSelectorPotentialAuditPayload",
    "FreeBBRadialNoGoAuditPayload",
    "FramedSterileAuditPayload",
    "HiggsBackreactionRadialStabilizerAuditPayload",
    "HiggsRadialLandauOriginAuditPayload",
    "ImpedanceAuditPayload",
    "JaynesPrimitiveErgodicityAuditPayload",
    "LabelConservingDynamicsAuditPayload",
    "LeptonicBoundaryHolonomyAuditPayload",
    "LeptonicPhaseWordAuditPayload",
    "LocalBoundaryFiberAuditPayload",
    "MicrocanonicalReductionAuditPayload",
    "MicroscopicFilledBandPotentialAuditPayload",
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
    "SchurShellLandauAuditPayload",
    "ResidualGraphTransferAuditPayload",
    "TetrahedralLandauAuditPayload",
    "TransferProbeTheoremAuditPayload",
    "UnitContinuationAuditPayload",
    "VacuumFramingAuditPayload",
    "VacuumSelectorAuditPayload",
    "VacuumSelectorClosureAuditPayload",
    "VacuumSelectorCondensationAuditPayload",
    "VacuumSelectorPotentialAuditPayload",
    "WeylSterileAuditPayload",
    "algebraic_intertwiner_audit_payload",
    "analytic_radial_breaking_theorem_audit_payload",
    "analytic_radial_energy",
    "arbitrary_label_preserving_degeneracies",
    "boundary_core_audit_payload",
    "bb_induced_radial_breaking_audit_payload",
    "bb_induced_radial_energy_samples",
    "bb_induced_radial_minimum_candidates",
    "bb_scalar_h2_quadratic_coefficients",
    "bb_scalar_h2_xyz_coefficient",
    "bb_trace_b3_xyz_coefficient",
    "bcc_unoriented_exit_representatives",
    "charged_lepton_leakage_audit_payload",
    "chiral_bb_selector_sign_audit_payload",
    "chiral_boundary_normalization_audit_payload",
    "ckm_conditional_audit_payload",
    "ckm_jarlskog",
    "ckm_magnitude_matrix",
    "ckm_sines",
    "completed_square_vev_squared",
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
    "filled_band_parity_odd_energy",
    "filled_band_selector_branch_energies",
    "filled_band_selector_branch_gap",
    "filled_band_selector_candidate_energy",
    "filled_band_effective_energy",
    "filled_band_even_radial_energy",
    "filled_band_odd_selector_energy",
    "filled_band_selector_potential_audit_payload",
    "filled_band_selector_ratio",
    "filled_band_selector_sign_passes",
    "framed_sterile_audit_payload",
    "framed_residual_exits",
    "free_bb_radial_stabilization_audit_payload",
    "ground_exit_indices",
    "higgs_backreaction_radial_stabilizer_audit_payload",
    "higgs_radial_landau_origin_audit_payload",
    "impedance_audit_payload",
    "invariant_space_dimension",
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
    "local_boundary_fiber_audit_payload",
    "local_fiber_degeneracies",
    "local_fiber_reduced_density",
    "lowest_selector_anisotropy_degree",
    "microcanonical_label_weights",
    "microcanonical_reduced_density",
    "microcanonical_reduction_audit_payload",
    "microscopic_selector_potential_audit_payload",
    "massless_quartic_stationary_radius",
    "mexican_hat_radial_potential",
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
    "quartic_backreaction_potential",
    "quark_boundary_shell_audit_payload",
    "quark_clebsch_audit_payload",
    "quark_coin_rigidity_audit_payload",
    "quark_transfer_hierarchy_audit_payload",
    "radial_energy_finite_differences",
    "radial_selector_energy_samples",
    "regular_boundary_fiber_audit_payload",
    "regular_fiber_degeneracies",
    "regular_fiber_reduced_density",
    "residual_complete_graph_adjacency",
    "residual_basis_continuation_couplings",
    "residual_graph_decaying_factor",
    "residual_graph_transfer_audit_payload",
    "residual_projectors",
    "schur_shell_cubic_coefficient",
    "schur_shell_landau_audit_payload",
    "schur_shell_landau_series",
    "selector_energies",
    "selector_gap",
    "selector_locking_potential",
    "selector_order_parameter",
    "selector_potential_energy",
    "spacetime_higgs_potential_pullback",
    "spectral_lift_operator",
    "stabilized_radial_energy_samples",
    "stabilized_radial_minimum_candidates",
    "tetrahedral_cubic_polynomial",
    "tetrahedral_landau_audit_payload",
    "tetrahedral_landau_locking_potential",
    "tetrahedral_rotation_group",
    "tetrahedral_selector_candidates",
    "tetrahedral_shell_power_sum",
    "transfer_probe_theorem_audit_payload",
    "unit_continuation_audit_payload",
    "unit_outward_matching",
    "vacuum_framing_audit_payload",
    "vacuum_selector_audit_payload",
    "vacuum_selector_closure_audit_payload",
    "vacuum_selector_condensation_audit_payload",
    "vacuum_selector_potential_audit_payload",
    "weyl_sterile_audit_payload",
]
