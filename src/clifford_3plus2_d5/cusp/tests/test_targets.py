"""Tests for the CUSP-PLAN target certificates."""

import sympy as sp

from clifford_3plus2_d5.cusp.targets import (
    BCCEdgeQClock,
    CenterChargeRecirculationAutomaton,
    CenterClosureSource,
    CenterHolonomyPath,
    CoefficientPathMeasure,
    CuspBoundaryDilationAudit,
    CuspBoundaryMaterialSource,
    CuspBoundaryMaterialOriginAudit,
    CuspCenterTopologyAudit,
    CuspCoefficientMeasureAudit,
    CuspRightChargeOriginAudit,
    CuspShearMatchingPrincipleAudit,
    MicroscopicColorZ3RecirculationAudit,
    MicroscopicQStiffnessAudit,
    MicroscopicQStiffnessControlAudit,
    MicroscopicQStiffnessMaterial,
    MicroscopicRetardedBoundaryAudit,
    MicroscopicRetardedClosureControls,
    MicroscopicTargetCModuleAudit,
    MicroscopicTargetDTopologyAudit,
    MicroscopicWeakZ2RecirculationAudit,
    NumericalSemigroup,
    QLocalLockingOriginAudit,
    RecirculationGraph,
    additive_center_control_powers,
    bilinear_center_linking_powers,
    complete_graph_distance_center_powers,
    cusp_boundary_dilation_audit,
    cusp_boundary_material_origin_audit,
    cusp_boundary_material_source,
    cusp_center_topology_audit,
    cusp_coefficient_measure_audit,
    cusp_module_path_amplitudes,
    cusp_module_path_count,
    cusp_right_charge_origin_audit,
    cusp_shear_matching_principle_audit,
    cusp_targets_payload,
    cyclic_difference_center_powers,
    exact_omega,
    finite_recirculation_graph_from_automata,
    finite_recirculation_graph_from_sources,
    flag_distance_center_powers_from_edges,
    graph_low_valuation_summary,
    microscopic_lambda_rec_audit,
    microscopic_color_z3_recirculation_audit,
    microscopic_q_stiffness_audit,
    microscopic_retarded_boundary_audit,
    microscopic_schur_semigroup_audit,
    microscopic_target_c_module_audit,
    microscopic_target_d_topology_audit,
    microscopic_weak_z2_recirculation_audit,
    minimal_center_recirculation_walk_graph,
    oriented_double_up_right_charges,
    path_flag_distance_center_powers,
    path_measure_from_center_powers,
    shortest_path_distance_on_flag,
    sm_global_quotient_cusp_audit,
    subtract_conductor_floor,
    standard_model_center_automata,
    target_a_neighbor_walk_graphs,
    target_a_payload,
    target_b_payload,
    target_c_payload,
    target_d_payload,
)


def test_bcc_edge_q_clock_splits_same_normal_and_leakage_branches() -> None:
    q_clock = BCCEdgeQClock()

    assert q_clock.same_normal_count == 2
    assert q_clock.mixed_normal_count == 2
    assert q_clock.same_normal_delta_q_values == (0, 0)
    assert sorted(q_clock.mixed_normal_delta_q_values) == [-2, 2]
    assert q_clock.q_clock_pass


def test_cusp_boundary_material_source_derives_weak_source_from_bcc_parity() -> None:
    material = cusp_boundary_material_source()

    assert isinstance(material, CuspBoundaryMaterialSource)
    assert material.q_clock.q_clock_pass
    assert material.same_normal_norm == sp.eye(2) / 2
    assert material.mixed_normal_norm == sp.eye(2) / 2
    assert material.total_norm_is_identity
    assert material.norm_split_pass
    assert material.weak_source.name == "weak_SU2_doublet"
    assert material.weak_source.center_order == material.q_clock.same_normal_count == 2
    assert material.weak_source.channel_count == material.q_clock.same_normal_count
    assert material.weak_source.primitive_closure_length == 2
    assert material.weak_source_from_bcc_pass
    assert material.color_source.name == "color_SU3_triplet"
    assert material.color_source.center_order == 3
    assert material.color_source.channel_count == 3
    assert material.color_source.primitive_closure_length == 3
    assert material.color_source_pass
    assert material.material_source_pass


def test_cusp_boundary_dilation_audit_completes_local_bb_isometry() -> None:
    audit = cusp_boundary_dilation_audit()

    assert isinstance(audit, CuspBoundaryDilationAudit)
    assert audit.local_isometry.shape == (8, 2)
    assert audit.local_isometry_norm == sp.eye(2)
    assert audit.local_isometry_pass
    assert audit.local_input_dimension == 2
    assert audit.local_output_dimension == 8
    assert audit.unitary_completion_nullity == 6
    assert audit.unitary_dilation_exists_pass
    assert audit.finite_gap_feedback_scalar
    assert audit.hard_gap_feedback_zero
    assert audit.retarded_feedback_limit_zero
    assert audit.retarded_visible_powers_match_survival
    assert audit.recurrent_wedge_return == sp.Matrix(
        [[-(1 + sp.I) / 4, 0], [0, -(1 - sp.I) / 4]]
    )
    assert audit.recurrent_wedge_return_nonzero
    assert audit.recurrent_visible_powers_differ
    assert audit.outgoing_leakage_pass
    assert audit.boundary_dilation_pass
    assert "single_clock_locking_field" in audit.remaining_declared_inputs[0]


def test_q_local_locking_origin_selects_unique_q_squared_penalty() -> None:
    audit = cusp_boundary_material_origin_audit()
    locking = audit.q_locking
    q = sp.symbols("q")
    g = locking.stiffness

    assert isinstance(audit, CuspBoundaryMaterialOriginAudit)
    assert isinstance(locking, QLocalLockingOriginAudit)
    assert locking.selected_polynomial == g * q**2
    assert locking.selected_penalty_values == (0, 4 * g, 4 * g)
    assert locking.selected_gap == 4 * g
    assert locking.visible_penalty_zero
    assert locking.mixed_penalties_degenerate
    assert locking.mixed_penalty_positive
    assert locking.uniqueness_solution[sp.symbols("a")] == g
    assert locking.uniqueness_solution[sp.symbols("b")] == 0
    assert locking.uniqueness_solution[sp.symbols("c")] == 0
    assert locking.linear_control_values == (0, -2, 2)
    assert locking.constant_control_values == (1, 1, 1)
    assert locking.linear_control_rejected
    assert locking.constant_control_rejected
    assert locking.q_locking_origin_pass


def test_microscopic_q_stiffness_derives_session_a_material_action() -> None:
    audit = microscopic_q_stiffness_audit()
    material = audit.material
    controls = audit.controls
    q = sp.symbols("q")
    g = sp.symbols("g", positive=True)
    r1, r2 = sp.symbols("r1 r2")

    assert isinstance(audit, MicroscopicQStiffnessAudit)
    assert isinstance(material, MicroscopicQStiffnessMaterial)
    assert isinstance(controls, MicroscopicQStiffnessControlAudit)
    assert material.stiffness_matrix == 2 * g * sp.Matrix([[1, -1], [-1, 1]])
    assert material.microscopic_action == g * (r1 - r2) ** 2
    assert material.action_in_q == g * q**2
    assert material.q_reflection_even
    assert material.no_linear_term
    assert material.curvature == 2 * g
    assert material.curvature_positive
    assert material.trace_mode_ungapped
    assert material.adjacent_leakage_gap == 4 * g
    assert material.stiffness_matrix_positive_semidefinite
    assert material.microscopic_q_stiffness_pass
    assert audit.effective_action_matches_finite_locking
    assert audit.adjacent_gap_matches_finite_locking
    assert audit.hard_gap_limit_suppresses_mixed_feedback
    assert audit.session_a_pass


def test_microscopic_q_stiffness_controls_reject_wrong_material_laws() -> None:
    controls = microscopic_q_stiffness_audit().controls
    q = sp.symbols("q")
    g = sp.symbols("g", positive=True)

    assert controls.zero_control_action == 0
    assert controls.zero_control_gap == 0
    assert controls.zero_control_rejected
    assert controls.linear_control_action == g * q
    assert controls.linear_control_mirror_defect == 2 * g * q
    assert controls.linear_control_rejected
    assert controls.absolute_control_action == g * sp.Abs(q)
    assert not controls.absolute_control_analytic_at_zero
    assert controls.absolute_control_rejected
    assert controls.controls_rejected


def test_microscopic_retarded_boundary_selects_no_incoming_closure() -> None:
    audit = microscopic_retarded_boundary_audit()
    controls = audit.controls

    assert isinstance(audit, MicroscopicRetardedBoundaryAudit)
    assert isinstance(controls, MicroscopicRetardedClosureControls)
    assert controls.no_incoming_reflection == sp.zeros(4)
    assert controls.recurrent_reflection == sp.eye(4)
    assert controls.hard_wall_reflection == -sp.eye(4)
    assert controls.symmetric_reflection == sp.eye(4) / 2
    assert audit.no_incoming_feedback == sp.zeros(2)
    assert audit.no_incoming_feedback_zero
    assert audit.local_unitary_dilation_exists
    assert audit.no_incoming_visible_powers_match
    assert audit.visible_schur_kernel_has_no_recurrent_mixed_return
    assert audit.finite_cusp_graph_preserved
    assert audit.session_b_pass


def test_microscopic_retarded_boundary_rejects_returning_leakage_controls() -> None:
    audit = microscopic_retarded_boundary_audit()

    assert audit.recurrent_feedback == audit.dilation.recurrent_wedge_return
    assert audit.recurrent_feedback_nonzero
    assert audit.hard_wall_feedback == -audit.recurrent_feedback
    assert audit.hard_wall_feedback_nonzero
    assert audit.symmetric_feedback == audit.recurrent_feedback / 2
    assert audit.symmetric_feedback_nonzero
    assert audit.controls_rejected
    assert "Z2_Z3_center_recirculation" in audit.microscopic_gate_remaining[0]


def test_microscopic_weak_z2_recirculation_comes_from_same_normal_branch_parity() -> None:
    audit = microscopic_weak_z2_recirculation_audit()

    assert isinstance(audit, MicroscopicWeakZ2RecirculationAudit)
    assert audit.branch_common_signs == (1, -1)
    assert audit.branch_parity_group == (1, -1)
    assert audit.branch_parity_group_closed
    assert audit.charge_table_through_six == (0, 1, 0, 1, 0, 1, 0)
    assert audit.sign_table_through_six == (1, -1, 1, -1, 1, -1, 1)
    assert audit.common_sign_after_ticks(1) == -1
    assert audit.common_sign_after_ticks(2) == 1
    assert not audit.visible_readout_allowed(1)
    assert audit.visible_readout_allowed(2)
    assert audit.primitive_return_length == 2
    assert audit.channel_count == 2
    assert audit.one_step_forbidden
    assert audit.automaton_matches_branch_parity
    assert audit.session_c_pass


def test_microscopic_weak_z2_recirculation_rejects_trivial_and_weak_only_controls() -> None:
    audit = microscopic_weak_z2_recirculation_audit()

    assert audit.trivial_charge_control_first_values == (0, 1, 2)
    assert audit.order_one_control_first_values == (0, 1, 2)
    assert audit.weak_only_control_first_values == (0, 2, 4)
    assert audit.controls_rejected
    assert "Z3_color_center_recirculation" in audit.microscopic_gate_remaining[0]


def test_microscopic_color_z3_recirculation_comes_from_closed_center_holonomy() -> None:
    audit = microscopic_color_z3_recirculation_audit()

    assert isinstance(audit, MicroscopicColorZ3RecirculationAudit)
    assert audit.center_phase == exact_omega()
    assert audit.charge_table_through_six == (0, 1, 2, 0, 1, 2, 0)
    assert audit.holonomy_table_through_three == (
        1,
        exact_omega(),
        sp.simplify(exact_omega() ** 2),
        1,
    )
    assert audit.open_holonomies_nontrivial
    assert audit.closed_holonomy_neutral
    assert not audit.visible_readout_allowed(1)
    assert not audit.visible_readout_allowed(2)
    assert audit.visible_readout_allowed(3)
    assert audit.primitive_return_length == 3
    assert audit.channel_count == 3
    assert audit.one_step_forbidden
    assert audit.automaton_matches_center_holonomy
    assert audit.combined_cusp_graph_preserved
    assert audit.session_d_pass


def test_microscopic_color_z3_recirculation_rejects_wrong_and_spectator_controls() -> None:
    audit = microscopic_color_z3_recirculation_audit()

    assert audit.wrong_color_length_two_first_values == (0, 2, 4)
    assert audit.wrong_color_length_four_first_values == (0, 2, 4)
    assert audit.spectator_color_control_first_values == (0, 2, 4)
    assert audit.gauged_away_open_phase_control_first_values == (0, 2, 4)
    assert audit.controls_rejected
    assert "Z6_quotient" in audit.microscopic_gate_remaining[0]


def test_sm_global_quotient_gate_selects_independent_center_axes() -> None:
    audit = sm_global_quotient_cusp_audit()

    assert audit.independent_tick_vectors == ((1, 0), (0, 1))
    assert audit.quotient_diagonal_tick_vector == (1, 1)
    assert audit.neutral_length_for_tick((1, 0)) == 2
    assert audit.neutral_length_for_tick((0, 1)) == 3
    assert audit.independent_primitive_lengths == (2, 3)
    assert audit.independent_low_valuations == (0, 2, 3)
    assert audit.independent_family_basis == ("1", "t^2", "t^3")
    assert audit.independent_axes_pass
    assert audit.session_global_quotient_pass


def test_sm_global_quotient_gate_rejects_correlated_z6_and_collapsed_controls() -> None:
    audit = sm_global_quotient_cusp_audit()

    assert audit.quotient_diagonal_primitive_length == 6
    assert audit.quotient_diagonal_low_valuations == (0, 6, 12)
    assert audit.quotient_diagonal_family_basis == ("1", "t^6", "t^12")
    assert audit.u1_collapsed_low_valuations == (0, 1, 2)
    assert audit.quotient_controls_rejected
    assert "Target_C_D_micro_topology" in audit.microscopic_gate_remaining[0]


def test_microscopic_schur_semigroup_recovers_two_three_from_return_moments() -> None:
    audit = microscopic_schur_semigroup_audit()

    assert audit.return_moments_through_six == (1, 0, 2, 3, 4, 12, 17)
    assert audit.m1_zero
    assert audit.m2_nonzero
    assert audit.m3_nonzero
    assert audit.primitive_return_semigroup == (2, 3)
    assert audit.recirculation_algebra == "C[t^2,t^3]"
    assert audit.maximal_ideal_generators == (2, 3)
    assert audit.family_module_basis == ("1", "t^2", "t^3")
    assert audit.low_valuations == (0, 2, 3)
    assert audit.session_e_pass


def test_microscopic_schur_semigroup_rejects_neighboring_controls() -> None:
    audit = microscopic_schur_semigroup_audit()

    assert audit.controls == {
        "S=<1>": (0, 1, 2),
        "S=<2>": (0, 2, 4),
        "S=<3>": (0, 3, 6),
        "S=<2,4>": (0, 2, 4),
        "S=<3,4>": (0, 3, 4),
    }
    assert audit.controls_rejected
    assert "lambda_rec" in audit.microscopic_gate_remaining[0]


def test_microscopic_lambda_rec_comes_from_one_sided_return_moments() -> None:
    audit = microscopic_lambda_rec_audit()

    assert audit.weak_return_moment == 2
    assert audit.color_return_moment == 3
    assert sp.simplify(audit.lambda_rec - (sp.sqrt(sp.Rational(3, 2)) - 1)) == 0
    assert audit.one_sided_residual == 0
    assert audit.one_sided_stable_minimum
    assert audit.session_f_pass


def test_microscopic_lambda_rec_rejects_two_sided_and_wrong_shear_controls() -> None:
    audit = microscopic_lambda_rec_audit()

    assert set(audit.control_residuals) == {
        "ordinary_reflection",
        "count_ratio_minus_one",
        "inverse_amplitude_shear",
    }
    assert all(sp.simplify(residual) != 0 for residual in audit.control_residuals.values())
    assert audit.controls_rejected
    assert "Target_C_D_micro_topology" in audit.microscopic_gate_remaining[0]


def test_microscopic_target_c_module_recovers_right_charges_without_targets() -> None:
    audit = microscopic_target_c_module_audit()

    assert isinstance(audit, MicroscopicTargetCModuleAudit)
    assert audit.primitive_return_semigroup == (2, 3)
    assert audit.family_module_basis == ("1", "t^2", "t^3")
    assert audit.module_valuations_heavy_to_light == (0, 2, 3)
    assert audit.left_charges_light_to_heavy == (3, 2, 0)
    assert audit.conductor == 2
    assert audit.frobenius_gap == 1
    assert audit.down_right_charges == (1, 0, 0)
    assert audit.up_right_charges == (5, 2, 0)
    assert audit.down_diagonal_exponents == (4, 2, 0)
    assert audit.up_diagonal_exponents == (8, 4, 0)
    assert audit.up_exponent_matrix == ((8, 5, 3), (7, 4, 2), (5, 2, 0))
    assert audit.down_exponent_matrix == ((4, 3, 3), (3, 2, 2), (1, 0, 0))
    assert audit.weak_double_cover_factor == 2
    assert audit.color_order_control_factor == 3
    assert audit.session_g_pass


def test_microscopic_target_c_module_rejects_controls_and_fits() -> None:
    audit = microscopic_target_c_module_audit()

    assert audit.wrong_conductor_controls == {
        1: (5, 3, 0),
        3: (3, 2, 0),
    }
    assert audit.lift_controls == {
        "trivial_lift": (4, 2, 0),
        "color_order_lift": (12, 6, 0),
    }
    assert audit.controls_rejected
    assert not audit.uses_diagonal_targets
    assert not audit.uses_mass_fits
    assert "Target_D" in audit.microscopic_gate_remaining[0]


def test_microscopic_target_d_topology_recovers_center_rules_and_cp() -> None:
    audit = microscopic_target_d_topology_audit()

    assert isinstance(audit, MicroscopicTargetDTopologyAudit)
    assert audit.family_flag_edges == ((0, 1), (1, 2))
    assert audit.center_labels == (0, 1, 2)
    assert shortest_path_distance_on_flag(0, 2, audit.family_flag_edges) == 2
    assert flag_distance_center_powers_from_edges(audit.family_flag_edges) == (
        (0, 1, 2),
        (1, 0, 1),
        (2, 1, 0),
    )
    assert audit.up_center_powers == path_flag_distance_center_powers()
    assert audit.down_center_powers == bilinear_center_linking_powers(unit=1)
    assert audit.up_amplitudes == (
        (sp.Integer(2), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert audit.down_amplitudes == (
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert sp.simplify(audit.cp_invariant) != 0
    assert sp.simplify(audit.real_control_invariant) == 0
    assert sp.simplify(audit.cp_invariant - audit.rephased_cp_invariant) == 0
    assert audit.session_h_pass
    assert audit.microscopic_gate_remaining == ()


def test_microscopic_target_d_topology_rejects_controls() -> None:
    audit = microscopic_target_d_topology_audit()

    assert audit.up_controls_rejected
    assert audit.down_controls_rejected_except_conjugate
    assert set(audit.one_sector_control_invariants) == {
        "up_only",
        "down_only",
        "separable_row_col",
    }
    assert all(
        sp.simplify(invariant) == 0
        for invariant in audit.one_sector_control_invariants.values()
    )
    assert audit.topology_controls_rejected


def test_causal_outgoing_origin_selects_retarded_closure_over_recurrence() -> None:
    audit = cusp_boundary_material_origin_audit()

    assert audit.outgoing.causal_rule == "no incoming mixed-normal clock-error data"
    assert audit.outgoing.retarded_visible_powers_match_survival
    assert audit.outgoing.recurrent_wedge_return == sp.Matrix(
        [[-(1 + sp.I) / 4, 0], [0, -(1 - sp.I) / 4]]
    )
    assert audit.outgoing.recurrent_wedge_return_nonzero
    assert audit.outgoing.recurrent_visible_powers_differ
    assert audit.outgoing.outgoing_origin_pass
    assert audit.material_origin_pass
    assert "q_reflection_positive_stiffness" in audit.remaining_physical_axioms[0]


def test_center_closure_source_forbids_one_step_and_closes_at_center_order() -> None:
    weak = CenterClosureSource("weak", center_order=2, channel_count=2)
    color = CenterClosureSource("color", center_order=3, channel_count=3)

    assert weak.center_charge(1) == 1
    assert color.center_charge(1) == 1
    assert weak.one_step_forbidden
    assert color.one_step_forbidden
    assert weak.primitive_closure_length == 2
    assert color.primitive_closure_length == 3


def test_center_charge_automata_force_visible_return_orders() -> None:
    weak, color = standard_model_center_automata()

    assert isinstance(weak, CenterChargeRecirculationAutomaton)
    assert weak.source.name == "weak_SU2_doublet"
    assert color.source.name == "color_SU3_triplet"
    assert weak.charge_table(6) == (0, 1, 0, 1, 0, 1, 0)
    assert color.charge_table(6) == (0, 1, 2, 0, 1, 2, 0)
    assert weak.return_ticks_through(6) == (2, 4, 6)
    assert color.return_ticks_through(6) == (3, 6)
    assert weak.one_step_forbidden
    assert color.one_step_forbidden
    assert weak.primitive_return_length == 2
    assert color.primitive_return_length == 3


def test_finite_recirculation_walk_graph_recovers_primitive_two_three_closures() -> None:
    graph = minimal_center_recirculation_walk_graph()
    automaton_graph = finite_recirculation_graph_from_automata(
        standard_model_center_automata()
    )
    source_graph = finite_recirculation_graph_from_sources(
        (
            CenterClosureSource("weak_SU2_doublet", center_order=2, channel_count=2),
            CenterClosureSource("color_SU3_triplet", center_order=3, channel_count=3),
        )
    )

    assert graph.edges == automaton_graph.edges
    assert graph.one_step_forbidden
    assert graph.closed_walk_counts_through(6) == (1, 0, 2, 3, 4, 12, 17)
    assert graph.closed_walk_lengths_through(6) == (0, 2, 3, 4, 5, 6)
    assert graph.primitive_closed_walk_lengths_through(6) == (2, 3)
    assert graph.closed_walk_count(2) == 2
    assert graph.closed_walk_count(3) == 3
    assert source_graph.closed_walk_counts_through(6) == graph.closed_walk_counts_through(6)
    assert source_graph.primitive_closed_walk_lengths_through(6) == (2, 3)


def test_target_a_neighbor_graph_controls_miss_the_cusp_valuations() -> None:
    controls = target_a_neighbor_walk_graphs()

    assert graph_low_valuation_summary(controls["one_step_visible_loop"]) == (0, 1, 2)
    assert graph_low_valuation_summary(controls["weak_only_two_cycle"]) == (0, 2, 4)
    assert graph_low_valuation_summary(controls["color_only_three_cycle"]) == (0, 3, 6)
    assert graph_low_valuation_summary(controls["wrong_color_length_four"]) == (0, 2, 4)
    assert all(
        graph_low_valuation_summary(control) != (0, 2, 3)
        for control in controls.values()
    )


def test_target_a_minimal_graph_has_cusp_valuations() -> None:
    payload = target_a_payload()

    assert payload.graph_closures == (2, 3)
    assert payload.bcc_same_normal_count == 2
    assert payload.bcc_mixed_normal_count == 2
    assert payload.bcc_same_normal_delta_q_values == (0, 0)
    assert sorted(payload.bcc_mixed_normal_delta_q_values) == [-2, 2]
    assert payload.bcc_q_clock_pass
    assert payload.closed_walk_counts_through_six == (1, 0, 2, 3, 4, 12, 17)
    assert payload.closed_walk_lengths_through_six == (0, 2, 3, 4, 5, 6)
    assert payload.graph_primitive_closures == (2, 3)
    assert payload.graph_one_step_forbidden
    assert payload.graph_model_pass
    assert payload.source_derived_graph_pass
    assert tuple(source.name for source in payload.closure_sources) == (
        "weak_SU2_doublet",
        "color_SU3_triplet",
    )
    assert payload.closure_source_orders == (2, 3)
    assert payload.closure_source_one_step_forbidden == (True, True)
    assert payload.sm_center_source_pass
    assert tuple(source.name for source in payload.material_closure_sources) == (
        "weak_SU2_doublet",
        "color_SU3_triplet",
    )
    assert payload.material_same_normal_norm == sp.eye(2) / 2
    assert payload.material_mixed_normal_norm == sp.eye(2) / 2
    assert payload.material_total_norm_is_identity
    assert payload.material_norm_split_pass
    assert payload.material_weak_source_from_bcc_pass
    assert payload.material_color_source_pass
    assert payload.material_source_pass
    assert payload.boundary_local_isometry_norm == sp.eye(2)
    assert payload.boundary_local_isometry_pass
    assert payload.boundary_local_input_dimension == 2
    assert payload.boundary_local_output_dimension == 8
    assert payload.boundary_unitary_completion_nullity == 6
    assert payload.boundary_unitary_dilation_exists_pass
    assert payload.boundary_finite_gap_feedback_scalar
    assert payload.boundary_hard_gap_feedback_zero
    assert payload.boundary_retarded_feedback_limit_zero
    assert payload.boundary_retarded_visible_powers_match_survival
    assert payload.boundary_recurrent_wedge_return_nonzero
    assert payload.boundary_recurrent_visible_powers_differ
    assert payload.boundary_outgoing_leakage_pass
    assert payload.boundary_dilation_pass
    assert "mixed_normal_clock_error_ports" in payload.boundary_remaining_declared_inputs[1]
    q = sp.symbols("q")
    g = sp.symbols("g", positive=True)
    assert payload.origin_q_locking_polynomial == g * q**2
    assert payload.origin_q_locking_values == (0, 4 * g, 4 * g)
    assert payload.origin_q_locking_gap == 4 * g
    assert payload.origin_q_locking_unique_solution[sp.symbols("a")] == g
    assert payload.origin_q_locking_unique_solution[sp.symbols("b")] == 0
    assert payload.origin_q_locking_unique_solution[sp.symbols("c")] == 0
    assert payload.origin_q_locking_pass
    assert payload.origin_linear_control_rejected
    assert payload.origin_constant_control_rejected
    assert payload.origin_causal_rule == "no incoming mixed-normal clock-error data"
    assert payload.origin_outgoing_pass
    assert payload.origin_material_pass
    assert "retarded_asymptotics" in payload.origin_remaining_physical_axioms[1]
    assert payload.automaton_return_orders == (2, 3)
    assert payload.automaton_one_step_forbidden == (True, True)
    assert payload.automaton_charge_tables_through_six == (
        (0, 1, 0, 1, 0, 1, 0),
        (0, 1, 2, 0, 1, 2, 0),
    )
    assert payload.automaton_return_ticks_through_six == ((2, 4, 6), (3, 6))
    assert len(payload.automaton_graph_edges) == 13
    assert payload.automaton_graph_pass
    assert payload.one_step_forbidden
    assert payload.first_three_valuations == (0, 2, 3)
    assert payload.gaps_through_six == (1,)
    assert payload.target_algebra == "C[t^2,t^3]"
    assert payload.graph_controls == {
        "one_step_visible_loop": (0, 1, 2),
        "weak_only_two_cycle": (0, 2, 4),
        "color_only_three_cycle": (0, 3, 6),
        "wrong_color_length_four": (0, 2, 4),
    }
    assert payload.graph_controls_rejected
    assert payload.target_a_pass


def test_target_a_controls_are_not_the_cusp_low_valuations() -> None:
    payload = target_a_payload()

    assert payload.controls["C[t]"] == (0, 1, 2)
    assert payload.controls["C[t^2]"] == (0, 2, 4)
    assert payload.controls["C[t^3]"] == (0, 3, 6)
    assert all(control != payload.first_three_valuations for control in payload.controls.values())


def test_one_step_return_collapses_to_vanilla_t_algebra() -> None:
    graph = RecirculationGraph((2, 3), one_step_return_allowed=True)

    assert graph.semigroup.first_values(3) == (0, 1, 2)
    assert not graph.one_step_forbidden


def test_numerical_semigroup_decompositions_are_exact() -> None:
    semigroup = NumericalSemigroup((2, 3))

    assert semigroup.values_through(6) == (0, 2, 3, 4, 5, 6)
    assert semigroup.decompositions(1) == ()
    assert semigroup.decompositions(6) == ((0, 2), (3, 0))


def test_target_b_cabibbo_shear_rejects_reflection_coefficient() -> None:
    payload = target_b_payload()

    assert payload.matching_principle == (
        "normalized amplitude matching: (1+t)sqrt(N_w)=sqrt(N_c)"
    )
    assert payload.weak_channels == 2
    assert payload.color_channels == 3
    assert payload.weak_amplitude == sp.sqrt(2)
    assert payload.color_amplitude == sp.sqrt(3)
    assert sp.simplify(payload.lambda_rec - (sp.sqrt(sp.Rational(3, 2)) - 1)) == 0
    assert payload.matching_residual == 0
    assert payload.matching_function_value == 0
    assert payload.matching_function_derivative_at_lambda == 0
    assert payload.matching_function_second_derivative == 4
    assert payload.matching_function_stable_minimum_pass
    assert payload.boundary_matching_condition.startswith(
        "one-sided retarded recirculation"
    )
    assert payload.one_sided_matching_residual_at_lambda == 0
    assert payload.ordinary_reflection_two_sided_residual == 0
    assert payload.ordinary_reflection_solves_two_sided_control
    assert payload.ordinary_reflection_excluded_by_one_sided_boundary
    assert payload.positive_oriented_uniformizer_pass
    assert not payload.uses_ckm_data
    assert payload.shear_matching_theorem_pass
    assert payload.matching_solution_unique
    assert payload.shear_formula_pass
    assert payload.control_shears == {
        "count_ratio_minus_one": sp.Rational(1, 2),
        "inverse_amplitude_shear": -1 + sp.sqrt(sp.Rational(2, 3)),
        "ordinary_reflection": (sp.sqrt(3) - sp.sqrt(2)) / (sp.sqrt(2) + sp.sqrt(3)),
    }
    assert set(payload.control_matching_residuals) == set(payload.control_shears)
    assert all(
        sp.simplify(residual) != 0
        for residual in payload.control_matching_residuals.values()
    )
    assert payload.reflection_control_rejected
    assert payload.wrong_shear_controls_rejected
    assert sp.simplify(payload.reflection_matching_residual) != 0
    assert sp.N(payload.lambda_rec, 10) == sp.N(sp.sqrt(sp.Rational(3, 2)) - 1, 10)


def test_cusp_shear_matching_principle_is_one_sided_not_reflection() -> None:
    audit = cusp_shear_matching_principle_audit()

    assert isinstance(audit, CuspShearMatchingPrincipleAudit)
    assert audit.matcher.weak_channels == 2
    assert audit.matcher.color_channels == 3
    assert sp.simplify(audit.lambda_rec - (sp.sqrt(sp.Rational(3, 2)) - 1)) == 0
    assert audit.one_sided_matching_residual_at_lambda == 0
    assert audit.ordinary_reflection_two_sided_residual == 0
    assert sp.simplify(audit.ordinary_reflection_one_sided_residual) != 0
    assert audit.ordinary_reflection_solves_two_sided_control
    assert audit.ordinary_reflection_excluded_by_one_sided_boundary
    assert audit.positive_oriented_uniformizer_pass
    assert not audit.uses_ckm_data
    assert audit.shear_matching_theorem_pass


def test_target_c_candidate_reproduces_fn_power_skeleton() -> None:
    payload = target_c_payload()

    assert payload.left_charges_light_to_heavy == (3, 2, 0)
    assert payload.cusp_semigroup_values_through_six == (0, 2, 3, 4, 5, 6)
    assert payload.cusp_conductor == 2
    assert payload.cusp_frobenius_number == 1
    assert payload.conductor_from_semigroup_pass
    assert payload.up_yukawa_hypercharge_sum == 0
    assert payload.down_yukawa_hypercharge_sum == 0
    assert payload.swapped_up_hypercharge_sum != 0
    assert payload.swapped_down_hypercharge_sum != 0
    assert payload.hypercharge_doors_forced
    assert payload.boundary_module_rule.name == "conductor_floor_plus_oriented_double_lift"
    assert payload.boundary_module_rule.conductor == 2
    assert payload.boundary_module_rule.frobenius_number == 1
    assert payload.boundary_module_rule.down_right_charges == (1, 0, 0)
    assert payload.boundary_module_rule.up_right_charges == (5, 2, 0)
    assert payload.boundary_module_rule.down_diagonal_exponents == (4, 2, 0)
    assert payload.boundary_module_rule.up_diagonal_exponents == (8, 4, 0)
    assert payload.boundary_module_rule_pass
    assert payload.wrong_conductor_down_diagonal_controls == {
        1: (5, 3, 0),
        2: (4, 2, 0),
        3: (3, 2, 0),
    }
    assert payload.orientation_lift_factor == 2
    assert payload.orientation_lift_up_diagonal_controls == {
        1: (4, 2, 0),
        2: (8, 4, 0),
        3: (12, 6, 0),
    }
    assert subtract_conductor_floor((3, 2, 0), 2) == (1, 0, 0)
    assert oriented_double_up_right_charges((3, 2, 0), (1, 0, 0)) == (5, 2, 0)
    assert payload.up_right_candidate == (5, 2, 0)
    assert payload.down_right_candidate == (1, 0, 0)
    assert payload.down_right_from_conductor_rule == (1, 0, 0)
    assert payload.up_right_from_oriented_double_rule == (5, 2, 0)
    assert isinstance(payload.right_charge_origin_audit, CuspRightChargeOriginAudit)
    assert payload.right_charge_origin_audit.down_right_charges == (1, 0, 0)
    assert payload.right_charge_origin_audit.up_right_charges == (5, 2, 0)
    assert payload.right_charge_origin_pass
    assert not payload.right_charge_origin_uses_mass_fits
    assert payload.right_charge_origin_down_rule == (
        "right residue of cusp conductor ideal: r_d=max(q-c,0)"
    )
    assert payload.right_charge_origin_up_rule == (
        "H_tilde oriented weak-double cover: diag_u=N_w diag_d"
    )
    assert payload.right_charge_origin_lift_factor_from_weak_order == 2
    assert payload.right_charge_origin_trivial_lift_control == (4, 2, 0)
    assert payload.right_charge_origin_color_lift_control == (12, 6, 0)
    assert isinstance(payload.microscopic_module_audit, MicroscopicTargetCModuleAudit)
    assert payload.microscopic_module_audit.left_charges_light_to_heavy == (3, 2, 0)
    assert payload.microscopic_module_audit.down_right_charges == (1, 0, 0)
    assert payload.microscopic_module_audit.up_right_charges == (5, 2, 0)
    assert payload.microscopic_module_pass
    assert payload.conductor_orientation_rule_pass
    assert payload.up_diagonal_exponents == (8, 4, 0)
    assert payload.down_diagonal_exponents == (4, 2, 0)
    assert payload.up_solver_solutions == ((5, 2, 0),)
    assert payload.down_solver_solutions == ((1, 0, 0),)
    assert payload.right_charges_forced_by_diagonal_targets
    assert payload.ckm_power_exponents == {"V_us": 1, "V_cb": 2, "V_ub": 3}
    assert payload.target_c_candidate_pass
    assert "semigroup derives conductor c=2" in payload.derivation_status


def test_cusp_right_charge_origin_uses_conductor_and_weak_order_not_mass_fits() -> None:
    audit = cusp_right_charge_origin_audit()

    assert isinstance(audit, CuspRightChargeOriginAudit)
    assert audit.left_charges == (3, 2, 0)
    assert audit.conductor == 2
    assert audit.weak_lift_factor == 2
    assert audit.color_lift_control_factor == 3
    assert audit.down_right_charges == (1, 0, 0)
    assert audit.up_right_charges == (5, 2, 0)
    assert audit.down_diagonal_exponents == (4, 2, 0)
    assert audit.up_diagonal_exponents == (8, 4, 0)
    assert audit.trivial_lift_control == (4, 2, 0)
    assert audit.color_lift_control == (12, 6, 0)
    assert audit.trivial_lift_control != audit.up_diagonal_exponents
    assert audit.color_lift_control != audit.up_diagonal_exponents
    assert audit.all_right_charges_nonnegative
    assert not audit.uses_mass_fits
    assert audit.right_charge_origin_pass


def test_target_d_center_holonomy_witness_has_nonzero_cp_invariant() -> None:
    payload = target_d_payload()

    assert sp.simplify(payload.center_phase**3 - 1) == 0
    assert sp.simplify(payload.center_phase - 1) != 0
    assert payload.up_holonomy_rule.name == "up_flag_winding"
    assert payload.down_holonomy_rule.name == "down_bilinear_linking"
    assert payload.holonomy_rule_pair_pass
    assert payload.up_holonomy_rule.center_powers == (
        (0, 1, 2),
        (1, 0, 1),
        (2, 1, 0),
    )
    assert payload.down_holonomy_rule.center_powers == (
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 1),
    )
    assert payload.up_path_measure.center_power_matrix() == (
        (0, 1, 2),
        (1, 0, 1),
        (2, 1, 0),
    )
    assert payload.down_path_measure.center_power_matrix() == (
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 1),
    )
    assert payload.up_coefficient_matrix[0][1] == payload.center_phase
    assert sp.simplify(payload.up_coefficient_matrix[0][2] - payload.center_phase**2) == 0
    assert payload.target_d_witness_pass
    assert sp.simplify(payload.cp_invariant) != 0
    assert payload.cp_invariant_numeric != 0
    assert payload.positive_up_path_measure.center_power_matrix() == (
        (0, 1, 2),
        (1, 0, 1),
        (2, 1, 0),
    )
    assert payload.positive_down_path_measure.center_power_matrix() == (
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 1),
    )
    assert payload.positive_up_amplitudes != ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    assert payload.positive_down_amplitudes != ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    assert all(
        entry.is_positive
        for row in payload.positive_up_amplitudes + payload.positive_down_amplitudes
        for entry in row
    )
    assert sp.simplify(payload.positive_amplitude_cp_invariant) != 0
    assert payload.positive_amplitude_cp_invariant_numeric != 0
    assert sp.simplify(payload.positive_real_control_invariant) == 0
    assert payload.positive_amplitude_witness_pass
    assert isinstance(payload.coefficient_measure_audit, CuspCoefficientMeasureAudit)
    assert payload.coefficient_measure_audit.amplitude_rule == (
        "A_ij=max(1,# decompositions of q_i+r_j in <2,3>)"
    )
    assert payload.derived_up_amplitudes == (
        (sp.Integer(2), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert payload.derived_down_amplitudes == (
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert sp.simplify(payload.derived_measure_cp_invariant) != 0
    assert payload.derived_measure_cp_invariant_numeric != 0
    assert sp.simplify(payload.derived_measure_real_control_invariant) == 0
    assert sp.simplify(
        payload.derived_measure_cp_invariant - payload.derived_measure_rephased_invariant
    ) == 0
    assert payload.derived_measure_pass
    assert payload.derived_measure_topology_selected
    assert isinstance(payload.center_topology_audit, CuspCenterTopologyAudit)
    assert payload.center_topology_finite_selection_pass
    assert payload.center_topology_boundary_selected
    assert isinstance(payload.microscopic_topology_audit, MicroscopicTargetDTopologyAudit)
    assert payload.microscopic_topology_pass
    assert payload.center_topology_up_control_powers == {
        "cyclic_group_difference": cyclic_difference_center_powers(),
        "complete_graph_distance": complete_graph_distance_center_powers(),
        "all_trivial": ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    }
    assert payload.center_topology_down_control_powers == {
        "all_trivial": ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        "additive_row_plus_col": additive_center_control_powers(),
        "cyclic_difference": cyclic_difference_center_powers(),
        "orientation_conjugate": bilinear_center_linking_powers(unit=2),
    }
    assert sp.simplify(payload.real_coefficient_control_invariant) == 0
    assert payload.real_control_rejected
    assert set(payload.holonomy_rule_control_invariants) == {
        "up_rule_down_trivial",
        "up_trivial_down_rule",
        "separable_row_col",
        "row_only_down_rule",
        "up_rule_col_only",
    }
    assert all(
        sp.simplify(invariant) == 0
        for invariant in payload.holonomy_rule_control_invariants.values()
    )
    assert payload.holonomy_rule_controls_rejected
    assert sp.simplify(payload.cp_invariant - payload.rephased_cp_invariant) == 0
    assert payload.rephasing_invariant_pass
    assert sp.simplify(payload.cp_invariant - payload.field_rephased_cp_invariant) == 0
    assert payload.full_field_rephasing_invariant_pass
    assert "derived cusp-module amplitudes" in payload.derivation_status


def test_cusp_center_topology_selects_flag_distance_and_bilinear_pairing() -> None:
    audit = cusp_center_topology_audit()

    assert isinstance(audit, CuspCenterTopologyAudit)
    assert path_flag_distance_center_powers() == (
        (0, 1, 2),
        (1, 0, 1),
        (2, 1, 0),
    )
    assert cyclic_difference_center_powers() == (
        (0, 1, 2),
        (2, 0, 1),
        (1, 2, 0),
    )
    assert complete_graph_distance_center_powers() == (
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    )
    assert bilinear_center_linking_powers(unit=1) == (
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 1),
    )
    assert bilinear_center_linking_powers(unit=2) == (
        (0, 0, 0),
        (0, 2, 1),
        (0, 1, 2),
    )
    assert additive_center_control_powers() == (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
    )
    assert audit.up_center_powers == path_flag_distance_center_powers()
    assert audit.down_center_powers == bilinear_center_linking_powers(unit=1)
    assert audit.up_rule.center_powers == audit.up_center_powers
    assert audit.down_rule.center_powers == audit.down_center_powers
    assert audit.up_controls_rejected
    assert audit.down_controls_rejected_except_conjugate
    assert audit.down_direct_row_col_zero
    assert audit.down_unit_pairing_normalized
    assert audit.finite_topology_selection_pass
    assert audit.boundary_microphysics_selected


def test_cusp_coefficient_measure_uses_module_path_counts() -> None:
    audit = cusp_coefficient_measure_audit()
    semigroup = NumericalSemigroup((2, 3))

    assert isinstance(audit, CuspCoefficientMeasureAudit)
    assert cusp_module_path_count(0, semigroup) == 1
    assert cusp_module_path_count(1, semigroup) == 1
    assert cusp_module_path_count(2, semigroup) == 1
    assert cusp_module_path_count(3, semigroup) == 1
    assert cusp_module_path_count(8, semigroup) == 2
    assert cusp_module_path_amplitudes(((8, 1, 0), (3, 2, 5)), semigroup) == (
        (sp.Integer(2), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert audit.up_amplitudes == (
        (sp.Integer(2), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert audit.down_amplitudes == (
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert audit.all_amplitudes_positive
    assert not audit.uses_fitted_amplitudes
    assert sp.simplify(audit.cp_invariant) != 0
    assert sp.simplify(audit.real_control_invariant) == 0
    assert sp.simplify(audit.cp_invariant - audit.rephased_cp_invariant) == 0
    assert audit.holonomy_topology_selected_by_boundary
    assert audit.coefficient_measure_pass


def test_center_holonomy_path_measure_sums_paths() -> None:
    measure = CoefficientPathMeasure(
        paths=(
            CenterHolonomyPath(row=0, col=0, amplitude=1, center_power=0),
            CenterHolonomyPath(row=0, col=0, amplitude=1, center_power=1),
        )
    )

    matrix = measure.coefficient_matrix()
    assert sp.simplify(matrix[0, 0] - (1 + exact_omega())) == 0
    assert matrix[1, 1] == 0


def test_path_measure_accepts_entrywise_positive_amplitudes() -> None:
    measure = path_measure_from_center_powers(
        ((0, 1, 2), (1, 0, 1), (2, 1, 0)),
        amplitude=((1, 2, 3), (4, 5, 6), (7, 8, 9)),
    )

    matrix = measure.coefficient_matrix()
    assert matrix[0, 0] == 1
    assert sp.simplify(matrix[0, 1] - 2 * exact_omega()) == 0
    assert sp.simplify(matrix[0, 2] - 3 * exact_omega() ** 2) == 0
    assert measure.center_power_matrix() == ((0, 1, 2), (1, 0, 1), (2, 1, 0))


def test_combined_payload_records_current_status_honestly() -> None:
    payload = cusp_targets_payload()

    assert payload.final_verdict == "CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS"
    assert payload.target_a.target_a_pass
    assert payload.target_b.shear_formula_pass
    assert payload.target_c.target_c_candidate_pass
    assert payload.target_d.target_d_witness_pass
    assert payload.target_c.microscopic_module_pass
    assert payload.target_d.microscopic_topology_pass
    assert "Target C proves the SM hypercharge Yukawa doors" in payload.interpretation
    assert "Target D has a finite center-holonomy topology selection" in (
        payload.interpretation
    )
